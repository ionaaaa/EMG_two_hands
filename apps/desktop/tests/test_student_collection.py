import json
from pathlib import Path

import numpy as np
from PySide6.QtWidgets import QApplication

from emg_live_marker.device.check_service import (
    CheckReason,
    ConnectionState,
    DeviceCheckResult,
    DeviceCheckService,
    SideCheckResult,
)
from emg_live_marker.paths import resolve_project_paths
from emg_live_marker.realtime.collection import CollectionController, CollectionPlan
from emg_live_marker.realtime.classroom_storage import ClassroomStorage
from emg_live_marker.ui.student_window import StudentMainWindow


class FakeRecorder:
    def __init__(self) -> None:
        self.session_dir: Path | None = None
        self.events: list[dict] = []
        self.emg_batches: list[list[object]] = []

    def start(self, *, session_dir: Path, metadata: dict, collection_mode: bool) -> Path:
        assert collection_mode is True
        session_dir.mkdir(parents=True)
        (session_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        self.session_dir = session_dir
        return session_dir

    def stop(self) -> None:
        self.session_dir = None

    def write_emg_packets(self, packets: list[object]) -> None:
        self.emg_batches.append(packets)

    def write_collection_event(self, **event) -> None:
        self.events.append(event)


class FakePacket:
    def __init__(self, index: int, values: np.ndarray) -> None:
        self.sample_index = index
        self.values_uv = values


def plan(trials: int = 5, *, hold_s: float = 0.1) -> CollectionPlan:
    return CollectionPlan(
        subject_id="group_01",
        side="left",
        course_id="yucai",
        trials_per_gesture=trials,
        gestures=("fist", "finger_spread", "thumb_index_pinch"),
        gesture_names={"fist": "全力握拳", "finger_spread": "五指完全张开", "thumb_index_pinch": "拇食两指轻捏"},
        rest_before_s=0.1,
        hold_s=hold_s,
        rest_after_s=0.1,
        randomize=False,
    )


def packets(count: int, value: float | None = None) -> list[FakePacket]:
    rng = np.random.default_rng(3)
    return [FakePacket(index, np.full(8, value) if value is not None else rng.normal(0, 10, 8)) for index in range(count)]


def test_trial_counts_follow_5_10_15_selection(tmp_path) -> None:
    for count in (5, 10, 15):
        controller = CollectionController(recorder=FakeRecorder())
        assert controller.start(plan(count), tmp_path, {"course": {"id": "yucai"}})
        assert len(controller.trials) == count * 3
        controller.end()


def test_pause_freezes_state_and_device_loss_pauses(tmp_path) -> None:
    ready = {"value": True}
    controller = CollectionController(recorder=FakeRecorder(), device_ready=lambda: ready["value"])
    assert controller.start(plan(), tmp_path, {})
    controller.pause()
    assert controller.paused is True
    assert controller.phase == "paused"
    ready["value"] = False
    controller.resume()
    assert controller.paused is True
    controller.end()


def test_repeat_and_invalid_trial_do_not_increase_completed_count(tmp_path) -> None:
    controller = CollectionController(recorder=FakeRecorder())
    assert controller.start(plan(), tmp_path, {})
    controller._timer.stop()
    controller._gesture_start()
    controller.on_emg_packets(packets(2))
    controller._trial_end()
    assert controller.completed_count == 0
    assert any(trial.status == "invalid" for trial in controller.trials)
    controller.repeat_current_or_last()
    assert controller.completed_count == 0
    assert any(trial.status == "repeated" for trial in controller.trials)
    controller.end()


def test_flat_and_nonfinite_samples_are_invalid(tmp_path) -> None:
    for bad_packets in (packets(30, 0.0), [FakePacket(index, np.full(8, np.nan)) for index in range(30)]):
        controller = CollectionController(recorder=FakeRecorder())
        assert controller.start(plan(), tmp_path, {})
        controller._timer.stop()
        controller._gesture_start()
        controller.on_emg_packets(bad_packets)
        assert controller._trial_is_valid() is False
        controller.end()


def test_completed_metadata_and_unique_session_id_are_saved(tmp_path) -> None:
    recorder = FakeRecorder()
    controller = CollectionController(recorder=recorder)
    assert controller.start(plan(), tmp_path, {"course": {"id": "yucai"}})
    session_dir = controller.session_dir
    assert session_dir is not None
    controller.end("partial")
    metadata = json.loads((session_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["anonymous_id"] == "group_01"
    assert metadata["selected_hand"] == "left"
    assert metadata["course_id"] == "yucai"
    assert metadata["collection_status"] == "partial"
    _, next_dir = CollectionController._new_session_target(tmp_path, "group_01")
    assert next_dir != session_dir


def test_new_collection_writes_to_classroom_group_sessions(tmp_path) -> None:
    classroom = ClassroomStorage(
        tmp_path / "data" / "classroom",
        "yucai_2026",
        "yucai",
        app_data_root=tmp_path / "app-data" / "EMGTwoHands",
    )
    controller = CollectionController(
        recorder=FakeRecorder(),
        classroom_storage=classroom,
    )
    assert controller.start(plan(), tmp_path / "legacy-datasets", {})
    assert controller.session_dir.parent == classroom.group_paths("group_01").sessions
    assert not (tmp_path / "legacy-datasets").exists()
    controller.end()


def test_student_page_rejects_bad_anonymous_id_and_locks_single_healthy_side() -> None:
    app = QApplication.instance() or QApplication([])
    service = DeviceCheckService()
    window = StudentMainWindow(paths=resolve_project_paths(), device_check_service=service)
    try:
        healthy = SideCheckResult(
            "left",
            ConnectionState.CONNECTED,
            CheckReason.HEALTHY,
            received_emg=True,
            valid_samples=True,
            signal_healthy=True,
            rate_stable=True,
        )
        unavailable = SideCheckResult("right", ConnectionState.DISCONNECTED, CheckReason.NO_DEVICE)
        service.result_changed.emit(DeviceCheckResult(healthy, unavailable, False, "检查完成"))
        app.processEvents()
        window.open_course_page(window.course_entries[3])
        assert window.collection_page.left_radio.isChecked()
        assert not window.collection_page.right_radio.isEnabled()
        window.collection_page.anonymous_id_edit.setText("../bad")
        window.start_collection()
        assert "只能使用" in window.collection_page.message_label.text()
    finally:
        window.close()
