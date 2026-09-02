import csv
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject
from PySide6.QtWidgets import QApplication, QDockWidget

from emg_live_marker.device.check_service import AssignedSerialDeviceProvider
from emg_live_marker.paths import ProjectPaths, resolve_project_paths
from emg_live_marker.realtime.game_mapping import GameMappingService
from emg_live_marker.realtime.classroom_storage import ClassroomStorage
from emg_live_marker.realtime.student_personal_training import StudentPersonalTrainingService
from emg_live_marker.realtime.teacher_classroom import (
    TeacherClassroomService,
    default_classroom_settings_path,
    load_bracelet_assignment,
    merge_classroom_overrides,
)
from emg_live_marker.ui.main_window import MainWindow
import emg_live_marker.ui.main_window as main_window_module
from emg_live_marker.ui.student_window import StudentMainWindow, load_yucai_course_config


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def base_config() -> dict:
    path = Path(__file__).parents[3] / "configs" / "teaching" / "yucai.json"
    return json.loads(path.read_text(encoding="utf-8"))


def project_paths(tmp_path: Path) -> ProjectPaths:
    artifacts = tmp_path / "apps" / "desktop"
    return ProjectPaths(
        project_root=tmp_path,
        dataset_root=tmp_path / "data" / "datasets",
        recordings_root=tmp_path / "data" / "recordings",
        artifacts_root=artifacts,
    )


def make_complete_model(model_dir: Path) -> Path:
    model_dir.mkdir(parents=True)
    model = model_dir / "gesture_classifier.pt"
    model.write_bytes(b"model")
    for name in ("gesture_labels.json", "normalization.json", "model_info.json"):
        (model_dir / name).write_text("{}", encoding="utf-8")
    return model


def make_service(tmp_path, *, settings_path=None):
    paths = project_paths(tmp_path)
    model = make_complete_model(paths.models_root / "standard_v1")
    config = base_config()
    config["realtime_decoding"]["standard_teaching_model_path"] = str(model)
    loader = lambda _path: SimpleNamespace(model_type="validated")
    service = TeacherClassroomService(
        paths,
        config,
        settings_path=settings_path or tmp_path / "app-data" / "classroom_settings.json",
        mapping_storage_root=tmp_path / "app-data" / "student-game-mappings",
        classroom_storage=ClassroomStorage.from_config(
            paths.classroom_root,
            config,
            app_data_root=tmp_path / "app-data" / "EMGTwoHands",
        ),
        model_loader=loader,
    )
    return service, paths, config, model


def test_classroom_settings_persist_outside_yucai_and_merge_into_student_config(tmp_path) -> None:
    settings_path = tmp_path / "app-data" / "classroom_settings.json"
    service, paths, config, model = make_service(tmp_path, settings_path=settings_path)
    saved, message = service.save_settings(
        standard_model_path=model,
        trials_per_action=12,
        personal_training_enabled=False,
    )
    assert saved and "已保存" in message
    payload = json.loads(settings_path.read_text(encoding="utf-8"))
    assert payload["trials_per_action"] == 12
    assert payload["personal_training_enabled"] is False
    assert payload["teacher_password_enabled"] is False
    assert not (paths.project_root / "configs" / "teaching" / "yucai.json").exists()

    merged = merge_classroom_overrides(config, paths.project_root, settings_path=settings_path)
    assert Path(merged["realtime_decoding"]["standard_teaching_model_path"]) == model.resolve()
    assert merged["collection"]["trials_per_action"] == 12
    assert merged["personal_training"]["enabled"] is False


def test_bracelet_assignment_is_machine_local_and_survives_settings_save(tmp_path) -> None:
    settings_path = tmp_path / "app-data" / "classroom_settings.json"
    service, _paths, _config, model = make_service(tmp_path, settings_path=settings_path)
    assert service.save_bracelet_assignment("COM6", "COM7")[0]
    assert load_bracelet_assignment(settings_path) == {
        "left_port": "COM6",
        "right_port": "COM7",
    }
    assert service.save_settings(
        standard_model_path=model,
        trials_per_action=15,
        personal_training_enabled=True,
    )[0]
    payload = json.loads(settings_path.read_text(encoding="utf-8"))
    assert payload["bracelet_assignment"] == {
        "left_port": "COM6",
        "right_port": "COM7",
    }


def test_bracelet_assignment_rejects_same_or_empty_port(tmp_path) -> None:
    service, _paths, _config, _model = make_service(tmp_path)
    assert service.save_bracelet_assignment("COM6", "COM6")[0] is False
    assert "不能使用同一个端口" in service.save_bracelet_assignment("COM6", "COM6")[1]
    assert service.save_bracelet_assignment("", "COM7")[0] is False
    assert service.bracelet_assignment is None


def test_student_window_uses_saved_machine_bracelet_assignment(app, tmp_path) -> None:
    settings_path = tmp_path / "app-data" / "classroom_settings.json"
    service, _paths, _config, _model = make_service(tmp_path, settings_path=settings_path)
    assert service.save_bracelet_assignment("COM6", "COM7")[0]

    window = StudentMainWindow(
        paths=resolve_project_paths(),
        classroom_settings_path=settings_path,
    )
    try:
        provider = window.device_check_service.provider
        assert isinstance(provider, AssignedSerialDeviceProvider)
        assert provider.left_port == "COM6"
        assert provider.right_port == "COM7"
    finally:
        window.close()


def test_default_teacher_settings_use_application_specific_directory() -> None:
    path = default_classroom_settings_path()
    assert path.name == "classroom_settings.json"
    assert path.parent.name == "EMGTwoHands"


def test_student_config_loader_applies_classroom_override(tmp_path) -> None:
    paths = resolve_project_paths(project_root=tmp_path, environ={})
    config_path = tmp_path / "configs" / "teaching" / "yucai.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps(base_config()), encoding="utf-8")
    settings = tmp_path / "classroom_settings.json"
    settings.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "standard_model_path": "models/lesson-v2/gesture_classifier.pt",
                "trials_per_action": 15,
                "personal_training_enabled": False,
            }
        ),
        encoding="utf-8",
    )
    loaded = load_yucai_course_config(paths, classroom_settings_path=settings)
    assert loaded["collection"]["trials_per_action"] == 15
    assert loaded["personal_training"]["enabled"] is False
    assert loaded["realtime_decoding"]["standard_teaching_model_path"] == str(
        (tmp_path / "models" / "lesson-v2" / "gesture_classifier.pt").resolve()
    )


class FakeObservation(QObject):
    def __init__(self, model_path: Path):
        super().__init__()
        self.settings = SimpleNamespace(model_path=model_path)
        self.model_source = "standard"


def test_personal_training_service_rejects_disabled_course(tmp_path) -> None:
    model = tmp_path / "standard.pt"
    model.write_bytes(b"standard")
    observation = FakeObservation(model)
    service = StudentPersonalTrainingService(
        tmp_path,
        tmp_path / "datasets",
        {"personal_training": {"enabled": False}},
        observation,
    )
    assert service.training_enabled is False
    assert service.start_training(None) is False
    assert service.snapshot.state == "disabled"
    assert "老师已关闭" in service.snapshot.message


def make_session(dataset_root: Path, *, classroom: bool = False) -> Path:
    session = (
        dataset_root / "session_01"
        if classroom
        else dataset_root / "group_01" / "session_01"
    )
    session.mkdir(parents=True)
    metadata = {
        "anonymous_id": "group_01",
        "session_id": "session_01",
        "selected_hand": "left",
        "collection_status": "completed",
        "valid_trial_counts": {"fist": 5, "finger_spread": 6, "thumb_index_pinch": 7},
        "trial_statuses": {"1": "completed", "2": "invalid", "3": "repeated"},
    }
    (session / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (session / "emg.csv").write_text("sample_index\n", encoding="utf-8")
    return session


def test_session_scan_recollect_marker_and_safe_deletion(tmp_path) -> None:
    service, paths, _config, _model = make_service(tmp_path)
    session = make_session(
        service.classroom_storage.group_paths("group_01", create=True).sessions,
        classroom=True,
    )
    scanned = service.scan_sessions()
    assert len(scanned) == 1
    assert scanned[0].student_id == "group_01"
    assert scanned[0].hand == "left"
    assert scanned[0].valid_counts == {
        "fist": 5,
        "finger_spread": 6,
        "thumb_index_pinch": 7,
    }
    assert scanned[0].invalid_count == 1
    assert scanned[0].repeated_count == 1

    marked, message = service.mark_recollect(session)
    assert marked and "旧数据仍然保留" in message
    assert session.is_dir()
    assert service.scan_sessions()[0].recollect_requested
    assert service.delete_session(session, confirmed=False)[0] is False
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "metadata.json").write_text("{}", encoding="utf-8")
    assert service.delete_session(outside, confirmed=True)[0] is False
    assert outside.is_dir()
    assert service.delete_session(session, confirmed=True)[0] is True
    assert not session.exists()


def test_standard_scan_requires_complete_loadable_model(tmp_path) -> None:
    service, paths, _config, model = make_service(tmp_path)
    incomplete = paths.models_root / "incomplete"
    incomplete.mkdir()
    (incomplete / "gesture_classifier.pt").write_bytes(b"bad")
    invalid = make_complete_model(paths.models_root / "invalid")

    def loader(path):
        if Path(path) == invalid:
            raise ValueError("bad checkpoint")
        return SimpleNamespace(model_type="real")

    service._model_loader = loader
    models = service.scan_standard_models()
    assert [item.path for item in models] == [model.resolve()]


def test_personal_model_report_and_confirmed_delete_never_delete_standard(tmp_path) -> None:
    service, paths, _config, standard = make_service(tmp_path)
    personal = service.classroom_storage.model_path("group_01", "run_01", create=True)
    (personal / "gesture_classifier.pt").write_bytes(b"personal")
    (personal / "personal_model.valid.json").write_text(
        json.dumps({"anonymous_group_id": "group_01", "validated_at": "2026-09-01T10:00:00Z"}),
        encoding="utf-8",
    )
    (personal / "train_report.json").write_text(json.dumps({"val_accuracy": 0.88}), encoding="utf-8")
    records = service.scan_personal_models()
    assert len(records) == 1
    assert records[0].validation_accuracy == pytest.approx(0.88)
    assert records[0].trained_at == "2026-09-01T10:00:00Z"
    assert service.delete_personal_model(personal, confirmed=False)[0] is False
    assert personal.exists()
    assert service.delete_personal_model(standard.parent, confirmed=True)[0] is False
    assert standard.is_file()
    assert service.delete_personal_model(personal, confirmed=True)[0] is True
    assert not personal.exists()


def competition_record(student_id="group_01", score=1234):
    return {
        "student_id": student_id,
        "model_type": "standard",
        "mapping": {"rest": "none", "fist": "A", "open-palm": "B", "pinch": "none"},
        "sensitivity": "standard",
        "stability": "balanced",
        "mode": "left",
        "score": score,
        "accuracy": 0.8,
        "max_combo": 12,
        "outcome": "completed",
        "timestamp": "2026-09-01T10:00:00+00:00",
    }


def test_competition_summary_and_csv_preserve_json(tmp_path) -> None:
    service, paths, _config, _model = make_service(tmp_path)
    result_dir = paths.reports_root / "teaching" / "yucai" / "round_01" / "group_01"
    result_dir.mkdir(parents=True)
    result_path = result_dir / "result.json"
    result_path.write_text(json.dumps(competition_record()), encoding="utf-8")
    records = service.scan_competition_results()
    assert len(records) == 1
    output = tmp_path / "export" / "results.csv"
    exported, message = service.export_competition_csv(output, student_id="group_01")
    assert exported and "1 条" in message
    with output.open(newline="", encoding="utf-8-sig") as file_obj:
        rows = list(csv.DictReader(file_obj))
    assert list(rows[0]) == [
        "student_id", "model_type", "mapping", "sensitivity", "stability", "mode",
        "score", "accuracy", "max_combo", "outcome", "timestamp",
    ]
    assert rows[0]["student_id"] == "group_01"
    assert rows[0]["score"] == "1234"
    assert result_path.is_file()


def test_prepare_next_group_only_clears_current_pointer_and_emits_reset(tmp_path) -> None:
    service, paths, _config, _model = make_service(tmp_path)
    service.mapping_storage_root.mkdir(parents=True)
    service.classroom_storage.set_active_group("group_01")
    current = service.classroom_storage.active_group_path
    saved_group = service.mapping_storage_root / "group_01.json"
    current.write_text("{}", encoding="utf-8")
    saved_group.write_text("{}", encoding="utf-8")
    session = make_session(paths.dataset_root)
    resets = []
    service.runtime_reset_requested.connect(lambda: resets.append(True))
    ok, _message = service.prepare_next_group()
    assert ok and resets == [True]
    assert not current.exists()
    assert saved_group.is_file()
    assert session.is_dir()


def test_device_diagnostics_reads_existing_runtime_counters_directly(tmp_path) -> None:
    service, _paths, _config, _model = make_service(tmp_path)
    runtime = SimpleNamespace(
        connected=True,
        emg_rate_sps=249.0,
        imu_rate_sps=50.0,
        aa_lost_count=2,
        bb_lost_count=3,
        global_lost_count=5,
        bad_header_count=7,
        bad_type_count=11,
        resync_count=13,
    )
    values = service.device_diagnostics({"left": runtime})["left"]
    assert values == {
        "connected": True,
        "emg_rate_sps": 249.0,
        "imu_rate_sps": 50.0,
        "aa_lost_count": 2,
        "bb_lost_count": 3,
        "global_lost_count": 5,
        "bad_header_count": 7,
        "bad_type_count": 11,
        "resync_count": 13,
    }


def test_classroom_dock_is_read_only_mirror_of_mainwindow_device_controls(app) -> None:
    window = MainWindow(simulate=False, paths=resolve_project_paths())
    try:
        assert type(window) is MainWindow
        dock = window._classroom_dock
        assert isinstance(dock, QDockWidget)
        assert dock.windowTitle() == "课堂管理"
        assert not hasattr(dock, "left_port_combo")
        assert not hasattr(dock, "right_port_combo")
        assert not hasattr(dock, "left_assign_button")
        assert not hasattr(dock, "right_assign_button")
        assert not hasattr(dock, "_assign_port")
        assert not hasattr(window._teacher_classroom_service, "serial_source")
        assert "Left" in [label.text() for label in window.findChildren(type(dock.left_device_status_label))]
    finally:
        window.close()


def test_classroom_device_status_tracks_mainwindow_ports_connections_and_refresh(app, monkeypatch) -> None:
    ports = ["COM6", "COM12", "COM13"]
    saved_assignments: list[tuple[str, str]] = []

    def save_assignment(_service, left_port: str, right_port: str) -> tuple[bool, str]:
        saved_assignments.append((left_port, right_port))
        return True, ""

    monkeypatch.setattr(main_window_module, "list_serial_ports", lambda: list(ports))
    monkeypatch.setattr(MainWindow, "_load_game_model", lambda _self: None)
    monkeypatch.setattr(TeacherClassroomService, "save_bracelet_assignment", save_assignment)
    window = MainWindow(simulate=False, paths=resolve_project_paths())
    try:
        dock = window._classroom_dock
        window._port_combo.setCurrentText("COM6")
        window._right_port_combo.setCurrentText("COM13")
        assert saved_assignments[-1] == ("COM6", "COM13")
        assert dock.left_device_status_label.text() == "COM6 · 未连接"
        assert dock.right_device_status_label.text() == "COM13 · 未连接"

        window._set_connected_ui_for_side("left", True)
        window._set_connected_ui_for_side("right", True)
        assert dock.left_device_status_label.text() == "COM6 · 已连接"
        assert dock.right_device_status_label.text() == "COM13 · 已连接"
        window._set_connected_ui_for_side("right", False)
        assert dock.right_device_status_label.text() == "COM13 · 未连接"

        window._set_connected_ui_for_side("left", False)
        ports[:] = ["COM6", "COM7"]
        save_count_before_refresh = len(saved_assignments)
        window._refresh_ports_button.click()

        assert [window._port_combo.itemText(index) for index in range(window._port_combo.count())] == ports
        assert [window._right_port_combo.itemText(index) for index in range(window._right_port_combo.count())] == ports
        assert window._port_combo.currentText() == "COM6"
        assert dock.left_device_status_label.text() == "COM6 · 未连接"
        assert window._right_port_combo.currentText() == ""
        assert dock.right_device_status_label.text() == "未选择端口 · 未连接"
        assert len(saved_assignments) == save_count_before_refresh

        window._right_port_combo.setCurrentText("COM7")
        assert saved_assignments[-1] == ("COM6", "COM7")
        save_count_before_dock_refresh = len(saved_assignments)
        assert dock.right_device_status_label.text() == "COM7 · 未连接"
        dock.refresh_ports_button.click()
        assert window._right_port_combo.currentText() == "COM7"
        assert len(saved_assignments) == save_count_before_dock_refresh
        assert "主窗口 Left / Right" in dock.ports_message.text()
    finally:
        window.close()


def test_student_next_group_reset_restores_defaults_without_deleting_artifacts(app, tmp_path) -> None:
    artifact = tmp_path / "keep.json"
    artifact.write_text("{}", encoding="utf-8")
    window = StudentMainWindow(paths=resolve_project_paths())
    try:
        assert window.collection_controller.classroom_storage is window.classroom_storage
        assert window.game_mapping_service.classroom_storage is window.classroom_storage
        assert window.personal_training_service.classroom_storage is window.classroom_storage
        assert window.competition_service.classroom_storage is window.classroom_storage
        window.collection_page.anonymous_id_edit.setText("group_old")
        window.game_mapping_service.swap_commands()
        window.game_mapping_service.set_control_preferences("high", "stable")
        window.observation_service.apply_control_profile("high", "stable")
        window.prepare_next_group()
        assert window.collection_page.anonymous_id_edit.text() == ""
        assert window.game_mapping_service.resolved_mapping == window.game_mapping_service.default_mapping
        assert window.observation_service.model_source == "standard"
        assert window.observation_service.control_profile == {
            "sensitivity": "standard",
            "control_style": "balanced",
        }
        assert artifact.is_file()
    finally:
        window.close()
