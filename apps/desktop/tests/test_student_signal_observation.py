import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from emg_live_marker.device.check_service import (
    CheckReason,
    ConnectionState,
    DeviceCheckResult,
    DeviceCheckService,
    SideCheckResult,
)
from emg_live_marker.paths import resolve_project_paths
from emg_live_marker.realtime.student_observation import StudentObservationService
from emg_live_marker.ui.student_pages import StudentSignalObservationPage
from emg_live_marker.ui.student_window import StudentMainWindow
from emg_live_marker.ui.waveform_view import MultiChannelWaveformView


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


def observation_config(model_path: Path) -> dict:
    return {
        "realtime_decoding": {
            "standard_teaching_model_path": str(model_path),
            "confidence_threshold": 0.61,
            "smoothing_frames": 3,
            "change_confirmations": 4,
        }
    }


@dataclass(frozen=True)
class Packet:
    t: float
    sample_index: int
    values_uv: np.ndarray


class Predictor:
    model_type = "test"
    signal_type = "filtered"
    normalization_loaded = True
    window_samples = 4
    model_info = {"window_s": 0.016, "stride_s": 0.1}

    def predict_window(self, _window: np.ndarray) -> dict:
        probs = {"rest": 0.1, "fist": 0.7, "open-palm": 0.15, "pinch": 0.05}
        return {"gesture": "fist", "confidence": 0.7, "probs": probs}


def test_service_reuses_stream_processing_and_loads_one_shared_model(tmp_path) -> None:
    model_path = tmp_path / "gesture_classifier.ts"
    model_path.touch()
    predictor = Predictor()
    loaded: list[Path] = []

    def loader(path: str | Path) -> Predictor:
        loaded.append(Path(path))
        return predictor

    service = StudentObservationService(
        tmp_path,
        observation_config(model_path),
        model_loader=loader,
    )
    packets = [Packet(i / 250.0, i, np.full(8, float(i + 1))) for i in range(8)]
    service.on_emg_packets("left", packets)
    service.start(left_ready=True, right_ready=True)
    try:
        left = service.decoder_for("left")
        right = service.decoder_for("right")
        assert loaded == [model_path]
        assert left is not None and right is not None
        assert left.predictor is predictor and right.predictor is predictor
        assert left.confidence_threshold == 0.61
        assert left.smoothing_frames == 3
        assert left.change_confirmations == 4
        for mode in ("raw", "filtered", "rms"):
            t, data, indices = service.display_window("left", mode)
            assert t.shape == (8,)
            assert data.shape == (8, 8)
            assert indices.tolist() == list(range(8))
        service.update_ready_sides(left_ready=True, right_ready=False)
        assert service.decoder_for("left") is left
        assert service.decoder_for("right") is None
    finally:
        service.stop()
    assert service.decoder_for("left") is None
    assert service.decoder_for("right") is None


def test_missing_model_keeps_waveforms_and_never_starts_demo_decoder(tmp_path) -> None:
    missing = tmp_path / "missing.ts"
    service = StudentObservationService(tmp_path, observation_config(missing))
    packets = [Packet(0.0, 0, np.ones(8)), Packet(0.004, 1, np.ones(8) * 2)]
    service.on_emg_packets("left", packets)
    service.start(left_ready=True, right_ready=False)
    try:
        assert service.decoder_for("left") is None
        assert "模型缺失" in service.model_error
        assert service.display_window("left", "raw")[1].shape == (2, 8)
    finally:
        service.stop()


def test_standard_and_personal_models_switch_on_the_shared_realtime_service(tmp_path) -> None:
    standard_path = tmp_path / "standard.pt"
    personal_path = tmp_path / "personal.pt"
    standard_path.touch()
    personal_path.touch()
    standard = Predictor()
    personal = Predictor()
    loaded = []

    def loader(path):
        loaded.append(Path(path))
        return standard

    service = StudentObservationService(
        tmp_path, observation_config(standard_path), model_loader=loader
    )
    service.start(left_ready=True, right_ready=False)
    first_decoder = service.decoder_for("left")
    try:
        assert service.activate_personal_model(personal_path, personal)
        personal_decoder = service.decoder_for("left")
        assert personal_decoder is not first_decoder
        assert personal_decoder.predictor is personal
        assert service.active_model_path == personal_path.resolve()
        assert service.use_standard_model()
        assert service.decoder_for("left").predictor is standard
        assert loaded == [standard_path]
    finally:
        service.stop()


def test_page_exposes_only_three_student_modes_and_chinese_results(app, tmp_path) -> None:
    service = StudentObservationService(
        tmp_path,
        observation_config(tmp_path / "missing.ts"),
    )
    page = StudentSignalObservationPage(service, lambda: None)
    try:
        assert isinstance(page.left_waveform_view, MultiChannelWaveformView)
        assert isinstance(page.right_waveform_view, MultiChannelWaveformView)
        assert [page.display_mode_combo.itemText(i) for i in range(page.display_mode_combo.count())] == [
            "原始肌电",
            "滤波肌电",
            "肌肉活动强度（RMS）",
        ]
        page.start(left_ready=True, right_ready=False)
        service.gesture_updated.emit(
            "left",
            "open-palm",
            0.8,
            {"rest": 0.05, "fist": 0.1, "open-palm": 0.8, "pinch": 0.05},
        )
        app.processEvents()
        assert page._timer.isActive()
        assert page.right_status_label.text() == "未连接"
        assert "伸掌" in page.left_result_label.text()
        assert all(name in page.left_probability_label.text() for name in ("放松", "握拳", "伸掌", "捏合"))
        assert "模型缺失" in page.model_message_label.text()
    finally:
        page.stop()
        page.close()
    assert not page._timer.isActive()


def _healthy_single_hand_result() -> DeviceCheckResult:
    healthy = SideCheckResult(
        "left",
        ConnectionState.CONNECTED,
        CheckReason.HEALTHY,
        received_emg=True,
        valid_samples=True,
        signal_healthy=True,
        rate_stable=True,
    )
    missing = SideCheckResult("right", ConnectionState.DISCONNECTED, CheckReason.NO_DEVICE)
    return DeviceCheckResult(healthy, missing, checking=False, message="检查完成")


def test_view_signals_gate_allows_one_ready_hand_and_home_stops_runtime(app) -> None:
    device_service = DeviceCheckService()
    window = StudentMainWindow(
        paths=resolve_project_paths(),
        device_check_service=device_service,
    )
    try:
        entry = next(item for item in window.course_entries if item.identifier == "view-signals")
        window.open_course_page(entry)
        assert window._stack.currentWidget() is window.signal_observation_gate_page

        device_service.result_changed.emit(_healthy_single_hand_result())
        app.processEvents()
        window.open_course_page(entry)
        assert window._stack.currentWidget() is window.signal_observation_page
        assert window.signal_observation_page.left_status_label.text() == "已连接"
        assert window.signal_observation_page.right_status_label.text() == "未连接"
        assert window.signal_observation_page._timer.isActive()

        window.show_home()
        assert window._stack.currentWidget() is window.home_page
        assert not window.signal_observation_page._timer.isActive()
        assert not window.observation_service.active
    finally:
        window.close()
