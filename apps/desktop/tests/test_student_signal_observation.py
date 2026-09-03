import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest

from emg_live_marker.device.check_service import (
    CheckReason,
    ConnectionState,
    DeviceCheckResult,
    DeviceCheckService,
    DeviceCheckThresholds,
    SideCheckResult,
)
from emg_live_marker.paths import resolve_project_paths
from emg_live_marker.realtime.student_observation import StudentObservationService
from emg_live_marker.realtime.signal_processing import (
    Y_RANGE_OPTIONS,
    normalize_y_range_option,
    notch_spec_from_option,
)
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


class RawPredictor(Predictor):
    signal_type = "raw"


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


@pytest.mark.parametrize(
    ("option", "expected"),
    [
        ("Off", None),
        ("50Hz", (50.0,)),
        ("50+100Hz", (50.0, 100.0)),
        ("60Hz", (60.0,)),
        ("60+120Hz", (60.0, 120.0)),
    ],
)
def test_observation_processors_share_the_teacher_notch_option(tmp_path, option, expected) -> None:
    model_path = tmp_path / "gesture_classifier.pt"
    model_path.touch()
    service = StudentObservationService(
        tmp_path,
        observation_config(model_path),
        model_loader=lambda _path: Predictor(),
        notch_option=option,
    )
    external_expected = expected[0] if isinstance(expected, tuple) and len(expected) == 1 else expected
    assert notch_spec_from_option(option) == external_expected
    left = service._runtime["left"].processor.config.notch_freq
    right = service._runtime["right"].processor.config.notch_freq
    assert left == expected
    assert right == expected


def test_teacher_display_notch_does_not_change_a_raw_model_input(tmp_path) -> None:
    model_path = tmp_path / "gesture_classifier.pt"
    model_path.touch()
    service = StudentObservationService(
        tmp_path,
        observation_config(model_path),
        model_loader=lambda _path: RawPredictor(),
        notch_option="50+100Hz",
    )
    packets = [Packet(i / 250.0, i, np.full(8, float(i + 1))) for i in range(8)]
    service.on_emg_packets("left", packets)
    service.start(left_ready=True, right_ready=False)
    try:
        decoder = service.decoder_for("left")
        assert decoder is not None
        assert decoder.signal_type == "raw"
        assert decoder._buffer_for_signal_type() is decoder.raw_emg_buffer
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


def test_observation_page_keeps_connected_quality_warning_stream_running(app, tmp_path) -> None:
    model_path = tmp_path / "gesture_classifier.pt"
    model_path.touch()
    service = StudentObservationService(
        tmp_path,
        observation_config(model_path),
        model_loader=lambda _path: Predictor(),
    )
    packets = [Packet(i / 250.0, i, np.full(8, float(i + 1))) for i in range(8)]
    service.on_emg_packets("left", packets)
    service.on_emg_packets("right", packets)
    page = StudentSignalObservationPage(service, lambda: None)
    try:
        page.start(
            left_ready=True,
            right_ready=True,
            left_collection_ready=False,
            right_collection_ready=True,
        )
        page.refresh_waveforms()
        assert "已连接" in page.left_status_label.text()
        assert "信号质量待调整" in page.left_status_label.text()
        assert page.right_status_label.text() == "已连接"
        assert service.decoder_for("left") is not None
        assert service.decoder_for("right") is not None
        assert page.left_waveform_view._curves[0].getData()[0].size > 0

        page.set_observation_sides(
            left_available=False,
            right_available=True,
            left_collection_ready=False,
            right_collection_ready=True,
            left_connected=False,
            right_connected=True,
        )
        assert page.left_status_label.text() == "未连接"
        assert page.right_status_label.text() == "已连接"
        assert service.decoder_for("left") is None
        assert service.decoder_for("right") is not None
    finally:
        page.stop()
        page.close()


@pytest.mark.parametrize("mode", list(Y_RANGE_OPTIONS))
def test_teacher_y_range_is_applied_to_both_student_views_only(app, tmp_path, mode) -> None:
    service = StudentObservationService(
        tmp_path,
        observation_config(tmp_path / "missing.pt"),
    )
    packets = [Packet(0.0, 0, np.ones(8)), Packet(0.004, 1, np.ones(8) * 2)]
    service.on_emg_packets("left", packets)
    before = service.display_window("left", "raw")[1].copy()
    page = StudentSignalObservationPage(service, lambda: None, y_range_mode=mode)
    try:
        expected = Y_RANGE_OPTIONS[normalize_y_range_option(mode)]
        assert page.left_waveform_view._y_range_uv == expected
        assert page.right_waveform_view._y_range_uv == expected
        assert page.left_waveform_view._auto_robust is (mode == "Auto robust")
        assert page.right_waveform_view._auto_robust is (mode == "Auto robust")
        page.refresh_waveforms()
        np.testing.assert_array_equal(service.display_window("left", "raw")[1], before)
    finally:
        page.close()


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


def _observable_but_not_collection_ready_result() -> DeviceCheckResult:
    flat = SideCheckResult(
        "left",
        ConnectionState.CONNECTED,
        CheckReason.FLAT_SIGNAL,
        received_emg=True,
        valid_samples=True,
    )
    missing = SideCheckResult("right", ConnectionState.DISCONNECTED, CheckReason.NO_DEVICE)
    return DeviceCheckResult(flat, missing, checking=False, message="检查完成")


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

        device_service.result = _healthy_single_hand_result()
        device_service.result_changed.emit(device_service.result)
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


def test_observation_accepts_connected_quality_warning_but_collection_stays_gated(app) -> None:
    device_service = DeviceCheckService()
    window = StudentMainWindow(
        paths=resolve_project_paths(),
        device_check_service=device_service,
    )
    try:
        device_service.result = _observable_but_not_collection_ready_result()
        device_service.result_changed.emit(device_service.result)
        app.processEvents()
        signal_entry = next(item for item in window.course_entries if item.identifier == "view-signals")
        collection_entry = next(item for item in window.course_entries if item.identifier == "collect-gestures")
        window.open_course_page(signal_entry)
        assert window._stack.currentWidget() is window.signal_observation_page
        assert "已连接" in window.signal_observation_page.left_status_label.text()
        assert "信号质量待调整" in window.signal_observation_page.left_status_label.text()
        assert window.observation_service.ready_sides["left"] is True

        window.open_course_page(collection_entry)
        assert window._stack.currentWidget() is window.collection_gate_page
    finally:
        window.close()


def test_dual_hand_check_keeps_both_sources_ready_when_observation_opens(app) -> None:
    device_service = DeviceCheckService(
        simulate=True,
        thresholds=DeviceCheckThresholds(
            observe_duration_ms=250,
            min_samples=10,
            min_rate_sps=1.0,
            max_rate_sps=1000.0,
        ),
    )
    device_service.start()
    QTest.qWait(450)
    app.processEvents()
    assert device_service.result.left.ready_for_collection is True
    assert device_service.result.right.ready_for_collection is True

    window = StudentMainWindow(
        paths=resolve_project_paths(),
        device_check_service=device_service,
    )
    try:
        assert window.session_device_result.left.ready_for_collection is True
        assert window.session_device_result.right.ready_for_collection is True
        entry = next(item for item in window.course_entries if item.identifier == "view-signals")
        window.open_course_page(entry)
        assert window.signal_observation_page._ready_sides == {"left": True, "right": True}
        assert len(device_service._sources) == 2
    finally:
        window.close()
