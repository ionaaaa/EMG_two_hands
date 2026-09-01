import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

from emg_live_marker.device.check_service import (
    CheckReason,
    ConnectionState,
    DeviceCheckResult,
    SideCheckResult,
)
from emg_live_marker.ml.gesture_model import DemoGesturePredictor, load_model
from emg_live_marker.paths import resolve_project_paths
from emg_live_marker.realtime.student_game_experience import StudentGameExperienceService
from emg_live_marker.realtime.student_observation import StudentObservationService
from emg_live_marker.ui.student_pages import StudentSignalObservationPage
from emg_live_marker.ui.student_window import load_yucai_course_config


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _standard_model_path() -> Path:
    paths = resolve_project_paths()
    return (
        paths.project_root
        / "apps"
        / "desktop"
        / "models"
        / "effie_real_full_v2_continue"
        / "gesture_classifier.pt"
    )


def test_yucai_standard_model_path_uses_checkpoint_file() -> None:
    config = load_yucai_course_config(resolve_project_paths())
    assert config["realtime_decoding"]["standard_teaching_model_path"] == (
        "apps/desktop/models/effie_real_full_v2_continue/gesture_classifier.pt"
    )


@pytest.mark.skipif(not _standard_model_path().is_file(), reason="local ignored teaching model is absent")
def test_standard_model_directory_and_real_predictor_metadata() -> None:
    model_path = _standard_model_path()
    for filename in (
        "gesture_classifier.pt",
        "model_info.json",
        "gesture_labels.json",
        "normalization.json",
    ):
        assert (model_path.parent / filename).is_file()

    predictor = load_model(model_path)

    assert not isinstance(predictor, DemoGesturePredictor)
    assert predictor.model_type == "effie_finetuned"
    assert predictor.labels == ["rest", "fist", "open-palm", "pinch"]
    assert predictor.signal_type == "raw"
    assert predictor.normalization_loaded is True
    assert predictor.window_samples == 125


class _PredictorStub:
    model_type = "standard-test"
    signal_type = "raw"
    normalization_loaded = True
    window_samples = 4
    model_info: dict[str, object] = {}

    def predict_window(self, _window):
        probabilities = {"rest": 1.0, "fist": 0.0, "open-palm": 0.0, "pinch": 0.0}
        return {"gesture": "rest", "confidence": 1.0, "probs": probabilities}


def _ready_result() -> DeviceCheckResult:
    left = SideCheckResult(
        "left",
        ConnectionState.CONNECTED,
        CheckReason.HEALTHY,
        received_emg=True,
        valid_samples=True,
        signal_healthy=True,
        rate_stable=True,
    )
    right = SideCheckResult("right", ConnectionState.DISCONNECTED, CheckReason.NO_DEVICE)
    return DeviceCheckResult(left, right, checking=False, message="检查完成")


class _ReadyDevice(QObject):
    result_changed = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.result = _ready_result()

    def start(self) -> None:
        raise AssertionError("ready device should be reused")

    def stop(self) -> None:
        pass


class _ConnectedGestureServer:
    def __init__(self) -> None:
        self.clients = 0

    @property
    def client_count(self) -> int:
        return self.clients

    def start(self) -> None:
        pass

    def stop(self) -> None:
        self.clients = 0

    def publish(self, _event: str, _payload: dict) -> None:
        pass


@pytest.mark.skipif(not _standard_model_path().is_file(), reason="local ignored teaching model is absent")
def test_observation_and_quick_experience_share_new_pt_path_and_single_load(app, tmp_path) -> None:
    paths = resolve_project_paths()
    config = load_yucai_course_config(paths)
    configured = config["realtime_decoding"]["standard_teaching_model_path"]
    assert configured == (
        "apps/desktop/models/effie_real_full_v2_continue/gesture_classifier.pt"
    )
    expected_path = _standard_model_path().resolve()
    loaded: list[Path] = []
    predictor = _PredictorStub()

    def loader(path: str | Path):
        loaded.append(Path(path).resolve())
        return predictor

    observation = StudentObservationService(
        paths.project_root,
        config,
        model_loader=loader,
    )
    page = StudentSignalObservationPage(observation, lambda: None)
    gesture_server = _ConnectedGestureServer()
    web_root = tmp_path / "web-game"
    web_root.mkdir()
    (web_root / "index.html").write_text("<title>game</title>", encoding="utf-8")

    try:
        page.start(left_ready=True, right_ready=False)
        assert observation.settings.model_path == expected_path
        assert loaded == [expected_path]
        assert observation.predictor is predictor
        page.stop()

        def opener(_url) -> bool:
            gesture_server.clients = 1
            return True

        experience = StudentGameExperienceService(
            _ReadyDevice(),
            observation,
            web_root,
            gesture_server=gesture_server,
            browser_opener=opener,
            client_wait_timeout_ms=500,
        )
        try:
            experience.start_experience()
            app.processEvents()
            assert experience.running
            assert experience.observation_service is observation
            assert loaded == [expected_path]
            assert observation.decoder_for("left") is not None
        finally:
            experience.stop()
    finally:
        page.stop()
        page.close()
