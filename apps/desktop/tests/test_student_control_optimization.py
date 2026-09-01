import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, Signal

from emg_live_marker.realtime.game_mapping import GameMappingService
from emg_live_marker.realtime.student_control_optimization import (
    StudentControlEffectTestService,
)
from emg_live_marker.realtime.student_observation import StudentObservationService


def yucai_config() -> dict:
    path = Path(__file__).parents[3] / "configs" / "teaching" / "yucai.json"
    return json.loads(path.read_text(encoding="utf-8"))


class Predictor:
    model_type = "effie_finetuned"
    signal_type = "raw"
    normalization_loaded = True
    window_samples = 100
    model_info = {"source_window_s": 0.5}

    def predict_window(self, _window):
        return {
            "gesture": "rest",
            "confidence": 1.0,
            "probs": {"rest": 1.0, "fist": 0.0, "open-palm": 0.0, "pinch": 0.0},
        }


class FakeObservation(QObject):
    gesture_updated = Signal(str, str, float, dict)

    def __init__(self, *, left=True, right=False, active=True):
        super().__init__()
        self.ready_sides = {"left": left, "right": right}
        self.active = active
        self.start_calls = []

    def start(self, *, left_ready, right_ready):
        self.active = True
        self.ready_sides = {"left": left_ready, "right": right_ready}
        self.start_calls.append((left_ready, right_ready))


def test_yucai_control_profile_mapping_is_exact() -> None:
    profiles = yucai_config()["student_control_profiles"]
    assert {
        name: value["confidence_threshold"] for name, value in profiles["sensitivity"].items()
    } == {"low": 0.80, "standard": 0.70, "high": 0.60}
    assert {
        name: (value["smoothing_frames"], value["change_confirmations"])
        for name, value in profiles["control_style"].items()
    } == {"fast": (3, 1), "balanced": (5, 2), "stable": (7, 3)}
    assert profiles["default"] == {
        "sensitivity": "standard",
        "control_style": "balanced",
    }


def test_profile_updates_both_decoders_and_survives_model_switch(tmp_path) -> None:
    standard_path = tmp_path / "standard.pt"
    personal_path = tmp_path / "personal.pt"
    standard_path.touch()
    personal_path.touch()
    standard = Predictor()
    personal = Predictor()
    config = yucai_config()
    config["realtime_decoding"]["standard_teaching_model_path"] = str(standard_path)
    service = StudentObservationService(tmp_path, config, model_loader=lambda _path: standard)
    service.start(left_ready=True, right_ready=True)
    try:
        assert service.apply_control_profile("high", "stable")
        for side in ("left", "right"):
            decoder = service.decoder_for(side)
            assert decoder.confidence_threshold == pytest.approx(0.60)
            assert decoder.smoothing_frames == 7
            assert decoder.change_confirmations == 3

        assert service.activate_personal_model(personal_path, personal)
        assert service.control_profile == {"sensitivity": "high", "control_style": "stable"}
        for side in ("left", "right"):
            decoder = service.decoder_for(side)
            assert decoder.predictor is personal
            assert decoder.confidence_threshold == pytest.approx(0.60)
            assert decoder.smoothing_frames == 7
            assert decoder.change_confirmations == 3
        assert service.use_standard_model()
        assert service.decoder_for("left").confidence_threshold == pytest.approx(0.60)
    finally:
        service.stop()


def test_group_file_persists_only_semantic_control_names(tmp_path) -> None:
    config = yucai_config()
    storage = tmp_path / "group-settings"
    service = GameMappingService(config, storage_root=storage)
    assert service.set_control_preferences("high", "fast")
    saved, _message = service.save_current_group("group_07")
    assert saved
    payload = json.loads((storage / "group_07.json").read_text(encoding="utf-8"))
    assert payload["control_profile"] == {"sensitivity": "high", "control_style": "fast"}
    serialized = json.dumps(payload["control_profile"])
    assert "threshold" not in serialized
    assert "smoothing" not in serialized
    assert "confirmations" not in serialized

    restored = GameMappingService(config, storage_root=storage)
    assert restored.current_group_id == "group_07"
    assert restored.control_preferences == {"sensitivity": "high", "control_style": "fast"}


def test_control_metrics_use_real_stable_transitions_and_single_hand() -> None:
    now = [0.0]
    observation = FakeObservation(left=True, right=False, active=False)
    service = StudentControlEffectTestService(
        observation, phase_duration_ms=60_000, clock=lambda: now[0]
    )
    assert service.start()
    assert observation.start_calls == [(True, False)]

    now[0] = 0.1
    observation.gesture_updated.emit("left", "rest", 0.9, {})
    now[0] = 0.2
    observation.gesture_updated.emit("left", "fist", 0.9, {})
    observation.gesture_updated.emit("left", "fist", 0.9, {})
    service.advance_phase()

    now[0] = 0.3
    observation.gesture_updated.emit("left", "rest", 0.9, {})
    now[0] = 0.7
    observation.gesture_updated.emit("left", "fist", 0.9, {})
    now[0] = 0.8
    observation.gesture_updated.emit("left", "open-palm", 0.9, {})
    service.advance_phase()

    now[0] = 1.0
    observation.gesture_updated.emit("left", "rest", 0.9, {})
    service.advance_phase()

    now[0] = 1.2
    observation.gesture_updated.emit("left", "open-palm", 0.9, {})
    service.advance_phase()

    result = service.result
    assert result.has_data
    assert result.false_triggers == 1
    assert result.wrong_switches == 2
    assert result.average_response_seconds == pytest.approx(0.35)
    assert result.stability_level == "较稳定"


def test_no_ready_hand_is_rejected_and_no_events_produce_no_result() -> None:
    missing = FakeObservation(left=False, right=False)
    blocked = StudentControlEffectTestService(missing)
    assert blocked.start() is False

    observation = FakeObservation(left=False, right=True)
    service = StudentControlEffectTestService(observation, phase_duration_ms=60_000)
    assert service.start()
    for _ in range(4):
        service.advance_phase()
    assert service.result.has_data is False
    assert service.result.false_triggers is None
    assert service.result.average_response_seconds is None
    assert service.result.wrong_switches is None
    assert service.result.stability_level == "暂无结果"
