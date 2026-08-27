import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from emg_live_marker.ml.game_bridge import GameBridge
from emg_live_marker.ml.dual_game_mapper import DualGameMapper
from emg_live_marker.ml.effie_adapter import EffieGestureNet, load_effie_finetuned_model
from emg_live_marker.ml.effie_preprocess import prepare_effie_window, resample_250hz_to_200hz
from emg_live_marker.ml.gesture_model import (
    EMG2PoseGestureTCN,
    EMGGestureNet,
    LABELS,
    DemoGesturePredictor,
    load_model,
)
from emg_live_marker.ml.preprocess import prepare_emg_window, preprocess_emg_window
from emg_live_marker.ml.realtime_decoder import RealtimeGestureDecoder
from emg_live_marker.realtime.ring_buffer import EmgRingBuffer
from emg_live_marker.cli.finetune_effie_gesture import finetune_effie
from emg_live_marker.cli.inspect_gesture_dataset import inspect_dataset
from emg_live_marker.cli.train_gesture_classifier import build_dataset, train_model


def test_preprocess_emg_window_outputs_conv1d_tensor():
    window = np.random.default_rng(1).normal(size=(260, 8)).astype(np.float32)
    window[0, 0] = np.nan
    window[1, 1] = np.inf

    out = preprocess_emg_window(window)

    assert out.shape == (1, 8, 250)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()


def test_prepare_emg_window_uses_training_normalization():
    window = np.ones((260, 8), dtype=np.float32) * 3.0
    mean = [1.0] * 8
    std = [2.0] * 8

    out = prepare_emg_window(window, mean=mean, std=std)

    assert tuple(out.shape) == (1, 8, 250)
    np.testing.assert_allclose(out.numpy(), 1.0, atol=1e-6)


def test_emg_gesture_net_forward_and_predictor_load(tmp_path):
    model = EMGGestureNet()
    logits = model(torch.zeros(2, 8, 250))
    assert logits.shape == (2, 4)

    model_path = tmp_path / "gesture_classifier.pt"
    labels_path = tmp_path / "gesture_labels.json"
    torch.save({"model_state_dict": model.state_dict(), "labels": LABELS}, model_path)
    labels_path.write_text(json.dumps({"labels": LABELS}), encoding="utf-8")

    predictor = load_model(model_path)
    result = predictor.predict_window(np.ones((250, 8), dtype=np.float32))

    assert result["gesture"] in LABELS
    assert 0.0 <= result["confidence"] <= 1.0
    assert set(result["probs"]) == set(LABELS)


def test_emg2pose_tcn_forward_shape():
    model = EMG2PoseGestureTCN()
    logits = model(torch.zeros(2, 8, 250))
    assert logits.shape == (2, 4)


def test_effie_preprocess_and_model_forward():
    window = np.random.default_rng(4).normal(size=(50, 8)).astype(np.float32)
    resampled = resample_250hz_to_200hz(window)
    tensor = prepare_effie_window(window)
    model = EffieGestureNet()

    logits = model(tensor)

    assert resampled.shape[1] == 8
    assert tuple(tensor.shape) == (1, 8, 32, 1)
    assert logits.shape == (1, 4)


def test_game_bridge_throttles_and_maps_low_confidence_to_rest(monkeypatch):
    bridge = GameBridge(min_interval_s=10.0, confidence_threshold=0.65)
    sent: list[tuple[str, float]] = []

    def fake_post(gesture: str, confidence: float) -> None:
        sent.append((gesture, confidence))

    monkeypatch.setattr(bridge, "_post_json", fake_post)

    assert bridge.send_gesture("fist", 0.9) is True
    time.sleep(0.05)
    assert bridge.send_gesture("fist", 0.9) is False
    assert bridge.send_gesture("pinch", 0.4) is True
    bridge.close()

    assert sent[0] == ("fist", 0.9)
    assert sent[1] == ("rest", 0.4)


def test_realtime_decoder_decode_window_uses_demo_predictor():
    buffer = EmgRingBuffer(seconds=2.0)
    decoder = RealtimeGestureDecoder(buffer)
    decoder.predictor = DemoGesturePredictor()

    result = decoder.decode_window(np.zeros((250, 8), dtype=np.float32))

    assert result["gesture"] == "rest"
    assert result["confidence"] >= 0.65
    decoder.close()


class _ShapeCheckingPredictor:
    model_type = "demo"
    signal_type = "filtered"
    normalization_loaded = False
    window_samples = 4
    model_info: dict[str, object] = {}

    def __init__(self) -> None:
        self.shapes: list[tuple[int, ...]] = []

    def predict_window(self, window: np.ndarray) -> dict[str, object]:
        self.shapes.append(tuple(window.shape))
        assert window.ndim == 2
        assert window.shape[1] == 8
        if float(window[0, 0]) == 1.0:
            probs = {"rest": 0.05, "fist": 0.90, "open-palm": 0.03, "pinch": 0.02}
        else:
            probs = {"rest": 0.05, "fist": 0.03, "open-palm": 0.90, "pinch": 0.02}
        gesture = max(probs, key=probs.get)
        return {"gesture": gesture, "confidence": probs[gesture], "probs": probs}


def test_dual_decoders_share_predictor_but_keep_independent_state():
    left_buffer = EmgRingBuffer(seconds=2.0)
    right_buffer = EmgRingBuffer(seconds=2.0)
    left_decoder = RealtimeGestureDecoder(
        left_buffer,
        smoothing_frames=1,
        change_confirmations=1,
        send_to_game_bridge=False,
    )
    right_decoder = RealtimeGestureDecoder(
        right_buffer,
        smoothing_frames=1,
        change_confirmations=1,
        send_to_game_bridge=False,
    )
    predictor = _ShapeCheckingPredictor()
    left_decoder.set_predictor(predictor)
    right_decoder.set_predictor(predictor)

    left = left_decoder.decode_window(np.ones((4, 8), dtype=np.float32))
    right = right_decoder.decode_window(np.full((4, 8), 2.0, dtype=np.float32))

    assert left["gesture"] == "fist"
    assert right["gesture"] == "open-palm"
    assert left_decoder._current_output_gesture == "fist"
    assert right_decoder._current_output_gesture == "open-palm"
    assert predictor.shapes == [(4, 8), (4, 8)]
    left_decoder.close()
    right_decoder.close()


def test_realtime_decoder_pinch_gate_preserves_raw_probs():
    class PinchBiasedPredictor:
        model_type = "demo"
        signal_type = "filtered"
        normalization_loaded = False
        window_samples = 4
        model_info: dict[str, object] = {}

        def predict_window(self, window: np.ndarray) -> dict[str, object]:
            _ = window
            probs = {"rest": 0.10, "fist": 0.35, "open-palm": 0.10, "pinch": 0.45}
            return {"gesture": "pinch", "confidence": 0.45, "probs": probs}

    buffer = EmgRingBuffer(seconds=2.0)
    decoder = RealtimeGestureDecoder(
        buffer,
        confidence_threshold=0.0,
        smoothing_frames=1,
        change_confirmations=1,
        send_to_game_bridge=False,
    )
    decoder.set_predictor(PinchBiasedPredictor())
    decoder.set_pinch_params(pinch_threshold=0.80, pinch_margin=0.10, pinch_boost=0.50)

    result = decoder.decode_window(np.ones((4, 8), dtype=np.float32))

    assert result["gesture"] == "fist"
    assert result["probs"]["pinch"] == 0.45
    decoder.close()


def test_dual_game_mapper_holds_and_releases_independent_side_keys():
    events: list[tuple[str, str]] = []

    class FakeSink:
        def key_down(self, key: str) -> None:
            events.append(("down", key))

        def key_up(self, key: str) -> None:
            events.append(("up", key))

    mapper = DualGameMapper(key_sink=FakeSink())

    mapper.update("fist", "open-palm", enabled=True)
    mapper.update("fist", "open-palm", enabled=True)
    mapper.update("rest", "fist", enabled=True)
    mapper.release_all()

    assert events == [
        ("down", "A"),
        ("down", "W"),
        ("up", "A"),
        ("up", "W"),
        ("down", "Space"),
        ("up", "Space"),
    ]


def test_realtime_decoder_loads_effie_model_defaults(tmp_path):
    model = EffieGestureNet()
    model_path = tmp_path / "gesture_classifier.ts"
    torch.jit.trace(model.eval(), torch.zeros(1, 8, 32, 1)).save(str(model_path))
    (tmp_path / "gesture_labels.json").write_text(json.dumps({"labels": LABELS}), encoding="utf-8")
    (tmp_path / "normalization.json").write_text(
        json.dumps({"mean": [0.0] * 8, "std": [1.0] * 8, "signal_type": "raw"}),
        encoding="utf-8",
    )
    (tmp_path / "model_info.json").write_text(
        json.dumps(
            {
                "model_type": "effie_finetuned",
                "source_fs": 250.0,
                "model_fs": 200.0,
                "window_s": 0.16,
                "source_window_s": 0.5,
                "stride_s": 0.08,
                "labels": LABELS,
                "signal_type": "raw",
            }
        ),
        encoding="utf-8",
    )
    buffer = EmgRingBuffer(seconds=2.0)
    decoder = RealtimeGestureDecoder(buffer, raw_emg_buffer=buffer, filtered_emg_buffer=buffer)

    decoder.load_model(model_path)

    assert decoder.model_type == "effie_finetuned"
    assert decoder.signal_type == "raw"
    assert decoder.window_s == 0.5
    assert decoder.smoothing_frames == 7
    assert decoder.change_confirmations == 3
    decoder.close()


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(header)
        writer.writerows(rows)


def test_training_script_builds_windows_and_saves_model(tmp_path):
    dataset_root = tmp_path / "dataset"
    session_dir = dataset_root / "subject_01" / "session_001"
    session_dir.mkdir(parents=True)
    sample_indices = list(range(1000))
    emg_rows = [
        [idx, f"{idx / 250.0:.6f}", *([float(idx % 17)] * 8)]
        for idx in sample_indices
    ]
    _write_csv(
        session_dir / "emg.csv",
        [
            "sample_index",
            "software_time",
            "ch1_uv",
            "ch2_uv",
            "ch3_uv",
            "ch4_uv",
            "ch5_uv",
            "ch6_uv",
            "ch7_uv",
            "ch8_uv",
        ],
        emg_rows,
    )
    _write_csv(
        session_dir / "events.csv",
        [
            "trial_id",
            "subject_id",
            "session_id",
            "gesture",
            "gesture_name",
            "phase",
            "software_time",
            "sample_index",
            "note",
        ],
        [
            ["0001", "subject_01", "session_001", "fist", "全力握拳", "trial_start", "0", 0, ""],
            ["0001", "subject_01", "session_001", "fist", "全力握拳", "gesture_start", "1", 250, ""],
            ["0001", "subject_01", "session_001", "fist", "全力握拳", "gesture_end", "2.5", 625, ""],
            ["0001", "subject_01", "session_001", "fist", "全力握拳", "trial_end", "4", 1000, ""],
        ],
    )
    (session_dir / "metadata.json").write_text("{}", encoding="utf-8")

    x, y = build_dataset(dataset_root)
    assert x.shape[1:] == (8, 250)
    assert set(y.tolist()) >= {0, 1}

    result = train_model(dataset_root, tmp_path / "models", epochs=1, batch_size=2)
    assert result["samples"] == len(y)
    assert (tmp_path / "models" / "gesture_classifier.pt").exists()
    assert (tmp_path / "models" / "gesture_classifier.ts").exists()
    assert (tmp_path / "models" / "gesture_labels.json").exists()
    assert (tmp_path / "models" / "normalization.json").exists()
    assert (tmp_path / "models" / "model_info.json").exists()
    assert (tmp_path / "models" / "train_report.json").exists()
    assert (tmp_path / "models" / "confusion_matrix.csv").exists()

    predictor = load_model(tmp_path / "models" / "gesture_classifier.ts")
    assert predictor.normalization_loaded is True
    assert predictor.signal_type == "raw"
    assert predictor.predict_window(np.ones((250, 8), dtype=np.float32))["gesture"] in LABELS

    report = inspect_dataset(dataset_root)
    assert report["session_count"] == 1
    assert report["total_trials"] == 1
    assert report["gesture_trial_counts"]["fist"] == 1

    effie_result = finetune_effie(
        dataset_root,
        tmp_path / "effie_models",
        epochs=1,
        batch_size=4,
        val_split="trial",
        device_arg="cpu",
    )
    assert effie_result["model_path"].exists()
    assert effie_result["torchscript_path"].exists()
    assert (tmp_path / "effie_models" / "normalization.json").exists()
    assert (tmp_path / "effie_models" / "model_info.json").exists()
    predictor = load_effie_finetuned_model(tmp_path / "effie_models" / "gesture_classifier.ts")
    assert predictor.predict_window(np.ones((125, 8), dtype=np.float32))["gesture"] in LABELS
