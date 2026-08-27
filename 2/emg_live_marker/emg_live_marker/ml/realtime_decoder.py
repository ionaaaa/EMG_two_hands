"""Qt realtime decoder that runs EMG gesture inference at a fixed stride."""

from __future__ import annotations

from collections import Counter, deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal

from emg_live_marker.ml.game_bridge import GameBridge
from emg_live_marker.ml.gesture_model import DemoGesturePredictor, load_model


class RealtimeGestureDecoder(QObject):
    gesture_changed = Signal(str, float, dict)

    def __init__(
        self,
        emg_buffer: object,
        *,
        raw_emg_buffer: object | None = None,
        filtered_emg_buffer: object | None = None,
        model_path: str | None = None,
        game_bridge: GameBridge | None = None,
        window_s: float = 1.0,
        stride_s: float = 0.1,
        confidence_threshold: float = 0.75,
        smoothing_frames: int = 5,
        change_confirmations: int = 2,
        send_to_game_bridge: bool = True,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.emg_buffer = emg_buffer
        self.raw_emg_buffer = raw_emg_buffer or emg_buffer
        self.filtered_emg_buffer = filtered_emg_buffer or emg_buffer
        self.window_s = float(window_s)
        self.stride_s = float(stride_s)
        self.confidence_threshold = float(confidence_threshold)
        self.smoothing_frames = max(1, int(smoothing_frames))
        self.change_confirmations = max(1, int(change_confirmations))
        self.send_to_game_bridge = bool(send_to_game_bridge)
        self.game_bridge = (
            game_bridge
            if game_bridge is not None
            else (GameBridge(confidence_threshold=confidence_threshold) if self.send_to_game_bridge else None)
        )
        self.predictor: Any = DemoGesturePredictor()
        self._inference_lock: Lock | None = None
        self.demo_mode = True
        self.model_info: dict[str, Any] = {}
        self.model_type = "demo"
        self.signal_type = "filtered"
        self.normalization_loaded = False
        self.pinch_threshold = 0.80
        self.pinch_boost = 0.0
        self.pinch_margin = 0.10
        self.enabled = False
        self._history: deque[str] = deque(maxlen=self.smoothing_frames)
        self._current_output_gesture: str | None = None
        self._candidate_gesture: str | None = None
        self._candidate_count = 0
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="GestureDecoder")
        self._pending: Future | None = None
        self._timer = QTimer(self)
        self._timer.setInterval(max(1, int(round(self.stride_s * 1000.0))))
        self._timer.timeout.connect(self._tick)
        if model_path:
            self.load_model(model_path)

    def load_model(self, path: str | Path) -> None:
        self.set_predictor(load_model(path))

    def set_predictor(self, predictor: Any, inference_lock: Lock | None = None) -> None:
        self.predictor = predictor
        self._inference_lock = inference_lock
        self.demo_mode = False
        self.model_info = dict(getattr(self.predictor, "model_info", {}))
        self.model_type = str(getattr(self.predictor, "model_type", "model"))
        self.signal_type = str(getattr(self.predictor, "signal_type", "filtered"))
        self.normalization_loaded = bool(getattr(self.predictor, "normalization_loaded", False))
        if self.model_type == "effie_finetuned":
            self.signal_type = "raw"
            self.window_s = float(self.model_info.get("source_window_s", 0.5))
            self.stride_s = 0.1
            self.set_confidence_threshold(0.70)
            self.set_smoothing_frames(7)
            self.set_change_confirmations(3)
        else:
            self.window_s = float(self.model_info.get("window_s", self.window_s))
            self.stride_s = float(self.model_info.get("stride_s", self.stride_s))
        self._timer.setInterval(max(1, int(round(self.stride_s * 1000.0))))
        self._history.clear()
        self._current_output_gesture = None
        self._candidate_gesture = None
        self._candidate_count = 0

    def use_demo_mode(self) -> None:
        self.predictor = DemoGesturePredictor()
        self._inference_lock = None
        self.demo_mode = True
        self.model_info = {}
        self.model_type = "demo"
        self.signal_type = "filtered"
        self.normalization_loaded = False
        self._history.clear()
        self._current_output_gesture = None
        self._candidate_gesture = None
        self._candidate_count = 0

    def set_confidence_threshold(self, value: float) -> None:
        self.confidence_threshold = float(value)
        if self.game_bridge is not None:
            self.game_bridge.confidence_threshold = float(value)

    def set_smoothing_frames(self, value: int) -> None:
        self.smoothing_frames = max(1, int(value))
        old_history = list(self._history)[-self.smoothing_frames :]
        self._history = deque(old_history, maxlen=self.smoothing_frames)

    def set_change_confirmations(self, value: int) -> None:
        self.change_confirmations = max(1, int(value))

    def set_pinch_params(
        self,
        *,
        pinch_threshold: float | None = None,
        pinch_boost: float | None = None,
        pinch_margin: float | None = None,
    ) -> None:
        if pinch_threshold is not None:
            self.pinch_threshold = float(pinch_threshold)
        if pinch_boost is not None:
            self.pinch_boost = float(pinch_boost)
        if pinch_margin is not None:
            self.pinch_margin = float(pinch_margin)

    def start(self) -> None:
        self.enabled = True
        if not self._timer.isActive():
            self._timer.start()

    def stop(self) -> None:
        self.enabled = False
        self._timer.stop()
        self._history.clear()
        self._candidate_gesture = None
        self._candidate_count = 0

    def close(self) -> None:
        self.stop()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def decode_window(self, window: np.ndarray) -> dict[str, Any]:
        if self._inference_lock is None:
            prediction = self.predictor.predict_window(window)
        else:
            with self._inference_lock:
                prediction = self.predictor.predict_window(window)
        probs = dict(prediction.get("probs", {}))
        gesture, confidence = self._select_decision_gesture(probs, prediction)
        if confidence < self.confidence_threshold:
            gesture = "rest"
        self._history.append(gesture)
        candidate = Counter(self._history).most_common(1)[0][0]
        smoothed = self._confirm_candidate(candidate)
        if smoothed != gesture:
            confidence = min(confidence, 0.99)
        return {"gesture": smoothed, "confidence": confidence, "probs": probs}

    def _tick(self) -> None:
        if not self.enabled or self._pending is not None:
            return
        buffer = self._buffer_for_signal_type()
        _t, data, _sample_index = buffer.get_window(self.window_s)
        window_samples = int(getattr(self.predictor, "window_samples", round(self.window_s * 250.0)))
        if data.shape[0] < window_samples:
            return
        window = data[-window_samples:, :].copy()
        self._pending = self._executor.submit(self.decode_window, window)
        self._pending.add_done_callback(self._on_prediction_done)

    def _on_prediction_done(self, future: Future) -> None:
        self._pending = None
        if not self.enabled:
            return
        try:
            result = future.result()
        except Exception:
            return
        gesture = str(result["gesture"])
        confidence = float(result["confidence"])
        probs = dict(result.get("probs", {}))
        if self.send_to_game_bridge and self.game_bridge is not None:
            self.game_bridge.send_gesture(gesture, confidence)
        self.gesture_changed.emit(gesture, confidence, probs)

    def _buffer_for_signal_type(self) -> object:
        if self.signal_type == "raw":
            return self.raw_emg_buffer
        return self.filtered_emg_buffer

    def _confirm_candidate(self, candidate: str) -> str:
        if self._current_output_gesture is None:
            self._current_output_gesture = candidate
            self._candidate_gesture = candidate
            self._candidate_count = 1
            return candidate
        if candidate == self._current_output_gesture:
            self._candidate_gesture = candidate
            self._candidate_count = 0
            return candidate
        if candidate == self._candidate_gesture:
            self._candidate_count += 1
        else:
            self._candidate_gesture = candidate
            self._candidate_count = 1
        if self._candidate_count >= self.change_confirmations:
            self._current_output_gesture = candidate
            self._candidate_count = 0
        return self._current_output_gesture

    def _select_decision_gesture(
        self,
        probs: dict[str, float],
        prediction: dict[str, Any],
    ) -> tuple[str, float]:
        if not probs:
            return str(prediction.get("gesture", "rest")), float(prediction.get("confidence", 0.0))

        decision_probs = {label: float(value) for label, value in probs.items()}
        if "pinch" in decision_probs:
            pinch_prob = float(probs.get("pinch", 0.0))
            other_max = max(
                (float(value) for label, value in probs.items() if label != "pinch"),
                default=0.0,
            )
            pinch_allowed = (
                pinch_prob >= self.pinch_threshold
                and (pinch_prob - other_max) >= self.pinch_margin
            )
            if pinch_allowed:
                decision_probs["pinch"] = pinch_prob + self.pinch_boost
            else:
                decision_probs["pinch"] = float("-inf")

        gesture = max(decision_probs, key=decision_probs.get)
        confidence = float(probs.get(gesture, 0.0))
        return gesture, confidence
