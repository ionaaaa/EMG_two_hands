"""Realtime data path for the student signal-observation lesson.

The service consumes packets already emitted by ``DeviceCheckService``.  It
does not enumerate devices or own serial sources.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Callable

import numpy as np
from PySide6.QtCore import QObject, Signal

from emg_live_marker.device.protocol import EMG_CHANNELS, EMG_FS
from emg_live_marker.ml.gesture_model import load_model
from emg_live_marker.ml.realtime_decoder import RealtimeGestureDecoder
from emg_live_marker.realtime.ring_buffer import EmgRingBuffer
from emg_live_marker.realtime.stream_processor import StreamingEMGProcessor

STUDENT_DISPLAY_MODES = ("raw", "filtered", "rms")
STUDENT_GESTURES = ("rest", "fist", "open-palm", "pinch")


@dataclass(frozen=True)
class StudentDecoderSettings:
    """Locked decoding values read from the teaching preset."""

    model_path: Path
    confidence_threshold: float
    smoothing_frames: int
    change_confirmations: int

    @classmethod
    def from_config(cls, project_root: Path, config: dict[str, Any]) -> "StudentDecoderSettings":
        realtime = config.get("realtime_decoding", {})
        if not isinstance(realtime, dict):
            realtime = {}
        configured_path = realtime.get("standard_teaching_model_path", "")
        path = Path(str(configured_path))
        if not path.is_absolute():
            path = project_root / path
        return cls(
            model_path=path.resolve(),
            confidence_threshold=float(realtime.get("confidence_threshold", 0.7)),
            smoothing_frames=max(1, int(realtime.get("smoothing_frames", 5))),
            change_confirmations=max(1, int(realtime.get("change_confirmations", 2))),
        )


@dataclass
class _SideRuntime:
    raw: EmgRingBuffer
    filtered: EmgRingBuffer
    rms: EmgRingBuffer
    processor: StreamingEMGProcessor
    decoder: RealtimeGestureDecoder | None = None


class StudentObservationService(QObject):
    """Process already-open device streams for waveform display and decoding."""

    gesture_updated = Signal(str, str, float, dict)
    model_status_changed = Signal(str)

    def __init__(
        self,
        project_root: Path,
        course_config: dict[str, Any],
        *,
        model_loader: Callable[[str | Path], Any] | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.settings = StudentDecoderSettings.from_config(project_root, course_config)
        self._model_loader = model_loader or load_model
        self._predictor: Any | None = None
        self._inference_lock = Lock()
        self._model_load_attempted = False
        self.model_error = ""
        self.active = False
        self.ready_sides = {"left": False, "right": False}
        self._runtime = {
            side: _SideRuntime(
                raw=EmgRingBuffer(seconds=20.0),
                filtered=EmgRingBuffer(seconds=20.0),
                rms=EmgRingBuffer(seconds=20.0),
                processor=StreamingEMGProcessor(),
            )
            for side in ("left", "right")
        }

    @property
    def predictor(self) -> Any | None:
        return self._predictor

    def on_emg_packets(self, side: str, packets: list[object]) -> None:
        """Append packets supplied by ``DeviceCheckService.emg_packets_received``."""

        if side not in self._runtime or not packets:
            return
        try:
            values = np.asarray([packet.values_uv for packet in packets], dtype=np.float64)
            times = np.asarray([packet.t for packet in packets], dtype=np.float64)
            indices = np.asarray([packet.sample_index for packet in packets], dtype=np.int64)
        except (AttributeError, TypeError, ValueError):
            return
        if values.ndim != 2 or values.shape[1] != EMG_CHANNELS:
            return

        runtime = self._runtime[side]
        processed = runtime.processor.process_block(values)
        runtime.raw.append_many(times, processed["raw"], indices)
        runtime.filtered.append_many(times, processed["filtered"], indices)
        runtime.rms.append_many(times, processed["rms"], indices)

    def start(self, *, left_ready: bool, right_ready: bool) -> None:
        self.stop()
        self.ready_sides = {"left": bool(left_ready), "right": bool(right_ready)}
        self.active = True
        if not self._ensure_predictor():
            self.model_status_changed.emit(self.model_error)
            return
        for side, ready in self.ready_sides.items():
            if ready:
                self._start_decoder(side)
        self.model_status_changed.emit("")

    def stop(self) -> None:
        self.active = False
        for runtime in self._runtime.values():
            decoder = runtime.decoder
            runtime.decoder = None
            if decoder is not None:
                decoder.close(wait=True)

    def update_ready_sides(self, *, left_ready: bool, right_ready: bool) -> None:
        """Follow connection loss without touching the device or its serial source."""

        updated = {"left": bool(left_ready), "right": bool(right_ready)}
        if not self.active:
            self.ready_sides = updated
            return
        for side, ready in updated.items():
            runtime = self._runtime[side]
            if not ready and runtime.decoder is not None:
                decoder = runtime.decoder
                runtime.decoder = None
                decoder.close(wait=True)
            elif ready and runtime.decoder is None and self._ensure_predictor():
                self._start_decoder(side)
        self.ready_sides = updated

    def display_window(
        self, side: str, mode: str, seconds: float = 6.0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if side not in self._runtime:
            raise ValueError(f"unsupported side: {side}")
        if mode not in STUDENT_DISPLAY_MODES:
            raise ValueError(f"unsupported student display mode: {mode}")
        return getattr(self._runtime[side], mode).get_window(seconds)

    def _ensure_predictor(self) -> bool:
        if self._predictor is not None:
            return True
        if self._model_load_attempted:
            return False
        self._model_load_attempted = True
        if not self.settings.model_path.is_file():
            self.model_error = "标准识别模型缺失，波形仍可正常观察，请联系老师。"
            return False
        try:
            self._predictor = self._model_loader(self.settings.model_path)
        except Exception:  # noqa: BLE001 - model backends provide heterogeneous errors.
            self.model_error = "标准识别模型加载失败，波形仍可正常观察，请联系老师。"
            return False
        self.model_error = ""
        return True

    def _start_decoder(self, side: str) -> None:
        runtime = self._runtime[side]
        decoder = RealtimeGestureDecoder(
            runtime.filtered,
            raw_emg_buffer=runtime.raw,
            filtered_emg_buffer=runtime.filtered,
            confidence_threshold=self.settings.confidence_threshold,
            smoothing_frames=self.settings.smoothing_frames,
            change_confirmations=self.settings.change_confirmations,
            send_to_game_bridge=False,
            parent=self,
        )
        decoder.set_predictor(self._predictor, self._inference_lock)
        # Some model adapters carry teacher-mode defaults.  The locked course
        # preset is authoritative for this student lesson.
        decoder.set_confidence_threshold(self.settings.confidence_threshold)
        decoder.set_smoothing_frames(self.settings.smoothing_frames)
        decoder.set_change_confirmations(self.settings.change_confirmations)
        decoder.gesture_changed.connect(
            lambda gesture, confidence, probs, selected_side=side: self.gesture_updated.emit(
                selected_side, gesture, confidence, probs
            )
        )
        runtime.decoder = decoder
        decoder.start()

    def decoder_for(self, side: str) -> RealtimeGestureDecoder | None:
        """Expose a side decoder for lifecycle checks and focused tests."""

        return self._runtime[side].decoder
