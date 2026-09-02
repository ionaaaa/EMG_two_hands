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
from emg_live_marker.realtime.signal_processing import (
    normalize_notch_option,
    notch_spec_from_option,
)
from emg_live_marker.realtime.stream_processor import StreamingEMGProcessor

STUDENT_DISPLAY_MODES = ("raw", "filtered", "rms")
STUDENT_GESTURES = ("rest", "fist", "open-palm", "pinch")
STUDENT_SENSITIVITIES = ("low", "standard", "high")
STUDENT_CONTROL_STYLES = ("fast", "balanced", "stable")


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
    model_source_changed = Signal(str, str)
    control_profile_changed = Signal(str, str)

    def __init__(
        self,
        project_root: Path,
        course_config: dict[str, Any],
        *,
        model_loader: Callable[[str | Path], Any] | None = None,
        notch_option: object = "Off",
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.settings = StudentDecoderSettings.from_config(project_root, course_config)
        self._control_profiles, defaults = self._parse_control_profiles(course_config)
        self.current_sensitivity = defaults["sensitivity"]
        self.current_control_style = defaults["control_style"]
        self._model_loader = model_loader or load_model
        self._predictor: Any | None = None
        self._standard_predictor: Any | None = None
        self._personal_predictor: Any | None = None
        self._personal_model_path: Path | None = None
        self.model_source = "standard"
        self._inference_lock = Lock()
        self._model_load_attempted = False
        self.model_error = ""
        self.active = False
        self.notch_option = normalize_notch_option(notch_option)
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
        notch = notch_spec_from_option(self.notch_option)
        for runtime in self._runtime.values():
            runtime.processor.update_config(notch_freq=notch)

    @property
    def predictor(self) -> Any | None:
        return self._predictor

    @property
    def personal_model_available(self) -> bool:
        return self._personal_predictor is not None and self._personal_model_path is not None

    @property
    def active_model_path(self) -> Path:
        if self.model_source == "personal" and self._personal_model_path is not None:
            return self._personal_model_path
        return self.settings.model_path

    @property
    def control_profile(self) -> dict[str, str]:
        return {
            "sensitivity": self.current_sensitivity,
            "control_style": self.current_control_style,
        }

    def apply_control_profile(self, sensitivity: str, control_style: str) -> bool:
        """Apply a validated student profile to every live decoder immediately."""

        sensitivity = str(sensitivity)
        control_style = str(control_style)
        if sensitivity not in STUDENT_SENSITIVITIES or control_style not in STUDENT_CONTROL_STYLES:
            return False
        if sensitivity not in self._control_profiles["sensitivity"]:
            return False
        if control_style not in self._control_profiles["control_style"]:
            return False
        if (
            sensitivity == self.current_sensitivity
            and control_style == self.current_control_style
        ):
            return True
        self.current_sensitivity = sensitivity
        self.current_control_style = control_style
        for runtime in self._runtime.values():
            if runtime.decoder is not None:
                self._apply_profile_to_decoder(runtime.decoder)
        self.control_profile_changed.emit(sensitivity, control_style)
        return True

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

    def activate_personal_model(self, path: str | Path, predictor: Any | None = None) -> bool:
        model_path = Path(path)
        if model_path.is_dir():
            model_path = model_path / "gesture_classifier.pt"
        try:
            loaded = predictor if predictor is not None else self._model_loader(model_path)
        except Exception:  # noqa: BLE001 - callers receive a student-safe failure.
            self.model_status_changed.emit("个人模型加载失败，继续使用训练前模型。")
            return False
        self._personal_predictor = loaded
        self._personal_model_path = model_path.resolve()
        self._switch_predictor("personal", loaded)
        return True

    def use_standard_model(self) -> bool:
        if not self._ensure_standard_predictor():
            self.model_status_changed.emit(self.model_error)
            return False
        self._switch_predictor("standard", self._standard_predictor)
        return True

    def use_personal_model(self) -> bool:
        if not self.personal_model_available:
            self.model_status_changed.emit("尚无有效的个人模型，请先完成训练。")
            return False
        self._switch_predictor("personal", self._personal_predictor)
        return True

    def _ensure_predictor(self) -> bool:
        if self.model_source == "personal" and self._personal_predictor is not None:
            self._predictor = self._personal_predictor
            return True
        return self._ensure_standard_predictor()

    def _ensure_standard_predictor(self) -> bool:
        if self._standard_predictor is not None:
            if self.model_source == "standard":
                self._predictor = self._standard_predictor
            return True
        if self._model_load_attempted:
            return False
        self._model_load_attempted = True
        if not self.settings.model_path.is_file():
            self.model_error = "标准识别模型缺失，波形仍可正常观察，请联系老师。"
            return False
        try:
            self._standard_predictor = self._model_loader(self.settings.model_path)
        except Exception:  # noqa: BLE001 - model backends provide heterogeneous errors.
            self.model_error = "标准识别模型加载失败，波形仍可正常观察，请联系老师。"
            return False
        if self.model_source == "standard":
            self._predictor = self._standard_predictor
        self.model_error = ""
        return True

    def _switch_predictor(self, source: str, predictor: Any) -> None:
        if predictor is None:
            return
        was_active = self.active
        if was_active:
            for runtime in self._runtime.values():
                decoder = runtime.decoder
                runtime.decoder = None
                if decoder is not None:
                    decoder.close(wait=True)
        self.model_source = source
        self._predictor = predictor
        if was_active:
            for side, ready in self.ready_sides.items():
                if ready:
                    self._start_decoder(side)
        display = "我的模型" if source == "personal" else "标准模型"
        self.model_status_changed.emit("")
        self.model_source_changed.emit(source, display)

    def _start_decoder(self, side: str) -> None:
        runtime = self._runtime[side]
        decoder = RealtimeGestureDecoder(
            runtime.filtered,
            raw_emg_buffer=runtime.raw,
            filtered_emg_buffer=runtime.filtered,
            confidence_threshold=self._resolved_control_values()[0],
            smoothing_frames=self._resolved_control_values()[1],
            change_confirmations=self._resolved_control_values()[2],
            send_to_game_bridge=False,
            parent=self,
        )
        decoder.set_predictor(self._predictor, self._inference_lock)
        # EffiE set_predictor installs model defaults, so the active student
        # profile must always be replayed after changing predictor.
        self._apply_profile_to_decoder(decoder)
        decoder.gesture_changed.connect(
            lambda gesture, confidence, probs, selected_side=side: self.gesture_updated.emit(
                selected_side, gesture, confidence, probs
            )
        )
        runtime.decoder = decoder
        decoder.start()

    def _resolved_control_values(self) -> tuple[float, int, int]:
        sensitivity = self._control_profiles["sensitivity"][self.current_sensitivity]
        style = self._control_profiles["control_style"][self.current_control_style]
        return (
            float(sensitivity["confidence_threshold"]),
            int(style["smoothing_frames"]),
            int(style["change_confirmations"]),
        )

    def _apply_profile_to_decoder(self, decoder: RealtimeGestureDecoder) -> None:
        threshold, smoothing, confirmations = self._resolved_control_values()
        decoder.set_confidence_threshold(threshold)
        decoder.set_smoothing_frames(smoothing)
        decoder.set_change_confirmations(confirmations)

    def _parse_control_profiles(
        self, course_config: dict[str, Any]
    ) -> tuple[dict[str, dict[str, dict[str, float | int]]], dict[str, str]]:
        configured = course_config.get("student_control_profiles", {})
        if not isinstance(configured, dict):
            configured = {}
        sensitivity_source = configured.get("sensitivity", {})
        style_source = configured.get("control_style", {})
        fallback_sensitivity = {
            "low": {"confidence_threshold": 0.80},
            "standard": {"confidence_threshold": self.settings.confidence_threshold},
            "high": {"confidence_threshold": 0.60},
        }
        fallback_styles = {
            "fast": {"smoothing_frames": 3, "change_confirmations": 1},
            "balanced": {
                "smoothing_frames": self.settings.smoothing_frames,
                "change_confirmations": self.settings.change_confirmations,
            },
            "stable": {"smoothing_frames": 7, "change_confirmations": 3},
        }
        sensitivities: dict[str, dict[str, float | int]] = {}
        styles: dict[str, dict[str, float | int]] = {}
        for name in STUDENT_SENSITIVITIES:
            value = sensitivity_source.get(name, {}) if isinstance(sensitivity_source, dict) else {}
            fallback = fallback_sensitivity[name]
            sensitivities[name] = {
                "confidence_threshold": float(
                    value.get("confidence_threshold", fallback["confidence_threshold"])
                    if isinstance(value, dict)
                    else fallback["confidence_threshold"]
                )
            }
        for name in STUDENT_CONTROL_STYLES:
            value = style_source.get(name, {}) if isinstance(style_source, dict) else {}
            fallback = fallback_styles[name]
            styles[name] = {
                "smoothing_frames": max(
                    1,
                    int(
                        value.get("smoothing_frames", fallback["smoothing_frames"])
                        if isinstance(value, dict)
                        else fallback["smoothing_frames"]
                    ),
                ),
                "change_confirmations": max(
                    1,
                    int(
                        value.get("change_confirmations", fallback["change_confirmations"])
                        if isinstance(value, dict)
                        else fallback["change_confirmations"]
                    ),
                ),
            }
        default_source = configured.get("default", {})
        if not isinstance(default_source, dict):
            default_source = {}
        default_sensitivity = str(default_source.get("sensitivity", "standard"))
        default_style = str(default_source.get("control_style", "balanced"))
        defaults = {
            "sensitivity": (
                default_sensitivity if default_sensitivity in sensitivities else "standard"
            ),
            "control_style": default_style if default_style in styles else "balanced",
        }
        return {"sensitivity": sensitivities, "control_style": styles}, defaults

    def decoder_for(self, side: str) -> RealtimeGestureDecoder | None:
        """Expose a side decoder for lifecycle checks and focused tests."""

        return self._runtime[side].decoder
