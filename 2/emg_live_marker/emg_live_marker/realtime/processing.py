"""Compatibility display processing API backed by the streaming processor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from emg_live_marker.device.protocol import EMG_CHANNELS, EMG_FS
from emg_live_marker.realtime.stream_processor import StreamProcessorConfig, StreamingEMGProcessor

__all__ = ["RealtimeEMGProcessor"]


@dataclass(frozen=True)
class RealtimeEMGProcessor:
    fs: float = EMG_FS
    channels: int = EMG_CHANNELS
    bandpass_low_hz: float = 20.0
    bandpass_high_hz: float = 90.0
    notch_hz: float | tuple[float, ...] | None = None
    rms_window_ms: float = 100.0

    @property
    def rms_window_samples(self) -> int:
        return max(1, int(round(self.fs * self.rms_window_ms / 1000.0)))

    def process_for_display(self, data_uv: np.ndarray, mode: str) -> np.ndarray:
        data = np.asarray(data_uv, dtype=np.float64)
        if data.ndim != 2 or data.shape[1] != self.channels:
            raise ValueError(f"data_uv must have shape (N, {self.channels})")
        if data.shape[0] == 0:
            return data.copy()

        normalized_mode = mode.lower()
        if normalized_mode == "raw":
            return data.copy()
        if normalized_mode not in {"filtered", "rectified", "rms"}:
            raise ValueError(f"unsupported display mode: {mode}")

        processor = StreamingEMGProcessor(
            StreamProcessorConfig(
                fs=self.fs,
                channels=self.channels,
                highpass=self.bandpass_low_hz,
                lowpass=self.bandpass_high_hz,
                notch_freq=self.notch_hz,
                rms_window_ms=self.rms_window_ms,
            )
        )
        return processor.process_block(data)[normalized_mode].astype(np.float64)
