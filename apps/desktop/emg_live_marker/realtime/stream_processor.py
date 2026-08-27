"""Streaming EMG display processing with preserved filter state."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from emg_live_marker.config import (
    DEFAULT_BANDPASS_HIGH_HZ,
    DEFAULT_BANDPASS_LOW_HZ,
    DEFAULT_EMG_CHANNELS,
    DEFAULT_EMG_FS,
    DEFAULT_RMS_WINDOW_MS,
)

NotchSpec = float | tuple[float, ...] | list[float] | None


@dataclass
class StreamProcessorConfig:
    fs: float = DEFAULT_EMG_FS
    channels: int = DEFAULT_EMG_CHANNELS
    highpass: float = DEFAULT_BANDPASS_LOW_HZ
    lowpass: float = DEFAULT_BANDPASS_HIGH_HZ
    notch_freq: NotchSpec = None
    rms_window_ms: float = DEFAULT_RMS_WINDOW_MS


class StreamingEMGProcessor:
    def __init__(self, config: StreamProcessorConfig | None = None) -> None:
        self.config = config or StreamProcessorConfig()
        self._build_filters()
        self.reset()

    def _build_filters(self) -> None:
        fs = float(self.config.fs)
        nyq = fs / 2.0
        if not (0 < self.config.highpass < self.config.lowpass < nyq):
            raise ValueError(
                "Invalid bandpass: "
                f"highpass={self.config.highpass}, lowpass={self.config.lowpass}, nyquist={nyq}"
            )
        self.band_sos = signal.butter(
            4,
            [self.config.highpass / nyq, self.config.lowpass / nyq],
            btype="bandpass",
            output="sos",
        )

        notch_freqs = self._normalized_notch_freqs()
        if not notch_freqs:
            self.notch_sos = []
            return

        self.notch_sos = []
        for notch_freq in notch_freqs:
            if not (0 < notch_freq < nyq):
                raise ValueError(f"Invalid notch frequency: {notch_freq}, nyquist={nyq}")
            b, a = signal.iirnotch(w0=notch_freq, Q=30.0, fs=fs)
            self.notch_sos.append(signal.tf2sos(b, a))

    def reset(self) -> None:
        channels = int(self.config.channels)
        band_zi_single = signal.sosfilt_zi(self.band_sos)
        self.band_zi = np.repeat(band_zi_single[:, :, None], channels, axis=2)
        self.notch_zi = []
        for notch_sos in self.notch_sos:
            notch_zi_single = signal.sosfilt_zi(notch_sos)
            self.notch_zi.append(np.repeat(notch_zi_single[:, :, None], channels, axis=2))

        self.rms_window = max(1, int(round(self.config.fs * self.config.rms_window_ms / 1000.0)))
        self.square_history = np.zeros((self.rms_window, channels), dtype=np.float64)

    def update_config(
        self,
        highpass: float | None = None,
        lowpass: float | None = None,
        notch_freq: NotchSpec | str = "unchanged",
        rms_window_ms: float | None = None,
    ) -> None:
        if highpass is not None:
            self.config.highpass = float(highpass)
        if lowpass is not None:
            self.config.lowpass = float(lowpass)
        if notch_freq != "unchanged":
            self.config.notch_freq = self._coerce_notch_spec(notch_freq)
        if rms_window_ms is not None:
            self.config.rms_window_ms = float(rms_window_ms)
        self._build_filters()
        self.reset()

    def process_block(self, raw_uv: np.ndarray) -> dict[str, np.ndarray]:
        raw_uv = np.asarray(raw_uv, dtype=np.float64)
        if raw_uv.ndim != 2:
            raise ValueError(f"raw_uv must be 2D, got shape={raw_uv.shape}")
        if raw_uv.shape[1] != self.config.channels:
            raise ValueError(f"expected {self.config.channels} channels, got {raw_uv.shape[1]}")

        if raw_uv.shape[0] == 0:
            empty = raw_uv.astype(np.float32)
            return {"raw": empty, "filtered": empty, "rectified": empty, "rms": empty}

        filtered, self.band_zi = signal.sosfilt(
            self.band_sos,
            raw_uv,
            axis=0,
            zi=self.band_zi,
        )
        for index, notch_sos in enumerate(self.notch_sos):
            filtered, self.notch_zi[index] = signal.sosfilt(
                notch_sos,
                filtered,
                axis=0,
                zi=self.notch_zi[index],
            )

        rectified = np.abs(filtered)
        rms = self._streaming_rms(filtered)
        return {
            "raw": raw_uv.astype(np.float32),
            "filtered": filtered.astype(np.float32),
            "rectified": rectified.astype(np.float32),
            "rms": rms.astype(np.float32),
        }

    def _normalized_notch_freqs(self) -> tuple[float, ...]:
        return self._coerce_notch_spec(self.config.notch_freq) or ()

    @staticmethod
    def _coerce_notch_spec(notch_freq: NotchSpec) -> tuple[float, ...] | None:
        if notch_freq is None:
            return None
        if isinstance(notch_freq, (list, tuple)):
            values = tuple(float(value) for value in notch_freq)
            return values or None
        return (float(notch_freq),)

    def _streaming_rms(self, filtered: np.ndarray) -> np.ndarray:
        squared = filtered * filtered
        out = np.empty_like(filtered, dtype=np.float64)
        for i in range(squared.shape[0]):
            self.square_history[:-1, :] = self.square_history[1:, :]
            self.square_history[-1, :] = squared[i, :]
            out[i, :] = np.sqrt(np.mean(self.square_history, axis=0))
        return out
