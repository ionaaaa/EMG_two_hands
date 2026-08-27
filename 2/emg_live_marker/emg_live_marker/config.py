"""Project-wide configuration values."""

from __future__ import annotations

from dataclasses import dataclass


APP_NAME = "emg_live_marker"
DEFAULT_BAUDRATE = 921_600
DEFAULT_EMG_CHANNELS = 8
DEFAULT_EMG_FS = 250.0
DEFAULT_IMU_FS = 104.0
DEFAULT_BUFFER_SECONDS = 60.0
DEFAULT_RMS_WINDOW_MS = 100.0
DEFAULT_BANDPASS_LOW_HZ = 20.0
DEFAULT_BANDPASS_HIGH_HZ = 90.0
DEFAULT_DISPLAY_OUTLIER_UV = 20_000.0


@dataclass(frozen=True)
class AppConfig:
    baudrate: int = DEFAULT_BAUDRATE
    emg_channels: int = DEFAULT_EMG_CHANNELS
    emg_fs: float = DEFAULT_EMG_FS
    imu_fs: float = DEFAULT_IMU_FS
    buffer_seconds: float = DEFAULT_BUFFER_SECONDS
    rms_window_ms: float = DEFAULT_RMS_WINDOW_MS
    bandpass_low_hz: float = DEFAULT_BANDPASS_LOW_HZ
    bandpass_high_hz: float = DEFAULT_BANDPASS_HIGH_HZ
    display_outlier_uv: float = DEFAULT_DISPLAY_OUTLIER_UV


DEFAULT_CONFIG = AppConfig()
