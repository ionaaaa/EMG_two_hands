"""EffiE-style preprocessing for 8-channel sEMG windows."""

from __future__ import annotations

import numpy as np
import torch
from scipy.signal import resample_poly

SOURCE_FS = 250.0
EFFIE_FS = 200.0
EFFIE_CHANNELS = 8
EFFIE_WINDOW_S = 0.16
EFFIE_WINDOW_SAMPLES = 32
EFFIE_STRIDE_S = 0.08
EFFIE_STRIDE_SAMPLES = 16


def resample_250hz_to_200hz(emg: np.ndarray) -> np.ndarray:
    """Resample [N, 8] EMG from 250Hz to 200Hz using polyphase filtering."""

    x = np.asarray(emg, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != EFFIE_CHANNELS:
        raise ValueError(f"emg must have shape [N, {EFFIE_CHANNELS}], got {x.shape}")
    if x.shape[0] == 0:
        return np.empty((0, EFFIE_CHANNELS), dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return resample_poly(x, up=4, down=5, axis=0).astype(np.float32, copy=False)


def prepare_effie_window(
    window_np: np.ndarray,
    *,
    mean: np.ndarray | list[float] | None = None,
    std: np.ndarray | list[float] | None = None,
    source_fs: float = SOURCE_FS,
) -> torch.Tensor:
    """Prepare a raw [N, 8] EMG window as EffiE input [1, 8, 32, 1]."""

    x = np.asarray(window_np, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != EFFIE_CHANNELS:
        raise ValueError(f"window_np must have shape [N, {EFFIE_CHANNELS}], got {x.shape}")
    if x.shape[0] < int(round(source_fs * EFFIE_WINDOW_S)):
        raise ValueError("not enough samples for an EffiE 160ms window")
    if abs(float(source_fs) - EFFIE_FS) > 1e-6:
        x = resample_250hz_to_200hz(x)
    if x.shape[0] < EFFIE_WINDOW_SAMPLES:
        raise ValueError("not enough resampled samples for an EffiE 32-sample window")
    x = x[-EFFIE_WINDOW_SAMPLES:, :]
    if mean is not None:
        if std is None:
            raise ValueError("std is required when mean is provided")
        mean_arr = np.asarray(mean, dtype=np.float32).reshape(1, EFFIE_CHANNELS)
        std_arr = np.asarray(std, dtype=np.float32).reshape(1, EFFIE_CHANNELS)
        x = (x - mean_arr) / (std_arr + 1e-6)
    # EffiE convention: [B, 8, 32, 1], channels x time x singleton feature.
    x = np.transpose(x, (1, 0))[:, :, None][None, :, :, :]
    return torch.from_numpy(x.astype(np.float32, copy=False))


def slice_effie_windows(
    segment_250hz: np.ndarray,
    *,
    label: str,
    stride_samples: int = EFFIE_STRIDE_SAMPLES,
) -> list[tuple[np.ndarray, str]]:
    """Resample one stable segment and slice EffiE 32-sample windows."""

    segment = resample_250hz_to_200hz(segment_250hz)
    if segment.shape[0] < EFFIE_WINDOW_SAMPLES:
        return []
    windows: list[tuple[np.ndarray, str]] = []
    for start in range(0, segment.shape[0] - EFFIE_WINDOW_SAMPLES + 1, stride_samples):
        window = segment[start : start + EFFIE_WINDOW_SAMPLES]
        windows.append((window.astype(np.float32, copy=True), label))
    return windows


def normalize_effie_batch(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Normalize [B, 8, 32, 1] by per-channel training-set statistics."""

    mean_arr = np.asarray(mean, dtype=np.float32).reshape(1, EFFIE_CHANNELS, 1, 1)
    std_arr = np.asarray(std, dtype=np.float32).reshape(1, EFFIE_CHANNELS, 1, 1)
    return ((x - mean_arr) / (std_arr + 1e-6)).astype(np.float32, copy=False)

