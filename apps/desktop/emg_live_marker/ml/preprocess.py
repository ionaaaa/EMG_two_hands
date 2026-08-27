"""Preprocessing for fixed-size EMG gesture classifier windows."""

from __future__ import annotations

import numpy as np
import torch


DEFAULT_WINDOW_SAMPLES = 250
DEFAULT_CHANNELS = 8


def prepare_emg_window(
    window_np: np.ndarray,
    *,
    mean: np.ndarray | list[float] | None = None,
    std: np.ndarray | list[float] | None = None,
    input_shape: str = "BCT",
    window_samples: int = DEFAULT_WINDOW_SAMPLES,
    channels: int = DEFAULT_CHANNELS,
) -> torch.Tensor:
    """Prepare one EMG window using training-set normalization.

    The realtime path intentionally does not recompute per-window statistics.
    ``mean`` and ``std`` are expected to come from ``normalization.json``.
    """

    x = np.asarray(window_np, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != channels:
        raise ValueError(f"window_np must have shape [N, {channels}], got {x.shape}")
    if x.shape[0] < window_samples:
        raise ValueError(f"need at least {window_samples} samples, got {x.shape[0]}")
    if x.shape[0] > window_samples:
        x = x[-window_samples:, :]

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if mean is not None:
        mean_arr = np.asarray(mean, dtype=np.float32).reshape(1, channels)
        if std is None:
            raise ValueError("std is required when mean is provided")
        std_arr = np.asarray(std, dtype=np.float32).reshape(1, channels)
        x = (x - mean_arr) / (std_arr + 1e-6)

    normalized_shape = input_shape.upper()
    if normalized_shape == "BCT":
        x = np.transpose(x, (1, 0))[None, :, :]
    elif normalized_shape == "BTC":
        x = x[None, :, :]
    else:
        raise ValueError("input_shape must be 'BCT' or 'BTC'")
    return torch.from_numpy(x.astype(np.float32, copy=False))


def preprocess_emg_window(
    window: np.ndarray,
    *,
    window_samples: int = 250,
    channels: int = 8,
    normalize: bool = True,
) -> np.ndarray:
    """Convert an EMG window shaped [N, 8] to Conv1d input [1, 8, 250].

    This legacy helper keeps per-window standardization for older tests/models.
    New training and realtime inference should use :func:`prepare_emg_window`
    with training-set mean/std.
    """

    x = np.asarray(window, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != channels:
        raise ValueError(f"window must have shape [N, {channels}], got {x.shape}")
    if x.shape[0] < window_samples:
        raise ValueError(f"need at least {window_samples} samples, got {x.shape[0]}")
    if x.shape[0] > window_samples:
        x = x[-window_samples:, :]

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = x - np.mean(x, axis=0, keepdims=True)
    if normalize:
        x = x / (np.std(x, axis=0, keepdims=True) + 1e-6)
    return np.transpose(x, (1, 0))[None, :, :].astype(np.float32, copy=False)
