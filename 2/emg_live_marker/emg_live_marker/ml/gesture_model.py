"""Realtime EMG gesture classifiers and loading helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from emg_live_marker.ml.preprocess import DEFAULT_CHANNELS, DEFAULT_WINDOW_SAMPLES, prepare_emg_window

LABELS = ["rest", "fist", "open-palm", "pinch"]
MODEL_CNN = "cnn"
MODEL_TCN = "emg2pose_tcn"


class EMGGestureNet(nn.Module):
    """Small baseline Conv1d classifier kept for compatibility."""

    def __init__(
        self,
        channels: int = DEFAULT_CHANNELS,
        classes: int = len(LABELS),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        dilation: int,
        kernel_size: int = 3,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            nn.GroupNorm(num_groups=8 if channels % 8 == 0 else 1, num_channels=channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.GroupNorm(num_groups=8 if channels % 8 == 0 else 1, num_channels=channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class EMG2PoseGestureTCN(nn.Module):
    """emg2pose-inspired temporal Conv1d classifier for 8ch/250Hz EMG.

    This does not use any emg2pose pretrained checkpoint. It only borrows the
    realtime temporal-window modeling idea and adapts it to four gesture labels.
    """

    def __init__(
        self,
        channels: int = DEFAULT_CHANNELS,
        classes: int = len(LABELS),
        hidden_channels: int = 64,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16),
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Conv1d(channels, hidden_channels, kernel_size=1),
            nn.GroupNorm(num_groups=8 if hidden_channels % 8 == 0 else 1, num_channels=hidden_channels),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[
                TemporalResidualBlock(hidden_channels, dilation=dilation, dropout=dropout)
                for dilation in dilations
            ]
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = self.blocks(x)
        return self.head(x)


class GesturePredictor:
    def __init__(
        self,
        model: nn.Module,
        *,
        labels: list[str] | None = None,
        mean: list[float] | np.ndarray | None = None,
        std: list[float] | np.ndarray | None = None,
        window_samples: int = DEFAULT_WINDOW_SAMPLES,
        channels: int = DEFAULT_CHANNELS,
        model_type: str = MODEL_CNN,
        signal_type: str = "filtered",
        normalization_loaded: bool = False,
        model_info: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.labels = labels or list(LABELS)
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float32)
        self.std = None if std is None else np.asarray(std, dtype=np.float32)
        self.window_samples = int(window_samples)
        self.channels = int(channels)
        self.model_type = model_type
        self.signal_type = signal_type
        self.normalization_loaded = bool(normalization_loaded)
        self.model_info = model_info or {}
        self.model.eval()

    def predict_window(self, window: np.ndarray) -> dict[str, Any]:
        tensor = prepare_emg_window(
            window,
            mean=self.mean,
            std=self.std,
            window_samples=self.window_samples,
            channels=self.channels,
        )
        with torch.no_grad():
            logits = self.model(tensor)
            probs_tensor = torch.softmax(logits, dim=1)[0]
        probs = {label: float(probs_tensor[index].item()) for index, label in enumerate(self.labels)}
        gesture = max(probs, key=probs.get)
        return {"gesture": gesture, "confidence": probs[gesture], "probs": probs}


class DemoGesturePredictor:
    """Deterministic fallback for testing the game bridge before a model is trained."""

    labels = list(LABELS)
    model_type = "demo"
    signal_type = "filtered"
    normalization_loaded = False
    window_samples = DEFAULT_WINDOW_SAMPLES
    model_info: dict[str, Any] = {}

    def predict_window(self, window: np.ndarray) -> dict[str, Any]:
        x = np.asarray(window, dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        rms = float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0
        channel_energy = np.sqrt(np.mean(np.square(x), axis=0)) if x.ndim == 2 and x.size else np.zeros(8)
        dominant = int(np.argmax(channel_energy)) if channel_energy.size else 0
        if rms < 25.0:
            gesture = "rest"
            confidence = 0.85
        elif dominant < 3:
            gesture = "fist"
            confidence = 0.82
        elif dominant < 6:
            gesture = "open-palm"
            confidence = 0.82
        else:
            gesture = "pinch"
            confidence = 0.82
        probs = {label: 0.05 for label in self.labels}
        probs[gesture] = confidence
        return {"gesture": gesture, "confidence": confidence, "probs": probs}


def create_model(
    model_name: str = MODEL_CNN,
    *,
    channels: int = DEFAULT_CHANNELS,
    classes: int = len(LABELS),
    dropout: float = 0.15,
) -> nn.Module:
    normalized = model_name.lower()
    if normalized == MODEL_TCN:
        return EMG2PoseGestureTCN(channels=channels, classes=classes, dropout=dropout)
    if normalized == MODEL_CNN:
        return EMGGestureNet(channels=channels, classes=classes, dropout=dropout)
    raise ValueError(f"Unsupported model_name: {model_name}")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _resolve_model_path(path: str | Path) -> Path:
    model_path = Path(path)
    if model_path.is_dir():
        ts_path = model_path / "gesture_classifier.ts"
        if ts_path.exists():
            return ts_path
        return model_path / "gesture_classifier.pt"
    return model_path


def _load_metadata(model_path: Path) -> tuple[list[str], dict[str, Any], dict[str, Any]]:
    base_dir = model_path.parent
    labels = list(LABELS)
    labels_json = _read_json(base_dir / "gesture_labels.json")
    if isinstance(labels_json.get("labels"), list):
        labels = [str(label) for label in labels_json["labels"]]
    model_info = _read_json(base_dir / "model_info.json")
    if isinstance(model_info.get("labels"), list):
        labels = [str(label) for label in model_info["labels"]]
    normalization = _read_json(base_dir / "normalization.json")
    return labels, model_info, normalization


def load_model(path: str | Path) -> GesturePredictor:
    model_path = _resolve_model_path(path)
    labels, model_info, normalization = _load_metadata(model_path)

    model_type = str(model_info.get("model_type", MODEL_CNN))
    if model_type == "effie_finetuned":
        from emg_live_marker.ml.effie_adapter import load_effie_finetuned_model

        return load_effie_finetuned_model(model_path)
    channels = int(model_info.get("channels", normalization.get("channels", DEFAULT_CHANNELS)))
    window_s = float(model_info.get("window_s", normalization.get("window_s", 1.0)))
    fs = float(model_info.get("fs", normalization.get("fs", 250.0)))
    window_samples = int(round(window_s * fs))
    signal_type = str(model_info.get("signal_type", normalization.get("signal_type", "filtered")))
    dropout = float(model_info.get("dropout", 0.15))
    mean = normalization.get("mean")
    std = normalization.get("std")
    normalization_loaded = isinstance(mean, list) and isinstance(std, list)

    if model_path.suffix.lower() == ".ts":
        model = torch.jit.load(str(model_path), map_location="cpu")
        return GesturePredictor(
            model,
            labels=labels,
            mean=mean,
            std=std,
            window_samples=window_samples,
            channels=channels,
            model_type=model_type,
            signal_type=signal_type,
            normalization_loaded=normalization_loaded,
            model_info=model_info,
        )

    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict: dict[str, Any]
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        labels = [str(label) for label in checkpoint.get("labels", labels)]
        model_type = str(checkpoint.get("model_type", model_type))
        channels = int(checkpoint.get("channels", channels))
    else:
        state_dict = checkpoint

    model = create_model(model_type, channels=channels, classes=len(labels), dropout=dropout)
    model.load_state_dict(state_dict)
    return GesturePredictor(
        model,
        labels=labels,
        mean=mean,
        std=std,
        window_samples=window_samples,
        channels=channels,
        model_type=model_type,
        signal_type=signal_type,
        normalization_loaded=normalization_loaded,
        model_info=model_info,
    )
