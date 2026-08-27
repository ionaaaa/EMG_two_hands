"""EffiE-style PyTorch adapter for fine-tuned sEMG gesture models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from emg_live_marker.ml.effie_preprocess import (
    EFFIE_CHANNELS,
    EFFIE_WINDOW_SAMPLES,
    prepare_effie_window,
)
from emg_live_marker.ml.gesture_model import LABELS

MODEL_TYPE = "effie_finetuned"


class EffieGestureNet(nn.Module):
    """Compact PyTorch implementation of the EffiE-style 8x32 CNN input path.

    The public EffiE repository uses TensorFlow/Keras. This model keeps the same
    input convention, [B, 8, 32, 1], and provides a practical PyTorch fine-tuning
    target for this application.
    """

    def __init__(self, classes: int = len(LABELS), dropout: float = 0.25) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=(3, 5), padding=(1, 2), bias=False),
            nn.BatchNorm2d(24),
            nn.PReLU(24),
            nn.Conv2d(24, 32, kernel_size=(3, 3), padding=(1, 1), groups=1, bias=False),
            nn.BatchNorm2d(32),
            nn.PReLU(32),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(dropout),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(64, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected [B, 8, 32, 1], got {tuple(x.shape)}")
        # PyTorch Conv2d expects [B, C, H, W].
        x = x.permute(0, 3, 1, 2).contiguous()
        return self.classifier(self.backbone(x))


def replace_classifier(model: EffieGestureNet, classes: int = len(LABELS)) -> EffieGestureNet:
    model.classifier = nn.Linear(model.classifier.in_features, classes)
    return model


def set_trainable_mode(model: EffieGestureNet, mode: str) -> None:
    if mode not in {"freeze_backbone", "finetune_all"}:
        raise ValueError("mode must be freeze_backbone or finetune_all")
    for parameter in model.backbone.parameters():
        parameter.requires_grad = mode == "finetune_all"
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def load_effie_checkpoint(
    model: EffieGestureNet,
    checkpoint_path: str | Path | None,
    *,
    strict: bool = False,
) -> dict[str, Any]:
    """Load a PyTorch EffiE-style checkpoint if available.

    TensorFlow/Keras checkpoint conversion is intentionally handled by
    ``external_models/effie/import_effie.py`` so the main app does not depend on
    TensorFlow.
    """

    if checkpoint_path is None:
        return {"loaded": False, "reason": "no checkpoint_path provided"}
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported checkpoint format: {path}")
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    return {
        "loaded": True,
        "path": str(path),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }


class EffieGesturePredictor:
    labels = list(LABELS)
    model_type = MODEL_TYPE
    signal_type = "raw"
    window_samples = 125

    def __init__(
        self,
        model: nn.Module,
        *,
        labels: list[str] | None = None,
        mean: list[float] | np.ndarray | None = None,
        std: list[float] | np.ndarray | None = None,
        model_info: dict[str, Any] | None = None,
        normalization_loaded: bool = False,
    ) -> None:
        self.model = model
        self.labels = labels or list(LABELS)
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float32)
        self.std = None if std is None else np.asarray(std, dtype=np.float32)
        self.model_info = model_info or {}
        self.signal_type = str(self.model_info.get("signal_type", "raw"))
        source_fs = float(self.model_info.get("source_fs", 250.0))
        source_window_s = float(self.model_info.get("source_window_s", 0.5))
        self.window_samples = max(40, int(round(source_fs * source_window_s)))
        self.normalization_loaded = bool(normalization_loaded)
        self.model.eval()

    def predict_window(self, window: np.ndarray) -> dict[str, Any]:
        tensor = prepare_effie_window(window, mean=self.mean, std=self.std)
        with torch.no_grad():
            logits = self.model(tensor)
            probs_tensor = torch.softmax(logits, dim=1)[0]
        probs = {label: float(probs_tensor[index].item()) for index, label in enumerate(self.labels)}
        gesture = max(probs, key=probs.get)
        return {"gesture": gesture, "confidence": probs[gesture], "probs": probs}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def load_effie_finetuned_model(path: str | Path) -> EffieGesturePredictor:
    model_path = Path(path)
    if model_path.is_dir():
        model_path = model_path / "gesture_classifier.ts"
    base_dir = model_path.parent
    labels_json = _read_json(base_dir / "gesture_labels.json")
    model_info = _read_json(base_dir / "model_info.json")
    normalization = _read_json(base_dir / "normalization.json")
    labels = [str(label) for label in model_info.get("labels", labels_json.get("labels", LABELS))]
    mean = normalization.get("mean")
    std = normalization.get("std")
    normalization_loaded = isinstance(mean, list) and isinstance(std, list)

    if model_path.suffix.lower() == ".ts":
        model = torch.jit.load(str(model_path), map_location="cpu")
    else:
        checkpoint = torch.load(model_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model = EffieGestureNet(classes=len(labels), dropout=float(model_info.get("dropout", 0.25)))
        model.load_state_dict(state_dict)
    return EffieGesturePredictor(
        model,
        labels=labels,
        mean=mean,
        std=std,
        model_info=model_info,
        normalization_loaded=normalization_loaded,
    )

