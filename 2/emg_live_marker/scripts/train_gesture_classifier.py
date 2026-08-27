"""Train a realtime EMG gesture classifier from collected dataset sessions."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emg_live_marker.ml.gesture_model import LABELS, MODEL_CNN, MODEL_TCN, create_model
from emg_live_marker.realtime.stream_processor import StreamingEMGProcessor

EMG_FS = 250.0
CHANNELS = 8
COLLECT_TO_GAME_LABEL = {
    "fist": "fist",
    "finger_spread": "open-palm",
    "thumb_index_pinch": "pinch",
    "rest": "rest",
}


@dataclass(frozen=True)
class WindowRecord:
    x: np.ndarray
    y: int
    session_id: str
    trial_id: str
    gesture: str
    phase: str


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj))


def discover_session_dirs(dataset_root: Path) -> list[Path]:
    return sorted(
        path.parent
        for path in dataset_root.rglob("metadata.json")
        if (path.parent / "emg.csv").exists() and (path.parent / "events.csv").exists()
    )


def resolve_dataset_root(dataset_root: Path) -> Path:
    if dataset_root.exists():
        return dataset_root
    package_dataset = Path(__file__).resolve().parents[1] / "emg_live_marker" / "dataset"
    if dataset_root == Path("dataset") and package_dataset.exists():
        return package_dataset
    return dataset_root


def load_emg(path: Path, *, signal_type: str = "raw") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = read_csv_dicts(path)
    sample_index = np.asarray([int(float(row["sample_index"])) for row in rows], dtype=np.int64)
    software_time = np.asarray([float(row.get("software_time", "nan")) for row in rows], dtype=np.float64)
    emg = np.asarray(
        [[float(row[f"ch{channel}_uv"]) for channel in range(1, CHANNELS + 1)] for row in rows],
        dtype=np.float32,
    )
    if signal_type == "filtered":
        processor = StreamingEMGProcessor()
        emg = processor.process_block(emg)["filtered"].astype(np.float32, copy=False)
    elif signal_type != "raw":
        raise ValueError("signal_type must be 'raw' or 'filtered'")
    return sample_index, software_time, emg


def event_groups(events: list[dict[str, str]]) -> dict[str, dict[str, dict[str, str]]]:
    grouped: dict[str, dict[str, dict[str, str]]] = {}
    for event in events:
        trial_id = event.get("trial_id", "")
        phase = event.get("phase", "")
        if trial_id and phase:
            grouped.setdefault(trial_id, {})[phase] = event
    return grouped


def _event_sample(event: dict[str, str]) -> int | None:
    try:
        return int(float(event.get("sample_index", "")))
    except (TypeError, ValueError):
        return None


def _event_time(event: dict[str, str]) -> float | None:
    try:
        value = float(event.get("software_time", ""))
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _slice_by_sample_index(
    emg: np.ndarray,
    sample_index: np.ndarray,
    *,
    start_index: int,
    end_index: int,
) -> np.ndarray:
    order = np.argsort(sample_index, kind="stable")
    sorted_index = sample_index[order]
    sorted_emg = emg[order]
    left = int(np.searchsorted(sorted_index, start_index, side="left"))
    right = int(np.searchsorted(sorted_index, end_index, side="left"))
    return sorted_emg[left:right]


def _slice_by_time(
    emg: np.ndarray,
    software_time: np.ndarray,
    *,
    start_time: float,
    end_time: float,
) -> np.ndarray:
    mask = np.isfinite(software_time) & (software_time >= start_time) & (software_time < end_time)
    return emg[mask]


def _segment_windows(
    segment: np.ndarray,
    *,
    label: str,
    session_id: str,
    trial_id: str,
    gesture: str,
    phase: str,
    window_samples: int,
    stride_samples: int,
) -> list[WindowRecord]:
    if segment.shape[0] < window_samples:
        print(
            f"skip segment session={session_id} trial={trial_id} phase={phase}: "
            f"{segment.shape[0]} samples < {window_samples}"
        )
        return []
    label_index = LABELS.index(label)
    out: list[WindowRecord] = []
    for start in range(0, segment.shape[0] - window_samples + 1, stride_samples):
        out.append(
            WindowRecord(
                x=segment[start : start + window_samples].astype(np.float32, copy=True),
                y=label_index,
                session_id=session_id,
                trial_id=trial_id,
                gesture=gesture,
                phase=phase,
            )
        )
    return out


def _trim_interval(start: int, end: int, desired_trim: int, window_samples: int) -> tuple[int, int]:
    length = end - start
    if length <= window_samples:
        return start, end
    trim = min(desired_trim, max(0, (length - window_samples) // 2))
    return start + trim, end - trim


def _trim_time_interval(start: float, end: float, desired_trim: float, window_s: float) -> tuple[float, float]:
    length = end - start
    if length <= window_s:
        return start, end
    trim = min(desired_trim, max(0.0, (length - window_s) / 2.0))
    return start + trim, end - trim


def collect_window_records(
    dataset_root: Path,
    *,
    signal_type: str = "raw",
    window_s: float = 1.0,
    stride_s: float = 0.1,
) -> list[WindowRecord]:
    window_samples = int(round(window_s * EMG_FS))
    stride_samples = max(1, int(round(stride_s * EMG_FS)))
    transition_gesture = int(round(0.3 * EMG_FS))
    transition_rest = int(round(0.2 * EMG_FS))
    records: list[WindowRecord] = []

    for session_dir in discover_session_dirs(dataset_root):
        session_id = session_dir.name
        sample_index, software_time, emg = load_emg(session_dir / "emg.csv", signal_type=signal_type)
        grouped = event_groups(read_csv_dicts(session_dir / "events.csv"))
        for trial_id, phases in sorted(grouped.items()):
            required = {"trial_start", "gesture_start", "gesture_end", "trial_end"}
            if not required <= set(phases):
                print(f"skip trial session={session_id} trial={trial_id}: missing phases")
                continue
            gesture = phases["gesture_start"].get("gesture", "")
            label = COLLECT_TO_GAME_LABEL.get(gesture)
            if label is None:
                print(f"skip trial session={session_id} trial={trial_id}: unknown gesture={gesture}")
                continue

            trial_start = _event_sample(phases["trial_start"])
            gesture_start = _event_sample(phases["gesture_start"])
            gesture_end = _event_sample(phases["gesture_end"])
            trial_end = _event_sample(phases["trial_end"])
            use_samples = None not in (trial_start, gesture_start, gesture_end, trial_end)

            if use_samples:
                intervals = [
                    (*_trim_interval(gesture_start, gesture_end, transition_gesture, window_samples), label, "gesture"),
                    (*_trim_interval(trial_start, gesture_start, transition_rest, window_samples), "rest", "rest_before"),
                    (*_trim_interval(gesture_end, trial_end, transition_rest, window_samples), "rest", "rest_after"),
                ]
                for start_index, end_index, out_label, phase in intervals:
                    if start_index >= end_index:
                        print(f"skip segment session={session_id} trial={trial_id} phase={phase}: empty")
                        continue
                    segment = _slice_by_sample_index(emg, sample_index, start_index=start_index, end_index=end_index)
                    records.extend(
                        _segment_windows(
                            segment,
                            label=out_label,
                            session_id=session_id,
                            trial_id=trial_id,
                            gesture=gesture if phase == "gesture" else "rest",
                            phase=phase,
                            window_samples=window_samples,
                            stride_samples=stride_samples,
                        )
                    )
                continue

            times = {phase: _event_time(event) for phase, event in phases.items()}
            if any(times[phase] is None for phase in required):
                print(f"skip trial session={session_id} trial={trial_id}: no usable sample_index/time")
                continue
            time_intervals = [
                (*_trim_time_interval(times["gesture_start"], times["gesture_end"], 0.3, window_s), label, "gesture"),
                (*_trim_time_interval(times["trial_start"], times["gesture_start"], 0.2, window_s), "rest", "rest_before"),
                (*_trim_time_interval(times["gesture_end"], times["trial_end"], 0.2, window_s), "rest", "rest_after"),
            ]
            for start_time, end_time, out_label, phase in time_intervals:
                if start_time >= end_time:
                    print(f"skip segment session={session_id} trial={trial_id} phase={phase}: empty")
                    continue
                segment = _slice_by_time(emg, software_time, start_time=start_time, end_time=end_time)
                records.extend(
                    _segment_windows(
                        segment,
                        label=out_label,
                        session_id=session_id,
                        trial_id=trial_id,
                        gesture=gesture if phase == "gesture" else "rest",
                        phase=phase,
                        window_samples=window_samples,
                        stride_samples=stride_samples,
                    )
                )
    if not records:
        raise ValueError(f"No training windows found under {dataset_root}")
    return records


def _label_counts(records: list[WindowRecord], indices: list[int] | None = None) -> dict[str, int]:
    selected = range(len(records)) if indices is None else indices
    counts = Counter(LABELS[records[index].y] for index in selected)
    return {label: int(counts.get(label, 0)) for label in LABELS}


def balance_rest_records(
    records: list[WindowRecord],
    *,
    max_rest_ratio: float = 1.0,
    seed: int = 42,
) -> tuple[list[WindowRecord], dict[str, int], dict[str, int]]:
    before = _label_counts(records)
    action_counts = [before[label] for label in LABELS if label != "rest"]
    max_action = max(action_counts) if action_counts else 0
    rest_limit = int(round(max_action * max(0.0, max_rest_ratio)))
    if before["rest"] <= rest_limit or rest_limit <= 0:
        return records, before, before

    rest_indices = [index for index, record in enumerate(records) if LABELS[record.y] == "rest"]
    action_indices = [index for index, record in enumerate(records) if LABELS[record.y] != "rest"]
    rng = random.Random(seed)
    rng.shuffle(rest_indices)
    keep = sorted(action_indices + rest_indices[:rest_limit])
    balanced = [records[index] for index in keep]
    return balanced, before, _label_counts(balanced)


def split_records(
    records: list[WindowRecord],
    *,
    val_split: str = "session",
    holdout_session: str | None = None,
    train_all: bool = False,
    seed: int = 42,
) -> tuple[list[int], list[int], str | None]:
    if train_all:
        return list(range(len(records))), [], None
    if val_split == "session":
        sessions = sorted({record.session_id for record in records})
        if len(sessions) < 2:
            return list(range(len(records))), [], None
        holdout = holdout_session or sessions[-1]
        val = [index for index, record in enumerate(records) if record.session_id == holdout]
        train = [index for index, record in enumerate(records) if record.session_id != holdout]
        if not train or not val:
            raise ValueError(f"session split failed for holdout_session={holdout}")
        return train, val, holdout
    if val_split == "trial":
        by_key: dict[tuple[int, str, str], list[int]] = defaultdict(list)
        for index, record in enumerate(records):
            by_key[(record.y, record.session_id, record.trial_id)].append(index)
        train: list[int] = []
        val: list[int] = []
        rng = random.Random(seed)
        by_label: dict[int, list[tuple[int, str, str]]] = defaultdict(list)
        for key in by_key:
            by_label[key[0]].append(key)
        for keys in by_label.values():
            rng.shuffle(keys)
            val_count = max(1, int(round(len(keys) * 0.2)))
            val_keys = set(keys[:val_count])
            for key in keys:
                if key in val_keys:
                    val.extend(by_key[key])
                else:
                    train.extend(by_key[key])
        if not train or not val:
            raise ValueError("trial split failed")
        return sorted(train), sorted(val), None
    raise ValueError("val_split must be 'session' or 'trial'")


def _stack(records: list[WindowRecord], indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
    x = np.stack([records[index].x for index in indices]).astype(np.float32)
    y = np.asarray([records[index].y for index in indices], dtype=np.int64)
    return x, y


def _normalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return np.transpose((x - mean.reshape(1, 1, CHANNELS)) / (std.reshape(1, 1, CHANNELS) + 1e-6), (0, 2, 1)).astype(
        np.float32,
        copy=False,
    )


class EMGWindowDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, *, augment: bool = False) -> None:
        self.x = x
        self.y = y
        self.augment = augment

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.x[index].copy()
        if self.augment:
            x = self._augment(x)
        return torch.from_numpy(x), torch.tensor(int(self.y[index]), dtype=torch.long)

    @staticmethod
    def _augment(x: np.ndarray) -> np.ndarray:
        gain = np.random.uniform(0.8, 1.2, size=(x.shape[0], 1)).astype(np.float32)
        x = x * gain
        noise_std = float(np.random.uniform(0.01, 0.03))
        x = x + np.random.normal(0.0, noise_std, size=x.shape).astype(np.float32)
        if np.random.random() < 0.10:
            channel = int(np.random.randint(0, x.shape[0]))
            x[channel, :] = 0.0
        shift = int(np.random.randint(-25, 26))
        if shift:
            x = np.roll(x, shift, axis=1)
        return x.astype(np.float32, copy=False)


def _select_device(device_arg: str) -> torch.device:
    cuda_available = torch.cuda.is_available()
    if device_arg == "cuda" and not cuda_available:
        raise RuntimeError("--device cuda was requested, but CUDA is not available")
    if device_arg == "auto":
        return torch.device("cuda" if cuda_available else "cpu")
    return torch.device(device_arg)


def _print_device_info(device: torch.device) -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
    print(f"torch version: {torch.__version__}")
    print(f"cuda available: {cuda_available}")
    print(f"device name: {device_name}")
    print(f"device used: {device}")
    return {
        "torch_version": torch.__version__,
        "cuda_available": bool(cuda_available),
        "device_name": device_name,
        "device_used": str(device),
    }


def _make_worker_init(seed: int):
    def _init(worker_id: int) -> None:
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return _init


def _class_weights(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y, minlength=len(LABELS)).astype(np.float32)
    return (float(y.size) / (len(LABELS) * np.maximum(counts, 1.0))).astype(np.float32)


def _make_train_loader(
    x: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    augment: bool,
    balanced_sampler: bool,
    num_workers: int,
    seed: int,
    device: torch.device,
) -> DataLoader:
    dataset = EMGWindowDataset(x, y, augment=augment)
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = None
    shuffle = True
    if balanced_sampler:
        weights = _class_weights(y)
        sample_weights = torch.as_tensor([weights[int(label)] for label in y], dtype=torch.double)
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
            generator=generator,
        )
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=_make_worker_init(seed) if num_workers else None,
        generator=generator if shuffle else None,
    )


def _make_eval_loader(
    x: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        EMGWindowDataset(x, y),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def _metrics_from_confusion(confusion: np.ndarray) -> dict[str, Any]:
    total = int(confusion.sum())
    accuracy = float(np.trace(confusion) / total) if total else 0.0
    per_class_accuracy: dict[str, float] = {}
    prf: dict[str, dict[str, float]] = {}
    for index, label in enumerate(LABELS):
        tp = float(confusion[index, index])
        fn = float(confusion[index, :].sum() - confusion[index, index])
        fp = float(confusion[:, index].sum() - confusion[index, index])
        support = int(confusion[index, :].sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class_accuracy[label] = recall
        prf[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    return {
        "accuracy": accuracy,
        "per_class_accuracy": per_class_accuracy,
        "per_class_precision_recall_f1": prf,
        "confusion_matrix": confusion.tolist(),
    }


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    records: list[WindowRecord] | None = None,
    indices: list[int] | None = None,
) -> dict[str, Any]:
    confusion = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    session_confusions: dict[str, np.ndarray] = {}
    model.eval()
    offset = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            pred = model(xb).argmax(dim=1)
            y_cpu = yb.detach().cpu().numpy()
            pred_cpu = pred.detach().cpu().numpy()
            for local_index, (target, predicted) in enumerate(zip(y_cpu, pred_cpu, strict=False)):
                confusion[int(target), int(predicted)] += 1
                if records is not None and indices is not None:
                    record = records[indices[offset + local_index]]
                    session_confusions.setdefault(
                        record.session_id,
                        np.zeros((len(LABELS), len(LABELS)), dtype=np.int64),
                    )[int(target), int(predicted)] += 1
            offset += len(y_cpu)
    metrics = _metrics_from_confusion(confusion)
    metrics["per_session_accuracy"] = {
        session: _metrics_from_confusion(matrix)["accuracy"]
        for session, matrix in sorted(session_confusions.items())
    }
    return metrics


def _accuracy_and_confusion(model: nn.Module, loader: DataLoader) -> tuple[float, list[list[int]], list[float]]:
    metrics = evaluate_model(model, loader, device=torch.device("cpu"))
    per_class = [metrics["per_class_accuracy"][label] for label in LABELS]
    return metrics["accuracy"], metrics["confusion_matrix"], per_class


def build_dataset(
    dataset_root: Path,
    *,
    signal_type: str = "raw",
    window_s: float = 1.0,
    stride_s: float = 0.1,
    max_rest_ratio: float | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    records = collect_window_records(
        dataset_root,
        signal_type=signal_type,
        window_s=window_s,
        stride_s=stride_s,
    )
    if max_rest_ratio is not None:
        records, _before, _after = balance_rest_records(records, max_rest_ratio=max_rest_ratio, seed=seed)
    x, y = _stack(records, list(range(len(records))))
    mean = x.mean(axis=(0, 1))
    std = x.std(axis=(0, 1)) + 1e-6
    return _normalize(x, mean, std), y


def _save_confusion_csv(path: Path, matrix: list[list[int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["actual\\predicted", *LABELS])
        for label, row in zip(LABELS, matrix, strict=False):
            writer.writerow([label, *row])


def _save_artifacts(
    *,
    output_dir: Path,
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    model_name: str,
    signal_type: str,
    window_s: float,
    stride_s: float,
    dropout: float,
    mean: np.ndarray,
    std: np.ndarray,
    report: dict[str, Any],
    export_torchscript: bool,
    training_preset: str | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "gesture_classifier.pt"
    labels_path = output_dir / "gesture_labels.json"
    norm_path = output_dir / "normalization.json"
    info_path = output_dir / "model_info.json"
    report_path = output_dir / "train_report.json"
    confusion_path = output_dir / "confusion_matrix.csv"
    script_path = output_dir / "gesture_classifier.ts"

    cpu_state = {key: value.detach().cpu() for key, value in state_dict.items()}
    torch.save(
        {
            "model_state_dict": cpu_state,
            "labels": LABELS,
            "model_type": model_name,
            "channels": CHANNELS,
        },
        model_path,
    )
    labels_path.write_text(json.dumps({"labels": LABELS}, indent=2), encoding="utf-8")
    normalization = {
        "signal_type": signal_type,
        "fs": EMG_FS,
        "window_s": window_s,
        "channels": CHANNELS,
        "mean": [float(value) for value in mean.tolist()],
        "std": [float(value) for value in std.tolist()],
    }
    norm_path.write_text(json.dumps(normalization, indent=2), encoding="utf-8")
    model_info = {
        "model_type": model_name,
        "input_shape": f"[1, {CHANNELS}, {int(round(window_s * EMG_FS))}]",
        "fs": EMG_FS,
        "window_s": window_s,
        "stride_s": stride_s,
        "channels": CHANNELS,
        "signal_type": signal_type,
        "labels": LABELS,
        "gesture_mapping": COLLECT_TO_GAME_LABEL,
        "preprocess": "training-set per-channel mean/std normalization",
        "dropout": dropout,
        "training_preset": training_preset,
    }
    info_path.write_text(json.dumps(model_info, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _save_confusion_csv(confusion_path, report["confusion_matrix"])

    if export_torchscript:
        cpu_model = create_model(model_name, channels=CHANNELS, classes=len(LABELS), dropout=dropout)
        cpu_model.load_state_dict(cpu_state)
        cpu_model.eval()
        traced = torch.jit.trace(cpu_model, torch.zeros(1, CHANNELS, int(round(window_s * EMG_FS))))
        traced.save(str(script_path))

    return {
        "model_path": model_path,
        "torchscript_path": script_path,
        "labels_path": labels_path,
        "normalization_path": norm_path,
        "model_info_path": info_path,
        "train_report_path": report_path,
        "confusion_matrix_path": confusion_path,
    }


def train_model(
    dataset_root: Path,
    output_dir: Path,
    *,
    model_name: str = MODEL_CNN,
    signal_type: str = "raw",
    window_s: float = 1.0,
    stride_s: float = 0.1,
    epochs: int = 150,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    dropout: float = 0.2,
    val_split: str = "session",
    holdout_session: str | None = None,
    train_all: bool = False,
    export_torchscript: bool = True,
    augment: bool = True,
    device_arg: str = "auto",
    num_workers: int = 0,
    seed: int = 42,
    patience: int = 25,
    max_rest_ratio: float = 1.0,
    balanced_sampler: bool = True,
    balanced_loss: bool = False,
    save_best: bool = True,
    verbose_device: bool = True,
    training_preset: str | None = None,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = _select_device(device_arg)
    device_info = _print_device_info(device) if verbose_device else {
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "device_used": str(device),
    }

    records = collect_window_records(
        dataset_root,
        signal_type=signal_type,
        window_s=window_s,
        stride_s=stride_s,
    )
    samples_before_balance = len(records)
    records, counts_before_balance, counts_after_balance = balance_rest_records(
        records,
        max_rest_ratio=max_rest_ratio,
        seed=seed,
    )
    print(f"class counts before balance: {counts_before_balance}")
    print(f"class counts after balance: {counts_after_balance}")

    train_indices, val_indices, holdout = split_records(
        records,
        val_split=val_split,
        holdout_session=holdout_session,
        train_all=train_all,
        seed=seed,
    )
    train_raw, train_y = _stack(records, train_indices)
    mean = train_raw.mean(axis=(0, 1))
    std = train_raw.std(axis=(0, 1)) + 1e-6
    train_x = _normalize(train_raw, mean, std)
    val_x = np.empty((0, CHANNELS, int(round(window_s * EMG_FS))), dtype=np.float32)
    val_y = np.empty(0, dtype=np.int64)
    if val_indices:
        val_raw, val_y = _stack(records, val_indices)
        val_x = _normalize(val_raw, mean, std)

    model = create_model(model_name, channels=CHANNELS, classes=len(LABELS), dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    class_weights = _class_weights(train_y)
    print(f"class weights: {dict(zip(LABELS, [float(value) for value in class_weights], strict=False))}")
    loss_weight = torch.from_numpy(class_weights).to(device) if balanced_loss else None
    loss_fn = nn.CrossEntropyLoss(weight=loss_weight)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=8)

    train_loader = _make_train_loader(
        train_x,
        train_y,
        batch_size=batch_size,
        augment=augment,
        balanced_sampler=balanced_sampler,
        num_workers=num_workers,
        seed=seed,
        device=device,
    )
    eval_train_loader = _make_eval_loader(train_x, train_y, batch_size=batch_size, num_workers=num_workers, device=device)
    val_loader = _make_eval_loader(val_x, val_y, batch_size=batch_size, num_workers=num_workers, device=device) if val_y.size else None

    best_score = -1.0
    best_epoch = 0
    best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
    epochs_without_improvement = 0
    final_train_metrics: dict[str, Any] = {}
    final_val_metrics: dict[str, Any] | None = None

    for epoch in range(1, int(epochs) + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * int(yb.shape[0])
            seen += int(yb.shape[0])

        final_train_metrics = evaluate_model(
            model,
            eval_train_loader,
            device=device,
            records=records,
            indices=train_indices,
        )
        if val_loader is not None:
            final_val_metrics = evaluate_model(
                model,
                val_loader,
                device=device,
                records=records,
                indices=val_indices,
            )
            score = float(final_val_metrics["accuracy"])
        else:
            score = float(final_train_metrics["accuracy"])
        scheduler.step(score)

        avg_loss = running_loss / max(1, seen)
        print(
            f"epoch {epoch:03d} loss={avg_loss:.4f} "
            f"train_acc={final_train_metrics['accuracy']:.3f} "
            f"val_acc={(final_val_metrics or {}).get('accuracy', float('nan')):.3f}"
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if patience > 0 and epochs_without_improvement >= patience:
            print(f"early stopping at epoch {epoch}; best_epoch={best_epoch}")
            break

    if save_best:
        model.load_state_dict(best_state)
        model.to(device)
        final_train_metrics = evaluate_model(model, eval_train_loader, device=device, records=records, indices=train_indices)
        final_val_metrics = (
            evaluate_model(model, val_loader, device=device, records=records, indices=val_indices)
            if val_loader is not None
            else None
        )

    chosen_metrics = final_val_metrics or final_train_metrics
    report = {
        "best_epoch": int(best_epoch),
        "best_val_accuracy": float(best_score) if final_val_metrics is not None else None,
        "final_train_accuracy": float(final_train_metrics["accuracy"]),
        "final_val_accuracy": float(final_val_metrics["accuracy"]) if final_val_metrics else None,
        "train_accuracy": float(final_train_metrics["accuracy"]),
        "val_accuracy": float(final_val_metrics["accuracy"]) if final_val_metrics else None,
        "train_per_class_accuracy": final_train_metrics["per_class_accuracy"],
        "val_per_class_accuracy": final_val_metrics["per_class_accuracy"] if final_val_metrics else None,
        "per_class_precision_recall_f1": chosen_metrics["per_class_precision_recall_f1"],
        "confusion_matrix": chosen_metrics["confusion_matrix"],
        "per_session_accuracy": chosen_metrics.get("per_session_accuracy", {}),
        "train_windows": int(train_y.size),
        "val_windows": int(val_y.size),
        "samples_before_balance": int(samples_before_balance),
        "samples_after_balance": int(len(records)),
        "class_counts_before_balance": counts_before_balance,
        "class_counts_after_balance": counts_after_balance,
        "class_weights": {label: float(class_weights[index]) for index, label in enumerate(LABELS)},
        "session_window_counts": dict(Counter(record.session_id for record in records)),
        "holdout_session": holdout,
        "val_split": val_split,
        "augment": augment,
        "balanced_sampler": balanced_sampler,
        "balanced_loss": balanced_loss,
        "max_rest_ratio": max_rest_ratio,
        "device_used": device_info["device_used"],
        "cuda_available": device_info["cuda_available"],
        "seed": seed,
        "training_args": {
            "model": model_name,
            "signal": signal_type,
            "window_s": window_s,
            "stride_s": stride_s,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": learning_rate,
            "weight_decay": weight_decay,
            "dropout": dropout,
            "patience": patience,
            "num_workers": num_workers,
            "save_best": save_best,
            "preset": training_preset,
        },
        **device_info,
    }
    artifacts = _save_artifacts(
        output_dir=output_dir,
        model=model,
        state_dict=best_state if save_best else {key: value.detach().cpu() for key, value in model.state_dict().items()},
        model_name=model_name,
        signal_type=signal_type,
        window_s=window_s,
        stride_s=stride_s,
        dropout=dropout,
        mean=mean,
        std=std,
        report=report,
        export_torchscript=export_torchscript,
        training_preset=training_preset,
    )

    return {
        **artifacts,
        "samples": samples_before_balance,
        "trained_samples": len(records),
        "train_accuracy": report["train_accuracy"],
        "val_accuracy": report["val_accuracy"],
        "best_epoch": best_epoch,
        "best_val_accuracy": report["best_val_accuracy"],
    }


def run_cross_session_cv(
    dataset_root: Path,
    output_dir: Path,
    **kwargs: Any,
) -> dict[str, Any]:
    records = collect_window_records(
        dataset_root,
        signal_type=kwargs.get("signal_type", "raw"),
        window_s=kwargs.get("window_s", 1.0),
        stride_s=kwargs.get("stride_s", 0.1),
    )
    sessions = sorted({record.session_id for record in records})
    output_dir.mkdir(parents=True, exist_ok=True)
    folds = []
    for session in sessions:
        fold_dir = output_dir / f"fold_{session}"
        print(f"cross-session fold holdout={session}")
        result = train_model(
            dataset_root,
            fold_dir,
            holdout_session=session,
            val_split="session",
            verbose_device=not folds,
            **kwargs,
        )
        report = json.loads((fold_dir / "train_report.json").read_text(encoding="utf-8"))
        folds.append(
            {
                "holdout_session": session,
                "val_accuracy": report["final_val_accuracy"],
                "per_class_accuracy": report["val_per_class_accuracy"],
                "confusion_matrix": report["confusion_matrix"],
                "fold_dir": str(fold_dir),
            }
        )

    accuracies = [fold["val_accuracy"] for fold in folds if fold["val_accuracy"] is not None]
    per_class_avg = {}
    for label in LABELS:
        values = [
            float(fold["per_class_accuracy"][label])
            for fold in folds
            if fold["per_class_accuracy"] is not None
        ]
        per_class_avg[label] = float(np.mean(values)) if values else 0.0
    report = {
        "folds": folds,
        "mean_val_accuracy": float(np.mean(accuracies)) if accuracies else None,
        "mean_per_class_accuracy": per_class_avg,
        "session_count": len(sessions),
    }
    (output_dir / "cv_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Train EMG gesture classifier.")
    parser.add_argument("--dataset-root", default="dataset", type=Path)
    parser.add_argument("--output-dir", default=None, type=Path)
    parser.add_argument("--preset", choices=["calibration"])
    parser.add_argument("--model", choices=[MODEL_CNN, MODEL_TCN], default=MODEL_TCN)
    parser.add_argument("--signal", choices=["raw", "filtered"], default="raw")
    parser.add_argument("--window-s", default=1.0, type=float)
    parser.add_argument("--stride-s", default=0.1, type=float)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--epochs", default=150, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--dropout", default=0.2, type=float)
    parser.add_argument("--val-split", choices=["session", "trial"], default="session")
    parser.add_argument("--holdout-session")
    parser.add_argument("--train-all", action="store_true")
    parser.add_argument("--export-torchscript", action="store_true", default=True)
    parser.add_argument("--no-export-torchscript", dest="export_torchscript", action="store_false")
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--patience", default=25, type=int)
    parser.add_argument("--max-rest-ratio", default=1.0, type=float)
    parser.add_argument("--balanced-sampler", action="store_true", default=True)
    parser.add_argument("--no-balanced-sampler", dest="balanced_sampler", action="store_false")
    parser.add_argument("--balanced-loss", action="store_true", default=False)
    parser.add_argument("--cross-session-cv", action="store_true")
    parser.add_argument("--save-best", action="store_true", default=True)
    parser.add_argument("--no-save-best", dest="save_best", action="store_false")
    args = parser.parse_args()
    if args.preset == "calibration":
        args.model = MODEL_TCN
        args.train_all = True
        args.epochs = 80
        args.batch_size = 128
        args.max_rest_ratio = 1.0
        args.balanced_sampler = True
        args.balanced_loss = True
        args.save_best = True
        args.export_torchscript = True
        args.window_s = 1.0
        args.stride_s = 0.1
    if args.output_dir is None:
        args.output_dir = (
            Path("models") / "calibration_game_model"
            if args.preset == "calibration"
            else Path("models") / "emg2pose_gesture_v1"
        )
    dataset_root = resolve_dataset_root(args.dataset_root)
    common_args = {
        "model_name": args.model,
        "signal_type": args.signal,
        "window_s": args.window_s,
        "stride_s": args.stride_s,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "train_all": args.train_all,
        "export_torchscript": args.export_torchscript,
        "augment": args.augment,
        "device_arg": args.device,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "patience": args.patience,
        "max_rest_ratio": args.max_rest_ratio,
        "balanced_sampler": args.balanced_sampler,
        "balanced_loss": args.balanced_loss,
        "save_best": args.save_best,
        "training_preset": args.preset,
    }
    if args.cross_session_cv:
        cv_dir = args.output_dir
        if cv_dir.name == "emg2pose_gesture_v1":
            cv_dir = cv_dir.parent / "emg2pose_gesture_cv"
        common_args["train_all"] = False
        report = run_cross_session_cv(dataset_root, cv_dir, **common_args)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    result = train_model(
        dataset_root,
        args.output_dir,
        val_split=args.val_split,
        holdout_session=args.holdout_session,
        **common_args,
    )
    print(json.dumps({key: str(value) for key, value in result.items()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
