"""Fine-tune an EffiE-style 8ch/200Hz sEMG classifier on collected gestures."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from emg_live_marker.ml.effie_adapter import (
    MODEL_TYPE,
    EffieGestureNet,
    load_effie_checkpoint,
    set_trainable_mode,
)
from emg_live_marker.ml.effie_preprocess import (
    EFFIE_CHANNELS,
    EFFIE_FS,
    EFFIE_STRIDE_S,
    EFFIE_STRIDE_SAMPLES,
    EFFIE_WINDOW_S,
    EFFIE_WINDOW_SAMPLES,
    SOURCE_FS,
    normalize_effie_batch,
    slice_effie_windows,
)
from emg_live_marker.ml.gesture_model import LABELS
from emg_live_marker.cli.train_gesture_classifier import (
    COLLECT_TO_GAME_LABEL,
    _event_sample,
    _metrics_from_confusion,
    _print_device_info,
    _select_device,
    _slice_by_sample_index,
    discover_session_dirs,
    event_groups,
    load_emg,
    read_csv_dicts,
    resolve_dataset_root,
)
from emg_live_marker.paths import add_path_arguments, resolve_paths_from_args, resolve_project_path
from emg_live_marker.run_naming import build_run_id


@dataclass(frozen=True)
class EffieWindowRecord:
    x: np.ndarray
    y: int
    session_id: str
    trial_id: str
    gesture: str
    phase: str


class EffieDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.x[index]), torch.tensor(int(self.y[index]), dtype=torch.long)


def _to_effie_input(window_200hz: np.ndarray) -> np.ndarray:
    return np.transpose(window_200hz, (1, 0))[:, :, None].astype(np.float32, copy=False)


def collect_effie_records(dataset_root: Path) -> list[EffieWindowRecord]:
    records: list[EffieWindowRecord] = []
    gesture_trim = int(round(0.3 * SOURCE_FS))
    rest_trim = int(round(0.2 * SOURCE_FS))
    for session_dir in discover_session_dirs(dataset_root):
        session_id = session_dir.name
        sample_index, _software_time, emg = load_emg(session_dir / "emg.csv", signal_type="raw")
        grouped = event_groups(read_csv_dicts(session_dir / "events.csv"))
        for trial_id, phases in sorted(grouped.items()):
            required = {"trial_start", "gesture_start", "gesture_end", "trial_end"}
            if not required <= set(phases):
                continue
            gesture = phases["gesture_start"].get("gesture", "")
            label = COLLECT_TO_GAME_LABEL.get(gesture)
            if label is None:
                continue
            trial_start = _event_sample(phases["trial_start"])
            gesture_start = _event_sample(phases["gesture_start"])
            gesture_end = _event_sample(phases["gesture_end"])
            trial_end = _event_sample(phases["trial_end"])
            if None in (trial_start, gesture_start, gesture_end, trial_end):
                continue
            intervals = [
                (gesture_start + gesture_trim, gesture_end - gesture_trim, label, gesture, "gesture"),
                (trial_start + rest_trim, gesture_start - rest_trim, "rest", "rest", "rest_before"),
                (gesture_end + rest_trim, trial_end - rest_trim, "rest", "rest", "rest_after"),
            ]
            for start_index, end_index, game_label, out_gesture, phase in intervals:
                if start_index >= end_index:
                    continue
                segment = _slice_by_sample_index(emg, sample_index, start_index=start_index, end_index=end_index)
                for window, window_label in slice_effie_windows(segment, label=game_label):
                    records.append(
                        EffieWindowRecord(
                            x=_to_effie_input(window),
                            y=LABELS.index(window_label),
                            session_id=session_id,
                            trial_id=trial_id,
                            gesture=out_gesture,
                            phase=phase,
                        )
                    )
    if not records:
        raise ValueError(f"No EffiE training windows found under {dataset_root}")
    return records


def _counts(records: list[EffieWindowRecord], indices: list[int] | None = None) -> dict[str, int]:
    selected = range(len(records)) if indices is None else indices
    counter = Counter(LABELS[records[index].y] for index in selected)
    return {label: int(counter.get(label, 0)) for label in LABELS}


def _balance_rest(
    records: list[EffieWindowRecord],
    *,
    max_rest_ratio: float,
    seed: int,
) -> tuple[list[EffieWindowRecord], dict[str, int], dict[str, int]]:
    before = _counts(records)
    max_action = max(before[label] for label in LABELS if label != "rest")
    rest_limit = int(round(max_action * max_rest_ratio))
    if before["rest"] <= rest_limit:
        return records, before, before
    rest_indices = [index for index, record in enumerate(records) if LABELS[record.y] == "rest"]
    action_indices = [index for index, record in enumerate(records) if LABELS[record.y] != "rest"]
    rng = random.Random(seed)
    rng.shuffle(rest_indices)
    keep = sorted(action_indices + rest_indices[:rest_limit])
    balanced = [records[index] for index in keep]
    return balanced, before, _counts(balanced)


def _split(
    records: list[EffieWindowRecord],
    *,
    val_split: str,
    holdout_session: str | None,
    seed: int,
) -> tuple[list[int], list[int], str | None]:
    if val_split == "session":
        sessions = sorted({record.session_id for record in records})
        if len(sessions) < 2:
            return list(range(len(records))), [], None
        holdout = holdout_session or sessions[-1]
        train = [index for index, record in enumerate(records) if record.session_id != holdout]
        val = [index for index, record in enumerate(records) if record.session_id == holdout]
        return train, val, holdout
    by_key: dict[tuple[int, str, str], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_key[(record.y, record.session_id, record.trial_id)].append(index)
    rng = random.Random(seed)
    train: list[int] = []
    val: list[int] = []
    for label in range(len(LABELS)):
        keys = [key for key in by_key if key[0] == label]
        if len(keys) < 2:
            for key in keys:
                train.extend(by_key[key])
            continue
        rng.shuffle(keys)
        val_count = max(1, int(round(len(keys) * 0.2)))
        val_keys = set(keys[:val_count])
        for key in keys:
            if key in val_keys:
                val.extend(by_key[key])
            else:
                train.extend(by_key[key])
    return sorted(train), sorted(val), None


def _stack(records: list[EffieWindowRecord], indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
    x = np.stack([records[index].x for index in indices]).astype(np.float32)
    y = np.asarray([records[index].y for index in indices], dtype=np.int64)
    return x, y


def _class_weights(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y, minlength=len(LABELS)).astype(np.float32)
    return (float(y.size) / (len(LABELS) * np.maximum(counts, 1.0))).astype(np.float32)


def _loader(
    x: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    balanced_sampler: bool,
    seed: int,
) -> DataLoader:
    sampler = None
    shuffle = True
    generator = torch.Generator().manual_seed(seed)
    if balanced_sampler:
        weights = _class_weights(y)
        sample_weights = torch.as_tensor([weights[int(label)] for label in y], dtype=torch.double)
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True, generator=generator)
        shuffle = False
    return DataLoader(EffieDataset(x, y), batch_size=batch_size, shuffle=shuffle, sampler=sampler)


def _evaluate(model: nn.Module, loader: DataLoader, *, device: torch.device) -> dict[str, Any]:
    confusion = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb).argmax(dim=1)
            for target, predicted in zip(yb.cpu().numpy(), pred.cpu().numpy(), strict=False):
                confusion[int(target), int(predicted)] += 1
    return _metrics_from_confusion(confusion)


def _write_confusion(path: Path, matrix: list[list[int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["actual\\predicted", *LABELS])
        for label, row in zip(LABELS, matrix, strict=False):
            writer.writerow([label, *row])


def finetune_effie(
    dataset_root: Path,
    output_dir: Path,
    *,
    checkpoint_path: Path | None = None,
    mode: str = "freeze_backbone",
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-4,
    device_arg: str = "auto",
    val_split: str = "trial",
    holdout_session: str | None = None,
    max_rest_ratio: float = 1.0,
    balanced_sampler: bool = True,
    export_torchscript: bool = True,
    seed: int = 42,
    run_id: str | None = None,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = _select_device(device_arg)
    device_info = _print_device_info(device)
    records = collect_effie_records(dataset_root)
    samples_before_balance = len(records)
    records, counts_before, counts_after = _balance_rest(records, max_rest_ratio=max_rest_ratio, seed=seed)
    train_indices, val_indices, holdout = _split(records, val_split=val_split, holdout_session=holdout_session, seed=seed)
    train_raw, train_y = _stack(records, train_indices)
    val_raw = np.empty((0, EFFIE_CHANNELS, EFFIE_WINDOW_SAMPLES, 1), dtype=np.float32)
    val_y = np.empty(0, dtype=np.int64)
    if val_indices:
        val_raw, val_y = _stack(records, val_indices)
    mean = train_raw.mean(axis=(0, 2, 3))
    std = train_raw.std(axis=(0, 2, 3)) + 1e-6
    train_x = normalize_effie_batch(train_raw, mean, std)
    val_x = normalize_effie_batch(val_raw, mean, std) if val_y.size else val_raw

    model = EffieGestureNet(classes=len(LABELS))
    checkpoint_info = load_effie_checkpoint(model, checkpoint_path, strict=False) if checkpoint_path else {"loaded": False}
    set_trainable_mode(model, mode)
    model.to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(_class_weights(train_y)).to(device))
    train_loader = _loader(train_x, train_y, batch_size=batch_size, balanced_sampler=balanced_sampler, seed=seed)
    eval_train_loader = _loader(train_x, train_y, batch_size=batch_size, balanced_sampler=False, seed=seed)
    val_loader = _loader(val_x, val_y, batch_size=batch_size, balanced_sampler=False, seed=seed) if val_y.size else None
    best_score = -1.0
    best_epoch = 0
    best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
    final_train = {}
    final_val = None
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
        final_train = _evaluate(model, eval_train_loader, device=device)
        final_val = _evaluate(model, val_loader, device=device) if val_loader is not None else None
        score = final_val["accuracy"] if final_val is not None else final_train["accuracy"]
        print(f"epoch {epoch:03d} train_acc={final_train['accuracy']:.3f} val_acc={(final_val or {}).get('accuracy', float('nan')):.3f}")
        if score > best_score:
            best_score = float(score)
            best_epoch = epoch
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})

    model.load_state_dict(best_state)
    model.to(device)
    final_train = _evaluate(model, eval_train_loader, device=device)
    final_val = _evaluate(model, val_loader, device=device) if val_loader is not None else None
    chosen = final_val or final_train
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "gesture_classifier.pt"
    ts_path = output_dir / "gesture_classifier.ts"
    torch.save({"model_state_dict": best_state, "labels": LABELS, "model_type": MODEL_TYPE}, model_path)
    (output_dir / "gesture_labels.json").write_text(json.dumps({"labels": LABELS}, indent=2), encoding="utf-8")
    normalization = {
        "signal_type": "raw",
        "source_fs": SOURCE_FS,
        "model_fs": EFFIE_FS,
        "window_s": EFFIE_WINDOW_S,
        "source_window_s": 0.5,
        "window_samples": EFFIE_WINDOW_SAMPLES,
        "channels": EFFIE_CHANNELS,
        "mean": [float(value) for value in mean.tolist()],
        "std": [float(value) for value in std.tolist()],
    }
    (output_dir / "normalization.json").write_text(json.dumps(normalization, indent=2), encoding="utf-8")
    model_info = {
        "model_type": MODEL_TYPE,
        "base_model": "MIC-Laboratory/IEEE-NER-2023-EffiE",
        "source_fs": SOURCE_FS,
        "model_fs": EFFIE_FS,
        "channels": EFFIE_CHANNELS,
        "window_s": EFFIE_WINDOW_S,
        "source_window_s": 0.5,
        "window_samples": EFFIE_WINDOW_SAMPLES,
        "stride_s": EFFIE_STRIDE_S,
        "labels": LABELS,
        "preprocess": "resample 250Hz to 200Hz, EffiE-style 8x32 windows, per-channel normalization",
        "signal_type": "raw",
    }
    (output_dir / "model_info.json").write_text(json.dumps(model_info, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {
        "run_id": run_id or output_dir.name,
        "model_type": MODEL_TYPE,
        "mode": mode,
        "checkpoint": checkpoint_info,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_score if final_val is not None else None,
        "train_accuracy": final_train["accuracy"],
        "val_accuracy": final_val["accuracy"] if final_val else None,
        "per_class_precision_recall_f1": chosen["per_class_precision_recall_f1"],
        "confusion_matrix": chosen["confusion_matrix"],
        "class_counts_before_balance": counts_before,
        "class_counts_after_balance": counts_after,
        "samples_before_balance": samples_before_balance,
        "samples_after_balance": len(records),
        "holdout_session": holdout,
        "val_split": val_split,
        "seed": seed,
        **device_info,
    }
    (output_dir / "train_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_confusion(output_dir / "confusion_matrix.csv", chosen["confusion_matrix"])
    if export_torchscript:
        cpu_model = EffieGestureNet(classes=len(LABELS))
        cpu_model.load_state_dict(best_state)
        cpu_model.eval()
        traced = torch.jit.trace(cpu_model, torch.zeros(1, EFFIE_CHANNELS, EFFIE_WINDOW_SAMPLES, 1))
        traced.save(str(ts_path))
    return {"model_path": model_path, "torchscript_path": ts_path, "train_report": output_dir / "train_report.json"}


def run_cross_session_cv(dataset_root: Path, output_dir: Path, **kwargs: Any) -> dict[str, Any]:
    sessions = sorted(path.name for path in discover_session_dirs(dataset_root))
    output_dir.mkdir(parents=True, exist_ok=True)
    folds = []
    for session in sessions:
        fold_dir = output_dir / f"fold_{session}"
        result = finetune_effie(dataset_root, fold_dir, holdout_session=session, val_split="session", **kwargs)
        report = json.loads((fold_dir / "train_report.json").read_text(encoding="utf-8"))
        folds.append(
            {
                "holdout_session": session,
                "val_accuracy": report.get("val_accuracy"),
                "confusion_matrix": report.get("confusion_matrix"),
                "fold_dir": str(fold_dir),
            }
        )
    values = [float(fold["val_accuracy"]) for fold in folds if fold["val_accuracy"] is not None]
    report = {
        "run_id": kwargs.get("run_id") or output_dir.name,
        "seed": kwargs.get("seed"),
        "folds": folds,
        "mean_val_accuracy": float(np.mean(values)) if values else None,
    }
    (output_dir / "cv_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune EffiE-style gesture classifier.")
    parser.add_argument("--effie-root", required=True, type=Path)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--output-dir", default=None, type=Path)
    parser.add_argument("--run-id", help="override the generated name when --output-dir is omitted")
    add_path_arguments(parser)
    parser.add_argument("--mode", choices=["freeze_backbone", "finetune_all"], default="freeze_backbone")
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--val-split", choices=["trial", "session"], default="trial")
    parser.add_argument("--holdout-session")
    parser.add_argument("--cross-session-cv", action="store_true")
    parser.add_argument("--max-rest-ratio", default=1.0, type=float)
    parser.add_argument("--balanced-sampler", action="store_true", default=True)
    parser.add_argument("--no-balanced-sampler", dest="balanced_sampler", action="store_false")
    parser.add_argument("--export-torchscript", action="store_true", default=True)
    parser.add_argument("--no-export-torchscript", dest="export_torchscript", action="store_false")
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()
    paths = resolve_paths_from_args(args)
    dataset_root = resolve_dataset_root(args.dataset_root, paths)
    args.effie_root = resolve_project_path(args.effie_root, paths)
    if args.checkpoint_path is not None:
        args.checkpoint_path = resolve_project_path(args.checkpoint_path, paths)
    split = "cross-session-cv" if args.cross_session_cv else f"{args.val_split}-split"
    if args.output_dir is None:
        run_id = args.run_id or build_run_id(
            model="effie",
            split=split,
            mode=args.mode,
            seed=args.seed,
        )
        args.output_dir = paths.models_root / run_id
    else:
        args.output_dir = resolve_project_path(args.output_dir, paths)
        run_id = args.run_id or args.output_dir.name
    common = {
        "checkpoint_path": args.checkpoint_path,
        "mode": args.mode,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "device_arg": args.device,
        "max_rest_ratio": args.max_rest_ratio,
        "balanced_sampler": args.balanced_sampler,
        "export_torchscript": args.export_torchscript,
        "seed": args.seed,
        "run_id": run_id,
    }
    if args.cross_session_cv:
        report = run_cross_session_cv(dataset_root, args.output_dir, **common)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    result = finetune_effie(
        dataset_root,
        args.output_dir,
        val_split=args.val_split,
        holdout_session=args.holdout_session,
        **common,
    )
    print(json.dumps({key: str(value) for key, value in result.items()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
