"""Offline replay evaluation for realtime gesture smoothing."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from emg_live_marker.ml.gesture_model import LABELS, load_model
from emg_live_marker.cli.train_gesture_classifier import (
    COLLECT_TO_GAME_LABEL,
    discover_session_dirs,
    event_groups,
    load_emg,
    read_csv_dicts,
    resolve_dataset_root,
)
from emg_live_marker.paths import add_path_arguments, resolve_paths_from_args, resolve_project_path

EMG_FS = 250.0
ACTION_LABELS = tuple(label for label in LABELS if label != "rest")
LOG_FIELDS = [
    "timestamp",
    "session_id",
    "trial_id",
    "true_phase",
    "true_gesture",
    "raw_gesture",
    "raw_confidence",
    "smooth_gesture",
    "smooth_confidence",
    "raw_prob_rest",
    "raw_prob_fist",
    "raw_prob_open_palm",
    "raw_prob_pinch",
    "smooth_prob_rest",
    "smooth_prob_fist",
    "smooth_prob_open_palm",
    "smooth_prob_pinch",
    "sent_to_game",
]


@dataclass(frozen=True)
class TrialInterval:
    session_id: str
    trial_id: str
    gesture: str
    game_gesture: str
    trial_start: float
    gesture_start: float
    gesture_end: float
    trial_end: float


def _event_sample_time(event: dict[str, str]) -> float:
    return float(int(float(event["sample_index"])) / EMG_FS)


def _prob_vector(probs: dict[str, float]) -> np.ndarray:
    return np.asarray([float(probs.get(label, 0.0)) for label in LABELS], dtype=np.float64)


def _prob_dict(values: np.ndarray) -> dict[str, float]:
    return {label: float(values[index]) for index, label in enumerate(LABELS)}


def _safe_div(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _parse_number_list(value: str | None, cast) -> list[Any] | None:
    if value is None or value.strip() == "":
        return None
    return [cast(part.strip()) for part in value.split(",") if part.strip()]


def load_trial_intervals(session_dir: Path) -> list[TrialInterval]:
    grouped = event_groups(read_csv_dicts(session_dir / "events.csv"))
    intervals: list[TrialInterval] = []
    required = {"trial_start", "gesture_start", "gesture_end", "trial_end"}
    for trial_id, phases in sorted(grouped.items()):
        if not required <= set(phases):
            continue
        gesture = phases["gesture_start"].get("gesture", "")
        game_gesture = COLLECT_TO_GAME_LABEL.get(gesture)
        if game_gesture is None:
            continue
        try:
            intervals.append(
                TrialInterval(
                    session_id=session_dir.name,
                    trial_id=trial_id,
                    gesture=gesture,
                    game_gesture=game_gesture,
                    trial_start=_event_sample_time(phases["trial_start"]),
                    gesture_start=_event_sample_time(phases["gesture_start"]),
                    gesture_end=_event_sample_time(phases["gesture_end"]),
                    trial_end=_event_sample_time(phases["trial_end"]),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return intervals


def true_label_at(timestamp: float, intervals: list[TrialInterval]) -> tuple[str, str, str, float | None]:
    for interval in intervals:
        if interval.trial_start <= timestamp < interval.gesture_start:
            return interval.trial_id, "rest_before", "rest", interval.gesture_start
        if interval.gesture_start <= timestamp < interval.gesture_end:
            return interval.trial_id, "gesture_hold", interval.game_gesture, interval.gesture_start
        if interval.gesture_end <= timestamp < interval.trial_end:
            return interval.trial_id, "rest_after", "rest", interval.gesture_start
    return "", "outside_trial", "rest", None


def discover_eval_sessions(dataset_root: Path, session: str | None = None) -> list[Path]:
    sessions = discover_session_dirs(resolve_dataset_root(dataset_root))
    if session:
        sessions = [path for path in sessions if path.name == session]
    if not sessions:
        raise ValueError(f"No dataset sessions found under {dataset_root}")
    return sessions


def _sorted_emg(session_dir: Path, signal_type: str) -> tuple[np.ndarray, np.ndarray]:
    sample_index, _software_time, emg = load_emg(session_dir / "emg.csv", signal_type=signal_type)
    if sample_index.size == 0:
        raise ValueError(f"{session_dir}: emg.csv has no samples")
    order = np.argsort(sample_index, kind="stable")
    sample_index = sample_index[order]
    emg = emg[order]
    if np.any(np.diff(sample_index) <= 0):
        raise ValueError(f"{session_dir}: sample_index must be strictly increasing for replay")
    return sample_index, emg


def generate_raw_predictions(
    *,
    dataset_root: Path,
    predictor: Any,
    window_s: float,
    stride_s: float,
    signal_type: str,
    session: str | None = None,
) -> list[dict[str, Any]]:
    del window_s  # The loaded predictor metadata determines the actual model window.
    window_samples = int(getattr(predictor, "window_samples", round(EMG_FS)))
    stride_samples = max(1, int(round(float(stride_s) * EMG_FS)))
    rows: list[dict[str, Any]] = []

    for session_dir in discover_eval_sessions(dataset_root, session=session):
        sample_index, emg = _sorted_emg(session_dir, signal_type)
        intervals = load_trial_intervals(session_dir)
        first_sample = int(sample_index[0])
        last_sample = int(sample_index[-1])
        first_end = first_sample + window_samples - 1
        for end_sample in range(first_end, last_sample + 1, stride_samples):
            start_sample = end_sample - window_samples + 1
            left = int(np.searchsorted(sample_index, start_sample, side="left"))
            right = int(np.searchsorted(sample_index, end_sample, side="right"))
            window = emg[left:right]
            if window.shape[0] < window_samples:
                continue
            if window.shape[0] > window_samples:
                window = window[-window_samples:]
            timestamp = end_sample / EMG_FS
            prediction = predictor.predict_window(window.astype(np.float32, copy=False))
            probs = {label: float(prediction.get("probs", {}).get(label, 0.0)) for label in LABELS}
            raw_gesture = max(probs, key=probs.get)
            trial_id, true_phase, true_gesture, gesture_start = true_label_at(timestamp, intervals)
            rows.append(
                {
                    "timestamp": float(timestamp),
                    "session_id": session_dir.name,
                    "trial_id": trial_id,
                    "true_phase": true_phase,
                    "true_gesture": true_gesture,
                    "gesture_start_timestamp": gesture_start,
                    "raw_gesture": raw_gesture,
                    "raw_confidence": float(probs[raw_gesture]),
                    "raw_probs": probs,
                }
            )
    if not rows:
        raise ValueError("No replay windows were generated")
    return rows


class SmoothingState:
    def __init__(self, smoothing_frames: int, change_confirm_frames: int, threshold: float) -> None:
        self.history: deque[np.ndarray] = deque(maxlen=max(1, int(smoothing_frames)))
        self.change_confirm_frames = max(1, int(change_confirm_frames))
        self.threshold = float(threshold)
        self.current_output: str | None = None
        self.candidate_gesture: str | None = None
        self.candidate_count = 0

    def update(self, probs: dict[str, float]) -> tuple[str, float, dict[str, float]]:
        self.history.append(_prob_vector(probs))
        average = np.mean(np.stack(list(self.history)), axis=0)
        smooth_probs = _prob_dict(average)
        candidate = LABELS[int(np.argmax(average))]
        if float(smooth_probs[candidate]) < self.threshold:
            candidate = "rest"
        gesture = self._confirm_candidate(candidate)
        return gesture, float(smooth_probs.get(gesture, 0.0)), smooth_probs

    def _confirm_candidate(self, candidate: str) -> str:
        if self.current_output is None:
            self.current_output = candidate
            self.candidate_gesture = candidate
            self.candidate_count = 1
            return candidate
        if candidate == self.current_output:
            self.candidate_gesture = candidate
            self.candidate_count = 0
            return candidate
        if candidate == self.candidate_gesture:
            self.candidate_count += 1
        else:
            self.candidate_gesture = candidate
            self.candidate_count = 1
        if self.candidate_count >= self.change_confirm_frames:
            self.current_output = candidate
            self.candidate_count = 0
        return self.current_output


def apply_smoothing(
    raw_rows: list[dict[str, Any]],
    *,
    smoothing_frames: int,
    change_confirm_frames: int,
    confidence_threshold: float,
) -> list[dict[str, Any]]:
    states: dict[str, SmoothingState] = {}
    previous_sent: dict[str, str] = {}
    out: list[dict[str, Any]] = []
    for row in raw_rows:
        session_id = str(row["session_id"])
        state = states.setdefault(
            session_id,
            SmoothingState(smoothing_frames, change_confirm_frames, confidence_threshold),
        )
        smooth_gesture, smooth_confidence, smooth_probs = state.update(dict(row["raw_probs"]))
        sent_to_game = previous_sent.get(session_id) != smooth_gesture
        previous_sent[session_id] = smooth_gesture
        merged = dict(row)
        merged.update(
            {
                "smooth_gesture": smooth_gesture,
                "smooth_confidence": smooth_confidence,
                "smooth_probs": smooth_probs,
                "sent_to_game": bool(sent_to_game),
            }
        )
        out.append(merged)
    return out


def _switch_count(rows: list[dict[str, Any]], pred_key: str) -> int:
    count = 0
    previous_by_session: dict[str, str] = {}
    for row in rows:
        session_id = str(row["session_id"])
        gesture = str(row[pred_key])
        previous = previous_by_session.get(session_id)
        if previous is not None and previous != gesture:
            count += 1
        previous_by_session[session_id] = gesture
    return count


def _duration_minutes(rows: list[dict[str, Any]]) -> float:
    spans = []
    for session_id in sorted({str(row["session_id"]) for row in rows}):
        timestamps = [float(row["timestamp"]) for row in rows if row["session_id"] == session_id]
        if timestamps:
            spans.append(max(timestamps) - min(timestamps))
    return max(sum(spans) / 60.0, 1e-9)


def _detection_delays(rows: list[dict[str, Any]], pred_key: str) -> dict[str, list[float]]:
    by_trial: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row["true_phase"] == "gesture_hold" and row["trial_id"]:
            by_trial.setdefault((str(row["session_id"]), str(row["trial_id"])), []).append(row)

    delays: dict[str, list[float]] = {label: [] for label in ACTION_LABELS}
    for trial_rows in by_trial.values():
        trial_rows = sorted(trial_rows, key=lambda item: float(item["timestamp"]))
        true_gesture = str(trial_rows[0]["true_gesture"])
        gesture_start = trial_rows[0].get("gesture_start_timestamp")
        if true_gesture not in delays or gesture_start is None:
            continue
        for row in trial_rows:
            if row[pred_key] == true_gesture:
                delays[true_gesture].append((float(row["timestamp"]) - float(gesture_start)) * 1000.0)
                break
    return delays


def _delay_stats(delays: dict[str, list[float]]) -> tuple[float | None, float | None, dict[str, dict[str, float | None]]]:
    all_delays = [value for values in delays.values() for value in values]
    per_class = {}
    for label in ACTION_LABELS:
        values = delays.get(label, [])
        per_class[label] = {
            "average": float(np.mean(values)) if values else None,
            "median": float(np.median(values)) if values else None,
            "count": int(len(values)),
        }
    return (
        float(np.mean(all_delays)) if all_delays else None,
        float(np.median(all_delays)) if all_delays else None,
        per_class,
    )


def compute_metrics(rows: list[dict[str, Any]], pred_key: str) -> dict[str, Any]:
    total = len(rows)
    correct = sum(1 for row in rows if row["true_gesture"] == row[pred_key])
    action_rows = [row for row in rows if row["true_phase"] == "gesture_hold"]
    rest_rows = [row for row in rows if row["true_gesture"] == "rest"]
    false_triggers = sum(1 for row in rest_rows if row[pred_key] in ACTION_LABELS)
    switch_count = _switch_count(rows, pred_key)
    duration_min = _duration_minutes(rows)

    per_class_accuracy = {}
    for label in LABELS:
        label_rows = [row for row in rows if row["true_gesture"] == label]
        per_class_accuracy[label] = _safe_div(
            sum(1 for row in label_rows if row[pred_key] == label),
            len(label_rows),
        )
    delays = _detection_delays(rows, pred_key)
    avg_delay, median_delay, per_class_delay = _delay_stats(delays)
    return {
        "accuracy": _safe_div(correct, total),
        "action_accuracy": _safe_div(
            sum(1 for row in action_rows if row["true_gesture"] == row[pred_key]),
            len(action_rows),
        ),
        "rest_accuracy": _safe_div(
            sum(1 for row in rest_rows if row[pred_key] == "rest"),
            len(rest_rows),
        ),
        "false_trigger_rate": _safe_div(false_triggers, len(rest_rows)),
        "switch_count": int(switch_count),
        "switches_per_min": float(switch_count / duration_min),
        "average_detection_delay_ms": avg_delay,
        "median_detection_delay_ms": median_delay,
        "per_class_accuracy": per_class_accuracy,
        "per_class_delay_ms": per_class_delay,
    }


def build_summary(rows: list[dict[str, Any]], *, include_sessions: bool = True) -> dict[str, Any]:
    raw = compute_metrics(rows, "raw_gesture")
    smooth = compute_metrics(rows, "smooth_gesture")
    summary = {
        "raw_accuracy": raw["accuracy"],
        "smoothed_accuracy": smooth["accuracy"],
        "raw_action_accuracy": raw["action_accuracy"],
        "smoothed_action_accuracy": smooth["action_accuracy"],
        "raw_rest_accuracy": raw["rest_accuracy"],
        "smoothed_rest_accuracy": smooth["rest_accuracy"],
        "raw_false_trigger_rate": raw["false_trigger_rate"],
        "smoothed_false_trigger_rate": smooth["false_trigger_rate"],
        "raw_switch_count": raw["switch_count"],
        "smoothed_switch_count": smooth["switch_count"],
        "raw_switches_per_min": raw["switches_per_min"],
        "smoothed_switches_per_min": smooth["switches_per_min"],
        "average_detection_delay_ms": smooth["average_detection_delay_ms"],
        "median_detection_delay_ms": smooth["median_detection_delay_ms"],
        "per_class_accuracy": {
            "raw": raw["per_class_accuracy"],
            "smoothed": smooth["per_class_accuracy"],
        },
        "per_class_delay_ms": smooth["per_class_delay_ms"],
        "session_results": {},
    }
    if include_sessions:
        for session_id in sorted({str(row["session_id"]) for row in rows}):
            session_rows = [row for row in rows if row["session_id"] == session_id]
            summary["session_results"][session_id] = build_summary(session_rows, include_sessions=False)
    return summary


def _log_row(row: dict[str, Any]) -> dict[str, Any]:
    raw_probs = dict(row["raw_probs"])
    smooth_probs = dict(row["smooth_probs"])
    return {
        "timestamp": f"{float(row['timestamp']):.6f}",
        "session_id": row["session_id"],
        "trial_id": row["trial_id"],
        "true_phase": row["true_phase"],
        "true_gesture": row["true_gesture"],
        "raw_gesture": row["raw_gesture"],
        "raw_confidence": f"{float(row['raw_confidence']):.6f}",
        "smooth_gesture": row["smooth_gesture"],
        "smooth_confidence": f"{float(row['smooth_confidence']):.6f}",
        "raw_prob_rest": f"{raw_probs.get('rest', 0.0):.6f}",
        "raw_prob_fist": f"{raw_probs.get('fist', 0.0):.6f}",
        "raw_prob_open_palm": f"{raw_probs.get('open-palm', 0.0):.6f}",
        "raw_prob_pinch": f"{raw_probs.get('pinch', 0.0):.6f}",
        "smooth_prob_rest": f"{smooth_probs.get('rest', 0.0):.6f}",
        "smooth_prob_fist": f"{smooth_probs.get('fist', 0.0):.6f}",
        "smooth_prob_open_palm": f"{smooth_probs.get('open-palm', 0.0):.6f}",
        "smooth_prob_pinch": f"{smooth_probs.get('pinch', 0.0):.6f}",
        "sent_to_game": "true" if row["sent_to_game"] else "false",
    }


def write_log(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=LOG_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(_log_row(row))


def write_sweep_summary(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "smoothing_frames",
        "change_confirm_frames",
        "confidence_threshold",
        "raw_accuracy",
        "smoothed_accuracy",
        "raw_switch_count",
        "smoothed_switch_count",
        "raw_false_trigger_rate",
        "smoothed_false_trigger_rate",
        "average_detection_delay_ms",
        "median_detection_delay_ms",
    ]
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def evaluate_realtime_smoothing(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictor = load_model(args.model_path)
    raw_rows = generate_raw_predictions(
        dataset_root=Path(args.dataset_root),
        predictor=predictor,
        window_s=float(args.window_s),
        stride_s=float(args.stride_s),
        signal_type=str(args.signal),
        session=args.session,
    )
    rows = apply_smoothing(
        raw_rows,
        smoothing_frames=int(args.smoothing_frames),
        change_confirm_frames=int(args.change_confirm_frames),
        confidence_threshold=float(args.confidence_threshold),
    )
    summary = build_summary(rows)
    summary.update(
        {
            "model_path": str(args.model_path),
            "signal": str(args.signal),
            "window_samples": int(getattr(predictor, "window_samples", round(float(args.window_s) * EMG_FS))),
            "stride_s": float(args.stride_s),
            "confidence_threshold": float(args.confidence_threshold),
            "smoothing_frames": int(args.smoothing_frames),
            "change_confirm_frames": int(args.change_confirm_frames),
            "row_count": int(len(rows)),
        }
    )
    write_log(output_dir / "realtime_smoothing_log.csv", rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    smoothing_list = _parse_number_list(args.smoothing_frames_list, int) or [int(args.smoothing_frames)]
    confirm_list = _parse_number_list(args.change_confirm_frames_list, int) or [int(args.change_confirm_frames)]
    threshold_list = _parse_number_list(args.confidence_threshold_list, float) or [float(args.confidence_threshold)]
    if args.smoothing_frames_list or args.change_confirm_frames_list or args.confidence_threshold_list:
        sweep_records = []
        for smoothing_frames, change_confirm_frames, threshold in itertools.product(
            smoothing_list,
            confirm_list,
            threshold_list,
        ):
            sweep_rows = apply_smoothing(
                raw_rows,
                smoothing_frames=smoothing_frames,
                change_confirm_frames=change_confirm_frames,
                confidence_threshold=threshold,
            )
            sweep_summary = build_summary(sweep_rows, include_sessions=False)
            sweep_records.append(
                {
                    "smoothing_frames": smoothing_frames,
                    "change_confirm_frames": change_confirm_frames,
                    "confidence_threshold": threshold,
                    "raw_accuracy": sweep_summary["raw_accuracy"],
                    "smoothed_accuracy": sweep_summary["smoothed_accuracy"],
                    "raw_switch_count": sweep_summary["raw_switch_count"],
                    "smoothed_switch_count": sweep_summary["smoothed_switch_count"],
                    "raw_false_trigger_rate": sweep_summary["raw_false_trigger_rate"],
                    "smoothed_false_trigger_rate": sweep_summary["smoothed_false_trigger_rate"],
                    "average_detection_delay_ms": sweep_summary["average_detection_delay_ms"],
                    "median_detection_delay_ms": sweep_summary["median_detection_delay_ms"],
                }
            )
        write_sweep_summary(output_dir / "sweep_summary.csv", sweep_records)
        summary["sweep_count"] = len(sweep_records)
        (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline replay evaluation for realtime EMG smoothing.")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    add_path_arguments(parser)
    parser.add_argument("--session")
    parser.add_argument("--window-s", default=1.0, type=float)
    parser.add_argument("--stride-s", default=0.1, type=float)
    parser.add_argument("--confidence-threshold", default=0.70, type=float)
    parser.add_argument("--smoothing-frames", default=5, type=int)
    parser.add_argument("--change-confirm-frames", default=2, type=int)
    parser.add_argument("--signal", choices=["raw", "filtered"], default="raw")
    parser.add_argument("--smoothing-frames-list")
    parser.add_argument("--change-confirm-frames-list")
    parser.add_argument("--confidence-threshold-list")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    paths = resolve_paths_from_args(args)
    args.dataset_root = resolve_dataset_root(args.dataset_root, paths)
    if args.model_path is None:
        args.model_path = paths.models_root / "effie_real_full_v2_continue" / "gesture_classifier.ts"
    else:
        args.model_path = resolve_project_path(args.model_path, paths)
    if args.output_dir is None:
        args.output_dir = paths.reports_root / "realtime_smoothing"
    else:
        args.output_dir = resolve_project_path(args.output_dir, paths)
    summary = evaluate_realtime_smoothing(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
