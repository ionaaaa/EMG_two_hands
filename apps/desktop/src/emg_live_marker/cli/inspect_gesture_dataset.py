"""Inspect collected EMG gesture dataset sessions before training."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from emg_live_marker.cli.train_gesture_classifier import (
    COLLECT_TO_GAME_LABEL,
    LABELS,
    collect_window_records,
    discover_session_dirs,
    load_emg,
    read_csv_dicts,
    resolve_dataset_root,
)


def _load_emg_summary(path: Path) -> dict[str, Any]:
    rows = read_csv_dicts(path)
    sample_index = np.asarray([int(float(row["sample_index"])) for row in rows], dtype=np.int64)
    software_time = np.asarray([float(row.get("software_time", "nan")) for row in rows], dtype=np.float64)
    sample_continuous = bool(sample_index.size < 2 or np.all(np.diff(sample_index) == 1))
    finite_time = software_time[np.isfinite(software_time)]
    time_monotonic = bool(finite_time.size < 2 or np.all(np.diff(finite_time) > 0))
    return {
        "rows": len(rows),
        "sample_index_continuous": sample_continuous,
        "software_time_monotonic": time_monotonic,
        "channel_rms_mean": [
            float(value)
            for value in np.sqrt(
                np.mean(
                    np.square(
                        np.asarray(
                            [[float(row[f"ch{channel}_uv"]) for channel in range(1, 9)] for row in rows],
                            dtype=np.float64,
                        )
                    ),
                    axis=0,
                )
            ).tolist()
        ],
    }


def inspect_session(session_dir: Path) -> dict[str, Any]:
    missing = [
        name
        for name in ("emg.csv", "events.csv", "metadata.json")
        if not (session_dir / name).exists()
    ]
    if missing:
        return {"path": str(session_dir), "ok": False, "reason": f"missing {', '.join(missing)}"}

    emg_summary = _load_emg_summary(session_dir / "emg.csv")
    events = read_csv_dicts(session_dir / "events.csv")
    trial_ids = {row.get("trial_id", "") for row in events if row.get("trial_id", "")}
    gestures_by_trial: dict[str, str] = {}
    phases = set()
    gestures = set()
    for row in events:
        trial_id = row.get("trial_id", "")
        phase = row.get("phase", "")
        gesture = row.get("gesture", "")
        if phase:
            phases.add(phase)
        if gesture:
            gestures.add(gesture)
        if trial_id and gesture and phase == "gesture_start":
            gestures_by_trial[trial_id] = gesture
    gesture_counts = Counter(gestures_by_trial.values())
    rest_window_estimate = 0
    for trial_id in trial_ids:
        trial_events = [row for row in events if row.get("trial_id", "") == trial_id]
        phase_to_row = {row.get("phase", ""): row for row in trial_events}
        try:
            trial_start = int(float(phase_to_row["trial_start"]["sample_index"]))
            gesture_start = int(float(phase_to_row["gesture_start"]["sample_index"]))
            gesture_end = int(float(phase_to_row["gesture_end"]["sample_index"]))
            trial_end = int(float(phase_to_row["trial_end"]["sample_index"]))
        except (KeyError, TypeError, ValueError):
            continue
        rest_window_estimate += max(0, (gesture_start - trial_start - 100) // 250 + 1)
        rest_window_estimate += max(0, (trial_end - gesture_end - 100) // 250 + 1)

    expected = {"fist", "finger_spread", "thumb_index_pinch"}
    unexpected = sorted(gestures - expected)
    ok = not unexpected
    return {
        "path": str(session_dir),
        "ok": ok,
        "reason": "" if ok else f"unexpected gestures: {unexpected}",
        "emg_rows": emg_summary["rows"],
        "events_rows": len(events),
        "trial_count": len(trial_ids),
        "gesture_trial_counts": dict(gesture_counts),
        "phases": sorted(phases),
        "gestures": sorted(gestures),
        "sample_index_continuous": emg_summary["sample_index_continuous"],
        "software_time_monotonic": emg_summary["software_time_monotonic"],
        "channel_rms_mean": emg_summary["channel_rms_mean"],
        "rest_window_estimate": int(rest_window_estimate),
    }


def inspect_dataset(dataset_root: Path) -> dict[str, Any]:
    sessions = [inspect_session(path) for path in discover_session_dirs(dataset_root)]
    window_counts_by_session: dict[str, dict[str, int]] = {}
    try:
        records = collect_window_records(dataset_root)
        for record in records:
            window_counts_by_session.setdefault(
                record.session_id,
                {label: 0 for label in LABELS},
            )[LABELS[record.y]] += 1
    except ValueError:
        records = []
    for session in sessions:
        session_name = Path(session["path"]).name
        session["window_counts"] = window_counts_by_session.get(session_name, {label: 0 for label in LABELS})

    total_trials = sum(int(session.get("trial_count", 0)) for session in sessions)
    gesture_counts: Counter[str] = Counter()
    rest_windows = 0
    for session in sessions:
        gesture_counts.update(session.get("gesture_trial_counts", {}))
        rest_windows += int(session.get("rest_window_estimate", 0))
    rms_total_by_session = {
        session["path"]: float(np.mean(session.get("channel_rms_mean", [0.0])))
        for session in sessions
        if session.get("channel_rms_mean")
    }
    median_rms = float(np.median(list(rms_total_by_session.values()))) if rms_total_by_session else 0.0
    warnings: list[str] = []
    for session in sessions:
        if not session.get("sample_index_continuous", True):
            warnings.append(f"{session['path']}: sample_index is not continuous")
        if not session.get("software_time_monotonic", True):
            warnings.append(f"{session['path']}: software_time is not monotonic; training can still use sample_index")
        rms_value = rms_total_by_session.get(session["path"], 0.0)
        if median_rms > 0 and (rms_value > median_rms * 2.5 or rms_value < median_rms * 0.4):
            warnings.append(
                f"{session['path']}: RMS amplitude differs from median "
                f"(session={rms_value:.3f}, median={median_rms:.3f})"
            )

    return {
        "session_count": len(sessions),
        "total_trials": total_trials,
        "gesture_trial_counts": {
            "fist": int(gesture_counts.get("fist", 0)),
            "finger_spread": int(gesture_counts.get("finger_spread", 0)),
            "thumb_index_pinch": int(gesture_counts.get("thumb_index_pinch", 0)),
        },
        "rest_window_estimate": rest_windows,
        "session_rms_mean": rms_total_by_session,
        "warnings": warnings,
        "sessions": sessions,
    }


def print_report(report: dict[str, Any]) -> None:
    for session in report["sessions"]:
        print(f"Session: {session['path']}")
        if not session["ok"]:
            print(f"  ERROR: {session['reason']}")
        print(f"  emg rows: {session.get('emg_rows', 0)}")
        print(f"  events rows: {session.get('events_rows', 0)}")
        print(f"  trials: {session.get('trial_count', 0)}")
        print(f"  gesture trials: {session.get('gesture_trial_counts', {})}")
        print(f"  phases: {session.get('phases', [])}")
        print(f"  gestures: {session.get('gestures', [])}")
        print(f"  sample_index continuous: {session.get('sample_index_continuous', False)}")
        print(f"  software_time monotonic: {session.get('software_time_monotonic', False)}")
        print(f"  window counts: {session.get('window_counts', {})}")
        print(f"  channel RMS mean: {session.get('channel_rms_mean', [])}")
        print()

    print("Summary")
    print(f"  sessions: {report['session_count']}")
    print(f"  trials: {report['total_trials']}")
    for gesture in COLLECT_TO_GAME_LABEL:
        if gesture == "rest":
            continue
        print(f"  {gesture} trials: {report['gesture_trial_counts'].get(gesture, 0)}")
    print(f"  rest cuttable windows estimate: {report['rest_window_estimate']}")
    if report.get("warnings"):
        print("Warnings")
        for warning in report["warnings"]:
            print(f"  WARNING: {warning}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect collected EMG gesture dataset.")
    parser.add_argument("--dataset-root", default="dataset", type=Path)
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = parser.parse_args()
    report = inspect_dataset(resolve_dataset_root(args.dataset_root))
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
