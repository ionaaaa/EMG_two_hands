import csv
import json
from pathlib import Path

import torch

from emg_live_marker.ml.gesture_model import EMGGestureNet, LABELS
from emg_live_marker.cli.evaluate_realtime_smoothing import LOG_FIELDS, main


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(header)
        writer.writerows(rows)


def _make_dataset(root: Path) -> Path:
    session_dir = root / "subject_01" / "session_001"
    session_dir.mkdir(parents=True)
    rows = []
    for index in range(1000):
        base = 150.0 if 250 <= index < 625 else 5.0
        rows.append([index, f"{index / 250.0:.6f}", *([base] * 8)])
    _write_csv(
        session_dir / "emg.csv",
        [
            "sample_index",
            "software_time",
            "ch1_uv",
            "ch2_uv",
            "ch3_uv",
            "ch4_uv",
            "ch5_uv",
            "ch6_uv",
            "ch7_uv",
            "ch8_uv",
        ],
        rows,
    )
    _write_csv(
        session_dir / "events.csv",
        [
            "trial_id",
            "subject_id",
            "session_id",
            "gesture",
            "gesture_name",
            "phase",
            "software_time",
            "sample_index",
            "note",
        ],
        [
            ["0001", "subject_01", "session_001", "fist", "fist", "trial_start", "0.0", 0, ""],
            ["0001", "subject_01", "session_001", "fist", "fist", "gesture_start", "1.0", 250, ""],
            ["0001", "subject_01", "session_001", "fist", "fist", "gesture_end", "2.5", 625, ""],
            ["0001", "subject_01", "session_001", "fist", "fist", "trial_end", "4.0", 1000, ""],
        ],
    )
    (session_dir / "metadata.json").write_text(json.dumps({"session_id": "session_001"}), encoding="utf-8")
    return session_dir


def _make_model(root: Path) -> Path:
    model_dir = root / "models"
    model_dir.mkdir()
    model = EMGGestureNet()
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "labels": LABELS,
            "model_type": "cnn",
            "channels": 8,
        },
        model_dir / "gesture_classifier.pt",
    )
    (model_dir / "gesture_labels.json").write_text(json.dumps({"labels": LABELS}), encoding="utf-8")
    (model_dir / "normalization.json").write_text(
        json.dumps(
            {
                "signal_type": "raw",
                "fs": 250.0,
                "window_s": 1.0,
                "channels": 8,
                "mean": [0.0] * 8,
                "std": [1.0] * 8,
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "model_info.json").write_text(
        json.dumps(
            {
                "model_type": "cnn",
                "fs": 250.0,
                "window_s": 1.0,
                "stride_s": 0.1,
                "channels": 8,
                "signal_type": "raw",
                "labels": LABELS,
            }
        ),
        encoding="utf-8",
    )
    return model_dir / "gesture_classifier.pt"


def test_evaluate_realtime_smoothing_writes_log_and_summary(tmp_path):
    dataset_root = tmp_path / "dataset"
    _make_dataset(dataset_root)
    model_path = _make_model(tmp_path)
    output_dir = tmp_path / "eval"

    assert main(
        [
            "--dataset-root",
            str(dataset_root),
            "--model-path",
            str(model_path),
            "--output-dir",
            str(output_dir),
            "--session",
            "session_001",
        ]
    ) == 0

    log_path = output_dir / "realtime_smoothing_log.csv"
    summary_path = output_dir / "summary.json"
    assert log_path.exists()
    assert summary_path.exists()

    with log_path.open(newline="", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        assert reader.fieldnames == LOG_FIELDS
        rows = list(reader)
    assert rows
    assert set(rows[0]) == set(LOG_FIELDS)
    assert rows[0]["sent_to_game"] in {"true", "false"}

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for key in [
        "raw_switch_count",
        "smoothed_switch_count",
        "raw_false_trigger_rate",
        "smoothed_false_trigger_rate",
        "average_detection_delay_ms",
        "median_detection_delay_ms",
        "session_results",
    ]:
        assert key in summary
    assert "session_001" in summary["session_results"]


def test_evaluate_realtime_smoothing_writes_sweep_summary(tmp_path):
    dataset_root = tmp_path / "dataset"
    _make_dataset(dataset_root)
    model_path = _make_model(tmp_path)
    output_dir = tmp_path / "eval"

    assert main(
        [
            "--dataset-root",
            str(dataset_root),
            "--model-path",
            str(model_path),
            "--output-dir",
            str(output_dir),
            "--smoothing-frames-list",
            "1,3",
            "--change-confirm-frames-list",
            "1,2",
            "--confidence-threshold-list",
            "0.6,0.7",
        ]
    ) == 0

    sweep_path = output_dir / "sweep_summary.csv"
    assert sweep_path.exists()
    with sweep_path.open(newline="", encoding="utf-8") as file_obj:
        rows = list(csv.DictReader(file_obj))
    assert len(rows) == 8
    assert {
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
    } <= set(rows[0])
