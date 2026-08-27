"""Compare available gesture model candidates and report metadata.

This script is intentionally lightweight: it records which candidate artifacts
exist and can be extended with full trial/CV evaluation as models mature.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CANDIDATE_MODELS = {
    "feature_rf": Path("models") / "feature_rf",
    "emg2pose_tcn": Path("models") / "emg2pose_gesture_v1",
    "tma_cnn": Path("models") / "tma_cnn",
    "tts_cnn": Path("models") / "tts_cnn",
    "effie_finetuned": Path("models") / "effie_finetuned",
}


def inspect_candidate(name: str, path: Path) -> dict[str, object]:
    model_info = path / "model_info.json"
    train_report = path / "train_report.json"
    return {
        "name": name,
        "path": str(path),
        "exists": path.exists(),
        "torchscript": str(path / "gesture_classifier.ts"),
        "torchscript_exists": (path / "gesture_classifier.ts").exists(),
        "checkpoint_exists": (path / "gesture_classifier.pt").exists(),
        "model_info": json.loads(model_info.read_text(encoding="utf-8")) if model_info.exists() else None,
        "train_report": json.loads(train_report.read_text(encoding="utf-8")) if train_report.exists() else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare gesture model candidates.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = {
        "candidates": [
            inspect_candidate(name, path)
            for name, path in CANDIDATE_MODELS.items()
        ],
        "comparison_modes": ["trial split", "cross-session-cv"],
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text if args.json or args.output is None else f"saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
