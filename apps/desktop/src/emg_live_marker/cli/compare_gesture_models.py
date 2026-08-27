"""Compare available gesture model candidates and report metadata.

This script is intentionally lightweight: it records which candidate artifacts
exist and can be extended with full trial/CV evaluation as models mature.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from emg_live_marker.paths import add_path_arguments, resolve_paths_from_args, resolve_project_path

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
    add_path_arguments(parser)
    args = parser.parse_args()
    paths = resolve_paths_from_args(args)
    candidate_models = {
        "feature_rf": paths.models_root / "feature_rf",
        "emg2pose_tcn": paths.models_root / "emg2pose_gesture_v1",
        "tma_cnn": paths.models_root / "tma_cnn",
        "tts_cnn": paths.models_root / "tts_cnn",
        "effie_finetuned": paths.models_root / "effie_finetuned",
    }
    if args.output is not None:
        args.output = resolve_project_path(args.output, paths)
    report = {
        "candidates": [
            inspect_candidate(name, path)
            for name, path in candidate_models.items()
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
