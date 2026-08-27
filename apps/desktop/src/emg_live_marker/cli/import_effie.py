"""Best-effort EffiE checkpoint import helper.

The public EffiE repository is TensorFlow/Keras-based. The main application is
PyTorch-only, so this helper keeps optional TensorFlow handling outside the app.

For PyTorch checkpoints already converted to the local EffiE-style architecture,
the script verifies and re-saves a normalized `model_state_dict` checkpoint.
TensorFlow checkpoint conversion is repository-version dependent; if TensorFlow
or the upstream model code is unavailable, the script exits with explicit
instructions instead of silently producing mismatched weights.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from emg_live_marker.ml.effie_adapter import EffieGestureNet
from emg_live_marker.ml.gesture_model import LABELS
from emg_live_marker.paths import add_path_arguments, resolve_paths_from_args, resolve_project_path


def import_pytorch_checkpoint(checkpoint_path: Path, output_path: Path) -> None:
    model = EffieGestureNet(classes=len(LABELS))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "labels": LABELS,
            "model_type": "effie_finetuned",
            "source_checkpoint": str(checkpoint_path),
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
        },
        output_path,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Import EffiE checkpoint for local fine-tuning.")
    parser.add_argument("--effie-root", required=True, type=Path)
    parser.add_argument("--checkpoint-path", required=True, type=Path)
    parser.add_argument("--output-path", default=None, type=Path)
    parser.add_argument("--format", choices=["auto", "pytorch", "tensorflow"], default="auto")
    add_path_arguments(parser)
    args = parser.parse_args()
    paths = resolve_paths_from_args(args)
    args.effie_root = resolve_project_path(args.effie_root, paths)
    args.checkpoint_path = resolve_project_path(args.checkpoint_path, paths)
    if args.output_path is None:
        args.output_path = paths.models_root / "effie_imported_backbone.pt"
    else:
        args.output_path = resolve_project_path(args.output_path, paths)

    if not args.checkpoint_path.exists():
        raise FileNotFoundError(args.checkpoint_path)
    suffixes = {suffix.lower() for suffix in args.checkpoint_path.suffixes}
    use_pytorch = args.format == "pytorch" or (
        args.format == "auto" and ({".pt", ".pth"} & suffixes)
    )
    if use_pytorch:
        import_pytorch_checkpoint(args.checkpoint_path, args.output_path)
        print(f"saved {args.output_path}")
        return 0

    message = (
        "TensorFlow/Keras EffiE checkpoint conversion is optional and depends on "
        "the exact upstream checkpoint/model version. Clone the repository with:\n"
        "  git clone https://github.com/MIC-Laboratory/IEEE-NER-2023-EffiE third_party\\EffiE\n"
        "Then inspect third_party\\EffiE\\checkpoints and convert the checkpoint "
        "to a PyTorch state_dict matching EffieGestureNet, or train without "
        "--checkpoint-path first."
    )
    raise RuntimeError(message)


if __name__ == "__main__":
    raise SystemExit(main())
