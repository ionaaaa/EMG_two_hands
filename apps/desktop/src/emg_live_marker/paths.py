"""Resolve project data paths independently of the current working directory.

Path values use the following precedence: command-line value, JSON config file,
environment variable, then the repository data locations.
"""

from __future__ import annotations

import json
import os
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / ".emg-paths.json"


@dataclass(frozen=True)
class ProjectPaths:
    """Absolute filesystem locations used by acquisition and ML workflows."""

    project_root: Path
    dataset_root: Path
    recordings_root: Path
    artifacts_root: Path
    config_path: Path | None = None

    @property
    def models_root(self) -> Path:
        return self.artifacts_root / "models"

    @property
    def reports_root(self) -> Path:
        return self.artifacts_root / "reports"


def add_path_arguments(parser: ArgumentParser) -> None:
    """Add the common path options shared by the desktop and offline CLI tools."""

    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--recordings-root", type=Path, default=None)
    parser.add_argument("--artifacts-root", type=Path, default=None)
    parser.add_argument(
        "--paths-config",
        type=Path,
        default=None,
        help="JSON path configuration file (default: .emg-paths.json at the repository root).",
    )


def resolve_project_paths(
    *,
    dataset_root: Path | str | None = None,
    recordings_root: Path | str | None = None,
    artifacts_root: Path | str | None = None,
    paths_config: Path | str | None = None,
    project_root: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> ProjectPaths:
    """Resolve absolute project paths using CLI > config > environment > defaults."""

    root = Path(project_root or PROJECT_ROOT).resolve()
    environment = os.environ if environ is None else environ
    config_path = _resolve_config_path(paths_config, root, environment)
    config = _load_config(config_path) if config_path is not None and config_path.exists() else {}

    defaults = {
        "dataset_root": root / "data" / "datasets",
        "recordings_root": root / "data" / "recordings",
        "artifacts_root": root / "apps" / "desktop",
    }
    values = {
        "dataset_root": dataset_root,
        "recordings_root": recordings_root,
        "artifacts_root": artifacts_root,
    }
    environment_names = {
        "dataset_root": "EMG_DATASET_ROOT",
        "recordings_root": "EMG_RECORDINGS_ROOT",
        "artifacts_root": "EMG_ARTIFACTS_ROOT",
    }

    resolved = {
        name: _resolve_path(
            values[name]
            if values[name] is not None
            else config.get(name, environment.get(environment_names[name], defaults[name])),
            root,
        )
        for name in defaults
    }
    return ProjectPaths(project_root=root, config_path=config_path, **resolved)


def resolve_paths_from_args(args: Namespace) -> ProjectPaths:
    """Resolve a parser namespace created with :func:`add_path_arguments`."""

    return resolve_project_paths(
        dataset_root=getattr(args, "dataset_root", None),
        recordings_root=getattr(args, "recordings_root", None),
        artifacts_root=getattr(args, "artifacts_root", None),
        paths_config=getattr(args, "paths_config", None),
    )


def resolve_project_path(value: Path | str, paths: ProjectPaths) -> Path:
    """Resolve an arbitrary CLI path relative to the repository root."""

    return _resolve_path(value, paths.project_root)


def _resolve_config_path(
    paths_config: Path | str | None,
    project_root: Path,
    environ: Mapping[str, str],
) -> Path | None:
    configured = paths_config if paths_config is not None else environ.get("EMG_PATHS_CONFIG")
    if configured is None:
        default = project_root / DEFAULT_CONFIG_PATH.name
        return default if default.exists() else None
    return _resolve_path(configured, project_root)


def _load_config(path: Path) -> dict[str, str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON path configuration: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Path configuration must be a JSON object: {path}")
    unsupported = set(data) - {"dataset_root", "recordings_root", "artifacts_root"}
    if unsupported:
        raise ValueError(f"Unsupported path configuration keys in {path}: {sorted(unsupported)}")
    return {name: str(value) for name, value in data.items() if value is not None}


def _resolve_path(value: Path | str, project_root: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()
