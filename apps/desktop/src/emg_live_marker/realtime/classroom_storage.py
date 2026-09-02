"""Canonical, traversal-safe storage for one student classroom."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from PySide6.QtCore import QStandardPaths

SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class ClassroomStorageError(ValueError):
    """Raised when classroom configuration or an identifier is unsafe."""


@dataclass(frozen=True)
class ClassroomGroupPaths:
    root: Path
    profile: Path
    sessions: Path
    models: Path
    settings: Path
    results: Path


def default_app_data_root() -> Path:
    """Return an application-specific directory, never the LocalAppData root itself."""

    base = Path(
        QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppLocalDataLocation)
    )
    return base if base.name.casefold() == "emgtwohands" else base / "EMGTwoHands"


class ClassroomStorage:
    """Generate every new group data path and own atomic JSON persistence."""

    SETTINGS_DEFAULTS: dict[str, Any] = {
        "mapping": {},
        "sensitivity": "standard",
        "control_style": "balanced",
        "active_model": {"type": "standard", "run_id": None},
        "recollect_requested": [],
    }

    def __init__(
        self,
        classroom_root: str | Path,
        classroom_id: str,
        course_id: str,
        *,
        app_data_root: str | Path | None = None,
    ) -> None:
        self.classroom_root = Path(classroom_root).resolve()
        self.classroom_id = self.validate_id(classroom_id, "classroom_id")
        self.course_id = self.validate_id(course_id, "course_id")
        self.root = (self.classroom_root / self.classroom_id).resolve()
        self.app_data_root = Path(app_data_root or default_app_data_root()).resolve()
        self.active_group_path = self.app_data_root / "current-classroom-group.json"

    @classmethod
    def from_config(
        cls,
        classroom_root: str | Path,
        course_config: dict[str, Any],
        *,
        app_data_root: str | Path | None = None,
    ) -> "ClassroomStorage":
        storage = course_config.get("storage", {})
        course = course_config.get("course", {})
        classroom_id = storage.get("classroom_id", "") if isinstance(storage, dict) else ""
        course_id = course.get("id", "") if isinstance(course, dict) else ""
        return cls(
            classroom_root,
            str(classroom_id),
            str(course_id),
            app_data_root=app_data_root,
        )

    @staticmethod
    def validate_id(value: object, field: str = "group_id") -> str:
        identifier = str(value).strip()
        if not SAFE_ID_PATTERN.fullmatch(identifier):
            raise ClassroomStorageError(
                f"{field} 只能使用字母、数字、下划线和短横线，且不能包含路径。"
            )
        if identifier in {".", ".."} or Path(identifier).is_absolute():
            raise ClassroomStorageError(f"{field} 路径无效。")
        return identifier

    def group_paths(self, group_id: object, *, create: bool = False) -> ClassroomGroupPaths:
        group = self.validate_id(group_id)
        root = (self.root / group).resolve()
        self._require_inside(root, self.root)
        paths = ClassroomGroupPaths(
            root=root,
            profile=root / "profile.json",
            sessions=root / "sessions",
            models=root / "models",
            settings=root / "settings.json",
            results=root / "results",
        )
        if create:
            paths.sessions.mkdir(parents=True, exist_ok=True)
            paths.models.mkdir(parents=True, exist_ok=True)
            paths.results.mkdir(parents=True, exist_ok=True)
            self.ensure_profile(group)
            if not paths.settings.exists():
                self.write_settings(group, {})
        return paths

    def ensure_profile(self, group_id: object) -> dict[str, Any]:
        group = self.validate_id(group_id)
        paths = self.group_paths(group, create=False)
        now = self._timestamp()
        try:
            existing = self.read_json(paths.profile)
        except (FileNotFoundError, OSError, json.JSONDecodeError, ValueError):
            existing = {}
        created = str(existing.get("created_at") or now) if isinstance(existing, dict) else now
        profile = {
            "schema_version": 1,
            "student_id": group,
            "course_id": self.course_id,
            "classroom_id": self.classroom_id,
            "created_at": created,
            "updated_at": now,
        }
        paths.root.mkdir(parents=True, exist_ok=True)
        self.atomic_write_json(paths.profile, profile)
        return profile

    def read_settings(self, group_id: object) -> dict[str, Any]:
        paths = self.group_paths(group_id, create=False)
        try:
            payload = self.read_json(paths.settings)
        except (FileNotFoundError, OSError, json.JSONDecodeError, ValueError):
            payload = {}
        return self._normalize_settings(payload)

    def write_settings(self, group_id: object, updates: dict[str, Any]) -> dict[str, Any]:
        group = self.validate_id(group_id)
        paths = self.group_paths(group, create=False)
        paths.root.mkdir(parents=True, exist_ok=True)
        current = self.read_settings(group)
        current.update(dict(updates))
        normalized = self._normalize_settings(current)
        normalized["schema_version"] = 1
        normalized["updated_at"] = self._timestamp()
        self.atomic_write_json(paths.settings, normalized)
        self.ensure_profile(group)
        return normalized

    def session_path(self, group_id: object, session_id: object, *, create: bool = False) -> Path:
        session = self.validate_id(session_id, "session_id")
        target = self.group_paths(group_id, create=create).sessions / session
        self._require_inside(target, self.group_paths(group_id).sessions)
        if create:
            target.mkdir(parents=True, exist_ok=False)
        return target

    def next_session_path(self, group_id: object, prefix: str) -> tuple[str, Path]:
        safe_prefix = self.validate_id(prefix, "session_id")
        paths = self.group_paths(group_id, create=True)
        session_id = safe_prefix
        suffix = 1
        while (paths.sessions / session_id).exists():
            suffix += 1
            session_id = f"{safe_prefix}_{suffix:02d}"
        return session_id, paths.sessions / session_id

    def model_path(self, group_id: object, run_id: object, *, create: bool = False) -> Path:
        run = self.validate_id(run_id, "run_id")
        target = self.group_paths(group_id, create=create).models / run
        self._require_inside(target, self.group_paths(group_id).models)
        if create:
            target.mkdir(parents=True, exist_ok=False)
        return target

    def result_path(self, group_id: object, competition_id: object) -> Path:
        competition = self.validate_id(competition_id, "competition_id")
        paths = self.group_paths(group_id, create=True)
        return paths.results / f"{competition}.json"

    def iter_groups(self) -> tuple[tuple[str, ClassroomGroupPaths], ...]:
        if not self.root.is_dir():
            return ()
        groups: list[tuple[str, ClassroomGroupPaths]] = []
        for candidate in sorted(self.root.iterdir()):
            if not candidate.is_dir():
                continue
            try:
                group = self.validate_id(candidate.name)
            except ClassroomStorageError:
                continue
            groups.append((group, self.group_paths(group)))
        return tuple(groups)

    def iter_sessions(self) -> tuple[tuple[str, Path], ...]:
        sessions: list[tuple[str, Path]] = []
        for group, paths in self.iter_groups():
            if not paths.sessions.is_dir():
                continue
            for metadata in sorted(paths.sessions.glob("*/metadata.json")):
                sessions.append((group, metadata.parent.resolve()))
        return tuple(sessions)

    def iter_models(self) -> tuple[tuple[str, str, Path], ...]:
        models: list[tuple[str, str, Path]] = []
        for group, paths in self.iter_groups():
            if not paths.models.is_dir():
                continue
            for marker in sorted(paths.models.glob("*/personal_model.valid.json")):
                try:
                    run_id = self.validate_id(marker.parent.name, "run_id")
                except ClassroomStorageError:
                    continue
                models.append((group, run_id, marker.parent.resolve()))
        return tuple(models)

    def iter_results(self) -> tuple[tuple[str, Path], ...]:
        results: list[tuple[str, Path]] = []
        for group, paths in self.iter_groups():
            if paths.results.is_dir():
                results.extend((group, path.resolve()) for path in sorted(paths.results.glob("*.json")))
        return tuple(results)

    def set_active_group(self, group_id: object) -> None:
        group = self.validate_id(group_id)
        self.app_data_root.mkdir(parents=True, exist_ok=True)
        self.atomic_write_json(
            self.active_group_path,
            {"schema_version": 1, "classroom_id": self.classroom_id, "group_id": group},
        )

    def active_group(self) -> str:
        try:
            payload = self.read_json(self.active_group_path)
            if payload.get("classroom_id") != self.classroom_id:
                return ""
            return self.validate_id(payload.get("group_id"))
        except (FileNotFoundError, OSError, json.JSONDecodeError, ValueError, AttributeError):
            return ""

    def clear_active_group(self) -> None:
        try:
            self.active_group_path.unlink(missing_ok=True)
        except OSError:
            raise

    @staticmethod
    def discover_legacy_sessions(dataset_root: str | Path) -> tuple[tuple[str, Path], ...]:
        root = Path(dataset_root).resolve()
        if not root.is_dir():
            return ()
        found: list[tuple[str, Path]] = []
        for metadata in sorted(root.rglob("metadata.json")):
            session = metadata.parent.resolve()
            try:
                relative = session.relative_to(root)
            except ValueError:
                continue
            group = relative.parts[0] if len(relative.parts) > 1 else "未分组"
            found.append((group, session))
        return tuple(found)

    @staticmethod
    def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as file_obj:
                json.dump(payload, file_obj, ensure_ascii=False, indent=2)
                file_obj.write("\n")
                file_obj.flush()
                os.fsync(file_obj.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def read_json(path: str | Path) -> dict[str, Any]:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("JSON root must be an object")
        return payload

    @classmethod
    def _normalize_settings(cls, payload: object) -> dict[str, Any]:
        source = payload if isinstance(payload, dict) else {}
        result = dict(cls.SETTINGS_DEFAULTS)
        mapping = source.get("mapping")
        if isinstance(mapping, dict):
            result["mapping"] = {str(key): str(value) for key, value in mapping.items()}
        sensitivity = source.get("sensitivity")
        if sensitivity in {"low", "standard", "high"}:
            result["sensitivity"] = sensitivity
        style = source.get("control_style")
        if style in {"fast", "balanced", "stable"}:
            result["control_style"] = style
        active = source.get("active_model")
        if isinstance(active, dict) and active.get("type") in {"standard", "personal"}:
            run_id = active.get("run_id")
            if active["type"] == "standard":
                result["active_model"] = {"type": "standard", "run_id": None}
            else:
                try:
                    result["active_model"] = {
                        "type": "personal",
                        "run_id": cls.validate_id(run_id, "run_id"),
                    }
                except ClassroomStorageError:
                    result["active_model"] = {"type": "standard", "run_id": None}
        recollect = source.get("recollect_requested")
        if isinstance(recollect, list):
            result["recollect_requested"] = [
                item for item in dict.fromkeys(str(value) for value in recollect)
                if SAFE_ID_PATTERN.fullmatch(item)
            ]
        if isinstance(source.get("updated_at"), str):
            result["updated_at"] = source["updated_at"]
        result["schema_version"] = 1
        return result

    @staticmethod
    def _require_inside(target: Path, root: Path) -> None:
        try:
            Path(target).resolve().relative_to(Path(root).resolve())
        except ValueError as exc:
            raise ClassroomStorageError("目标路径超出课堂数据目录。") from exc

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")
