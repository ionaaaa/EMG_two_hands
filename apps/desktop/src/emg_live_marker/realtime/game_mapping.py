"""Validated student game-command mapping and anonymous-group persistence."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Signal

from emg_live_marker.realtime.classroom_storage import (
    ClassroomStorage,
    ClassroomStorageError,
    default_app_data_root,
)

GESTURES = ("rest", "fist", "open-palm", "pinch")
EDITABLE_GESTURES = ("fist", "open-palm")
COMMANDS = ("A", "B", "none")
SENSITIVITIES = ("low", "standard", "high")
CONTROL_STYLES = ("fast", "balanced", "stable")


class GameMappingConfigError(ValueError):
    """Raised when the locked teaching mapping configuration is invalid."""


class GameMappingService(QObject):
    """Own all validation, mutation, and storage for the student mapping page."""

    mapping_changed = Signal(dict)
    test_feedback = Signal(dict)
    control_preferences_changed = Signal(str, str)

    def __init__(
        self,
        course_config: dict[str, Any],
        *,
        storage_root: Path | None = None,
        classroom_storage: ClassroomStorage | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        game = course_config.get("game", {})
        if not isinstance(game, dict):
            raise GameMappingConfigError("课程游戏配置无效")
        self._commands = self._validate_commands(game.get("commands"))
        self._default_mapping = self._validate_mapping(game.get("default_mapping"))
        self._mapping = dict(self._default_mapping)
        profile_config = course_config.get("student_control_profiles", {})
        defaults = profile_config.get("default", {}) if isinstance(profile_config, dict) else {}
        self._default_control_preferences = self._validate_control_preferences(
            defaults, fallback={"sensitivity": "standard", "control_style": "balanced"}
        )
        self._control_preferences = dict(self._default_control_preferences)
        self.current_group_id = ""
        self.classroom_storage = classroom_storage
        default_root = default_app_data_root() / "legacy-student-game-mappings"
        self.storage_root = Path(storage_root or default_root).resolve()
        self._restore_last_group()

    @property
    def commands(self) -> dict[str, dict[str, str]]:
        return {command: dict(details) for command, details in self._commands.items()}

    @property
    def default_mapping(self) -> dict[str, str]:
        return dict(self._default_mapping)

    @property
    def resolved_mapping(self) -> dict[str, str]:
        return dict(self._mapping)

    @property
    def control_preferences(self) -> dict[str, str]:
        return dict(self._control_preferences)

    def set_control_preferences(self, sensitivity: str, control_style: str) -> bool:
        try:
            validated = self._validate_control_preferences(
                {"sensitivity": sensitivity, "control_style": control_style}
            )
        except GameMappingConfigError:
            return False
        if validated == self._control_preferences:
            return True
        self._control_preferences = validated
        self.control_preferences_changed.emit(
            validated["sensitivity"], validated["control_style"]
        )
        self._persist_current_settings()
        return True

    def set_editable_mapping(self, fist_command: str, open_palm_command: str) -> None:
        candidate = dict(self._mapping)
        candidate["fist"] = str(fist_command)
        candidate["open-palm"] = str(open_palm_command)
        self._apply_mapping(candidate)

    def swap_commands(self) -> None:
        self.set_editable_mapping(self._mapping["open-palm"], self._mapping["fist"])

    def restore_default(self) -> None:
        self._apply_mapping(self._default_mapping)

    def reset_temporary_state(self) -> None:
        """Forget the active group and restore classroom defaults without deleting saved groups."""

        self.current_group_id = ""
        self._apply_mapping(self._default_mapping)
        self._apply_control_preferences(self._default_control_preferences)
        if self.classroom_storage is not None:
            try:
                self.classroom_storage.clear_active_group()
            except OSError:
                pass

    def save_current_group(self, group_id: str) -> tuple[bool, str]:
        normalized = self._valid_group_id(group_id)
        if normalized is None:
            return False, "请先填写匿名小组编号，只能使用字母、数字、下划线和短横线。"
        try:
            if self.classroom_storage is not None:
                self.classroom_storage.group_paths(normalized, create=True)
                self.classroom_storage.write_settings(
                    normalized,
                    {
                        "mapping": self.resolved_mapping,
                        "sensitivity": self._control_preferences["sensitivity"],
                        "control_style": self._control_preferences["control_style"],
                    },
                )
                self.classroom_storage.set_active_group(normalized)
            else:
                payload = {
                    "schema_version": 1,
                    "anonymous_group_id": normalized,
                    "mapping": self.resolved_mapping,
                    "control_profile": self.control_preferences,
                }
                self.storage_root.mkdir(parents=True, exist_ok=True)
                self._write_json(self._group_path(normalized), payload)
                self._write_json(
                    self.storage_root / "current-group.json",
                    {"schema_version": 1, "anonymous_group_id": normalized},
                )
        except OSError:
            return False, "本组设置保存失败，请稍后重试。"
        self.current_group_id = normalized
        return True, f"已保存匿名小组 {normalized} 的课程设置。"

    def load_group(self, group_id: str) -> tuple[bool, str]:
        normalized = self._valid_group_id(group_id)
        if normalized is None:
            return False, "请先填写有效的匿名小组编号。"
        try:
            if self.classroom_storage is not None:
                path = self.classroom_storage.group_paths(normalized).settings
                if not path.is_file():
                    return False, "未找到该匿名小组已保存的设置，将使用当前设置。"
                payload = self.classroom_storage.read_settings(normalized)
                preferences_value = {
                    "sensitivity": payload.get("sensitivity"),
                    "control_style": payload.get("control_style"),
                }
                self.classroom_storage.set_active_group(normalized)
            else:
                path = self._group_path(normalized)
                payload = json.loads(path.read_text(encoding="utf-8"))
                preferences_value = payload.get("control_profile")
            mapping = self._validate_mapping(payload.get("mapping"))
            preferences = self._validate_control_preferences(
                preferences_value, fallback=self._default_control_preferences
            )
        except FileNotFoundError:
            return False, "未找到该匿名小组已保存的设置，将使用当前设置。"
        except (OSError, json.JSONDecodeError, AttributeError, GameMappingConfigError):
            return False, "该匿名小组的设置文件无效，将使用当前设置。"
        self.current_group_id = normalized
        self._apply_mapping(mapping)
        self._apply_control_preferences(preferences)
        return True, f"已加载匿名小组 {normalized} 的课程设置。"

    def test_mapping(self) -> str:
        fist = self._commands[self._mapping["fist"]]["display_name_zh"]
        open_palm = self._commands[self._mapping["open-palm"]]["display_name_zh"]
        message = f"映射测试：握拳 → {fist}；伸掌 → {open_palm}。未生成任何识别结果。"
        self.test_feedback.emit(
            {
                "test": True,
                "kind": "mapping-test",
                "resolved_mapping": self.resolved_mapping,
                "message": message,
            }
        )
        return message

    def runtime_config(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "commands": self.commands,
            "default_mapping": self.default_mapping,
            "resolved_mapping": self.resolved_mapping,
            "control_profile": self.control_preferences,
            "anonymous_group_id": self.current_group_id,
        }

    def _apply_mapping(self, mapping: object) -> None:
        validated = self._validate_mapping(mapping)
        if validated == self._mapping:
            return
        self._mapping = validated
        self.mapping_changed.emit(self.runtime_config())
        self._persist_current_settings()

    def _restore_last_group(self) -> None:
        if self.classroom_storage is not None:
            group_id = self.classroom_storage.active_group()
            if not group_id:
                return
            payload = self.classroom_storage.read_settings(group_id)
            try:
                self._mapping = self._validate_mapping(payload.get("mapping"))
                self._control_preferences = self._validate_control_preferences(
                    {
                        "sensitivity": payload.get("sensitivity"),
                        "control_style": payload.get("control_style"),
                    },
                    fallback=self._default_control_preferences,
                )
            except GameMappingConfigError:
                return
            self.current_group_id = group_id
            return
        path = self.storage_root / "current-group.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            group_id = payload.get("anonymous_group_id", "")
        except (FileNotFoundError, OSError, json.JSONDecodeError, AttributeError):
            return
        normalized = self._valid_group_id(str(group_id))
        if normalized is None:
            return
        group_path = self._group_path(normalized)
        try:
            saved = json.loads(group_path.read_text(encoding="utf-8"))
            self._mapping = self._validate_mapping(saved.get("mapping"))
            self._control_preferences = self._validate_control_preferences(
                saved.get("control_profile"), fallback=self._default_control_preferences
            )
        except (FileNotFoundError, OSError, json.JSONDecodeError, AttributeError, GameMappingConfigError):
            return
        self.current_group_id = normalized

    def _apply_control_preferences(self, preferences: dict[str, str]) -> None:
        if preferences == self._control_preferences:
            return
        self._control_preferences = dict(preferences)
        self.control_preferences_changed.emit(
            preferences["sensitivity"], preferences["control_style"]
        )
        self._persist_current_settings()

    def _persist_current_settings(self) -> None:
        if self.classroom_storage is None or not self.current_group_id:
            return
        try:
            self.classroom_storage.write_settings(
                self.current_group_id,
                {
                    "mapping": self.resolved_mapping,
                    "sensitivity": self._control_preferences["sensitivity"],
                    "control_style": self._control_preferences["control_style"],
                },
            )
        except (OSError, ClassroomStorageError):
            return

    def _group_path(self, group_id: str) -> Path:
        return self.storage_root / f"{group_id}.json"

    @staticmethod
    def _valid_group_id(group_id: str) -> str | None:
        normalized = str(group_id).strip()
        if not normalized or ".." in normalized or not re.fullmatch(r"[A-Za-z0-9_-]+", normalized):
            return None
        return normalized

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        ClassroomStorage.atomic_write_json(path, payload)

    @staticmethod
    def _validate_commands(value: object) -> dict[str, dict[str, str]]:
        if not isinstance(value, dict) or set(value) != set(COMMANDS):
            raise GameMappingConfigError("课程指令配置必须包含 A、B 和 none")
        commands: dict[str, dict[str, str]] = {}
        expected_gestures = {"A": "fist", "B": "open-palm", "none": "rest"}
        for command in COMMANDS:
            details = value.get(command)
            if not isinstance(details, dict):
                raise GameMappingConfigError(f"课程指令 {command} 配置无效")
            display_name = details.get("display_name_zh")
            game_gesture = details.get("game_gesture")
            if not isinstance(display_name, str) or not display_name.strip():
                raise GameMappingConfigError(f"课程指令 {command} 缺少中文名称")
            if game_gesture != expected_gestures[command]:
                raise GameMappingConfigError(f"课程指令 {command} 的游戏手势无效")
            commands[command] = {
                "display_name_zh": display_name.strip(),
                "game_gesture": str(game_gesture),
            }
        return commands

    @staticmethod
    def _validate_mapping(value: object) -> dict[str, str]:
        if not isinstance(value, dict) or set(value) != set(GESTURES):
            raise GameMappingConfigError("游戏映射必须包含四类课程手势")
        mapping = {gesture: str(value[gesture]) for gesture in GESTURES}
        if mapping["rest"] != "none" or mapping["pinch"] != "none":
            raise GameMappingConfigError("放松和捏合必须固定为无操作")
        if {mapping["fist"], mapping["open-palm"]} != {"A", "B"}:
            raise GameMappingConfigError("握拳和伸掌必须一一对应指令 A/B")
        return mapping

    @staticmethod
    def _validate_control_preferences(
        value: object, *, fallback: dict[str, str] | None = None
    ) -> dict[str, str]:
        if not isinstance(value, dict):
            if fallback is not None:
                return dict(fallback)
            raise GameMappingConfigError("学生控制设置无效")
        sensitivity = str(value.get("sensitivity", ""))
        control_style = str(value.get("control_style", ""))
        if sensitivity not in SENSITIVITIES or control_style not in CONTROL_STYLES:
            if fallback is not None:
                return dict(fallback)
            raise GameMappingConfigError("学生控制设置必须使用课程预设")
        return {"sensitivity": sensitivity, "control_style": control_style}
