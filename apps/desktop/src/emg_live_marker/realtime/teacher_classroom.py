"""Teacher classroom settings and safe management of existing student artifacts."""

from __future__ import annotations

import copy
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from PySide6.QtCore import QObject, QStandardPaths, Signal

from emg_live_marker.ml.gesture_model import DemoGesturePredictor, load_model
from emg_live_marker.paths import ProjectPaths


def load_teaching_course_config(project_root: Path) -> dict[str, Any]:
    path = Path(project_root) / "configs" / "teaching" / "yucai.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def default_classroom_settings_path() -> Path:
    root = Path(
        QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppLocalDataLocation)
    )
    return root / "classroom_settings.json"


def merge_classroom_overrides(
    course_config: dict[str, Any],
    project_root: Path,
    *,
    settings_path: Path | None = None,
) -> dict[str, Any]:
    """Merge the small, teacher-owned override surface into a course copy."""

    merged = copy.deepcopy(course_config)
    path = Path(settings_path or default_classroom_settings_path())
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return merged
    if not isinstance(payload, dict):
        return merged
    model_path = payload.get("standard_model_path")
    if isinstance(model_path, str) and model_path.strip():
        configured = Path(model_path)
        if not configured.is_absolute():
            configured = Path(project_root) / configured
        merged.setdefault("realtime_decoding", {})["standard_teaching_model_path"] = str(
            configured.resolve()
        )
    trials = payload.get("trials_per_action")
    if isinstance(trials, int) and not isinstance(trials, bool) and 5 <= trials <= 100:
        merged.setdefault("collection", {})["trials_per_action"] = trials
    enabled = payload.get("personal_training_enabled")
    if isinstance(enabled, bool):
        merged.setdefault("personal_training", {})["enabled"] = enabled
    return merged


@dataclass(frozen=True)
class ClassroomSettings:
    standard_model_path: str
    trials_per_action: int
    personal_training_enabled: bool
    teacher_password_enabled: bool = False


@dataclass(frozen=True)
class ClassroomSession:
    student_id: str
    session_id: str
    hand: str
    status: str
    valid_counts: dict[str, int]
    invalid_count: int
    repeated_count: int
    recollect_requested: bool
    path: Path


@dataclass(frozen=True)
class ClassroomModel:
    version: str
    path: Path
    model_type: str


@dataclass(frozen=True)
class PersonalModelRecord:
    student_id: str
    path: Path
    validation_accuracy: float | None
    trained_at: str


class TeacherClassroomService(QObject):
    settings_changed = Signal(object)
    sessions_changed = Signal(object)
    runtime_reset_requested = Signal()

    def __init__(
        self,
        paths: ProjectPaths,
        course_config: dict[str, Any],
        *,
        settings_path: Path | None = None,
        mapping_storage_root: Path | None = None,
        model_loader: Callable[[str | Path], Any] | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.paths = paths
        self.course_config = course_config
        self.settings_path = Path(settings_path or default_classroom_settings_path()).resolve()
        default_mapping_root = Path(
            QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppLocalDataLocation)
        ) / "student-game-mappings"
        self.mapping_storage_root = Path(mapping_storage_root or default_mapping_root).resolve()
        self._model_loader = model_loader or load_model
        self._base_settings = self._settings_from_course(course_config)
        self.settings = self.load_settings()

    def load_settings(self) -> ClassroomSettings:
        merged = merge_classroom_overrides(
            self.course_config,
            self.paths.project_root,
            settings_path=self.settings_path,
        )
        settings = self._settings_from_course(merged)
        try:
            payload = json.loads(self.settings_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            payload = {}
        password_enabled = bool(payload.get("teacher_password_enabled", False)) if isinstance(payload, dict) else False
        self.settings = ClassroomSettings(
            settings.standard_model_path,
            settings.trials_per_action,
            settings.personal_training_enabled,
            password_enabled,
        )
        return self.settings

    @property
    def configured_standard_model_path(self) -> Path | None:
        return self._resolve_setting_model_path(self.settings.standard_model_path)

    def save_settings(
        self,
        *,
        standard_model_path: str | Path,
        trials_per_action: int,
        personal_training_enabled: bool,
        teacher_password_enabled: bool = False,
    ) -> tuple[bool, str]:
        model_path = Path(standard_model_path).resolve()
        valid_models = {model.path.resolve() for model in self.scan_standard_models()}
        if model_path not in valid_models:
            return False, "请选择已验证的完整标准模型。"
        if not 5 <= int(trials_per_action) <= 100:
            return False, "每类采集次数必须在 5 到 100 之间。"
        try:
            stored_path = str(model_path.relative_to(self.paths.project_root))
        except ValueError:
            stored_path = str(model_path)
        payload = {
            "schema_version": 1,
            "standard_model_path": stored_path.replace("\\", "/"),
            "trials_per_action": int(trials_per_action),
            "personal_training_enabled": bool(personal_training_enabled),
            "teacher_password_enabled": bool(teacher_password_enabled),
        }
        try:
            self.settings_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_json(self.settings_path, payload)
        except OSError:
            return False, "课堂设置保存失败，请检查应用数据目录。"
        self.settings = ClassroomSettings(
            stored_path,
            int(trials_per_action),
            bool(personal_training_enabled),
            bool(teacher_password_enabled),
        )
        self.settings_changed.emit(self.settings)
        return True, "课堂设置已保存，下次进入学生模式时生效。"

    def scan_standard_models(self) -> tuple[ClassroomModel, ...]:
        required = ("gesture_labels.json", "normalization.json", "model_info.json")
        discovered: list[ClassroomModel] = []
        if not self.paths.models_root.is_dir():
            return ()
        model_dirs = [self.paths.models_root]
        model_dirs.extend(path for path in self.paths.models_root.rglob("*") if path.is_dir())
        for model_dir in sorted(model_dirs):
            if not all((model_dir / name).is_file() for name in required):
                continue
            candidates = [model_dir / "gesture_classifier.pt", model_dir / "gesture_classifier.ts"]
            model_path = next((path for path in candidates if path.is_file()), None)
            if model_path is None:
                continue
            try:
                predictor = self._model_loader(model_path)
                if isinstance(predictor, DemoGesturePredictor):
                    continue
            except Exception:  # noqa: BLE001 - invalid candidates are omitted.
                continue
            discovered.append(
                ClassroomModel(
                    version=model_dir.name,
                    path=model_path.resolve(),
                    model_type=str(getattr(predictor, "model_type", "model")),
                )
            )
        return tuple(discovered)

    def scan_sessions(self) -> tuple[ClassroomSession, ...]:
        sessions: list[ClassroomSession] = []
        if not self.paths.dataset_root.is_dir():
            return ()
        for metadata_path in sorted(self.paths.dataset_root.rglob("metadata.json")):
            session_dir = metadata_path.parent.resolve()
            if not self._inside(session_dir, self.paths.dataset_root):
                continue
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(metadata, dict):
                continue
            statuses = metadata.get("trial_statuses", {})
            valid_counts = metadata.get("valid_trial_counts", {})
            if not isinstance(statuses, dict):
                statuses = {}
            if not isinstance(valid_counts, dict):
                valid_counts = {}
            try:
                relative = session_dir.relative_to(self.paths.dataset_root.resolve())
            except ValueError:
                continue
            student_id = str(
                metadata.get("anonymous_id") or metadata.get("subject_id") or relative.parts[0]
            )
            sessions.append(
                ClassroomSession(
                    student_id=student_id,
                    session_id=str(metadata.get("session_id") or session_dir.name),
                    hand=str(metadata.get("selected_hand") or metadata.get("side") or "未知"),
                    status=str(metadata.get("collection_status") or "未知"),
                    valid_counts={
                        gesture: self._safe_int(valid_counts.get(gesture, 0))
                        for gesture in ("fist", "finger_spread", "thumb_index_pinch")
                    },
                    invalid_count=self._safe_int(
                        metadata.get("invalid_trial_count", sum(v == "invalid" for v in statuses.values()))
                    ),
                    repeated_count=self._safe_int(
                        metadata.get("repeated_trial_count", sum(v == "repeated" for v in statuses.values()))
                    ),
                    recollect_requested=(session_dir / "recollect_requested.json").is_file(),
                    path=session_dir,
                )
            )
        return tuple(sessions)

    def mark_recollect(self, session_path: str | Path, reason: str = "教师标记重新采集") -> tuple[bool, str]:
        session = self._valid_session_path(session_path)
        if session is None:
            return False, "采集会话路径无效，不能标记。"
        try:
            self._write_json(
                session / "recollect_requested.json",
                {"schema_version": 1, "requested": True, "reason": str(reason)},
            )
        except OSError:
            return False, "重新采集标记保存失败。"
        self.sessions_changed.emit(self.scan_sessions())
        return True, "已标记重新采集，旧数据仍然保留。"

    def delete_session(self, session_path: str | Path, *, confirmed: bool) -> tuple[bool, str]:
        if not confirmed:
            return False, "删除已取消。"
        session = self._valid_session_path(session_path)
        if session is None:
            return False, "采集会话路径无效，禁止删除。"
        try:
            shutil.rmtree(session)
        except OSError:
            return False, "本次采集删除失败。"
        self.sessions_changed.emit(self.scan_sessions())
        return True, "本次采集已删除，无法从程序内恢复。"

    def scan_personal_models(self) -> tuple[PersonalModelRecord, ...]:
        records: list[PersonalModelRecord] = []
        if not self.paths.models_root.is_dir():
            return ()
        for marker_path in sorted(self.paths.models_root.rglob("personal_model.valid.json")):
            model_dir = marker_path.parent.resolve()
            report_path = model_dir / "train_report.json"
            model_path = model_dir / "gesture_classifier.pt"
            if not report_path.is_file() or not model_path.is_file():
                continue
            try:
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            validation = report.get("val_accuracy", report.get("best_val_accuracy"))
            try:
                validation_value = None if validation is None else float(validation)
            except (TypeError, ValueError):
                validation_value = None
            records.append(
                PersonalModelRecord(
                    student_id=str(marker.get("anonymous_group_id", "未知")),
                    path=model_dir,
                    validation_accuracy=validation_value,
                    trained_at=str(
                        marker.get("validated_at")
                        or report.get("completed_at")
                        or report.get("trained_at")
                        or "未知"
                    ),
                )
            )
        return tuple(records)

    def delete_personal_model(
        self, model_dir: str | Path, *, confirmed: bool
    ) -> tuple[bool, str]:
        if not confirmed:
            return False, "删除已取消。"
        target = Path(model_dir).resolve()
        if not self._inside(target, self.paths.models_root):
            return False, "个人模型路径无效，禁止删除。"
        if not (target / "personal_model.valid.json").is_file():
            return False, "标准模型或未验证模型禁止删除。"
        standard = self.configured_standard_model_path
        if standard is not None and target == standard.parent:
            return False, "标准模型禁止删除。"
        try:
            shutil.rmtree(target)
        except OSError:
            return False, "个人模型删除失败。"
        return True, "个人模型已删除，无法从程序内恢复。"

    def scan_competition_results(self) -> tuple[dict[str, Any], ...]:
        required = {
            "student_id", "model_type", "mapping", "sensitivity", "stability",
            "mode", "score", "accuracy", "max_combo", "outcome", "timestamp",
        }
        records: list[dict[str, Any]] = []
        if not self.paths.reports_root.is_dir():
            return ()
        for path in sorted(self.paths.reports_root.rglob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict) and required <= set(payload):
                record = dict(payload)
                record["_path"] = str(path)
                records.append(record)
        return tuple(records)

    def export_competition_csv(
        self, output_path: str | Path, *, student_id: str = ""
    ) -> tuple[bool, str]:
        fields = (
            "student_id", "model_type", "mapping", "sensitivity", "stability", "mode",
            "score", "accuracy", "max_combo", "outcome", "timestamp",
        )
        records = [
            record for record in self.scan_competition_results()
            if not student_id or record.get("student_id") == student_id
        ]
        path = Path(output_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="", encoding="utf-8-sig") as file_obj:
                writer = csv.DictWriter(file_obj, fieldnames=fields)
                writer.writeheader()
                for record in records:
                    row = {field: record.get(field, "") for field in fields}
                    row["mapping"] = json.dumps(row["mapping"], ensure_ascii=False, sort_keys=True)
                    writer.writerow(row)
        except OSError:
            return False, "比赛 CSV 导出失败。"
        return True, f"已导出 {len(records)} 条比赛记录。"

    def prepare_next_group(self) -> tuple[bool, str]:
        current_group_path = self.mapping_storage_root / "current-group.json"
        try:
            current_group_path.unlink(missing_ok=True)
        except OSError:
            return False, "当前小组状态清理失败。"
        self.runtime_reset_requested.emit()
        return True, "已准备下一组：运行状态已清理，数据、模型和比赛成绩均保留。"

    @staticmethod
    def device_diagnostics(runtimes: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        fields = (
            "connected", "emg_rate_sps", "imu_rate_sps", "aa_lost_count", "bb_lost_count",
            "global_lost_count", "bad_header_count", "bad_type_count", "resync_count",
        )
        return {
            str(side): {field: getattr(runtime, field) for field in fields}
            for side, runtime in runtimes.items()
        }

    def _valid_session_path(self, value: str | Path) -> Path | None:
        target = Path(value).resolve()
        if not self._inside(target, self.paths.dataset_root):
            return None
        if target == self.paths.dataset_root.resolve() or not (target / "metadata.json").is_file():
            return None
        return target

    @staticmethod
    def _inside(target: Path, root: Path) -> bool:
        try:
            target.resolve().relative_to(Path(root).resolve())
            return True
        except ValueError:
            return False

    def _settings_from_course(self, config: dict[str, Any]) -> ClassroomSettings:
        realtime = config.get("realtime_decoding", {})
        collection = config.get("collection", {})
        training = config.get("personal_training", {})
        return ClassroomSettings(
            standard_model_path=str(realtime.get("standard_teaching_model_path", "")),
            trials_per_action=int(collection.get("trials_per_action", 10)),
            personal_training_enabled=bool(training.get("enabled", True)),
        )

    def _resolve_setting_model_path(self, value: str) -> Path | None:
        if not value:
            return None
        path = Path(value)
        return (path if path.is_absolute() else self.paths.project_root / path).resolve()

    @staticmethod
    def _safe_int(value: object) -> int:
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(path)
