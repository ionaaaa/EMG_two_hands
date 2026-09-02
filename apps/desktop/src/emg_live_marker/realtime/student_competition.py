"""Trusted student competition-result validation and persistence."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from PySide6.QtCore import QObject, Signal

from emg_live_marker.realtime.game_mapping import GameMappingService
from emg_live_marker.realtime.classroom_storage import ClassroomStorage, ClassroomStorageError
from emg_live_marker.realtime.student_observation import StudentObservationService

VALID_MODES = ("left", "right", "both")


class CompetitionResultError(ValueError):
    """Raised when an untrusted browser result is invalid."""


class StudentCompetitionService(QObject):
    """Complete browser scores with the current trusted desktop state."""

    result_saved = Signal(object)
    save_failed = Signal(str)

    def __init__(
        self,
        project_root: Path,
        course_config: dict[str, Any],
        observation_service: StudentObservationService,
        mapping_service: GameMappingService,
        *,
        classroom_storage: ClassroomStorage | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.project_root = Path(project_root).resolve()
        self.observation_service = observation_service
        self.mapping_service = mapping_service
        self.classroom_storage = classroom_storage
        storage = course_config.get("storage", {})
        template = storage.get("competition_results_path", "") if isinstance(storage, dict) else ""
        self.results_path_template = str(template).strip()
        self.student_id = ""
        self.latest_result: dict[str, Any] | None = None
        self.latest_result_path: Path | None = None
        self._seen_result_ids: set[str] = set()
        self._lock = Lock()

    def begin_competition(self, student_id: str) -> tuple[bool, str]:
        normalized = self._valid_student_id(student_id)
        if normalized is None:
            return False, "请先填写匿名小组编号，才能开始挑战赛。"
        if self.classroom_storage is None and not self.results_path_template:
            return False, "比赛结果保存位置未配置，请联系老师。"
        try:
            if self.classroom_storage is not None:
                self.classroom_storage.group_paths(normalized, create=True)
                self.classroom_storage.set_active_group(normalized)
                if self.mapping_service.current_group_id != normalized:
                    loaded, _message = self.mapping_service.load_group(normalized)
                    if not loaded:
                        self.mapping_service.save_current_group(normalized)
        except (OSError, ClassroomStorageError):
            return False, "匿名小组课堂目录创建失败，请联系老师。"
        with self._lock:
            if normalized != self.student_id:
                self._seen_result_ids.clear()
            self.student_id = normalized
        return True, "挑战赛准备完成。"

    def save_browser_result(
        self, payload: object, *, result_id: str = ""
    ) -> tuple[int, dict[str, Any]]:
        try:
            browser_result = self._validate_browser_result(payload)
        except CompetitionResultError as exc:
            return 400, {"ok": False, "message": f"比赛结果无效：{exc}"}
        with self._lock:
            if not self.student_id:
                return 409, {"ok": False, "message": "匿名小组编号为空，不能保存成绩。"}
            dedupe_id = self._dedupe_id(browser_result, result_id)
            if dedupe_id in self._seen_result_ids:
                return 200, {
                    "ok": True,
                    "duplicate": True,
                    "message": "本局成绩已经保存，无需重复提交。",
                }
            self._seen_result_ids.add(dedupe_id)
            student_id = self.student_id

        timestamp = datetime.now(timezone.utc)
        competition_id = timestamp.strftime("%Y%m%d_%H%M%S_%f") + "_" + uuid4().hex[:8]
        result = {
            "student_id": student_id,
            "model_type": (
                "personal" if self.observation_service.model_source == "personal" else "standard"
            ),
            "mapping": self.mapping_service.resolved_mapping,
            "sensitivity": self.observation_service.current_sensitivity,
            "stability": self.observation_service.current_control_style,
            "score": browser_result["score"],
            "accuracy": browser_result["accuracy"],
            "max_combo": browser_result["max_combo"],
            "timestamp": timestamp.isoformat(timespec="milliseconds"),
            "mode": browser_result["mode"],
            "outcome": browser_result["outcome"],
        }
        for name in ("left_accuracy", "right_accuracy"):
            if name in browser_result:
                result[name] = browser_result[name]
        try:
            result_path = self._result_path(competition_id, student_id)
            result_path.parent.mkdir(parents=True, exist_ok=True)
            ClassroomStorage.atomic_write_json(result_path, result)
        except (OSError, KeyError, ValueError) as exc:
            with self._lock:
                self._seen_result_ids.discard(dedupe_id)
            message = f"成绩保存失败：{exc}"
            self.save_failed.emit(message)
            return 500, {"ok": False, "message": message}

        self.latest_result = dict(result)
        self.latest_result_path = result_path
        self.result_saved.emit(dict(result))
        return 201, {"ok": True, "duplicate": False, "message": "成绩已保存。"}

    def _result_path(self, competition_id: str, student_id: str) -> Path:
        if self.classroom_storage is not None:
            return self.classroom_storage.result_path(student_id, competition_id)
        configured = Path(
            self.results_path_template.format(
                competition_id=competition_id,
                student_id=student_id,
            )
        )
        resolved = configured if configured.is_absolute() else self.project_root / configured
        resolved = resolved.resolve()
        if resolved.suffix.lower() == ".json":
            return resolved
        return resolved / "result.json"

    @staticmethod
    def _validate_browser_result(payload: object) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise CompetitionResultError("提交内容必须是对象")
        mode = str(payload.get("mode", ""))
        if mode not in VALID_MODES:
            raise CompetitionResultError("比赛模式无效")
        score = StudentCompetitionService._integer(payload.get("score"), "得分")
        max_combo = StudentCompetitionService._integer(
            payload.get("max_combo"), "最大连击"
        )
        accuracy = StudentCompetitionService._accuracy(payload.get("accuracy"), "正确率")
        outcome = payload.get("outcome")
        if not isinstance(outcome, str) or not outcome.strip() or len(outcome) > 100:
            raise CompetitionResultError("最终结果无效")
        outcome = outcome.strip()
        allowed_outcomes = (
            {"draw", "left_win", "right_win"} if mode == "both" else {"completed"}
        )
        if outcome not in allowed_outcomes:
            raise CompetitionResultError("最终结果与比赛模式不匹配")
        validated: dict[str, Any] = {
            "mode": mode,
            "score": score,
            "accuracy": accuracy,
            "max_combo": max_combo,
            "outcome": outcome,
        }
        if mode == "both":
            validated["left_accuracy"] = StudentCompetitionService._accuracy(
                payload.get("left_accuracy"), "左手正确率"
            )
            validated["right_accuracy"] = StudentCompetitionService._accuracy(
                payload.get("right_accuracy"), "右手正确率"
            )
        else:
            for name, display in (
                ("left_accuracy", "左手正确率"),
                ("right_accuracy", "右手正确率"),
            ):
                if name in payload:
                    validated[name] = StudentCompetitionService._accuracy(payload[name], display)
        return validated

    @staticmethod
    def _integer(value: object, display: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 1_000_000_000:
            raise CompetitionResultError(f"{display}必须是非负整数")
        return int(value)

    @staticmethod
    def _accuracy(value: object, display: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise CompetitionResultError(f"{display}必须是数值")
        result = float(value)
        if not math.isfinite(result) or not 0.0 <= result <= 1.0:
            raise CompetitionResultError(f"{display}必须在 0 到 1 之间")
        return result

    @staticmethod
    def _valid_student_id(value: str) -> str | None:
        normalized = str(value).strip()
        if not normalized or ".." in normalized:
            return None
        if not re.fullmatch(r"[A-Za-z0-9_-]+", normalized):
            return None
        return normalized

    @staticmethod
    def _dedupe_id(result: dict[str, Any], result_id: str) -> str:
        normalized = str(result_id).strip()
        if normalized and len(normalized) <= 128:
            return "id:" + normalized
        canonical = json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return "payload:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
