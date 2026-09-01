"""Student-safe discovery and QProcess orchestration for personal EffiE training."""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal

from emg_live_marker.ml.gesture_model import DemoGesturePredictor, load_model
from emg_live_marker.realtime.student_observation import StudentObservationService

COLLECT_GESTURES = ("fist", "finger_spread", "thumb_index_pinch")
GAME_GESTURES = ("rest", "fist", "open-palm", "pinch")
GESTURE_NAMES_ZH = {
    "fist": "握拳",
    "finger_spread": "伸掌",
    "thumb_index_pinch": "捏合",
    "rest": "放松",
    "open-palm": "伸掌",
    "pinch": "捏合",
}
REQUIRED_ARTIFACTS = (
    "gesture_classifier.pt",
    "gesture_labels.json",
    "normalization.json",
    "model_info.json",
    "train_report.json",
)
EPOCH_PATTERN = re.compile(
    r"epoch\s+(?P<epoch>\d+)\s+train_acc=(?P<train>[0-9.]+)\s+val_acc=(?P<val>[0-9.]+|nan)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class StudentTrainingSession:
    group_id: str
    session_id: str
    path: Path
    counts: dict[str, int]
    ready: bool
    message: str


@dataclass(frozen=True)
class PersonalTrainingSnapshot:
    state: str
    message: str
    epoch: int = 0
    total_epochs: int = 0
    progress_percent: int = 0
    validation_accuracy: float | None = None
    counts: dict[str, int] | None = None
    per_class_performance: dict[str, float | None] | None = None
    model_source: str = "standard"


class StudentPersonalTrainingService(QObject):
    """Launch the existing EffiE fine-tuning CLI without exposing technical knobs."""

    state_changed = Signal(object)
    sessions_changed = Signal(object)

    def __init__(
        self,
        project_root: Path,
        dataset_root: Path,
        course_config: dict[str, Any],
        observation_service: StudentObservationService,
        *,
        process_factory: Callable[[QObject], object] | None = None,
        model_loader: Callable[[str | Path], Any] | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.project_root = Path(project_root).resolve()
        self.dataset_root = Path(dataset_root).resolve()
        self.course_config = course_config
        self.observation_service = observation_service
        self._process_factory = process_factory or (lambda parent: QProcess(parent))
        self._model_loader = model_loader or load_model
        training = course_config.get("personal_training", {})
        if not isinstance(training, dict):
            training = {}
        self.minimum_trials = max(
            5, int(training.get("minimum_completed_trials_per_gesture", 5))
        )
        self.epochs = max(1, int(training.get("epochs", 50)))
        self.batch_size = max(1, int(training.get("batch_size", 128)))
        self.learning_rate = float(training.get("learning_rate", 1e-4))
        self.device = str(training.get("device", "auto"))
        self.validation_split = str(training.get("validation_split", "trial"))
        self.max_rest_ratio = float(training.get("max_rest_ratio", 1.0))
        self.seed = int(training.get("seed", 42))
        self.export_torchscript = bool(training.get("export_torchscript", False))
        output_value = Path(str(training.get("output_directory", "apps/desktop/models/student_personal")))
        self.output_root = (
            output_value.resolve()
            if output_value.is_absolute()
            else (self.project_root / output_value).resolve()
        )
        self.sessions: tuple[StudentTrainingSession, ...] = ()
        self.process: object | None = None
        self.running = False
        self.cancel_requested = False
        self.temporary_output_dir: Path | None = None
        self.last_valid_model_dir: Path | None = None
        self.last_arguments: list[str] = []
        self.report_summary: dict[str, Any] = {}
        self._stdout_buffer = ""
        self._last_output_line = ""
        self._selected_session: StudentTrainingSession | None = None
        self._process_handled = False
        self._cancel_timer = QTimer(self)
        self._cancel_timer.setSingleShot(True)
        self._cancel_timer.setInterval(1500)
        self._cancel_timer.timeout.connect(self._kill_if_running)
        self.snapshot = PersonalTrainingSnapshot(
            "idle", "请选择已有采集数据。", total_epochs=self.epochs
        )

    def discover_sessions(self) -> tuple[StudentTrainingSession, ...]:
        discovered: list[StudentTrainingSession] = []
        if self.dataset_root.is_dir():
            for metadata_path in sorted(self.dataset_root.rglob("metadata.json")):
                session_dir = metadata_path.parent
                try:
                    relative = session_dir.relative_to(self.dataset_root)
                except ValueError:
                    continue
                group_id = relative.parts[0] if len(relative.parts) > 1 else "未分组"
                discovered.append(self._inspect_session(group_id, session_dir))
        self.sessions = tuple(discovered)
        self.sessions_changed.emit(self.sessions)
        return self.sessions

    def inspect_session(self, group_id: str, session_dir: str | Path) -> StudentTrainingSession:
        """Validate one discovered session without exposing file selection in the UI."""

        return self._inspect_session(group_id, Path(session_dir).resolve())

    def start_training(self, session: StudentTrainingSession) -> bool:
        if self.running:
            self._emit("running", "训练正在进行，请勿重复启动。")
            return False
        inspected = self._inspect_session(session.group_id, session.path)
        if not inspected.ready:
            self._emit("error", inspected.message, counts=inspected.counts)
            return False
        standard_path = self.observation_service.settings.model_path.resolve()
        if not standard_path.is_file():
            self._emit("error", "标准教学模型缺失，无法开始个人训练。", counts=inspected.counts)
            return False

        self.output_root.mkdir(parents=True, exist_ok=True)
        temp_path = Path(tempfile.mkdtemp(prefix=".training-", dir=self.output_root))
        if temp_path == standard_path.parent or standard_path.is_relative_to(temp_path):
            self._cleanup_output(temp_path)
            self._emit("error", "训练输出位置无效，标准模型不会被覆盖。")
            return False
        self.temporary_output_dir = temp_path
        self._selected_session = inspected
        self.cancel_requested = False
        self._process_handled = False
        self._stdout_buffer = ""
        self._last_output_line = ""
        self.report_summary = {}
        self.last_arguments = self._build_arguments(inspected, temp_path, standard_path)
        process = self._process_factory(self)
        self.process = process
        process.readyReadStandardOutput.connect(self._read_stdout)
        process.finished.connect(self._on_finished)
        process.errorOccurred.connect(self._on_process_error)
        if hasattr(process, "setProcessChannelMode"):
            process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.setProgram(sys.executable)
        process.setArguments(self.last_arguments)
        if hasattr(process, "setWorkingDirectory"):
            process.setWorkingDirectory(str(self.project_root / "apps" / "desktop"))
        if hasattr(process, "setProcessEnvironment"):
            environment = QProcessEnvironment.systemEnvironment()
            source_root = str(self.project_root / "apps" / "desktop" / "src")
            previous = environment.value("PYTHONPATH")
            environment.insert(
                "PYTHONPATH", source_root if not previous else source_root + os.pathsep + previous
            )
            process.setProcessEnvironment(environment)
        self.running = True
        self._emit(
            "running",
            "正在训练个人模型，请保持程序开启。",
            counts=inspected.counts,
        )
        process.start()
        return True

    def cancel_training(self) -> bool:
        if not self.running or self.process is None:
            return False
        self.cancel_requested = True
        self._emit("cancelling", "正在取消训练……")
        self.process.terminate()
        self._cancel_timer.start()
        return True

    def use_standard_model(self) -> bool:
        switched = self.observation_service.use_standard_model()
        self._emit(
            "model-selected" if switched else "error",
            "已切换到标准模型。" if switched else "标准模型切换失败。",
        )
        return switched

    def use_personal_model(self) -> bool:
        switched = self.observation_service.use_personal_model()
        self._emit(
            "model-selected" if switched else "error",
            "已切换到我的模型。" if switched else "尚无有效的个人模型，请先完成训练。",
        )
        return switched

    def close(self) -> None:
        if not self.running or self.process is None:
            return
        self.cancel_requested = True
        self.process.terminate()
        if hasattr(self.process, "waitForFinished") and not self.process.waitForFinished(1500):
            self.process.kill()
            self.process.waitForFinished(1500)
        self._cancel_timer.stop()
        if self.running:
            self._process_handled = True
            self.running = False
            self._cleanup_output(self.temporary_output_dir)
            self.temporary_output_dir = None

    def _inspect_session(self, group_id: str, session_dir: Path) -> StudentTrainingSession:
        metadata_path = session_dir / "metadata.json"
        counts = {gesture: 0 for gesture in COLLECT_GESTURES}
        missing = [
            filename
            for filename in ("metadata.json", "emg.csv", "events.csv")
            if not (session_dir / filename).is_file()
        ]
        if missing:
            message = "数据文件不完整：缺少 " + "、".join(missing)
            return StudentTrainingSession(group_id, session_dir.name, session_dir, counts, False, message)
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return StudentTrainingSession(
                group_id, session_dir.name, session_dir, counts, False, "metadata.json 无效。"
            )
        if not isinstance(metadata, dict):
            return StudentTrainingSession(
                group_id, session_dir.name, session_dir, counts, False, "metadata.json 必须是对象。"
            )
        counts = self._completed_counts(metadata, session_dir / "events.csv")
        ready = all(counts[gesture] >= self.minimum_trials for gesture in COLLECT_GESTURES)
        count_text = "，".join(
            f"{GESTURE_NAMES_ZH[gesture]} {counts[gesture]} 次" for gesture in COLLECT_GESTURES
        )
        message = (
            f"有效数据：{count_text}。"
            if ready
            else f"数据不足：{count_text}；每类至少需要 {self.minimum_trials} 次有效采集。"
        )
        return StudentTrainingSession(group_id, session_dir.name, session_dir, counts, ready, message)

    @staticmethod
    def _completed_counts(metadata: dict[str, Any], events_path: Path) -> dict[str, int]:
        counts = {gesture: 0 for gesture in COLLECT_GESTURES}
        statuses = metadata.get("trial_statuses")
        if isinstance(statuses, dict):
            gesture_by_trial = StudentPersonalTrainingService._trial_gestures(events_path)
            for trial_id, status in statuses.items():
                gesture = gesture_by_trial.get(str(trial_id))
                if status == "completed" and gesture in counts:
                    counts[gesture] += 1
            return counts
        valid_counts = metadata.get("valid_trial_counts")
        if isinstance(valid_counts, dict):
            for gesture in counts:
                try:
                    counts[gesture] = max(0, int(valid_counts.get(gesture, 0)))
                except (TypeError, ValueError):
                    counts[gesture] = 0
            return counts
        completed_trials: set[str] = set()
        gesture_by_trial: dict[str, str] = {}
        try:
            with events_path.open(newline="", encoding="utf-8") as file_obj:
                for row in csv.DictReader(file_obj):
                    trial_id = row.get("trial_id", "")
                    phase = row.get("phase", "")
                    if phase == "gesture_start":
                        gesture_by_trial[trial_id] = row.get("gesture", "")
                    elif phase == "trial_end" and row.get("note", "") != "invalid":
                        completed_trials.add(trial_id)
        except OSError:
            return counts
        for trial_id in completed_trials:
            gesture = gesture_by_trial.get(trial_id)
            if gesture in counts:
                counts[gesture] += 1
        return counts

    @staticmethod
    def _trial_gestures(events_path: Path) -> dict[str, str]:
        result: dict[str, str] = {}
        try:
            with events_path.open(newline="", encoding="utf-8") as file_obj:
                for row in csv.DictReader(file_obj):
                    if row.get("phase") == "gesture_start":
                        result[row.get("trial_id", "")] = row.get("gesture", "")
        except OSError:
            pass
        return result

    def _build_arguments(
        self,
        session: StudentTrainingSession,
        output_dir: Path,
        checkpoint_path: Path,
    ) -> list[str]:
        arguments = [
            "-u",
            "-m",
            "emg_live_marker.cli.finetune_effie_gesture",
            "--effie-root",
            str(self.project_root),
            "--dataset-root",
            str(session.path),
            "--checkpoint-path",
            str(checkpoint_path),
            "--output-dir",
            str(output_dir),
            "--run-id",
            output_dir.name,
            "--mode",
            "freeze_backbone",
            "--epochs",
            str(self.epochs),
            "--batch-size",
            str(self.batch_size),
            "--lr",
            str(self.learning_rate),
            "--device",
            self.device,
            "--val-split",
            self.validation_split,
            "--max-rest-ratio",
            str(self.max_rest_ratio),
            "--seed",
            str(self.seed),
        ]
        if not self.export_torchscript:
            arguments.append("--no-export-torchscript")
        return arguments

    def _read_stdout(self) -> None:
        if self.process is None:
            return
        chunk = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._stdout_buffer += chunk
        lines = self._stdout_buffer.splitlines(keepends=True)
        self._stdout_buffer = ""
        if lines and not lines[-1].endswith(("\n", "\r")):
            self._stdout_buffer = lines.pop()
        for line in lines:
            self._consume_output_line(line.strip())

    def _consume_output_line(self, line: str) -> None:
        if not line:
            return
        self._last_output_line = line
        match = EPOCH_PATTERN.search(line)
        if match is None:
            return
        epoch = int(match.group("epoch"))
        val_text = match.group("val").lower()
        validation = None if val_text == "nan" else float(val_text)
        self._emit(
            "running",
            f"正在训练：第 {epoch} / {self.epochs} 轮",
            epoch=epoch,
            validation_accuracy=validation,
        )

    def _on_finished(self, exit_code: int, _exit_status: object = None) -> None:
        if self._process_handled:
            return
        self._process_handled = True
        self._cancel_timer.stop()
        if self._stdout_buffer.strip():
            self._consume_output_line(self._stdout_buffer.strip())
        self._stdout_buffer = ""
        self.running = False
        if self.cancel_requested:
            self._cleanup_output(self.temporary_output_dir)
            self.temporary_output_dir = None
            self._emit("cancelled", "训练已取消，继续使用训练前模型。")
            return
        if int(exit_code) != 0:
            reason = self._last_output_line or f"训练进程退出码 {exit_code}"
            self._cleanup_output(self.temporary_output_dir)
            self.temporary_output_dir = None
            self._emit("failed", f"训练失败：{reason}。继续使用训练前模型。")
            return
        self._complete_success()

    def _on_process_error(self, _error: object) -> None:
        if not self.running or self._process_handled:
            return
        self._process_handled = True
        self._cancel_timer.stop()
        self.running = False
        self._cleanup_output(self.temporary_output_dir)
        self.temporary_output_dir = None
        self._emit("failed", "训练进程无法启动，继续使用训练前模型。")

    def _complete_success(self) -> None:
        temp_dir = self.temporary_output_dir
        session = self._selected_session
        if temp_dir is None or session is None:
            self._emit("failed", "训练输出状态无效，继续使用训练前模型。")
            return
        missing = [filename for filename in REQUIRED_ARTIFACTS if not (temp_dir / filename).is_file()]
        if missing:
            self._cleanup_output(temp_dir)
            self.temporary_output_dir = None
            self._emit("failed", "训练产物不完整：缺少 " + "、".join(missing) + "。")
            return
        try:
            predictor = self._model_loader(temp_dir / "gesture_classifier.pt")
            if isinstance(predictor, DemoGesturePredictor):
                raise ValueError("demo predictor is not a valid personal model")
            report = json.loads((temp_dir / "train_report.json").read_text(encoding="utf-8"))
            summary = self._report_summary(report)
        except Exception as exc:  # noqa: BLE001 - invalid outputs never become active.
            self._cleanup_output(temp_dir)
            self.temporary_output_dir = None
            self._emit("failed", f"个人模型验证失败：{exc}。继续使用训练前模型。")
            return
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        safe_group = re.sub(r"[^A-Za-z0-9_-]", "_", session.group_id)
        final_dir = self.output_root / f"{safe_group}_{stamp}_{uuid4().hex[:8]}"
        try:
            temp_dir.replace(final_dir)
            (final_dir / "personal_model.valid.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "anonymous_group_id": session.group_id,
                        "validated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except OSError:
            self._cleanup_output(temp_dir)
            self.temporary_output_dir = None
            self._emit("failed", "无法保存已验证的个人模型，继续使用训练前模型。")
            return
        if not self.observation_service.activate_personal_model(
            final_dir / "gesture_classifier.pt", predictor
        ):
            self._cleanup_output(final_dir)
            self.temporary_output_dir = None
            self._emit("failed", "个人模型激活失败，继续使用训练前模型。")
            return
        self.last_valid_model_dir = final_dir
        self.temporary_output_dir = None
        self.report_summary = summary
        self._emit(
            "completed",
            "个人模型训练完成，已自动切换到我的模型。",
            epoch=self.epochs,
            validation_accuracy=summary["validation_accuracy"],
            per_class=summary["per_class_performance"],
        )

    @staticmethod
    def _report_summary(report: object) -> dict[str, Any]:
        if not isinstance(report, dict):
            raise ValueError("train_report.json 必须是对象")
        validation = report.get("val_accuracy")
        if validation is None:
            validation = report.get("best_val_accuracy")
        validation_value = None if validation is None else float(validation)
        metrics = report.get("per_class_precision_recall_f1", {})
        performance: dict[str, float | None] = {}
        for gesture in GAME_GESTURES:
            values = metrics.get(gesture, {}) if isinstance(metrics, dict) else {}
            try:
                performance[gesture] = float(values.get("f1"))
            except (AttributeError, TypeError, ValueError):
                performance[gesture] = None
        return {
            "validation_accuracy": validation_value,
            "per_class_performance": performance,
        }

    def _kill_if_running(self) -> None:
        if self.running and self.process is not None:
            self.process.kill()

    def _emit(
        self,
        state: str,
        message: str,
        *,
        epoch: int = 0,
        validation_accuracy: float | None = None,
        counts: dict[str, int] | None = None,
        per_class: dict[str, float | None] | None = None,
    ) -> None:
        self.snapshot = PersonalTrainingSnapshot(
            state=state,
            message=message,
            epoch=epoch,
            total_epochs=self.epochs,
            progress_percent=min(100, round(epoch * 100 / self.epochs)) if epoch else 0,
            validation_accuracy=validation_accuracy,
            counts=dict(counts) if counts is not None else None,
            per_class_performance=dict(per_class) if per_class is not None else None,
            model_source=self.observation_service.model_source,
        )
        self.state_changed.emit(self.snapshot)

    @staticmethod
    def _cleanup_output(path: Path | None) -> None:
        if path is not None and path.exists():
            shutil.rmtree(path, ignore_errors=True)
