"""Student-mode course navigation window.

The student window intentionally stays separate from :mod:`main_window` and
only exposes the guided course features implemented for student mode.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGridLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from emg_live_marker.device.check_service import (
    DeviceCheckResult,
    DeviceCheckService,
    DeviceCheckThresholds,
)
from emg_live_marker.paths import ProjectPaths, resolve_project_paths
from emg_live_marker.realtime.collection import CollectionController, CollectionPlan
from emg_live_marker.realtime.game_mapping import GameMappingService
from emg_live_marker.realtime.student_game_experience import StudentGameExperienceService
from emg_live_marker.realtime.student_observation import StudentObservationService
from emg_live_marker.realtime.student_control_optimization import StudentControlEffectTestService
from emg_live_marker.realtime.student_competition import StudentCompetitionService
from emg_live_marker.realtime.teacher_classroom import merge_classroom_overrides
from emg_live_marker.realtime.student_personal_training import StudentPersonalTrainingService
from emg_live_marker.ui.student_pages import (
    COURSE_ENTRIES,
    CourseEntry,
    DeviceCheckPage,
    StudentCollectionPage,
    StudentChallengePage,
    StudentGameMappingPage,
    StudentQuickExperiencePage,
    StudentPersonalTrainingPage,
    StudentSignalObservationPage,
    create_collection_gate_page,
    create_course_page,
    create_signal_observation_gate_page,
)

TEACHING_CONFIG_RELATIVE_PATH = Path("configs") / "teaching" / "yucai.json"


class TeachingConfigError(ValueError):
    """Raised when the student-mode teaching preset cannot be used."""


def load_yucai_course_config(
    paths: ProjectPaths | None = None,
    *,
    classroom_settings_path: Path | None = None,
) -> dict[str, Any]:
    """Load the course metadata required to open student mode.

    The location is resolved from the existing project-root path mechanism so no
    host-specific absolute path is stored in the preset or in this module.
    """

    project_paths = paths or resolve_project_paths()
    config_path = project_paths.project_root / TEACHING_CONFIG_RELATIVE_PATH
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise TeachingConfigError(f"学生模式课程配置不存在: {TEACHING_CONFIG_RELATIVE_PATH}") from exc
    except json.JSONDecodeError as exc:
        raise TeachingConfigError(f"学生模式课程配置 JSON 无效: {TEACHING_CONFIG_RELATIVE_PATH}") from exc

    if not isinstance(payload, dict):
        raise TeachingConfigError(f"学生模式课程配置必须是对象: {TEACHING_CONFIG_RELATIVE_PATH}")
    if payload.get("schema_version") != 1:
        raise TeachingConfigError("学生模式课程配置的 schema_version 必须为 1")

    course = payload.get("course")
    if not isinstance(course, dict):
        raise TeachingConfigError("学生模式课程配置缺少 course 对象")
    for field in ("id", "name", "language"):
        if not isinstance(course.get(field), str) or not course[field].strip():
            raise TeachingConfigError(f"学生模式课程配置的 course.{field} 必须是非空字符串")
    return merge_classroom_overrides(
        payload,
        project_paths.project_root,
        settings_path=classroom_settings_path,
    )


class StudentMainWindow(QMainWindow):
    """Chinese course navigation with isolated student-mode services."""

    def __init__(
        self,
        paths: ProjectPaths | None = None,
        *,
        simulate: bool = False,
        device_check_service: DeviceCheckService | None = None,
        game_mapping_service: GameMappingService | None = None,
        game_experience_service: StudentGameExperienceService | None = None,
        personal_training_service: StudentPersonalTrainingService | None = None,
        competition_service: StudentCompetitionService | None = None,
        classroom_settings_path: Path | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.paths = paths or resolve_project_paths()
        self.course_config = load_yucai_course_config(
            self.paths, classroom_settings_path=classroom_settings_path
        )
        self.course = self.course_config["course"]
        self.course_entries = COURSE_ENTRIES
        self.course_entry_buttons: list[QPushButton] = []
        self._entry_page_indexes: dict[str, int] = {}
        self.device_check_service = device_check_service or DeviceCheckService(
            thresholds=DeviceCheckThresholds.from_config(self.course_config),
            simulate=simulate,
            parent=self,
        )
        self.session_device_result: DeviceCheckResult = self.device_check_service.result
        self.device_check_service.result_changed.connect(self._on_device_check_result)
        self._collection_side: str | None = None
        self.collection_controller = CollectionController(
            device_ready=self._selected_side_ready,
            parent=self,
        )
        self.collection_controller.state_changed.connect(self._on_collection_snapshot)
        self.collection_controller.finished.connect(self._on_collection_finished)
        self.observation_service = StudentObservationService(
            self.paths.project_root,
            self.course_config,
            parent=self,
        )
        self.game_mapping_service = game_mapping_service or GameMappingService(
            self.course_config,
            parent=self,
        )
        preferences = self.game_mapping_service.control_preferences
        self.observation_service.apply_control_profile(
            preferences["sensitivity"], preferences["control_style"]
        )
        self.game_mapping_service.control_preferences_changed.connect(
            self.observation_service.apply_control_profile
        )
        profile_config = self.course_config.get("student_control_profiles", {})
        phase_duration_ms = (
            profile_config.get("test_phase_duration_ms", 3000)
            if isinstance(profile_config, dict)
            else 3000
        )
        self.control_effect_test_service = StudentControlEffectTestService(
            self.observation_service,
            phase_duration_ms=int(phase_duration_ms),
            parent=self,
        )
        self.competition_service = competition_service or StudentCompetitionService(
            self.paths.project_root,
            self.course_config,
            self.observation_service,
            self.game_mapping_service,
            parent=self,
        )
        self.game_experience_service = game_experience_service or StudentGameExperienceService(
            self.device_check_service,
            self.observation_service,
            self.paths.project_root / "apps" / "web-game",
            mapping_service=self.game_mapping_service,
            competition_service=self.competition_service,
            parent=self,
        )
        self.game_experience_service.set_competition_service(self.competition_service)
        self.personal_training_service = (
            personal_training_service
            or StudentPersonalTrainingService(
                self.paths.project_root,
                self.paths.dataset_root,
                self.course_config,
                self.observation_service,
                parent=self,
            )
        )
        self.device_check_service.emg_packets_received.connect(self._on_device_emg_packets)

        self.setWindowTitle(f"{self.course['name']} - 学生模式")
        self.resize(1000, 720)
        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

        self.home_page = self._create_home_page()
        self._home_page_index = self._stack.addWidget(self.home_page)
        for entry in self.course_entries:
            if entry.identifier == "connect-bracelet":
                self.device_check_page = DeviceCheckPage(self.start_device_check, self.show_home)
                self._entry_page_indexes[entry.identifier] = self._stack.addWidget(self.device_check_page)
            elif entry.identifier == "quick-experience":
                self.quick_experience_page = StudentQuickExperiencePage(
                    self.game_experience_service,
                    self.show_home,
                )
                self._entry_page_indexes[entry.identifier] = self._stack.addWidget(
                    self.quick_experience_page
                )
            elif entry.identifier == "collect-gestures":
                self.collection_page = StudentCollectionPage(
                    self.start_collection,
                    self.toggle_collection_pause,
                    self.collection_controller.repeat_current_or_last,
                    self.end_collection,
                    self.show_home,
                )
                configured_group = str(
                    self.course_config.get("student_mode", {})
                    .get("student", {})
                    .get("student_id", "")
                ).strip()
                if configured_group:
                    self.collection_page.anonymous_id_edit.setText(configured_group)
                configured_trials = str(
                    self.course_config.get("collection", {}).get("trials_per_action", 10)
                )
                if self.collection_page.trials_combo.findText(configured_trials) < 0:
                    self.collection_page.trials_combo.addItem(configured_trials)
                self.collection_page.trials_combo.setCurrentText(configured_trials)
                self._entry_page_indexes[entry.identifier] = self._stack.addWidget(self.collection_page)
            elif entry.identifier == "view-signals":
                self.signal_observation_page = StudentSignalObservationPage(
                    self.observation_service,
                    self.show_home,
                )
                self._entry_page_indexes[entry.identifier] = self._stack.addWidget(
                    self.signal_observation_page
                )
            elif entry.identifier == "configure-game":
                self.game_mapping_page = StudentGameMappingPage(
                    self.game_mapping_service,
                    self._current_anonymous_group_id,
                    self.show_home,
                )
                self._entry_page_indexes[entry.identifier] = self._stack.addWidget(
                    self.game_mapping_page
                )
            elif entry.identifier == "train-model":
                self.personal_training_page = StudentPersonalTrainingPage(
                    self.personal_training_service,
                    self.show_home,
                    mapping_service=self.game_mapping_service,
                    control_test_service=self.control_effect_test_service,
                )
                self._entry_page_indexes[entry.identifier] = self._stack.addWidget(
                    self.personal_training_page
                )
            elif entry.identifier == "challenge":
                self.challenge_page = StudentChallengePage(
                    self.game_experience_service,
                    self.competition_service,
                    self.game_mapping_service,
                    self.observation_service,
                    self._current_anonymous_group_id,
                    self.show_home,
                )
                self._entry_page_indexes[entry.identifier] = self._stack.addWidget(
                    self.challenge_page
                )
            else:
                self._entry_page_indexes[entry.identifier] = self._stack.addWidget(
                    create_course_page(entry, self.show_home)
                )
        self.collection_gate_page = create_collection_gate_page(self.show_home, self.open_device_check)
        self._collection_gate_index = self._stack.addWidget(self.collection_gate_page)
        self.signal_observation_gate_page = create_signal_observation_gate_page(
            self.show_home, self.open_device_check
        )
        self._signal_observation_gate_index = self._stack.addWidget(
            self.signal_observation_gate_page
        )
        self._on_device_check_result(self.session_device_result)

        self.show_home()

    def _create_home_page(self) -> QWidget:
        page = QWidget()
        page.setObjectName("student-home-page")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(48, 36, 48, 36)
        layout.setSpacing(16)

        title = QLabel(self.course["name"])
        title.setObjectName("student-course-title")
        title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        title.setStyleSheet("font-size: 30px; font-weight: 700;")
        layout.addWidget(title)

        subtitle = QLabel(f"学生模式 · 课程语言：{self.course['language']}")
        subtitle.setObjectName("student-course-language")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        subtitle.setStyleSheet("font-size: 16px;")
        layout.addWidget(subtitle)

        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(18)
        for index, entry in enumerate(self.course_entries):
            button = QPushButton(self._entry_button_text(entry))
            button.setObjectName(f"course-entry-{index + 1}")
            button.setMinimumHeight(86)
            button.setStyleSheet("font-size: 19px; font-weight: 600;")
            button.setEnabled(entry.available)
            if entry.available:
                button.clicked.connect(lambda _checked=False, item=entry: self.open_course_page(item))
            self.course_entry_buttons.append(button)
            grid.addWidget(button, index // 2, index % 2)
        layout.addLayout(grid)
        layout.addStretch(1)
        return page

    @staticmethod
    def _entry_button_text(entry: CourseEntry) -> str:
        return entry.title if entry.available else f"{entry.title}\n{entry.status}"

    def open_course_page(self, entry: CourseEntry) -> None:
        """Open an enabled course page; disabled future lessons stay unavailable."""

        if not entry.available:
            return
        if entry.identifier == "collect-gestures" and not self.session_device_result.collection_ready:
            self._stack.setCurrentIndex(self._collection_gate_index)
            return
        if entry.identifier == "view-signals":
            if not self.session_device_result.collection_ready:
                self._stack.setCurrentIndex(self._signal_observation_gate_index)
                return
            self.signal_observation_page.start(
                left_ready=self.session_device_result.left.ready_for_collection,
                right_ready=self.session_device_result.right.ready_for_collection,
            )
        if entry.identifier == "configure-game":
            self.game_mapping_page.activate()
        if entry.identifier == "train-model":
            self.personal_training_page.activate()
        if entry.identifier == "challenge":
            self.challenge_page.activate()
        self._stack.setCurrentIndex(self._entry_page_indexes[entry.identifier])

    def _current_anonymous_group_id(self) -> str:
        entered = self.collection_page.anonymous_id_edit.text().strip()
        return entered or self.game_mapping_service.current_group_id

    def open_device_check(self) -> None:
        self._stack.setCurrentIndex(self._entry_page_indexes["connect-bracelet"])

    def start_device_check(self) -> None:
        self.device_check_service.start()

    def _on_device_check_result(self, result: DeviceCheckResult) -> None:
        self.session_device_result = result
        self.device_check_page.set_result(result)
        self.collection_page.set_available_sides(
            result.left.ready_for_collection,
            result.right.ready_for_collection,
        )
        self.signal_observation_page.set_ready_sides(
            result.left.ready_for_collection,
            result.right.ready_for_collection,
        )
        self.collection_controller.check_device_state()

    def _on_device_emg_packets(self, side: str, packets: list) -> None:
        self.observation_service.on_emg_packets(side, packets)
        if side == self._collection_side:
            self.collection_controller.on_emg_packets(packets)

    def _selected_side_ready(self) -> bool:
        side = self._collection_side
        if side == "left":
            return self.session_device_result.left.ready_for_collection
        if side == "right":
            return self.session_device_result.right.ready_for_collection
        return False

    def start_collection(self) -> None:
        anonymous_id = self.collection_page.anonymous_id_edit.text().strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]+", anonymous_id) or ".." in anonymous_id:
            self.collection_page.message_label.setText("编号不能为空，且只能使用字母、数字、下划线和短横线")
            return
        side = self.collection_page.selected_side()
        if side is None or not self._selected_result_ready(side):
            self.collection_page.message_label.setText("请先完成手环连接与信号检查")
            return
        collection = self.course_config.get("collection", {})
        actions = collection.get("actions", []) if isinstance(collection, dict) else []
        gestures = tuple(action["id"] for action in actions if isinstance(action, dict) and "id" in action)
        names = {
            action["id"]: action.get("display_name_zh", action["id"])
            for action in actions
            if isinstance(action, dict) and "id" in action
        }
        if not gestures or set(gestures) != {"fist", "finger_spread", "thumb_index_pinch"}:
            self.collection_page.message_label.setText("课程动作配置无效，请联系老师")
            return
        self._collection_side = side
        plan = CollectionPlan(
            subject_id=anonymous_id,
            side=side,
            course_id=str(self.course["id"]),
            trials_per_gesture=int(self.collection_page.trials_combo.currentText()),
            gestures=gestures,
            gesture_names=names,
            rest_before_s=float(collection.get("rest_before_s", 0.5)),
            hold_s=float(collection.get("hold_s", 1.5)),
            rest_after_s=float(collection.get("rest_after_s", 1.0)),
            randomize=bool(collection.get("randomize_action_order", True)),
            min_sample_ratio=float(self.course_config.get("student_device_check", {}).get("trial_min_sample_ratio", 0.8)),
        )
        if not self.collection_controller.start(plan, self.paths.dataset_root, self.course_config):
            self.collection_page.message_label.setText("无法创建采集会话，请稍后重试")

    def _selected_result_ready(self, side: str) -> bool:
        return self.session_device_result.left.ready_for_collection if side == "left" else self.session_device_result.right.ready_for_collection

    def toggle_collection_pause(self) -> None:
        if self.collection_controller.paused:
            self.collection_controller.resume()
        else:
            self.collection_controller.pause()

    def end_collection(self) -> None:
        if not self.collection_controller.active:
            return
        answer = QMessageBox.question(self, "提前结束", "确认提前结束并保存已完成数据吗？")
        if answer == QMessageBox.StandardButton.Yes:
            self.collection_controller.end("partial")

    def _on_collection_snapshot(self, snapshot: object) -> None:
        self.collection_page.set_snapshot(snapshot)

    def _on_collection_finished(self, result: dict) -> None:
        counts = {
            gesture: sum(
                trial.gesture == gesture and trial.status == "completed"
                for trial in self.collection_controller.trials
            )
            for gesture in ("fist", "finger_spread", "thumb_index_pinch")
        }
        session_dir = result.get("session_dir")
        relative = ""
        if isinstance(session_dir, Path):
            try:
                relative = str(session_dir.relative_to(self.paths.project_root))
            except ValueError:
                relative = session_dir.name
        message = "采集完成并已保存" if result.get("status") == "completed" else "部分完成，已安全保存"
        self.collection_page.show_completion(
            f"{message}\n编号：{self.collection_controller.plan.subject_id}\n"
            f"{('左手' if self._collection_side == 'left' else '右手')}\n"
            f"握拳 {counts['fist']} 次，张开 {counts['finger_spread']} 次，轻捏 {counts['thumb_index_pinch']} 次\n"
            f"重做/无效：{sum(t.status == 'repeated' for t in self.collection_controller.trials)}/"
            f"{sum(t.status == 'invalid' for t in self.collection_controller.trials)}\n"
            f"会话：{result.get('session_id')}\n保存位置：{relative}"
        )

    def show_home(self) -> None:
        """Return from a course page to the course home."""

        if self.collection_controller.active:
            answer = QMessageBox.question(self, "采集进行中", "返回首页将提前结束并保存已完成数据，是否继续？")
            if answer != QMessageBox.StandardButton.Yes:
                return
            self.collection_controller.end("partial")
        self.signal_observation_page.stop()
        self.control_effect_test_service.stop()
        self.game_experience_service.stop()
        self._stack.setCurrentIndex(self._home_page_index)

    def prepare_next_group(self) -> None:
        """Clear only volatile student state while retaining all saved artifacts."""

        if self.collection_controller.active:
            self.collection_controller.end("interrupted")
        self.control_effect_test_service.stop()
        self.personal_training_service.close()
        self.game_experience_service.stop()
        self.signal_observation_page.stop()
        self.observation_service.use_standard_model()
        self.game_mapping_service.reset_temporary_state()
        self.observation_service.apply_control_profile("standard", "balanced")
        self.collection_page.anonymous_id_edit.clear()
        self._collection_side = None
        self._stack.setCurrentIndex(self._home_page_index)

    def closeEvent(self, event) -> None:
        if self.collection_controller.active:
            self.collection_controller.end("interrupted")
        self.signal_observation_page.stop()
        self.control_effect_test_service.stop()
        self.game_experience_service.stop()
        self.personal_training_service.close()
        self.device_check_service.close()
        super().closeEvent(event)
