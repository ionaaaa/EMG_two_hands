"""Course pages used by the student-mode window."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from emg_live_marker.device.check_service import ConnectionState, DeviceCheckResult
from emg_live_marker.realtime.game_mapping import GameMappingService
from emg_live_marker.realtime.student_observation import (
    STUDENT_DISPLAY_MODES,
    STUDENT_GESTURES,
    StudentObservationService,
)
from emg_live_marker.realtime.student_game_experience import StudentGameExperienceService
from emg_live_marker.ui.waveform_view import MultiChannelWaveformView


@dataclass(frozen=True)
class CourseEntry:
    """One item in the student course navigation."""

    identifier: str
    title: str
    description: str
    available: bool = True

    @property
    def status(self) -> str:
        return "功能接入中" if self.available else "后续课程开放"


COURSE_ENTRIES: tuple[CourseEntry, ...] = (
    CourseEntry("emg-and-gestures", "认识肌电与手势", "了解肌电信号和三种课程手势。"),
    CourseEntry("connect-bracelet", "连接并检查手环", "检查手环连接和信号状态。"),
    CourseEntry("quick-experience", "快速体验", "使用课程示例体验手势识别。"),
    CourseEntry("collect-gestures", "采集我的手势", "按课程节奏采集自己的手势数据。"),
    CourseEntry("view-signals", "查看信号与识别结果", "查看肌电信号和识别结果。"),
    CourseEntry(
        "configure-game",
        "设置游戏指令",
        "交换握拳和伸掌对应的红蓝游戏指令。",
    ),
    CourseEntry(
        "train-model",
        "训练和优化我的模型",
        "模型训练与优化将在后续课程中开放。",
        available=False,
    ),
    CourseEntry("challenge", "进入挑战赛", "进入课程挑战赛页面。"),
)


def create_course_page(entry: CourseEntry, go_home: Callable[[], None]) -> QWidget:
    """Build one clearly marked placeholder or future-course page."""

    page = QWidget()
    page.setObjectName(f"student-page-{entry.identifier}")
    layout = QVBoxLayout(page)
    layout.setContentsMargins(48, 48, 48, 48)
    layout.setSpacing(18)

    title = QLabel(entry.title)
    title.setObjectName("course-page-title")
    title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
    title.setStyleSheet("font-size: 28px; font-weight: 700;")
    layout.addWidget(title)

    description = QLabel(entry.description)
    description.setObjectName("course-page-description")
    description.setAlignment(Qt.AlignmentFlag.AlignHCenter)
    description.setWordWrap(True)
    description.setStyleSheet("font-size: 16px;")
    layout.addWidget(description)

    status = QLabel(entry.status)
    status.setObjectName("course-page-status")
    status.setAlignment(Qt.AlignmentFlag.AlignHCenter)
    status.setStyleSheet("font-size: 20px; font-weight: 600; color: #9a6700;")
    layout.addWidget(status)
    layout.addStretch(1)

    back_button = QPushButton("返回首页")
    back_button.setObjectName(f"return-home-{entry.identifier}")
    back_button.setMinimumHeight(44)
    back_button.clicked.connect(go_home)
    layout.addWidget(back_button, alignment=Qt.AlignmentFlag.AlignHCenter)
    return page


class DeviceCheckPage(QWidget):
    """Student-facing device check summary with no technical connection details."""

    def __init__(self, start_check: Callable[[], None], go_home: Callable[[], None]) -> None:
        super().__init__()
        self.setObjectName("student-device-check-page")
        self._start_check = start_check
        layout = QVBoxLayout(self)
        layout.setContentsMargins(48, 42, 48, 42)
        layout.setSpacing(18)

        title = QLabel("连接并检查手环")
        title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        title.setStyleSheet("font-size: 28px; font-weight: 700;")
        layout.addWidget(title)

        description = QLabel("系统会检查连接、肌电数据、基础信号质量和数据接收稳定性。")
        description.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        description.setWordWrap(True)
        layout.addWidget(description)

        grid = QGridLayout()
        grid.setHorizontalSpacing(28)
        grid.setVerticalSpacing(14)
        self._left_status = self._add_status_row(grid, 0, "左手环")
        self._right_status = self._add_status_row(grid, 1, "右手环")
        self._signal_status = self._add_status_row(grid, 2, "信号")
        self._stream_status = self._add_status_row(grid, 3, "数据接收")
        layout.addLayout(grid)

        self._demo_label = QLabel("演示模式：使用模拟数据，不代表真实手环检查结果")
        self._demo_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self._demo_label.setStyleSheet("font-weight: 600; color: #9a6700;")
        self._demo_label.hide()
        layout.addWidget(self._demo_label)

        self._message = QLabel("点击下方按钮开始检查。")
        self._message.setObjectName("device-check-message")
        self._message.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self._message.setWordWrap(True)
        self._message.setStyleSheet("font-size: 16px;")
        layout.addWidget(self._message)
        layout.addStretch(1)

        self.check_button = QPushButton("连接并检查手环")
        self.check_button.setObjectName("device-check-button")
        self.check_button.setMinimumHeight(52)
        self.check_button.setStyleSheet("font-size: 19px; font-weight: 600;")
        self.check_button.clicked.connect(self._start_check)
        layout.addWidget(self.check_button)

        back_button = QPushButton("返回首页")
        back_button.setObjectName("return-home-device-check")
        back_button.setMinimumHeight(44)
        back_button.clicked.connect(go_home)
        layout.addWidget(back_button, alignment=Qt.AlignmentFlag.AlignHCenter)

    @staticmethod
    def _add_status_row(grid: QGridLayout, row: int, name: str) -> QLabel:
        label = QLabel(f"{name}：")
        label.setStyleSheet("font-size: 18px; font-weight: 600;")
        value = QLabel("未连接" if name.endswith("手环") else "检测中")
        value.setStyleSheet("font-size: 18px;")
        grid.addWidget(label, row, 0)
        grid.addWidget(value, row, 1)
        return value

    def set_result(self, result: DeviceCheckResult) -> None:
        self._left_status.setText(self._connection_text(result.left.connection))
        self._right_status.setText(self._connection_text(result.right.connection))
        self._signal_status.setText(result.signal_status)
        self._stream_status.setText(result.stream_status)
        self._message.setText(result.message)
        self._demo_label.setVisible(result.simulate)
        self.check_button.setEnabled(not result.checking)
        if result.message == "尚未开始检查":
            self.check_button.setText("连接并检查手环")
        elif result.checking:
            self.check_button.setText("检测中")
        else:
            self.check_button.setText("重新检测")

    @staticmethod
    def _connection_text(connection: ConnectionState) -> str:
        if connection is ConnectionState.CHECKING:
            return "检测中"
        if connection is ConnectionState.CONNECTED:
            return "已连接"
        if connection is ConnectionState.UNASSIGNED:
            return "无法确认"
        return "未连接"


def create_collection_gate_page(go_home: Callable[[], None], open_device_check: Callable[[], None]) -> QWidget:
    """Explain why collection cannot be entered before a successful check."""

    page = QWidget()
    page.setObjectName("student-collection-gate-page")
    layout = QVBoxLayout(page)
    layout.setContentsMargins(48, 48, 48, 48)
    layout.setSpacing(18)
    title = QLabel("采集我的手势")
    title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
    title.setStyleSheet("font-size: 28px; font-weight: 700;")
    layout.addWidget(title)
    message = QLabel("请先完成手环连接与信号检查")
    message.setObjectName("collection-gate-message")
    message.setAlignment(Qt.AlignmentFlag.AlignHCenter)
    message.setStyleSheet("font-size: 20px; font-weight: 600; color: #9a6700;")
    layout.addWidget(message)
    layout.addStretch(1)
    check_button = QPushButton("前往检查手环")
    check_button.setObjectName("open-device-check-from-gate")
    check_button.clicked.connect(open_device_check)
    layout.addWidget(check_button, alignment=Qt.AlignmentFlag.AlignHCenter)
    back_button = QPushButton("返回首页")
    back_button.setObjectName("return-home-collection-gate")
    back_button.clicked.connect(go_home)
    layout.addWidget(back_button, alignment=Qt.AlignmentFlag.AlignHCenter)
    return page


def create_signal_observation_gate_page(
    go_home: Callable[[], None], open_device_check: Callable[[], None]
) -> QWidget:
    """Require at least one checked hand before entering signal observation."""

    page = QWidget()
    page.setObjectName("student-signal-observation-gate-page")
    layout = QVBoxLayout(page)
    layout.setContentsMargins(48, 48, 48, 48)
    layout.setSpacing(18)
    title = QLabel("查看信号与识别结果")
    title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
    title.setStyleSheet("font-size: 28px; font-weight: 700;")
    layout.addWidget(title)
    message = QLabel("请先完成手环连接与信号检查，至少一只手环检查通过后即可观察。")
    message.setObjectName("signal-observation-gate-message")
    message.setAlignment(Qt.AlignmentFlag.AlignHCenter)
    message.setWordWrap(True)
    message.setStyleSheet("font-size: 20px; font-weight: 600; color: #9a6700;")
    layout.addWidget(message)
    layout.addStretch(1)
    check_button = QPushButton("前往检查手环")
    check_button.setObjectName("open-device-check-from-observation-gate")
    check_button.clicked.connect(open_device_check)
    layout.addWidget(check_button, alignment=Qt.AlignmentFlag.AlignHCenter)
    back_button = QPushButton("返回首页")
    back_button.setObjectName("return-home-observation-gate")
    back_button.clicked.connect(go_home)
    layout.addWidget(back_button, alignment=Qt.AlignmentFlag.AlignHCenter)
    return page


GESTURE_NAMES_ZH = {
    "rest": "放松",
    "fist": "握拳",
    "open-palm": "伸掌",
    "pinch": "捏合",
}


class StudentQuickExperiencePage(QWidget):
    """One-button entry to the locked standard-model web game experience."""

    def __init__(
        self,
        service: StudentGameExperienceService,
        go_home: Callable[[], None],
    ) -> None:
        super().__init__()
        self.setObjectName("student-quick-experience-page")
        self.service = service
        layout = QVBoxLayout(self)
        layout.setContentsMargins(64, 48, 64, 48)
        layout.setSpacing(20)

        title = QLabel("快速体验")
        title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        title.setStyleSheet("font-size: 28px; font-weight: 700;")
        layout.addWidget(title)
        description = QLabel(
            "系统会自动检查手环、启动课程标准模型并打开本地游戏。\n"
            "至少一只手环检查通过即可开始。"
        )
        description.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        description.setWordWrap(True)
        description.setStyleSheet("font-size: 17px;")
        layout.addWidget(description)

        self.status_label = QLabel("点击按钮开始体验。")
        self.status_label.setObjectName("student-quick-experience-status")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("font-size: 18px; font-weight: 600;")
        layout.addWidget(self.status_label)
        layout.addStretch(1)

        self.start_button = QPushButton("使用标准模型开始体验")
        self.start_button.setObjectName("start-standard-model-experience")
        self.start_button.setMinimumHeight(58)
        self.start_button.setStyleSheet("font-size: 20px; font-weight: 700;")
        self.start_button.clicked.connect(self.service.start_experience)
        layout.addWidget(self.start_button)

        self.home_button = QPushButton("返回首页")
        self.home_button.setObjectName("return-home-quick-experience")
        self.home_button.clicked.connect(go_home)
        layout.addWidget(self.home_button, alignment=Qt.AlignmentFlag.AlignHCenter)
        self.service.status_changed.connect(self.set_status)
        self.set_status(self.service.state, self.service.message)

    def set_status(self, state: str, message: str) -> None:
        self.status_label.setText(message)
        starting = state in {"checking", "starting", "waiting-client"}
        self.start_button.setEnabled(not starting)
        self.start_button.setText(
            "重新打开游戏" if state == "running" else "使用标准模型开始体验"
        )
        color = "#137333" if state == "running" else "#b42318" if state == "error" else "#22577a"
        self.status_label.setStyleSheet(f"font-size: 18px; font-weight: 600; color: {color};")


class StudentGameMappingPage(QWidget):
    """Visual controls for the two editable student game commands."""

    def __init__(
        self,
        service: GameMappingService,
        group_id_provider: Callable[[], str],
        go_home: Callable[[], None],
    ) -> None:
        super().__init__()
        self.setObjectName("student-game-mapping-page")
        self.service = service
        self._group_id_provider = group_id_provider
        layout = QVBoxLayout(self)
        layout.setContentsMargins(48, 32, 48, 32)
        layout.setSpacing(16)

        title = QLabel("设置游戏指令")
        title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        title.setStyleSheet("font-size: 28px; font-weight: 700;")
        layout.addWidget(title)
        hint = QLabel("只能交换握拳和伸掌对应的红蓝指令；放松和捏合规则保持固定。")
        hint.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        hint.setWordWrap(True)
        layout.addWidget(hint)

        mapping_grid = QGridLayout()
        mapping_grid.setHorizontalSpacing(24)
        mapping_grid.setVerticalSpacing(12)
        mapping_grid.addWidget(QLabel("动作"), 0, 0)
        mapping_grid.addWidget(QLabel("当前游戏指令"), 0, 1)
        self.fist_mapping_label = self._add_mapping_row(mapping_grid, 1, "握拳")
        self.open_palm_mapping_label = self._add_mapping_row(mapping_grid, 2, "伸掌")
        self.rest_mapping_label = self._add_mapping_row(mapping_grid, 3, "放松（固定）")
        self.pinch_mapping_label = self._add_mapping_row(mapping_grid, 4, "捏合（本轮不参与）")
        layout.addLayout(mapping_grid)

        self.group_label = QLabel("匿名小组：未填写")
        self.group_label.setObjectName("game-mapping-group")
        self.group_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self.group_label)
        self.message_label = QLabel("")
        self.message_label.setObjectName("game-mapping-message")
        self.message_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.message_label.setWordWrap(True)
        layout.addWidget(self.message_label)
        layout.addStretch(1)

        buttons = QGridLayout()
        self.swap_button = QPushButton("交换指令")
        self.test_button = QPushButton("测试映射")
        self.restore_button = QPushButton("恢复默认")
        self.save_button = QPushButton("保存本组设置")
        self.home_button = QPushButton("返回首页")
        self.swap_button.setObjectName("swap-game-mapping")
        self.test_button.setObjectName("test-game-mapping")
        self.restore_button.setObjectName("restore-default-game-mapping")
        self.save_button.setObjectName("save-group-game-mapping")
        for button in (self.swap_button, self.test_button, self.restore_button, self.save_button):
            button.setMinimumHeight(44)
        self.swap_button.clicked.connect(self.service.swap_commands)
        self.test_button.clicked.connect(self._test_mapping)
        self.restore_button.clicked.connect(self.service.restore_default)
        self.save_button.clicked.connect(self._save_mapping)
        self.home_button.clicked.connect(go_home)
        buttons.addWidget(self.swap_button, 0, 0)
        buttons.addWidget(self.test_button, 0, 1)
        buttons.addWidget(self.restore_button, 1, 0)
        buttons.addWidget(self.save_button, 1, 1)
        buttons.addWidget(self.home_button, 2, 0, 1, 2)
        layout.addLayout(buttons)

        self.service.mapping_changed.connect(self._on_mapping_changed)
        self._render_mapping()

    @staticmethod
    def _add_mapping_row(grid: QGridLayout, row: int, gesture_name: str) -> QLabel:
        gesture = QLabel(gesture_name)
        gesture.setStyleSheet("font-size: 18px; font-weight: 600;")
        command = QLabel()
        command.setStyleSheet("font-size: 18px; font-weight: 700;")
        grid.addWidget(gesture, row, 0)
        grid.addWidget(command, row, 1)
        return command

    def activate(self) -> None:
        active_group = self._group_id_provider().strip() or self.service.current_group_id
        if active_group:
            _loaded, message = self.service.load_group(active_group)
            self.message_label.setText(message)
        self.group_label.setText(f"匿名小组：{active_group}" if active_group else "匿名小组：未填写")
        self._render_mapping()

    def _on_mapping_changed(self, _runtime_config: dict) -> None:
        self._render_mapping()
        self.message_label.setText("映射已更新；如需下次恢复，请保存本组设置。")

    def _render_mapping(self) -> None:
        mapping = self.service.resolved_mapping
        commands = self.service.commands
        labels = {
            gesture: commands[command]["display_name_zh"]
            for gesture, command in mapping.items()
        }
        self.fist_mapping_label.setText(labels["fist"])
        self.open_palm_mapping_label.setText(labels["open-palm"])
        self.rest_mapping_label.setText(labels["rest"])
        self.pinch_mapping_label.setText(labels["pinch"])
        for label, command in (
            (self.fist_mapping_label, mapping["fist"]),
            (self.open_palm_mapping_label, mapping["open-palm"]),
        ):
            color = "#c62828" if command == "A" else "#0077a8"
            label.setStyleSheet(f"font-size: 18px; font-weight: 700; color: {color};")

    def _test_mapping(self) -> None:
        self.message_label.setText(self.service.test_mapping())

    def _save_mapping(self) -> None:
        group_id = self._group_id_provider().strip() or self.service.current_group_id
        saved, message = self.service.save_current_group(group_id)
        self.message_label.setText(message)
        if saved:
            self.group_label.setText(f"匿名小组：{self.service.current_group_id}")


class StudentSignalObservationPage(QWidget):
    """Student-facing dual-hand waveform and real-model observation page."""

    DISPLAY_MODE_LABELS = {
        "raw": "原始肌电",
        "filtered": "滤波肌电",
        "rms": "肌肉活动强度（RMS）",
    }

    def __init__(
        self,
        service: StudentObservationService,
        go_home: Callable[[], None],
    ) -> None:
        super().__init__()
        self.setObjectName("student-signal-observation-page")
        self.service = service
        self._ready_sides = {"left": False, "right": False}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 14, 20, 14)
        layout.setSpacing(10)
        title = QLabel("查看信号与识别结果")
        title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        title.setStyleSheet("font-size: 25px; font-weight: 700;")
        layout.addWidget(title)

        prompt = QLabel("观察提示：请依次尝试 放松 → 握拳 → 伸掌，留意波形与识别结果的变化。")
        prompt.setObjectName("signal-observation-prompt")
        prompt.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        prompt.setStyleSheet("font-size: 16px; font-weight: 600; color: #22577a;")
        layout.addWidget(prompt)

        selector_row = QHBoxLayout()
        selector_row.addWidget(QLabel("观察内容："))
        self.display_mode_combo = QComboBox()
        self.display_mode_combo.setObjectName("student-signal-display-mode")
        for mode in STUDENT_DISPLAY_MODES:
            self.display_mode_combo.addItem(self.DISPLAY_MODE_LABELS[mode], mode)
        selector_row.addWidget(self.display_mode_combo)
        selector_row.addStretch(1)
        self.model_message_label = QLabel("")
        self.model_message_label.setObjectName("student-observation-model-message")
        self.model_message_label.setStyleSheet("color: #b42318; font-weight: 600;")
        selector_row.addWidget(self.model_message_label)
        layout.addLayout(selector_row)

        hands = QHBoxLayout()
        hands.setSpacing(12)
        self.left_waveform_view, left_panel = self._build_hand_panel("left", "左手")
        self.right_waveform_view, right_panel = self._build_hand_panel("right", "右手")
        hands.addWidget(left_panel, 1)
        hands.addWidget(right_panel, 1)
        layout.addLayout(hands, 1)

        self.home_button = QPushButton("返回首页")
        self.home_button.setObjectName("return-home-view-signals")
        self.home_button.clicked.connect(go_home)
        layout.addWidget(self.home_button, alignment=Qt.AlignmentFlag.AlignHCenter)

        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self.refresh_waveforms)
        self.service.gesture_updated.connect(self._on_gesture_updated)
        self.service.model_status_changed.connect(self.model_message_label.setText)

    def _build_hand_panel(self, side: str, name: str) -> tuple[MultiChannelWaveformView, QWidget]:
        panel = QWidget()
        panel.setObjectName(f"student-observation-{side}-panel")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(4)
        header = QHBoxLayout()
        hand_label = QLabel(name)
        hand_label.setStyleSheet("font-size: 18px; font-weight: 700;")
        status = QLabel("未连接")
        status.setObjectName(f"student-observation-{side}-status")
        status.setStyleSheet("font-weight: 600; color: #b42318;")
        header.addWidget(hand_label)
        header.addStretch(1)
        header.addWidget(status)
        panel_layout.addLayout(header)

        result = QLabel("当前手势：--    置信度：--")
        result.setObjectName(f"student-observation-{side}-result")
        result.setStyleSheet("font-size: 15px; font-weight: 600;")
        panel_layout.addWidget(result)
        probabilities = QLabel(self._probability_text({}))
        probabilities.setObjectName(f"student-observation-{side}-probabilities")
        probabilities.setWordWrap(True)
        panel_layout.addWidget(probabilities)
        waveform = MultiChannelWaveformView(display_seconds=6.0)
        waveform.setObjectName(f"student-observation-{side}-waveform")
        panel_layout.addWidget(waveform, 1)

        setattr(self, f"{side}_status_label", status)
        setattr(self, f"{side}_result_label", result)
        setattr(self, f"{side}_probability_label", probabilities)
        return waveform, panel

    def start(self, *, left_ready: bool, right_ready: bool) -> None:
        self.set_ready_sides(left_ready, right_ready)
        self.service.start(left_ready=left_ready, right_ready=right_ready)
        self.model_message_label.setText(self.service.model_error)
        self._timer.start()
        self.refresh_waveforms()

    def stop(self) -> None:
        self._timer.stop()
        self.service.stop()

    def set_ready_sides(self, left_ready: bool, right_ready: bool) -> None:
        self._ready_sides = {"left": bool(left_ready), "right": bool(right_ready)}
        self.service.update_ready_sides(left_ready=left_ready, right_ready=right_ready)
        for side, ready in self._ready_sides.items():
            status = getattr(self, f"{side}_status_label")
            status.setText("已连接" if ready else "未连接")
            status.setStyleSheet(
                "font-weight: 600; color: #137333;" if ready else "font-weight: 600; color: #b42318;"
            )
            if not ready:
                getattr(self, f"{side}_result_label").setText("当前手势：--    置信度：--")
                getattr(self, f"{side}_probability_label").setText(self._probability_text({}))
                getattr(self, f"{side}_waveform_view").clear()

    def refresh_waveforms(self) -> None:
        mode = str(self.display_mode_combo.currentData())
        for side, ready in self._ready_sides.items():
            if not ready:
                continue
            t, data, sample_index = self.service.display_window(side, mode, seconds=6.0)
            getattr(self, f"{side}_waveform_view").update_data(t, data, sample_index)

    def _on_gesture_updated(
        self, side: str, gesture: str, confidence: float, probabilities: dict[str, float]
    ) -> None:
        if not self._ready_sides.get(side, False):
            return
        name = GESTURE_NAMES_ZH.get(gesture, "未知")
        getattr(self, f"{side}_result_label").setText(
            f"当前手势：{name}    置信度：{float(confidence):.1%}"
        )
        getattr(self, f"{side}_probability_label").setText(self._probability_text(probabilities))

    @staticmethod
    def _probability_text(probabilities: dict[str, float]) -> str:
        aliases = {
            "rest": ("rest",),
            "fist": ("fist",),
            "open-palm": ("open-palm", "open_palm", "finger_spread"),
            "pinch": ("pinch", "thumb_index_pinch"),
        }
        parts = []
        for gesture in STUDENT_GESTURES:
            value = next((probabilities[key] for key in aliases[gesture] if key in probabilities), None)
            shown = "--" if value is None else f"{float(value):.1%}"
            parts.append(f"{GESTURE_NAMES_ZH[gesture]} {shown}")
        return "  |  ".join(parts)


class StudentCollectionPage(QWidget):
    """Chinese guided collection controls with no system or model settings."""

    def __init__(
        self,
        start: Callable[[], None],
        pause: Callable[[], None],
        repeat: Callable[[], None],
        end: Callable[[], None],
        go_home: Callable[[], None],
    ) -> None:
        super().__init__()
        self.setObjectName("student-collection-page")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(44, 30, 44, 30)
        layout.setSpacing(12)
        title = QLabel("采集我的手势")
        title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        title.setStyleSheet("font-size: 28px; font-weight: 700;")
        layout.addWidget(title)
        hint = QLabel("请输入匿名编号，不要填写真实姓名。")
        hint.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(hint)

        self.anonymous_id_edit = QLineEdit()
        self.anonymous_id_edit.setObjectName("anonymous-id")
        self.anonymous_id_edit.setPlaceholderText("匿名学生/小组编号，例如 group_01")
        layout.addWidget(self.anonymous_id_edit)

        side_row = QHBoxLayout()
        side_row.addWidget(QLabel("采集手："))
        self.left_radio = QRadioButton("左手")
        self.right_radio = QRadioButton("右手")
        self.side_group = QButtonGroup(self)
        self.side_group.addButton(self.left_radio)
        self.side_group.addButton(self.right_radio)
        side_row.addWidget(self.left_radio)
        side_row.addWidget(self.right_radio)
        side_row.addStretch(1)
        layout.addLayout(side_row)

        count_row = QHBoxLayout()
        count_row.addWidget(QLabel("每类动作次数："))
        self.trials_combo = QComboBox()
        self.trials_combo.setObjectName("trials-per-gesture")
        self.trials_combo.addItems(["5", "10", "15"])
        self.trials_combo.setCurrentText("10")
        count_row.addWidget(self.trials_combo)
        count_row.addStretch(1)
        layout.addLayout(count_row)

        self.prompt_label = QLabel("请先完成手环连接与信号检查")
        self.prompt_label.setObjectName("collection-prompt")
        self.prompt_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.prompt_label.setStyleSheet("font-size: 24px; font-weight: 700;")
        layout.addWidget(self.prompt_label)
        self.countdown_label = QLabel("倒计时：--")
        self.countdown_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self.countdown_label)
        self.progress_label = QLabel("完成进度：0 / 0")
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self.progress_label)
        self.message_label = QLabel("")
        self.message_label.setObjectName("collection-message")
        self.message_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.message_label.setWordWrap(True)
        layout.addWidget(self.message_label)
        layout.addStretch(1)

        buttons = QGridLayout()
        self.start_button = QPushButton("开始采集")
        self.pause_button = QPushButton("暂停")
        self.repeat_button = QPushButton("重做当前或最近一次动作")
        self.end_button = QPushButton("提前结束")
        self.home_button = QPushButton("返回首页")
        for button in (self.start_button, self.pause_button, self.repeat_button, self.end_button):
            button.setMinimumHeight(42)
        self.pause_button.setEnabled(False)
        self.repeat_button.setEnabled(False)
        self.end_button.setEnabled(False)
        self.start_button.clicked.connect(start)
        self.pause_button.clicked.connect(pause)
        self.repeat_button.clicked.connect(repeat)
        self.end_button.clicked.connect(end)
        self.home_button.clicked.connect(go_home)
        buttons.addWidget(self.start_button, 0, 0)
        buttons.addWidget(self.pause_button, 0, 1)
        buttons.addWidget(self.repeat_button, 1, 0, 1, 2)
        buttons.addWidget(self.end_button, 2, 0)
        buttons.addWidget(self.home_button, 2, 1)
        layout.addLayout(buttons)

    def selected_side(self) -> str | None:
        if self.left_radio.isChecked():
            return "left"
        if self.right_radio.isChecked():
            return "right"
        return None

    def set_available_sides(self, left: bool, right: bool) -> None:
        self.left_radio.setEnabled(left)
        self.right_radio.setEnabled(right)
        if left and not right:
            self.left_radio.setChecked(True)
        elif right and not left:
            self.right_radio.setChecked(True)
        elif not left and not right:
            self.left_radio.setAutoExclusive(False)
            self.right_radio.setAutoExclusive(False)
            self.left_radio.setChecked(False)
            self.right_radio.setChecked(False)
            self.left_radio.setAutoExclusive(True)
            self.right_radio.setAutoExclusive(True)

    def set_snapshot(self, snapshot: object) -> None:
        prompt = getattr(snapshot, "prompt", "请准备")
        self.prompt_label.setText(prompt)
        self.countdown_label.setText(f"倒计时：{getattr(snapshot, 'remaining_s', 0.0):.1f} 秒")
        self.progress_label.setText(
            f"完成进度：{getattr(snapshot, 'completed', 0)} / {getattr(snapshot, 'total', 0)}"
        )
        self.message_label.setText(getattr(snapshot, "message", ""))
        active = bool(getattr(snapshot, "active", False))
        paused = bool(getattr(snapshot, "paused", False))
        self.start_button.setEnabled(not active)
        self.pause_button.setEnabled(active)
        self.pause_button.setText("继续" if paused else "暂停")
        self.repeat_button.setEnabled(active)
        self.end_button.setEnabled(active)

    def show_completion(self, summary: str) -> None:
        self.prompt_label.setText(summary)
        self.message_label.setText(summary)
