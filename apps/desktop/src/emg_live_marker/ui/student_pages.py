"""Static course pages used by the student-mode window.

These widgets deliberately provide course navigation only. They do not create
devices, start collection, load models, or start game services.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGridLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from emg_live_marker.device.check_service import ConnectionState, DeviceCheckResult


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
        "游戏指令设置将在后续课程中开放。",
        available=False,
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
