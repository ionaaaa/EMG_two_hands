"""Static course pages used by the student-mode window.

These widgets deliberately provide course navigation only. They do not create
devices, start collection, load models, or start game services.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget


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
