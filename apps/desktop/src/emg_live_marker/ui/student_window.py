"""Student-mode course navigation window.

The student window intentionally stays separate from :mod:`main_window`: it is
only a course shell and never starts hardware, collection, inference, training,
or game services.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGridLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from emg_live_marker.paths import ProjectPaths, resolve_project_paths
from emg_live_marker.ui.student_pages import COURSE_ENTRIES, CourseEntry, create_course_page

TEACHING_CONFIG_RELATIVE_PATH = Path("configs") / "teaching" / "yucai.json"


class TeachingConfigError(ValueError):
    """Raised when the student-mode teaching preset cannot be used."""


def load_yucai_course_config(paths: ProjectPaths | None = None) -> dict[str, Any]:
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
    return payload


class StudentMainWindow(QMainWindow):
    """A Chinese course-navigation shell with no real-time business logic."""

    def __init__(self, paths: ProjectPaths | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.paths = paths or resolve_project_paths()
        self.course_config = load_yucai_course_config(self.paths)
        self.course = self.course_config["course"]
        self.course_entries = COURSE_ENTRIES
        self.course_entry_buttons: list[QPushButton] = []
        self._entry_page_indexes: dict[str, int] = {}

        self.setWindowTitle(f"{self.course['name']} - 学生模式")
        self.resize(1000, 720)
        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

        self.home_page = self._create_home_page()
        self._home_page_index = self._stack.addWidget(self.home_page)
        for entry in self.course_entries:
            self._entry_page_indexes[entry.identifier] = self._stack.addWidget(
                create_course_page(entry, self.show_home)
            )

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
        self._stack.setCurrentIndex(self._entry_page_indexes[entry.identifier])

    def show_home(self) -> None:
        """Return from a course page to the course home."""

        self._stack.setCurrentIndex(self._home_page_index)
