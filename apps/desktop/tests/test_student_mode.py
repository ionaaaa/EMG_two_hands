import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QPushButton

import emg_live_marker.__main__ as app_main
from emg_live_marker.paths import resolve_project_paths
from emg_live_marker.ui.student_pages import COURSE_ENTRIES
from emg_live_marker.ui.student_window import (
    StudentMainWindow,
    TeachingConfigError,
    load_yucai_course_config,
)


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_mode_argument_defaults_to_teacher_and_restricts_choices() -> None:
    parser = app_main.build_parser()

    assert parser.parse_args([]).mode == "teacher"
    assert parser.parse_args(["--mode", "student"]).mode == "student"
    assert parser.parse_args(["--mode", "teacher"]).mode == "teacher"
    with pytest.raises(SystemExit):
        parser.parse_args(["--mode", "invalid"])


def test_create_window_routes_each_mode_without_starting_real_components(monkeypatch, tmp_path) -> None:
    created: list[tuple[str, object]] = []

    class FakeTeacherWindow:
        def __init__(self, **kwargs) -> None:
            created.append(("teacher", kwargs))

    class FakeStudentWindow:
        def __init__(self, **kwargs) -> None:
            created.append(("student", kwargs))

    monkeypatch.setattr(app_main, "MainWindow", FakeTeacherWindow)
    monkeypatch.setattr(app_main, "StudentMainWindow", FakeStudentWindow)
    paths = resolve_project_paths(project_root=tmp_path, environ={})
    parser = app_main.build_parser()

    teacher_window = app_main.create_window(parser.parse_args([]), paths)
    student_window = app_main.create_window(parser.parse_args(["--mode", "student"]), paths)

    assert isinstance(teacher_window, FakeTeacherWindow)
    assert isinstance(student_window, FakeStudentWindow)
    assert created[0][1]["simulate"] is True
    assert created[1][1] == {"paths": paths, "simulate": False}


def test_student_window_contains_all_course_entries(app) -> None:
    window = StudentMainWindow(paths=resolve_project_paths())
    try:
        assert [entry.title for entry in window.course_entries] == [
            "认识肌电与手势",
            "连接并检查手环",
            "快速体验",
            "采集我的手势",
            "查看信号与识别结果",
            "设置游戏指令",
            "训练我的模型",
            "进入挑战赛",
        ]
        assert len(window.course_entry_buttons) == 8
        assert window.course_config["course"]["name"] in window.windowTitle()
    finally:
        window.close()


def test_personal_training_course_is_enabled(app) -> None:
    window = StudentMainWindow(paths=resolve_project_paths())
    try:
        button = window.course_entry_buttons[6]
        assert button.isEnabled() is True
        assert button.text() == "训练我的模型"
        assert window.course_entry_buttons[5].isEnabled() is True
    finally:
        window.close()


def test_enabled_entry_navigates_to_page_and_back_home(app) -> None:
    window = StudentMainWindow(paths=resolve_project_paths())
    try:
        entry = COURSE_ENTRIES[0]
        window.course_entry_buttons[0].click()
        app.processEvents()

        page = window._stack.currentWidget()
        assert page.objectName() == f"student-page-{entry.identifier}"
        back_button = page.findChild(QPushButton, f"return-home-{entry.identifier}")
        assert back_button is not None
        back_button.click()
        app.processEvents()
        assert window._stack.currentWidget() is window.home_page
    finally:
        window.close()


def test_student_config_failure_is_clear_and_isolated_from_teacher_paths(tmp_path) -> None:
    paths = resolve_project_paths(project_root=tmp_path, environ={})

    with pytest.raises(TeachingConfigError, match="课程配置不存在"):
        load_yucai_course_config(paths)
