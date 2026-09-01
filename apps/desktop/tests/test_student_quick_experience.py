import http.client
import os
import time
import urllib.request
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

from emg_live_marker.device.check_service import (
    CheckReason,
    ConnectionState,
    DeviceCheckResult,
    DeviceCheckService,
    SideCheckResult,
)
from emg_live_marker.paths import resolve_project_paths
from emg_live_marker.realtime.gesture_server import GestureServer
from emg_live_marker.realtime.student_game_experience import StudentGameExperienceService
from emg_live_marker.ui.student_pages import StudentQuickExperiencePage
from emg_live_marker.ui.student_window import StudentMainWindow


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


def disconnected_result(*, checking: bool = False) -> DeviceCheckResult:
    connection = ConnectionState.CHECKING if checking else ConnectionState.DISCONNECTED
    side = SideCheckResult("left", connection, CheckReason.NO_DEVICE)
    other = SideCheckResult("right", connection, CheckReason.NO_DEVICE)
    return DeviceCheckResult(side, other, checking=checking, message="未检测到手环")


def single_hand_ready_result() -> DeviceCheckResult:
    ready = SideCheckResult(
        "left",
        ConnectionState.CONNECTED,
        CheckReason.HEALTHY,
        received_emg=True,
        valid_samples=True,
        signal_healthy=True,
        rate_stable=True,
    )
    missing = SideCheckResult("right", ConnectionState.DISCONNECTED, CheckReason.NO_DEVICE)
    return DeviceCheckResult(ready, missing, checking=False, message="检查完成")


class FakeDeviceCheckService(QObject):
    result_changed = Signal(object)

    def __init__(self, *, ready_on_start: bool = True) -> None:
        super().__init__()
        self.result = disconnected_result()
        self.ready_on_start = ready_on_start
        self.start_calls = 0
        self.stop_calls = 0

    def start(self) -> None:
        self.start_calls += 1
        self.result = disconnected_result(checking=True)
        self.result_changed.emit(self.result)
        self.result = single_hand_ready_result() if self.ready_on_start else disconnected_result()
        self.result_changed.emit(self.result)

    def stop(self) -> None:
        self.stop_calls += 1


class FakeDecoder:
    model_type = "standard-test-model"


class FakeObservationService(QObject):
    gesture_updated = Signal(str, str, float, dict)

    def __init__(self, *, model_available: bool = True) -> None:
        super().__init__()
        self.model_available = model_available
        self.predictor = object() if model_available else None
        self.model_error = "" if model_available else "标准识别模型缺失，请联系老师。"
        self.ready_sides = {"left": False, "right": False}
        self.start_calls = 0
        self.stop_calls = 0
        self.update_calls = 0
        self._decoders: dict[str, FakeDecoder] = {}

    def start(self, *, left_ready: bool, right_ready: bool) -> None:
        self.start_calls += 1
        self.ready_sides = {"left": left_ready, "right": right_ready}
        self._decoders = (
            {side: FakeDecoder() for side, ready in self.ready_sides.items() if ready}
            if self.model_available
            else {}
        )

    def stop(self) -> None:
        self.stop_calls += 1
        self._decoders.clear()

    def update_ready_sides(self, *, left_ready: bool, right_ready: bool) -> None:
        self.update_calls += 1
        self.ready_sides = {"left": left_ready, "right": right_ready}

    def decoder_for(self, side: str) -> FakeDecoder | None:
        return self._decoders.get(side)


class FakeGestureServer:
    def __init__(self) -> None:
        self.start_calls = 0
        self.stop_calls = 0
        self.events: list[tuple[str, dict]] = []
        self.clients = 0

    @property
    def client_count(self) -> int:
        return self.clients

    def start(self) -> None:
        self.start_calls += 1

    def stop(self) -> None:
        self.stop_calls += 1
        self.clients = 0

    def publish(self, event: str, payload: dict) -> None:
        self.events.append((event, payload))


def _write_web_game(root: Path) -> None:
    root.mkdir()
    (root / "index.html").write_text("<!doctype html><title>学生游戏</title>", encoding="utf-8")


def test_one_click_checks_device_starts_single_hand_services_and_prevents_duplicates(
    app, tmp_path
) -> None:
    web_root = tmp_path / "web-game"
    _write_web_game(web_root)
    device = FakeDeviceCheckService()
    observation = FakeObservationService()
    gesture = FakeGestureServer()
    opened: list[str] = []

    def opener(url) -> bool:
        opened.append(url.toString())
        gesture.clients = 1
        return True

    service = StudentGameExperienceService(
        device,
        observation,
        web_root,
        gesture_server=gesture,
        browser_opener=opener,
        client_wait_timeout_ms=500,
    )
    try:
        service.start_experience()
        app.processEvents()
        assert service.running
        assert device.start_calls == 1
        assert observation.start_calls == 1
        assert observation.ready_sides == {"left": True, "right": False}
        assert gesture.start_calls == 1
        assert opened == [service.game_url]
        assert service.http_server is not None

        html = urllib.request.urlopen(service.game_url, timeout=2).read().decode("utf-8")
        assert "学生游戏" in html
        statuses = [payload for event, payload in gesture.events if event == "hand_status"]
        assert statuses == [
            {"hand": "left", "connected": True, "game_control": True},
            {"hand": "right", "connected": False, "game_control": False},
        ]

        observation.gesture_updated.emit(
            "left",
            "fist",
            0.9,
            {"rest": 0.02, "fist": 0.9, "open-palm": 0.06, "pinch": 0.02},
        )
        app.processEvents()
        gesture_event = next(payload for event, payload in gesture.events if event == "gesture")
        assert gesture_event["hand"] == "left"
        assert gesture_event["gesture"] == "fist"
        assert gesture_event["model_type"] == "standard-test-model"

        server = service.http_server
        service.start_experience()
        assert opened == [service.game_url, service.game_url]
        assert observation.start_calls == 1
        assert gesture.start_calls == 1
        assert service.http_server is server
    finally:
        service.stop()
    assert service.http_server is None
    assert observation.stop_calls >= 1
    assert gesture.stop_calls >= 1


def test_model_missing_stops_before_sse_http_and_browser(app, tmp_path) -> None:
    web_root = tmp_path / "web-game"
    _write_web_game(web_root)
    device = FakeDeviceCheckService()
    device.result = single_hand_ready_result()
    observation = FakeObservationService(model_available=False)
    gesture = FakeGestureServer()
    opened: list[str] = []
    service = StudentGameExperienceService(
        device,
        observation,
        web_root,
        gesture_server=gesture,
        browser_opener=lambda url: opened.append(url.toString()),
    )

    service.start_experience()
    app.processEvents()

    assert service.state == "error"
    assert "模型缺失" in service.message
    assert gesture.start_calls == 0
    assert service.http_server is None
    assert opened == []
    assert observation.stop_calls == 1


def test_failed_device_check_shows_chinese_message_without_starting_model(app, tmp_path) -> None:
    web_root = tmp_path / "web-game"
    _write_web_game(web_root)
    device = FakeDeviceCheckService(ready_on_start=False)
    observation = FakeObservationService()
    gesture = FakeGestureServer()
    service = StudentGameExperienceService(
        device,
        observation,
        web_root,
        gesture_server=gesture,
        browser_opener=lambda _url: True,
    )

    service.start_experience()
    app.processEvents()

    assert service.state == "error"
    assert "设备检查未通过" in service.message
    assert observation.start_calls == 0
    assert gesture.start_calls == 0


def test_page_uses_required_primary_button_and_reflects_starting_and_running(app, tmp_path) -> None:
    web_root = tmp_path / "web-game"
    _write_web_game(web_root)
    device = FakeDeviceCheckService()
    observation = FakeObservationService()
    gesture = FakeGestureServer()
    service = StudentGameExperienceService(
        device,
        observation,
        web_root,
        gesture_server=gesture,
        browser_opener=lambda _url: True,
    )
    page = StudentQuickExperiencePage(service, lambda: None)
    try:
        assert page.start_button.text() == "使用标准模型开始体验"
        service.status_changed.emit("checking", "正在检查")
        assert not page.start_button.isEnabled()
        service.status_changed.emit("running", "已连接")
        assert page.start_button.isEnabled()
        assert page.start_button.text() == "重新打开游戏"
    finally:
        service.stop()
        page.close()


def test_student_window_quick_experience_entry_is_real_page(app) -> None:
    device_service = DeviceCheckService()
    window = StudentMainWindow(
        paths=resolve_project_paths(),
        device_check_service=device_service,
    )
    try:
        entry = next(item for item in window.course_entries if item.identifier == "quick-experience")
        window.open_course_page(entry)
        assert window._stack.currentWidget() is window.quick_experience_page
        assert window.quick_experience_page.start_button.text() == "使用标准模型开始体验"
    finally:
        window.close()


def test_gesture_server_client_count_tracks_real_sse_connection_and_stop_releases_thread() -> None:
    server = GestureServer(host="127.0.0.1", port=0)
    server.start()
    connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=2)
    response = None
    try:
        connection.request("GET", "/events")
        response = connection.getresponse()
        assert response.status == 200
        assert response.readline().strip() == b"event: status"
        deadline = time.monotonic() + 2.0
        while server.client_count == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server.client_count == 1
        with pytest.raises(AttributeError):
            server.client_count = 9
    finally:
        server.stop()
        if response is not None:
            response.close()
        connection.close()
    assert server.client_count == 0
    assert server._thread is None
