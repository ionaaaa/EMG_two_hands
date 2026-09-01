import json
import os
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
from emg_live_marker.ml.dual_game_mapper import DualGameMapper
from emg_live_marker.paths import resolve_project_paths
from emg_live_marker.realtime.game_mapping import GameMappingConfigError, GameMappingService
from emg_live_marker.realtime.student_game_experience import StudentGameExperienceService
from emg_live_marker.ui.student_pages import StudentGameMappingPage
from emg_live_marker.ui.student_window import StudentMainWindow


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


def mapping_config() -> dict:
    return {
        "game": {
            "commands": {
                "A": {"display_name_zh": "红色音符/指令 A", "game_gesture": "fist"},
                "B": {
                    "display_name_zh": "蓝色音符/指令 B",
                    "game_gesture": "open-palm",
                },
                "none": {"display_name_zh": "无操作", "game_gesture": "rest"},
            },
            "default_mapping": {
                "fist": "A",
                "open-palm": "B",
                "rest": "none",
                "pinch": "none",
            },
        }
    }


def test_yucai_declares_the_unified_command_source(tmp_path) -> None:
    project_root = Path(__file__).resolve().parents[3]
    config = json.loads(
        (project_root / "configs" / "teaching" / "yucai.json").read_text(encoding="utf-8")
    )
    service = GameMappingService(config, storage_root=tmp_path)
    assert service.resolved_mapping == service.default_mapping
    assert set(service.commands) == {"A", "B", "none"}


def test_mapping_defaults_swap_uniqueness_locked_gestures_and_restore(tmp_path) -> None:
    service = GameMappingService(mapping_config(), storage_root=tmp_path)
    assert service.resolved_mapping == {
        "rest": "none",
        "fist": "A",
        "open-palm": "B",
        "pinch": "none",
    }
    assert service.commands["A"]["display_name_zh"] == "红色音符/指令 A"
    assert service.commands["B"]["display_name_zh"] == "蓝色音符/指令 B"

    service.swap_commands()
    assert service.resolved_mapping == {
        "rest": "none",
        "fist": "B",
        "open-palm": "A",
        "pinch": "none",
    }
    with pytest.raises(GameMappingConfigError, match="一一对应"):
        service.set_editable_mapping("A", "A")
    with pytest.raises(GameMappingConfigError, match="放松和捏合"):
        service._apply_mapping(
            {"rest": "A", "fist": "B", "open-palm": "A", "pinch": "none"}
        )

    service.restore_default()
    assert service.resolved_mapping == service.default_mapping


def test_anonymous_group_save_load_and_restart_restore_without_git_writes(tmp_path) -> None:
    service = GameMappingService(mapping_config(), storage_root=tmp_path)
    service.swap_commands()
    saved, message = service.save_current_group("group_01")
    assert saved is True
    assert "group_01" in message
    payload = json.loads((tmp_path / "group_01.json").read_text(encoding="utf-8"))
    assert set(payload) == {"schema_version", "anonymous_group_id", "mapping"}
    assert payload["mapping"]["fist"] == "B"

    restarted = GameMappingService(mapping_config(), storage_root=tmp_path)
    assert restarted.current_group_id == "group_01"
    assert restarted.resolved_mapping["fist"] == "B"
    restarted.restore_default()
    loaded, _message = restarted.load_group("group_01")
    assert loaded is True
    assert restarted.resolved_mapping["fist"] == "B"


def test_empty_or_unsafe_group_id_never_writes(app, tmp_path) -> None:
    service = GameMappingService(mapping_config(), storage_root=tmp_path)
    page = StudentGameMappingPage(service, lambda: "", lambda: None)
    try:
        page.save_button.click()
        app.processEvents()
        assert "请先填写匿名小组编号" in page.message_label.text()
        assert list(tmp_path.iterdir()) == []
        saved, _message = service.save_current_group("../真实姓名")
        assert saved is False
        assert list(tmp_path.iterdir()) == []
    finally:
        page.close()


def test_student_window_uses_real_mapping_page_and_restores_saved_group(app, tmp_path) -> None:
    mapping = GameMappingService(mapping_config(), storage_root=tmp_path)
    device = DeviceCheckService()
    window = StudentMainWindow(
        paths=resolve_project_paths(),
        device_check_service=device,
        game_mapping_service=mapping,
    )
    try:
        window.collection_page.anonymous_id_edit.setText("group_02")
        entry = next(item for item in window.course_entries if item.identifier == "configure-game")
        assert entry.available is True
        window.open_course_page(entry)
        assert window._stack.currentWidget() is window.game_mapping_page
        window.game_mapping_page.swap_button.click()
        window.game_mapping_page.save_button.click()
        assert mapping.resolved_mapping["fist"] == "B"

        window.show_home()
        mapping.restore_default()
        window.open_course_page(entry)
        assert mapping.resolved_mapping["fist"] == "B"
        assert "group_02" in window.game_mapping_page.group_label.text()
    finally:
        window.close()


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    def key_down(self, key: str) -> None:
        self.events.append(("down", key))

    def key_up(self, key: str) -> None:
        self.events.append(("up", key))


def test_dual_game_mapper_consumes_unified_mapping_and_keeps_legacy_defaults(tmp_path) -> None:
    legacy_sink = RecordingSink()
    legacy = DualGameMapper(key_sink=legacy_sink)
    legacy.update("fist", "open-palm", enabled=True)
    assert legacy_sink.events == [("down", "A"), ("down", "W")]
    legacy.release_all()

    service = GameMappingService(mapping_config(), storage_root=tmp_path)
    unified_sink = RecordingSink()
    unified = DualGameMapper(
        key_sink=unified_sink,
        command_mapping=service.resolved_mapping,
    )
    unified.update("fist", "open-palm", enabled=True)
    assert unified_sink.events == [("down", "A"), ("down", "B")]
    unified.set_command_mapping(
        {"rest": "none", "fist": "B", "open-palm": "A", "pinch": "none"}
    )
    unified.update("fist", "pinch", enabled=True)
    assert unified.pressed_keys("left") == {"B"}
    assert unified.pressed_keys("right") == set()


def ready_result() -> DeviceCheckResult:
    left = SideCheckResult(
        "left",
        ConnectionState.CONNECTED,
        CheckReason.HEALTHY,
        received_emg=True,
        valid_samples=True,
        signal_healthy=True,
        rate_stable=True,
    )
    right = SideCheckResult("right", ConnectionState.DISCONNECTED, CheckReason.NO_DEVICE)
    return DeviceCheckResult(left, right, checking=False, message="检查完成")


class FakeDevice(QObject):
    result_changed = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.result = ready_result()

    def start(self) -> None:
        raise AssertionError("ready device must be reused")

    def stop(self) -> None:
        pass


class FakeDecoder:
    model_type = "standard"


class FakeObservation(QObject):
    gesture_updated = Signal(str, str, float, dict)

    def __init__(self) -> None:
        super().__init__()
        self.predictor = object()
        self.model_error = ""
        self.ready_sides = {"left": False, "right": False}
        self.decoder = None

    def start(self, *, left_ready: bool, right_ready: bool) -> None:
        self.ready_sides = {"left": left_ready, "right": right_ready}
        self.decoder = FakeDecoder()

    def stop(self) -> None:
        self.decoder = None

    def update_ready_sides(self, *, left_ready: bool, right_ready: bool) -> None:
        self.ready_sides = {"left": left_ready, "right": right_ready}

    def decoder_for(self, side: str):
        return self.decoder if self.ready_sides.get(side, False) else None


class FakeGestureServer:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []
        self.started = False
        self.clients = 0

    @property
    def client_count(self) -> int:
        return self.clients

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False
        self.clients = 0

    def publish(self, event: str, payload: dict) -> None:
        self.events.append((event, payload))


def test_runtime_config_mapping_sse_and_marked_mapping_test(app, tmp_path) -> None:
    web_root = tmp_path / "web-game"
    web_root.mkdir()
    (web_root / "index.html").write_text("<title>game</title>", encoding="utf-8")
    mapping = GameMappingService(mapping_config(), storage_root=tmp_path / "settings")
    gesture = FakeGestureServer()

    def opener(_url) -> bool:
        gesture.clients = 1
        return True

    experience = StudentGameExperienceService(
        FakeDevice(),
        FakeObservation(),
        web_root,
        mapping_service=mapping,
        gesture_server=gesture,
        browser_opener=opener,
        client_wait_timeout_ms=500,
    )
    try:
        experience.start_experience()
        app.processEvents()
        assert experience.running
        config_url = experience.game_url.replace("index.html", "runtime-config.json")
        runtime = json.loads(urllib.request.urlopen(config_url, timeout=2).read().decode("utf-8"))
        assert runtime["resolved_mapping"]["fist"] == "A"
        assert runtime["resolved_mapping"]["rest"] == "none"

        gesture.events.clear()
        mapping.swap_commands()
        app.processEvents()
        assert gesture.events == [("mapping", mapping.runtime_config())]

        gesture.events.clear()
        message = mapping.test_mapping()
        app.processEvents()
        assert "未生成任何识别结果" in message
        assert len(gesture.events) == 1
        event, payload = gesture.events[0]
        assert event == "mapping_test"
        assert payload["test"] is True
        assert not any(name == "gesture" for name, _payload in gesture.events)
    finally:
        experience.stop()


def test_web_game_loads_and_listens_for_runtime_mapping() -> None:
    project_root = Path(__file__).resolve().parents[3]
    script = (project_root / "apps" / "web-game" / "game.js").read_text(encoding="utf-8")
    assert 'fetch("runtime-config.json"' in script
    assert 'source.addEventListener("mapping", onMapping)' in script
    assert "mapRealtimeGesture(recognizedGesture)" in script
