import json
import os
import urllib.error
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
    SideCheckResult,
)
from emg_live_marker.paths import resolve_project_paths
from emg_live_marker.realtime.game_mapping import GameMappingService
from emg_live_marker.realtime.student_competition import StudentCompetitionService
from emg_live_marker.realtime.student_game_experience import StudentGameExperienceService
from emg_live_marker.ui.student_pages import StudentChallengePage
from emg_live_marker.ui.student_window import StudentMainWindow


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def course_config(tmp_path: Path) -> dict:
    path = Path(__file__).parents[3] / "configs" / "teaching" / "yucai.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["storage"]["competition_results_path"] = str(
        tmp_path / "results" / "{competition_id}" / "{student_id}"
    )
    return config


class FakeObservation(QObject):
    gesture_updated = Signal(str, str, float, dict)
    model_source_changed = Signal(str, str)

    def __init__(self):
        super().__init__()
        self.model_source = "standard"
        self.current_sensitivity = "standard"
        self.current_control_style = "balanced"
        self.predictor = object()
        self.model_error = ""
        self.ready_sides = {"left": False, "right": False}
        self.start_calls = 0
        self.stop_calls = 0
        self.update_calls = 0
        self._decoders = {}

    def start(self, *, left_ready, right_ready):
        self.start_calls += 1
        self.ready_sides = {"left": left_ready, "right": right_ready}
        self._decoders = {
            side: type("Decoder", (), {"model_type": "private-backend-name"})()
            for side, ready in self.ready_sides.items()
            if ready
        }

    def stop(self):
        self.stop_calls += 1
        self._decoders.clear()

    def update_ready_sides(self, *, left_ready, right_ready):
        self.update_calls += 1
        self.ready_sides = {"left": left_ready, "right": right_ready}

    def decoder_for(self, side):
        return self._decoders.get(side)


class FakeDevice(QObject):
    result_changed = Signal(object)

    def __init__(self):
        super().__init__()
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
        self.result = DeviceCheckResult(ready, missing, checking=False, message="检查完成")
        self.start_calls = 0
        self.stop_calls = 0

    def start(self):
        self.start_calls += 1

    def stop(self):
        self.stop_calls += 1


class FakeGestureServer:
    def __init__(self):
        self.clients = 0
        self.start_calls = 0
        self.stop_calls = 0
        self.events = []

    @property
    def client_count(self):
        return self.clients

    def start(self):
        self.start_calls += 1

    def stop(self):
        self.stop_calls += 1
        self.clients = 0

    def publish(self, event, payload):
        self.events.append((event, payload))


class FakeExperience(QObject):
    status_changed = Signal(str, str)

    def __init__(self):
        super().__init__()
        self.running = False
        self.calls = []

    def start_challenge(self, student_id):
        self.calls.append(student_id)
        self.running = True
        self.status_changed.emit("running", "挑战赛网页已连接。")
        return True


def make_competition(tmp_path):
    config = course_config(tmp_path)
    observation = FakeObservation()
    mapping = GameMappingService(config, storage_root=tmp_path / "settings")
    service = StudentCompetitionService(tmp_path, config, observation, mapping)
    return service, observation, mapping, config


def single_result(**overrides):
    result = {
        "mode": "left",
        "score": 12340,
        "accuracy": 0.875,
        "max_combo": 18,
        "outcome": "completed",
    }
    result.update(overrides)
    return result


def dual_result(**overrides):
    result = single_result(
        mode="both",
        outcome="left_win",
        left_accuracy=0.9,
        right_accuracy=0.75,
    )
    result.update(overrides)
    return result


def test_result_validation_rejects_untrusted_invalid_values(tmp_path) -> None:
    service, _observation, _mapping, _config = make_competition(tmp_path)
    assert service.begin_competition("group_01")[0]
    for payload in (
        single_result(score=-1),
        single_result(accuracy=1.1),
        single_result(max_combo=2.5),
        single_result(mode="teacher"),
        single_result(outcome="left_win"),
        dual_result(left_accuracy=None),
    ):
        status, response = service.save_browser_result(payload)
        assert status == 400
        assert response["ok"] is False


def test_single_result_uses_trusted_current_model_mapping_and_profile(tmp_path) -> None:
    service, observation, mapping, _config = make_competition(tmp_path)
    observation.model_source = "personal"
    observation.current_sensitivity = "high"
    observation.current_control_style = "stable"
    mapping.swap_commands()
    assert service.begin_competition("group_02")[0]
    status, response = service.save_browser_result(single_result(), result_id="round-1")
    assert status == 201
    assert response["ok"]
    saved = json.loads(service.latest_result_path.read_text(encoding="utf-8"))
    assert saved["student_id"] == "group_02"
    assert saved["model_type"] == "personal"
    assert saved["mapping"]["fist"] == "B"
    assert saved["sensitivity"] == "high"
    assert saved["stability"] == "stable"
    assert saved["score"] == 12340
    assert saved["accuracy"] == pytest.approx(0.875)
    assert saved["max_combo"] == 18
    assert "timestamp" in saved
    assert "private-backend-name" not in json.dumps(saved)


def test_dual_hand_fields_and_each_round_gets_an_independent_file(tmp_path) -> None:
    service, _observation, _mapping, _config = make_competition(tmp_path)
    assert service.begin_competition("group_03")[0]
    assert service.save_browser_result(dual_result(), result_id="round-a")[0] == 201
    first = service.latest_result_path
    assert service.latest_result["left_accuracy"] == pytest.approx(0.9)
    assert service.latest_result["right_accuracy"] == pytest.approx(0.75)
    assert service.save_browser_result(dual_result(score=999), result_id="round-b")[0] == 201
    second = service.latest_result_path
    assert first != second
    assert first.is_file() and second.is_file()


def test_duplicate_post_is_idempotent(tmp_path) -> None:
    service, _observation, _mapping, _config = make_competition(tmp_path)
    assert service.begin_competition("group_04")[0]
    assert service.save_browser_result(single_result(), result_id="same-round")[0] == 201
    first_path = service.latest_result_path
    status, response = service.save_browser_result(single_result(), result_id="same-round")
    assert status == 200
    assert response["duplicate"] is True
    assert service.latest_result_path == first_path
    assert len(list((tmp_path / "results").rglob("result.json"))) == 1


def test_save_failure_is_chinese_and_does_not_claim_success(tmp_path) -> None:
    service, _observation, _mapping, config = make_competition(tmp_path)
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory", encoding="utf-8")
    config["storage"]["competition_results_path"] = str(
        blocked / "{competition_id}" / "{student_id}"
    )
    service = StudentCompetitionService(
        tmp_path, config, service.observation_service, service.mapping_service
    )
    errors = []
    service.save_failed.connect(errors.append)
    assert service.begin_competition("group_05")[0]
    status, response = service.save_browser_result(single_result(), result_id="failed-round")
    assert status == 500
    assert "成绩保存失败" in response["message"]
    assert errors and "成绩保存失败" in errors[0]
    assert service.latest_result is None


def _post(url, payload, round_id):
    request = urllib.request.Request(
        url.split("/index.html", 1)[0] + "/competition-result",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "X-Competition-Round-ID": round_id,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=2) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def test_shared_http_service_separates_quick_experience_and_challenge_results(app, tmp_path) -> None:
    web_root = tmp_path / "web-game"
    web_root.mkdir()
    (web_root / "index.html").write_text("<!doctype html><title>game</title>", encoding="utf-8")
    competition, observation, mapping, _config = make_competition(tmp_path)
    device = FakeDevice()
    gesture = FakeGestureServer()

    def opener(_url):
        gesture.clients = 1
        return True

    experience = StudentGameExperienceService(
        device,
        observation,
        web_root,
        mapping_service=mapping,
        competition_service=competition,
        gesture_server=gesture,
        browser_opener=opener,
        client_wait_timeout_ms=500,
    )
    try:
        experience.start_experience()
        app.processEvents()
        assert experience.running
        assert "mode=challenge" not in experience.game_url
        status, response = _post(experience.game_url, single_result(), "quick-round")
        assert status == 403
        assert response["ok"] is False
        assert not list((tmp_path / "results").rglob("result.json"))
        experience.stop()

        assert experience.start_challenge("group_06")
        app.processEvents()
        assert experience.running
        assert experience.game_url.endswith("index.html?mode=challenge")
        status, response = _post(experience.game_url, dual_result(), "challenge-round")
        assert status == 201
        assert response["ok"] is True
        assert competition.latest_result["mode"] == "both"
        server = experience.http_server
        experience.start_challenge("group_06")
        assert experience.http_server is server
    finally:
        experience.stop()
    assert experience.http_server is None
    assert experience._http_thread is None
    assert observation.stop_calls >= 2
    assert gesture.stop_calls >= 2


def test_challenge_page_blocks_empty_group_and_displays_saved_result(app, tmp_path) -> None:
    competition, observation, mapping, _config = make_competition(tmp_path)
    experience = FakeExperience()
    group = [""]
    page = StudentChallengePage(
        experience,
        competition,
        mapping,
        observation,
        lambda: group[0],
        lambda: None,
    )
    page.start_button.click()
    assert experience.calls == []
    assert "匿名小组编号" in page.status_label.text()

    group[0] = "group_07"
    page.activate()
    page.start_button.click()
    assert experience.calls == ["group_07"]
    assert page.group_label.text() == "group_07"
    assert page.model_label.text() == "标准模型"
    assert "握拳" in page.mapping_label.text()
    assert page.sensitivity_label.text() == "标准"
    assert page.style_label.text() == "均衡"
    result = single_result()
    competition.result_saved.emit(result)
    app.processEvents()
    assert "得分：12340" in page.recent_result_label.text()
    assert "正确率：87.5%" in page.recent_result_label.text()
    assert "最大连击：18" in page.recent_result_label.text()
    assert "成绩已保存" in page.recent_result_label.text()
    assert "private-backend-name" not in page.model_label.text()
    competition.save_failed.emit("成绩保存失败：磁盘不可写")
    app.processEvents()
    assert "网页中的本局结果不受影响" in page.status_label.text()


def test_challenge_web_mode_uses_one_shared_page_and_hides_technical_sections() -> None:
    web_root = Path(__file__).parents[2] / "web-game"
    html = (web_root / "index.html").read_text(encoding="utf-8")
    script = (web_root / "game.js").read_text(encoding="utf-8")
    styles = (web_root / "styles.css").read_text(encoding="utf-8")
    assert "data-challenge-hide" in html
    assert "body.challenge-mode [data-challenge-hide]" in styles
    assert 'APP_MODE === "challenge"' in script
    assert 'fetch("/competition-result"' in script
    assert "competitionResultSubmitted" in script
    assert "student_id" not in script
    assert "confidence_threshold" not in script


def test_student_window_challenge_entry_uses_real_page(app) -> None:
    window = StudentMainWindow(paths=resolve_project_paths())
    try:
        entry = next(item for item in window.course_entries if item.identifier == "challenge")
        window.open_course_page(entry)
        assert window._stack.currentWidget() is window.challenge_page
        assert window.challenge_page.objectName() == "student-challenge-page"
    finally:
        window.close()
