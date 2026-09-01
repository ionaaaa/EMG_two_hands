"""One-click standard-model game experience for student mode."""

from __future__ import annotations

import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread, current_thread
from time import monotonic
from typing import TYPE_CHECKING, Any, Callable
from urllib.parse import urlsplit

from PySide6.QtCore import QObject, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices

from emg_live_marker.device.check_service import DeviceCheckResult, DeviceCheckService
from emg_live_marker.realtime.gesture_server import GestureServer
from emg_live_marker.realtime.game_mapping import GameMappingService
from emg_live_marker.realtime.student_observation import (
    STUDENT_GESTURES,
    StudentObservationService,
)

if TYPE_CHECKING:
    from emg_live_marker.realtime.student_competition import StudentCompetitionService


class _StudentGameRequestHandler(SimpleHTTPRequestHandler):
    def __init__(
        self,
        *args: object,
        runtime_config_provider: Callable[[], dict],
        competition_result_handler: Callable[[object, str], tuple[int, dict[str, Any]]],
        **kwargs: object,
    ) -> None:
        self._runtime_config_provider = runtime_config_provider
        self._competition_result_handler = competition_result_handler
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:  # noqa: N802
        if urlsplit(self.path).path == "/runtime-config.json":
            body = json.dumps(
                self._runtime_config_provider(), ensure_ascii=False
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        if urlsplit(self.path).path != "/competition-result":
            self.send_error(404)
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            content_length = 0
        if content_length <= 0 or content_length > 65_536:
            self._send_json(400, {"ok": False, "message": "比赛结果内容无效。"})
            return
        try:
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(400, {"ok": False, "message": "比赛结果 JSON 无效。"})
            return
        result_id = self.headers.get("X-Competition-Round-ID", "")
        status, response = self._competition_result_handler(payload, result_id)
        self._send_json(status, response)

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        return


class StudentGameExperienceService(QObject):
    """Coordinate device readiness, shared decoding, SSE, and the local game site."""

    status_changed = Signal(str, str)

    def __init__(
        self,
        device_check_service: DeviceCheckService,
        observation_service: StudentObservationService,
        web_game_root: Path,
        *,
        mapping_service: GameMappingService | None = None,
        competition_service: "StudentCompetitionService | None" = None,
        gesture_server: GestureServer | None = None,
        browser_opener: Callable[[QUrl], bool | None] | None = None,
        client_wait_timeout_ms: int = 8000,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.device_check_service = device_check_service
        self.observation_service = observation_service
        self.web_game_root = Path(web_game_root).resolve()
        self.mapping_service = mapping_service
        self.competition_service = competition_service
        self.gesture_server = gesture_server or GestureServer(host="127.0.0.1", port=8766)
        self.browser_opener = browser_opener or QDesktopServices.openUrl
        self.client_wait_timeout_ms = max(100, int(client_wait_timeout_ms))
        self.state = "idle"
        self.message = "点击按钮开始体验。"
        self.game_url = ""
        self.launch_mode = "experience"
        self._phase = "idle"
        self._client_deadline = 0.0
        self._http_server: ThreadingHTTPServer | None = None
        self._http_thread: Thread | None = None
        self._client_timer = QTimer(self)
        self._client_timer.setInterval(100)
        self._client_timer.timeout.connect(self._poll_sse_client)
        self.device_check_service.result_changed.connect(self._on_device_result)
        self.observation_service.gesture_updated.connect(self._publish_gesture)
        if self.mapping_service is not None:
            self.mapping_service.mapping_changed.connect(self._publish_mapping)
            self.mapping_service.test_feedback.connect(self._publish_mapping_test)

    @property
    def starting(self) -> bool:
        return self.state in {"checking", "starting", "waiting-client"}

    @property
    def running(self) -> bool:
        return self.state == "running"

    @property
    def http_server(self) -> ThreadingHTTPServer | None:
        return self._http_server

    def start_experience(self) -> None:
        """Start once, or reopen the existing URL when already running."""

        if self.running and self.launch_mode == "experience":
            self._open_game_url()
            return
        if self.running:
            return
        if self.starting:
            return
        self.launch_mode = "experience"
        self._start_runtime()

    def start_challenge(self, student_id: str) -> bool:
        """Start the same runtime and page with challenge-only behavior enabled."""

        if self.competition_service is None:
            self._set_status("error", "比赛结果服务不可用，请联系老师。")
            return False
        if self.running:
            if (
                self.launch_mode == "challenge"
                and str(student_id).strip() == self.competition_service.student_id
            ):
                return self._open_game_url()
            return False
        if self.starting:
            return False
        ready, message = self.competition_service.begin_competition(student_id)
        if not ready:
            self._set_status("error", message)
            return False
        self.launch_mode = "challenge"
        self._start_runtime()
        return True

    def _start_runtime(self) -> None:
        self._set_status("checking", "正在检查手环，请稍候……")
        self._phase = "device-check"
        result = self.device_check_service.result
        if result.collection_ready:
            self._start_model_and_servers(result)
            return
        if result.checking:
            return
        self.device_check_service.start()

    def set_competition_service(self, service: "StudentCompetitionService") -> None:
        self.competition_service = service

    def stop(self) -> None:
        """Stop all runtime components owned by the quick-experience flow."""

        was_checking = self._phase == "device-check" and self.device_check_service.result.checking
        self._phase = "stopping"
        self._client_timer.stop()
        self.observation_service.stop()
        self.gesture_server.stop()
        self._stop_http_server()
        if was_checking:
            self.device_check_service.stop()
        self.game_url = ""
        self._phase = "idle"
        self._set_status("idle", "点击按钮开始体验。")

    def _on_device_result(self, result: DeviceCheckResult) -> None:
        if self._phase in {"waiting-client", "running"}:
            self.observation_service.update_ready_sides(
                left_ready=result.left.ready_for_collection,
                right_ready=result.right.ready_for_collection,
            )
            self._publish_hand_statuses()
            return
        if self._phase != "device-check":
            return
        if result.checking:
            self._set_status("checking", "正在检查手环和肌电信号……")
            return
        if not result.collection_ready:
            self._fail(f"设备检查未通过：{result.message}")
            return
        self._start_model_and_servers(result)

    def _start_model_and_servers(self, result: DeviceCheckResult) -> None:
        self._phase = "starting-runtime"
        model_name = (
            "我的模型"
            if getattr(self.observation_service, "model_source", "standard") == "personal"
            else "标准模型"
        )
        self._set_status("starting", f"设备检查通过，正在启动{model_name}……")
        left_ready = result.left.ready_for_collection
        right_ready = result.right.ready_for_collection
        self.observation_service.start(left_ready=left_ready, right_ready=right_ready)
        if self.observation_service.predictor is None:
            error = self.observation_service.model_error or "标准识别模型无法使用，请联系老师。"
            self._fail(error)
            return
        if not any(
            ready and self.observation_service.decoder_for(side) is not None
            for side, ready in (("left", left_ready), ("right", right_ready))
        ):
            self._fail("标准识别模型未能启动，请联系老师。")
            return

        try:
            self.gesture_server.start()
        except OSError:
            self._fail("手势服务启动失败，请关闭占用端口的程序后重试。")
            return
        try:
            self._start_http_server()
        except OSError:
            self._fail("游戏网页服务启动失败，请稍后重试。")
            return
        if not self._open_game_url():
            self._fail("无法打开游戏网页，请检查系统浏览器设置。")
            return

        self._phase = "waiting-client"
        self._client_deadline = monotonic() + self.client_wait_timeout_ms / 1000.0
        self._set_status("waiting-client", "游戏已打开，正在等待网页连接……")
        self._client_timer.start()
        self._poll_sse_client()

    def _start_http_server(self) -> None:
        index_path = self.web_game_root / "index.html"
        if not index_path.is_file():
            raise OSError("web game index is missing")
        handler = partial(
            _StudentGameRequestHandler,
            directory=str(self.web_game_root),
            runtime_config_provider=self.runtime_config,
            competition_result_handler=self._handle_competition_result,
        )
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        server.daemon_threads = True
        self._http_server = server
        port = int(server.server_address[1])
        query = "?mode=challenge" if self.launch_mode == "challenge" else ""
        self.game_url = f"http://127.0.0.1:{port}/index.html{query}"
        self._http_thread = Thread(
            target=server.serve_forever,
            daemon=True,
            name="StudentGameStaticServer",
        )
        self._http_thread.start()

    def _stop_http_server(self) -> None:
        server = self._http_server
        thread = self._http_thread
        self._http_server = None
        self._http_thread = None
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None and thread is not current_thread():
            thread.join(timeout=2.0)

    def _open_game_url(self) -> bool:
        if not self.game_url:
            return False
        try:
            opened = self.browser_opener(QUrl(self.game_url))
        except Exception:  # noqa: BLE001 - injected desktop openers are external.
            return False
        return opened is not False

    def _poll_sse_client(self) -> None:
        if self._phase != "waiting-client":
            return
        if self.gesture_server.client_count > 0:
            self._client_timer.stop()
            self._phase = "running"
            self._publish_mapping(self.runtime_config())
            self._publish_hand_statuses()
            model_name = (
                "我的模型"
                if getattr(self.observation_service, "model_source", "standard") == "personal"
                else "标准模型"
            )
            self._set_status("running", f"{model_name}和游戏已启动，网页连接正常。")
            return
        if monotonic() >= self._client_deadline:
            self._fail("游戏网页未连接到手势服务，请确认浏览器已正常打开。")

    def _publish_gesture(
        self, side: str, gesture: str, confidence: float, probabilities: dict[str, float]
    ) -> None:
        if self._phase not in {"waiting-client", "running"}:
            return
        decoder = self.observation_service.decoder_for(side)
        self.gesture_server.publish(
            "gesture",
            {
                "hand": side,
                "gesture": str(gesture),
                "confidence": float(confidence),
                "probs": {
                    label: float(probabilities.get(label, 0.0)) for label in STUDENT_GESTURES
                },
                "game_control": True,
                "model_type": getattr(decoder, "model_type", "model"),
                "source": "emg_live_marker",
                "connected": bool(self.observation_service.ready_sides.get(side, False)),
            },
        )

    def _publish_hand_statuses(self) -> None:
        for side in ("left", "right"):
            connected = bool(self.observation_service.ready_sides.get(side, False))
            self.gesture_server.publish(
                "hand_status",
                {
                    "hand": side,
                    "connected": connected,
                    "game_control": connected,
                },
            )

    def runtime_config(self) -> dict:
        if self.mapping_service is not None:
            config = self.mapping_service.runtime_config()
            config["app_mode"] = self.launch_mode
            return config
        return {
            "schema_version": 1,
            "app_mode": self.launch_mode,
            "commands": {
                "A": {"display_name_zh": "红色音符/指令 A", "game_gesture": "fist"},
                "B": {
                    "display_name_zh": "蓝色音符/指令 B",
                    "game_gesture": "open-palm",
                },
                "none": {"display_name_zh": "无操作", "game_gesture": "rest"},
            },
            "default_mapping": {
                "rest": "none",
                "fist": "A",
                "open-palm": "B",
                "pinch": "none",
            },
            "resolved_mapping": {
                "rest": "none",
                "fist": "A",
                "open-palm": "B",
                "pinch": "none",
            },
            "anonymous_group_id": "",
        }

    def _handle_competition_result(
        self, payload: object, result_id: str
    ) -> tuple[int, dict[str, Any]]:
        if self.launch_mode != "challenge" or self.competition_service is None:
            return 403, {"ok": False, "message": "快速体验不会保存比赛成绩。"}
        return self.competition_service.save_browser_result(payload, result_id=result_id)

    def _publish_mapping(self, runtime_config: dict) -> None:
        if self._phase not in {"waiting-client", "running"}:
            return
        self.gesture_server.publish("mapping", dict(runtime_config))

    def _publish_mapping_test(self, payload: dict) -> None:
        if self._phase not in {"waiting-client", "running"}:
            return
        marked = dict(payload)
        marked["test"] = True
        self.gesture_server.publish("mapping_test", marked)

    def _fail(self, message: str) -> None:
        self._phase = "failing"
        self._client_timer.stop()
        self.observation_service.stop()
        self.gesture_server.stop()
        self._stop_http_server()
        self.game_url = ""
        self._phase = "error"
        self._set_status("error", message)

    def _set_status(self, state: str, message: str) -> None:
        self.state = state
        self.message = message
        self.status_changed.emit(state, message)
