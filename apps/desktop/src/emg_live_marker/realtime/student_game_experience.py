"""One-click standard-model game experience for student mode."""

from __future__ import annotations

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread, current_thread
from time import monotonic
from typing import Callable

from PySide6.QtCore import QObject, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices

from emg_live_marker.device.check_service import DeviceCheckResult, DeviceCheckService
from emg_live_marker.realtime.gesture_server import GestureServer
from emg_live_marker.realtime.student_observation import (
    STUDENT_GESTURES,
    StudentObservationService,
)


class _QuietStaticRequestHandler(SimpleHTTPRequestHandler):
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
        gesture_server: GestureServer | None = None,
        browser_opener: Callable[[QUrl], bool | None] | None = None,
        client_wait_timeout_ms: int = 8000,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.device_check_service = device_check_service
        self.observation_service = observation_service
        self.web_game_root = Path(web_game_root).resolve()
        self.gesture_server = gesture_server or GestureServer(host="127.0.0.1", port=8766)
        self.browser_opener = browser_opener or QDesktopServices.openUrl
        self.client_wait_timeout_ms = max(100, int(client_wait_timeout_ms))
        self.state = "idle"
        self.message = "点击按钮开始体验。"
        self.game_url = ""
        self._phase = "idle"
        self._client_deadline = 0.0
        self._http_server: ThreadingHTTPServer | None = None
        self._http_thread: Thread | None = None
        self._client_timer = QTimer(self)
        self._client_timer.setInterval(100)
        self._client_timer.timeout.connect(self._poll_sse_client)
        self.device_check_service.result_changed.connect(self._on_device_result)
        self.observation_service.gesture_updated.connect(self._publish_gesture)

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

        if self.running:
            self._open_game_url()
            return
        if self.starting:
            return
        self._set_status("checking", "正在检查手环，请稍候……")
        self._phase = "device-check"
        result = self.device_check_service.result
        if result.collection_ready:
            self._start_model_and_servers(result)
            return
        if result.checking:
            return
        self.device_check_service.start()

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
        self._set_status("starting", "设备检查通过，正在启动标准模型……")
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
        handler = partial(_QuietStaticRequestHandler, directory=str(self.web_game_root))
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        server.daemon_threads = True
        self._http_server = server
        port = int(server.server_address[1])
        self.game_url = f"http://127.0.0.1:{port}/index.html"
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
            self._publish_hand_statuses()
            self._set_status("running", "标准模型和游戏已启动，网页连接正常。")
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
