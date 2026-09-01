"""Local SSE server streaming per-hand gesture events to the web UI.

Runs on port 8766 by default. The desktop app publishes ``gesture`` and
``hand_status`` events; browsers connect to ``/events`` to receive them.
"""

from __future__ import annotations

import json
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class GestureEventBroadcaster:
    def __init__(self) -> None:
        self._clients: set[queue.Queue[dict[str, Any]]] = set()
        self._lock = threading.Lock()
        self._closing = False

    def subscribe(self) -> queue.Queue[dict[str, Any]]:
        client: queue.Queue[dict[str, Any]] = queue.Queue()
        with self._lock:
            self._clients.add(client)
            closing = self._closing
        if closing:
            client.put({"event": "__close__", "data": {}})
        return client

    def unsubscribe(self, client: queue.Queue[dict[str, Any]]) -> None:
        with self._lock:
            self._clients.discard(client)

    def publish(self, event: str, payload: dict[str, Any]) -> None:
        with self._lock:
            clients = list(self._clients)
        for client in clients:
            client.put({"event": event, "data": payload})

    @property
    def client_count(self) -> int:
        """Return the number of currently subscribed SSE clients."""

        with self._lock:
            return len(self._clients)

    def close_clients(self) -> None:
        """Wake connected request threads so server shutdown cannot strand them."""

        with self._lock:
            self._closing = True
            clients = list(self._clients)
        for client in clients:
            client.put({"event": "__close__", "data": {}})

    def reopen(self) -> None:
        with self._lock:
            self._closing = False


class _GestureRequestHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send_json({"ok": True})
            return
        if self.path == "/events":
            self._stream_events()
            return
        self.send_error(404, "Not found")

    def _stream_events(self) -> None:
        self.send_response(200)
        self._send_cors_headers()
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        self.wfile.write(b"event: status\ndata: online\n\n")
        self.wfile.flush()
        broadcaster: GestureEventBroadcaster = self.server.broadcaster  # type: ignore[attr-defined]
        client = broadcaster.subscribe()
        try:
            while True:
                try:
                    msg = client.get(timeout=15)
                    if msg["event"] == "__close__":
                        break
                    message = f"event: {msg['event']}\ndata: {json.dumps(msg['data'])}\n\n"
                except queue.Empty:
                    message = ": keepalive\n\n"
                self.wfile.write(message.encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass
        finally:
            broadcaster.unsubscribe(client)

    def _send_json(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return


class GestureServer:
    """SSE server publishing per-hand gesture/status events."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8766) -> None:
        self.host = host
        self.port = int(port)
        self.broadcaster = GestureEventBroadcaster()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._server is not None:
            return
        self.broadcaster.reopen()
        server = ThreadingHTTPServer((self.host, self.port), _GestureRequestHandler)
        server.broadcaster = self.broadcaster  # type: ignore[attr-defined]
        self.port = int(server.server_address[1])
        self._server = server
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name="GestureServer",
        )
        self._thread.start()

    def stop(self) -> None:
        if self._server is not None:
            self.broadcaster.close_clients()
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)
        self._thread = None

    def publish(self, event: str, payload: dict[str, Any]) -> None:
        self.broadcaster.publish(event, payload)

    @property
    def client_count(self) -> int:
        """Read-only count used to verify that a browser EventSource connected."""

        return self.broadcaster.client_count
