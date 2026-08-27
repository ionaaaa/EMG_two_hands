from __future__ import annotations

import argparse
import json
import queue
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


GESTURES = {"rest", "fist", "open-palm", "pinch"}


class GestureBroadcaster:
    def __init__(self) -> None:
        self._clients: set[queue.Queue[dict[str, Any]]] = set()
        self._lock = threading.Lock()

    def subscribe(self) -> queue.Queue[dict[str, Any]]:
        client: queue.Queue[dict[str, Any]] = queue.Queue()
        with self._lock:
            self._clients.add(client)
        return client

    def unsubscribe(self, client: queue.Queue[dict[str, Any]]) -> None:
        with self._lock:
            self._clients.discard(client)

    def publish(self, payload: dict[str, Any]) -> None:
        with self._lock:
            clients = list(self._clients)
        for client in clients:
            client.put(payload)


class GestureClassifier:
    """Replace this class with your trained model loading and inference code."""

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path
        self.model = self._load_model(model_path)

    def _load_model(self, model_path: str | None) -> Any:
        if not model_path:
            return None

        # Example for a scikit-learn model:
        # import joblib
        # return joblib.load(model_path)
        #
        # Example for a PyTorch model:
        # import torch
        # model = torch.load(model_path, map_location="cpu")
        # model.eval()
        # return model
        return None

    def predict(self, samples: list[Any]) -> dict[str, Any]:
        if self.model is None:
            return self._demo_predict(samples)

        # Adapt this block to your model input shape and label order.
        # features = preprocess_emg_window(samples)
        # label_index = self.model.predict([features])[0]
        # confidence = max(self.model.predict_proba([features])[0])
        # gesture = LABELS[label_index]
        # return {"gesture": gesture, "confidence": float(confidence)}
        raise NotImplementedError("Connect your trained model in GestureClassifier.predict().")

    def _demo_predict(self, samples: list[Any]) -> dict[str, Any]:
        # Small deterministic-ish fallback so the API can be tested immediately.
        flat = flatten_numeric(samples)
        if not flat:
            gesture = "rest"
        else:
            energy = sum(abs(value) for value in flat) / len(flat)
            if energy < 0.15:
                gesture = "rest"
            elif energy < 0.38:
                gesture = "fist"
            elif energy < 0.68:
                gesture = "pinch"
            else:
                gesture = "open-palm"
        return {"gesture": gesture, "confidence": round(random.uniform(0.82, 0.96), 2)}


def flatten_numeric(value: Any) -> list[float]:
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, list):
        result: list[float] = []
        for item in value:
            result.extend(flatten_numeric(item))
        return result
    return []


def normalize_gesture(value: Any) -> str:
    gesture = str(value or "").strip().lower().replace("_", "-")
    aliases = {
        "relax": "rest",
        "idle": "rest",
        "open": "open-palm",
        "palm": "open-palm",
        "open-palm": "open-palm",
        "openpalm": "open-palm",
        "handopen": "open-palm",
        "open-hand": "open-palm",
        "grip": "fist",
        "jump": "fist",
        "high": "open-palm",
        "pinch": "pinch",
        "eat": "pinch",
        "bite": "pinch",
        "index-thumb-pinch": "pinch",
        "thumbindexpinch": "pinch",
    }
    return aliases.get(gesture, gesture)


def normalize_hand(value: Any) -> str | None:
    hand = str(value or "").strip().lower()
    aliases = {"l": "left", "left": "left", "r": "right", "right": "right"}
    return aliases.get(hand)


class EmgRequestHandler(BaseHTTPRequestHandler):
    broadcaster: GestureBroadcaster
    classifier: GestureClassifier

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json({"ok": True})
            return

        if self.path == "/events":
            self._stream_events()
            return

        self.send_error(404, "Not found")

    def do_POST(self) -> None:
        if self.path == "/gesture":
            payload = self._read_json()
            gesture = normalize_gesture(payload.get("gesture"))
            confidence = float(payload.get("confidence", 0.9))
            hand = normalize_hand(payload.get("hand"))
            self._publish_gesture(gesture, confidence, hand)
            return

        if self.path == "/emg":
            payload = self._read_json()
            samples = payload.get("samples", [])
            prediction = self.classifier.predict(samples)
            gesture = normalize_gesture(prediction.get("gesture"))
            confidence = float(prediction.get("confidence", 0.8))
            hand = normalize_hand(payload.get("hand") or prediction.get("hand"))
            self._publish_gesture(gesture, confidence, hand)
            return

        self.send_error(404, "Not found")

    def _publish_gesture(self, gesture: str, confidence: float, hand: str | None = None) -> None:
        if gesture not in GESTURES:
            self.send_error(400, f"Unknown gesture: {gesture}")
            return

        payload = {
            "gesture": gesture,
            "confidence": max(0.0, min(confidence, 1.0)),
            "timestamp": time.time(),
        }
        if hand:
            payload["hand"] = hand
        self.broadcaster.publish(payload)
        self._send_json({"ok": True, **payload})

    def _stream_events(self) -> None:
        self.send_response(200)
        self._send_cors_headers()
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        self.wfile.write(b"event: status\ndata: online\n\n")
        self.wfile.flush()

        client = self.broadcaster.subscribe()
        try:
            while True:
                try:
                    payload = client.get(timeout=15)
                    message = f"event: gesture\ndata: {json.dumps(payload)}\n\n"
                except queue.Empty:
                    message = ": keepalive\n\n"
                self.wfile.write(message.encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass
        finally:
            self.broadcaster.unsubscribe(client)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8") if length else "{}"
        return json.loads(raw or "{}")

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
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def log_message(self, format: str, *args: Any) -> None:
        return


def run(host: str, port: int, model_path: str | None) -> None:
    EmgRequestHandler.broadcaster = GestureBroadcaster()
    EmgRequestHandler.classifier = GestureClassifier(model_path)
    server = ThreadingHTTPServer((host, port), EmgRequestHandler)
    print(f"EMG API running at http://{host}:{port}")
    print("Open index.html, then POST EMG windows to /emg or gestures to /gesture.")
    server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local EMG-to-game gesture bridge.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    parser.add_argument("--model", default=None, help="Optional trained model path.")
    args = parser.parse_args()
    run(args.host, args.port, args.model)
