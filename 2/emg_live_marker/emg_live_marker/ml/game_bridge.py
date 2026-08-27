"""Non-blocking bridge from EMG gesture predictions to the browser game API."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor


class GameBridge:
    def __init__(
        self,
        url: str = "http://127.0.0.1:8765/gesture",
        *,
        min_interval_s: float = 0.1,
        timeout_s: float = 0.25,
        confidence_threshold: float = 0.75,
    ) -> None:
        self.url = url
        self.min_interval_s = float(min_interval_s)
        self.timeout_s = float(timeout_s)
        self.confidence_threshold = float(confidence_threshold)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="GameBridge")
        self._last_gesture: str | None = None
        self._last_sent_at = 0.0
        self.last_status = "idle"

    def send_gesture(self, gesture: str, confidence: float) -> bool:
        if confidence < self.confidence_threshold:
            gesture = "rest"
        now = time.monotonic()
        if gesture == self._last_gesture and (now - self._last_sent_at) < self.min_interval_s:
            return False
        self._last_gesture = gesture
        self._last_sent_at = now
        self._executor.submit(self._post_json, gesture, float(confidence))
        return True

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)

    def _post_json(self, gesture: str, confidence: float) -> None:
        payload = json.dumps({"gesture": gesture, "confidence": confidence}).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                response.read()
            self.last_status = "online"
        except (OSError, urllib.error.URLError):
            self.last_status = "offline"
