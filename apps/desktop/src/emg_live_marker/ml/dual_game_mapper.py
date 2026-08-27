"""Map independent left/right gestures to held game keys."""

from __future__ import annotations

import sys
from typing import Literal, Protocol

BraceletSide = Literal["left", "right"]


class KeySink(Protocol):
    def key_down(self, key: str) -> None: ...

    def key_up(self, key: str) -> None: ...


class NoopKeySink:
    def key_down(self, key: str) -> None:
        _ = key

    def key_up(self, key: str) -> None:
        _ = key


class WindowsKeySink:
    _VK = {
        "A": 0x41,
        "D": 0x44,
        "W": 0x57,
        "Space": 0x20,
    }
    _KEYEVENTF_KEYUP = 0x0002

    def __init__(self) -> None:
        if not sys.platform.startswith("win"):
            raise RuntimeError("WindowsKeySink is only available on Windows")
        import ctypes

        self._user32 = ctypes.windll.user32

    def key_down(self, key: str) -> None:
        self._send(key, key_up=False)

    def key_up(self, key: str) -> None:
        self._send(key, key_up=True)

    def _send(self, key: str, *, key_up: bool) -> None:
        vk = self._VK.get(key)
        if vk is None:
            return
        flags = self._KEYEVENTF_KEYUP if key_up else 0
        self._user32.keybd_event(vk, 0, flags, 0)


def default_key_sink() -> KeySink:
    if sys.platform.startswith("win"):
        try:
            return WindowsKeySink()
        except RuntimeError:
            return NoopKeySink()
    return NoopKeySink()


class DualGameMapper:
    def __init__(self, key_sink: KeySink | None = None) -> None:
        self.key_sink = key_sink or default_key_sink()
        self._mapping: dict[BraceletSide, dict[str, tuple[str, ...]]] = {
            "left": {
                "fist": ("A",),
                "open-palm": ("D",),
            },
            "right": {
                "fist": ("Space",),
                "open-palm": ("W",),
            },
        }
        self._pressed: dict[BraceletSide, set[str]] = {"left": set(), "right": set()}
        self.enabled = False

    def update(self, left_gesture: str, right_gesture: str, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        if not self.enabled:
            self.release_all()
            return
        self.update_side("left", left_gesture)
        self.update_side("right", right_gesture)

    def update_side(self, side: BraceletSide, gesture: str, enabled: bool = True) -> None:
        if not enabled:
            self.release_side(side)
            return
        desired = set(self._mapping[side].get(str(gesture), ()))
        pressed = self._pressed[side]
        for key in sorted(pressed - desired):
            self.key_sink.key_up(key)
            pressed.remove(key)
        for key in sorted(desired - pressed):
            self.key_sink.key_down(key)
            pressed.add(key)

    def release_side(self, side: BraceletSide) -> None:
        pressed = self._pressed[side]
        for key in sorted(pressed):
            self.key_sink.key_up(key)
        pressed.clear()

    def release_all(self) -> None:
        self.release_side("left")
        self.release_side("right")

    def pressed_keys(self, side: BraceletSide) -> set[str]:
        return set(self._pressed[side])
