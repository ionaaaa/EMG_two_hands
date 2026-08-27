"""Trial list helpers for EMG gesture collection sessions."""

from __future__ import annotations

import random
from dataclasses import dataclass


COLLECTION_GESTURES = ("fist", "finger_spread", "thumb_index_pinch")
GESTURE_DISPLAY_NAMES = {
    "fist": "全力握拳",
    "finger_spread": "五指完全张开",
    "thumb_index_pinch": "拇食两指轻捏",
}


def gesture_display_name(gesture: str) -> str:
    return GESTURE_DISPLAY_NAMES.get(gesture, gesture)


@dataclass
class CollectionTrial:
    trial_id: str
    gesture: str
    status: str = "pending"
    start_time: float | None = None
    end_time: float | None = None


def build_trial_list(
    trials_per_gesture: int,
    *,
    randomize: bool = True,
    gestures: tuple[str, ...] = COLLECTION_GESTURES,
) -> list[CollectionTrial]:
    gesture_order = [gesture for gesture in gestures for _ in range(int(trials_per_gesture))]
    if randomize:
        random.shuffle(gesture_order)
    return [
        CollectionTrial(trial_id=f"{index + 1:04d}", gesture=gesture)
        for index, gesture in enumerate(gesture_order)
    ]
