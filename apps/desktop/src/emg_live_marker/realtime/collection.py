"""Trial list helpers for EMG gesture collection sessions."""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Literal

import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal

from emg_live_marker.device.protocol import EMG_FS
from emg_live_marker.realtime.classroom_storage import ClassroomStorage
from emg_live_marker.realtime.recorder import SessionRecorder

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


CollectionPhase = Literal["idle", "rest_before", "gesture", "rest_after", "paused", "done"]


@dataclass(frozen=True)
class CollectionPlan:
    subject_id: str
    side: str
    course_id: str
    trials_per_gesture: int
    gestures: tuple[str, ...]
    gesture_names: dict[str, str]
    rest_before_s: float
    hold_s: float
    rest_after_s: float
    randomize: bool
    min_sample_ratio: float = 0.8


@dataclass(frozen=True)
class CollectionSnapshot:
    phase: CollectionPhase
    prompt: str
    remaining_s: float
    completed: int
    total: int
    current_gesture: str | None
    active: bool
    paused: bool
    message: str = ""


class CollectionController(QObject):
    """UI-independent student collection workflow using the existing recorder.

    It owns only trial timing, event markers and quality checks. Device I/O and
    packet parsing remain in ``DeviceCheckService`` / ``SerialSource``.
    """

    state_changed = Signal(object)
    finished = Signal(object)

    def __init__(
        self,
        *,
        recorder: SessionRecorder | None = None,
        classroom_storage: ClassroomStorage | None = None,
        device_ready: Callable[[], bool] | None = None,
        clock: Callable[[], float] = monotonic,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.recorder = recorder or SessionRecorder()
        self.classroom_storage = classroom_storage
        self.device_ready = device_ready or (lambda: True)
        self.clock = clock
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._advance)
        self._display_timer = QTimer(self)
        self._display_timer.setInterval(100)
        self._display_timer.timeout.connect(self._emit)
        self.plan: CollectionPlan | None = None
        self.session_dir: Path | None = None
        self.session_id = ""
        self.trials: list[CollectionTrial] = []
        self.current_index = -1
        self.current: CollectionTrial | None = None
        self.phase: CollectionPhase = "idle"
        self.active = False
        self.paused = False
        self._deadline = 0.0
        self._remaining_s = 0.0
        self._step: Callable[[], None] | None = None
        self._gesture_samples: list[np.ndarray] = []
        self._latest_sample_index: int | None = None
        self._invalid_count = 0
        self._repeated_count = 0
        self._prompt = ""

    @property
    def completed_count(self) -> int:
        return sum(trial.status == "completed" for trial in self.trials)

    def start(self, plan: CollectionPlan, dataset_root: Path, course_config: dict) -> bool:
        if self.active or not self.device_ready():
            return False
        self.plan = plan
        self.trials = build_trial_list(
            plan.trials_per_gesture,
            randomize=plan.randomize,
            gestures=plan.gestures,
        )
        self.current_index = -1
        self._invalid_count = 0
        self._repeated_count = 0
        if self.classroom_storage is None:
            self.session_id, self.session_dir = self._new_session_target(
                dataset_root, plan.subject_id
            )
        else:
            prefix = datetime.now(timezone.utc).strftime("student_%Y%m%d_%H%M%S")
            try:
                self.session_id, self.session_dir = self.classroom_storage.next_session_path(
                    plan.subject_id, prefix
                )
                self.classroom_storage.set_active_group(plan.subject_id)
            except (OSError, ValueError):
                self.session_dir = None
                return False
        metadata = {
            "anonymous_id": plan.subject_id,
            "subject_id": plan.subject_id,
            "session_id": self.session_id,
            "selected_hand": plan.side,
            "course_id": plan.course_id,
            "trials_per_gesture": plan.trials_per_gesture,
            "gesture_order": [trial.gesture for trial in self.trials],
            "gestures": list(plan.gestures),
            "gesture_display_names": plan.gesture_names,
            "course_config": course_config,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "collection_status": "in_progress",
        }
        try:
            self.recorder.start(session_dir=self.session_dir, metadata=metadata, collection_mode=True)
        except Exception:  # noqa: BLE001 - recorder failure is shown as a concise UI message.
            self.session_dir = None
            return False
        self.active = True
        self.paused = False
        self._next_trial()
        return True

    def on_emg_packets(self, packets: list[object]) -> None:
        if not self.active or not packets:
            return
        self.recorder.write_emg_packets(packets)  # Existing recorder/file format.
        self._latest_sample_index = getattr(packets[-1], "sample_index", None)
        if self.phase == "gesture" and self.current is not None:
            self._gesture_samples.extend(
                np.asarray(getattr(packet, "values_uv", ()), dtype=np.float64) for packet in packets
            )

    def pause(self, message: str = "已暂停") -> None:
        if not self.active or self.paused:
            return
        self._remaining_s = max(0.0, self._deadline - self.clock())
        self._timer.stop()
        self._display_timer.stop()
        self.paused = True
        self.phase = "paused"
        self._emit(message=message)

    def resume(self) -> None:
        if not self.active or not self.paused:
            return
        if not self.device_ready():
            self._emit(message="请调整手环后再继续")
            return
        self.paused = False
        if self._step is None:
            self._next_trial()
            return
        self._schedule(self._remaining_s, self._step)

    def repeat_current_or_last(self) -> None:
        target = self.current
        if target is None and self.current_index >= 0:
            target = self.trials[self.current_index]
        if target is None:
            return
        target.status = "repeated"
        self._repeated_count += 1
        self._write_event(target, "repeat_trial", "repeated")
        replacement = CollectionTrial(self._next_trial_id(), target.gesture)
        insert_at = self.current_index + 1
        self.trials.insert(insert_at, replacement)
        self._timer.stop()
        self.current = None
        self.paused = False
        self._schedule(0.0, self._next_trial)

    def end(self, status: str = "partial") -> None:
        if not self.active:
            return
        self._timer.stop()
        if self.current is not None:
            self.current.status = "interrupted"
            self._write_event(self.current, "trial_end", "interrupted")
        self._finalize(status)

    def check_device_state(self) -> None:
        if self.active and not self.paused and not self.device_ready():
            self.pause("手环状态异常，已自动暂停，请调整后继续或重做")

    def _next_trial(self) -> None:
        if not self.active:
            return
        if not self.device_ready():
            self.pause("请调整手环后继续")
            return
        self.current_index += 1
        if self.current_index >= len(self.trials):
            self._finalize("completed")
            return
        self.current = self.trials[self.current_index]
        self.current.status = "pending"
        self._write_event(self.current, "trial_start")
        self._set_phase("rest_before", "请放松", self.plan.rest_before_s, self._gesture_start)

    def _gesture_start(self) -> None:
        if self.current is None:
            return
        self._gesture_samples = []
        self._write_event(self.current, "gesture_start")
        self._set_phase("gesture", f"请{self.plan.gesture_names[self.current.gesture]}", self.plan.hold_s, self._gesture_end)

    def _gesture_end(self) -> None:
        if self.current is None:
            return
        self._write_event(self.current, "gesture_end")
        self._set_phase("rest_after", "请放松", self.plan.rest_after_s, self._trial_end)

    def _trial_end(self) -> None:
        if self.current is None:
            return
        valid = self._trial_is_valid()
        self._write_event(self.current, "trial_end", "completed" if valid else "invalid")
        self.current.status = "completed" if valid else "invalid"
        if not valid:
            self._invalid_count += 1
            self.trials.insert(self.current_index + 1, CollectionTrial(self._next_trial_id(), self.current.gesture))
        self.current = None
        self._emit(message="" if valid else "本次数据不完整，请重新完成该动作")
        self._schedule(0.4, self._next_trial)

    def _trial_is_valid(self) -> bool:
        if self.plan is None:
            return False
        samples = np.asarray(self._gesture_samples, dtype=np.float64)
        required = int(np.ceil(EMG_FS * self.plan.hold_s * self.plan.min_sample_ratio))
        if samples.ndim != 2 or samples.shape[0] < required or samples.shape[1] != 8:
            return False
        if not np.isfinite(samples).all() or np.all(samples == 0.0):
            return False
        return bool(np.all(np.std(samples, axis=0) >= 0.5))

    def _set_phase(self, phase: CollectionPhase, prompt: str, seconds: float, step: Callable[[], None]) -> None:
        self.phase = phase
        self._prompt = prompt
        self._schedule(seconds, step, prompt)

    def _schedule(self, seconds: float, step: Callable[[], None], prompt: str | None = None) -> None:
        self._step = step
        self._remaining_s = max(0.0, seconds)
        self._deadline = self.clock() + self._remaining_s
        self._timer.start(max(1, round(self._remaining_s * 1000)))
        if not self.paused:
            self._display_timer.start()
        self._emit(prompt=prompt)

    def _advance(self) -> None:
        step = self._step
        self._step = None
        if self.active and not self.paused and step is not None:
            step()

    def _write_event(self, trial: CollectionTrial, phase: str, note: str = "") -> None:
        self.recorder.write_collection_event(
            trial_id=trial.trial_id,
            subject_id=self.plan.subject_id,
            session_id=self.session_id,
            gesture=trial.gesture,
            gesture_name=self.plan.gesture_names[trial.gesture],
            phase=phase,
            sample_index=self._latest_sample_index,
            note=note,
        )

    def _emit(self, prompt: str | None = None, message: str = "") -> None:
        current = self.current.gesture if self.current else None
        default_prompt = "采集完成" if self.phase == "done" else (self._prompt or "请准备")
        self.state_changed.emit(
            CollectionSnapshot(
                self.phase,
                prompt or default_prompt,
                max(0.0, self._deadline - self.clock()) if self.active and not self.paused else self._remaining_s,
                self.completed_count,
                sum(trial.status not in {"repeated", "invalid"} for trial in self.trials),
                current,
                self.active,
                self.paused,
                message,
            )
        )

    def _finalize(self, status: str) -> None:
        self.active = False
        self.paused = False
        self.phase = "done"
        self._timer.stop()
        self._display_timer.stop()
        session_dir = self.session_dir
        if session_dir is not None:
            metadata_path = session_dir / "metadata.json"
            try:
                import json

                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                metadata.update(
                    {
                        "collection_status": status,
                        "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "trial_statuses": {trial.trial_id: trial.status for trial in self.trials},
                        "valid_trial_counts": {
                            gesture: sum(
                                trial.gesture == gesture and trial.status == "completed" for trial in self.trials
                            )
                            for gesture in self.plan.gestures
                        },
                        "repeated_trial_count": self._repeated_count,
                        "invalid_trial_count": self._invalid_count,
                    }
                )
                ClassroomStorage.atomic_write_json(metadata_path, metadata)
            finally:
                self.recorder.stop()
        self._emit(message="采集完成并已保存" if status == "completed" else "部分完成，已安全保存")
        self.finished.emit({"status": status, "session_id": self.session_id, "session_dir": session_dir})

    @staticmethod
    def _new_session_target(dataset_root: Path, subject_id: str) -> tuple[str, Path]:
        root = dataset_root / subject_id
        root.mkdir(parents=True, exist_ok=True)
        base = datetime.now(timezone.utc).strftime("student_%Y%m%d_%H%M%S")
        suffix = 1
        candidate = root / base
        while candidate.exists():
            suffix += 1
            candidate = root / f"{base}_{suffix:02d}"
        return candidate.name, candidate

    def _next_trial_id(self) -> str:
        return f"{len(self.trials) + 1:04d}"
