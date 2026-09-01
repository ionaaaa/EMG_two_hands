"""Guided, real-output control-profile evaluation for student mode."""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from typing import Callable

from PySide6.QtCore import QObject, QTimer, Signal

from emg_live_marker.realtime.student_observation import StudentObservationService

TEST_PHASES = (
    ("rest", "放松"),
    ("fist", "握拳"),
    ("rest", "放松"),
    ("open-palm", "伸掌"),
)


@dataclass(frozen=True)
class ControlTestResult:
    has_data: bool
    false_triggers: int | None
    average_response_seconds: float | None
    wrong_switches: int | None
    stability_level: str


class StudentControlEffectTestService(QObject):
    """Score only stable gestures emitted by the existing realtime decoders."""

    phase_changed = Signal(int, str, str)
    state_changed = Signal(str, str)
    result_changed = Signal(object)

    def __init__(
        self,
        observation_service: StudentObservationService,
        *,
        phase_duration_ms: int = 3000,
        clock: Callable[[], float] | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.observation_service = observation_service
        self.phase_duration_ms = max(250, int(phase_duration_ms))
        self._clock = clock or monotonic
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.advance_phase)
        self.running = False
        self.phase_index = -1
        self.ready_sides: set[str] = set()
        self._phase_started_at = 0.0
        self._last_gesture: dict[str, str | None] = {}
        self._correct_seen: dict[str, bool] = {}
        self._event_count = 0
        self._false_triggers = 0
        self._wrong_switches = 0
        self._response_times: list[float] = []
        self.result = ControlTestResult(False, None, None, None, "暂无结果")
        observation_service.gesture_updated.connect(self.on_gesture)

    @property
    def current_target(self) -> str | None:
        if 0 <= self.phase_index < len(TEST_PHASES):
            return TEST_PHASES[self.phase_index][0]
        return None

    def start(self) -> bool:
        if self.running:
            self.state_changed.emit("running", "控制效果测试正在进行。")
            return False
        self.ready_sides = {
            side for side, ready in self.observation_service.ready_sides.items() if ready
        }
        if not self.ready_sides:
            self.state_changed.emit("error", "请先完成设备检查，至少连接一只手环。")
            return False
        if not self.observation_service.active:
            self.observation_service.start(
                left_ready="left" in self.ready_sides,
                right_ready="right" in self.ready_sides,
            )
        self.running = True
        self.phase_index = 0
        self._event_count = 0
        self._false_triggers = 0
        self._wrong_switches = 0
        self._response_times = []
        self.result = ControlTestResult(False, None, None, None, "暂无结果")
        self.state_changed.emit("running", "测试开始，请按提示完成动作。")
        self._begin_current_phase()
        return True

    def stop(self) -> None:
        if not self.running:
            return
        self.running = False
        self._timer.stop()
        self.state_changed.emit("stopped", "控制效果测试已停止。")

    def advance_phase(self) -> None:
        if not self.running:
            return
        self._timer.stop()
        self.phase_index += 1
        if self.phase_index >= len(TEST_PHASES):
            self._finish()
            return
        self._begin_current_phase()

    def on_gesture(self, side: str, gesture: str, _confidence: float, _probs: dict) -> None:
        if not self.running or side not in self.ready_sides:
            return
        target = self.current_target
        if target is None:
            return
        self._event_count += 1
        gesture = str(gesture)
        if self._last_gesture.get(side) == gesture:
            return
        self._last_gesture[side] = gesture
        if target == "rest" and gesture != "rest":
            self._false_triggers += 1
        if gesture == target:
            if not self._correct_seen.get(side, False):
                self._correct_seen[side] = True
                if target != "rest":
                    self._response_times.append(max(0.0, self._clock() - self._phase_started_at))
        elif self._correct_seen.get(side, False):
            self._wrong_switches += 1

    def _begin_current_phase(self) -> None:
        target, display = TEST_PHASES[self.phase_index]
        self._phase_started_at = self._clock()
        self._last_gesture = {side: None for side in self.ready_sides}
        self._correct_seen = {side: False for side in self.ready_sides}
        self.phase_changed.emit(self.phase_index + 1, target, display)
        self._timer.start(self.phase_duration_ms)

    def _finish(self) -> None:
        self.running = False
        self._timer.stop()
        if self._event_count == 0:
            self.result = ControlTestResult(False, None, None, None, "暂无结果")
            self.state_changed.emit("completed", "测试结束：暂无结果。")
        else:
            average = (
                sum(self._response_times) / len(self._response_times)
                if self._response_times
                else None
            )
            stability = self._stability_level(self._wrong_switches)
            self.result = ControlTestResult(
                True,
                self._false_triggers,
                average,
                self._wrong_switches,
                stability,
            )
            self.state_changed.emit("completed", "控制效果测试完成。")
        self.result_changed.emit(self.result)

    @staticmethod
    def _stability_level(wrong_switches: int) -> str:
        if wrong_switches == 0:
            return "稳定"
        if wrong_switches <= 2:
            return "较稳定"
        return "建议选择更稳定的控制风格"
