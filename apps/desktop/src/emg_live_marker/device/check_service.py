"""Asynchronous, student-safe EMG wristband health checks.

The service reuses :class:`SerialSource` for serial I/O and its packet parser.
It deliberately does not infer left/right from serial-port enumeration order.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from time import monotonic
from typing import Literal, Protocol

import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal

from emg_live_marker.config import DEFAULT_BAUDRATE, DEFAULT_DISPLAY_OUTLIER_UV
from emg_live_marker.device.protocol import EMG_CHANNELS
from emg_live_marker.device.serial_source import SerialSource, list_serial_ports
from emg_live_marker.device.simulator import SimulatedDevice, SimulatorConfig

BraceletSide = Literal["left", "right"]


class CheckReason(str, Enum):
    NO_DEVICE = "no_device"
    SIDE_UNASSIGNED = "side_unassigned"
    NO_EMG_DATA = "no_emg_data"
    INSUFFICIENT_SAMPLES = "insufficient_samples"
    FLAT_SIGNAL = "flat_signal"
    INVALID_VALUES = "invalid_values"
    ABNORMAL_CHANNEL = "abnormal_channel"
    UNSTABLE_RATE = "unstable_rate"
    HEALTHY = "healthy"


class ConnectionState(str, Enum):
    DISCONNECTED = "disconnected"
    CHECKING = "checking"
    CONNECTED = "connected"
    UNASSIGNED = "unassigned"


@dataclass(frozen=True)
class DeviceDescriptor:
    """A candidate device and an optional pre-assigned bracelet side.

    ``side`` must come from a reliable external assignment. The production
    provider cannot fill it because the current protocol and port list contain
    no left/right identity.
    """

    port: str
    side: BraceletSide | None = None


class DeviceProvider(Protocol):
    def list_devices(self) -> Sequence[DeviceDescriptor]: ...


class SerialDeviceProvider:
    """Production provider that enumerates ports without assigning a side."""

    def list_devices(self) -> Sequence[DeviceDescriptor]:
        return [DeviceDescriptor(port=port) for port in list_serial_ports()]


class AssignedSerialDeviceProvider:
    """Enumerate only teacher-assigned ports, preserving explicit left/right sides."""

    def __init__(
        self,
        left_port: str,
        right_port: str,
        *,
        port_lister: Callable[[], Sequence[str]] = list_serial_ports,
    ) -> None:
        self.left_port = str(left_port).strip()
        self.right_port = str(right_port).strip()
        self._port_lister = port_lister
        self.assignment_configured = bool(
            self.left_port and self.right_port and self.left_port != self.right_port
        )

    def list_devices(self) -> Sequence[DeviceDescriptor]:
        if not self.assignment_configured:
            return ()
        available = {str(port).strip() for port in self._port_lister()}
        return tuple(
            DeviceDescriptor(port=port, side=side)
            for side, port in (("left", self.left_port), ("right", self.right_port))
            if port in available
        )


@dataclass(frozen=True)
class DeviceCheckThresholds:
    observe_duration_ms: int = 1500
    min_samples: int = 50
    min_rate_sps: float = 200.0
    max_rate_sps: float = 300.0
    max_rate_spread_sps: float = 40.0
    flat_channel_std_uv: float = 0.5
    max_abs_value_uv: float = DEFAULT_DISPLAY_OUTLIER_UV

    @classmethod
    def from_config(cls, config: object) -> DeviceCheckThresholds:
        if not isinstance(config, dict):
            return cls()
        values = dict(config.get("student_device_check", {})) if isinstance(config.get("student_device_check"), dict) else {}
        defaults = cls()
        try:
            return cls(
                observe_duration_ms=max(250, int(values.get("observe_duration_ms", defaults.observe_duration_ms))),
                min_samples=max(1, int(values.get("min_samples", defaults.min_samples))),
                min_rate_sps=float(values.get("min_rate_sps", defaults.min_rate_sps)),
                max_rate_sps=float(values.get("max_rate_sps", defaults.max_rate_sps)),
                max_rate_spread_sps=max(0.0, float(values.get("max_rate_spread_sps", defaults.max_rate_spread_sps))),
                flat_channel_std_uv=max(0.0, float(values.get("flat_channel_std_uv", defaults.flat_channel_std_uv))),
                max_abs_value_uv=max(1.0, float(values.get("max_abs_value_uv", defaults.max_abs_value_uv))),
            )
        except (TypeError, ValueError):
            return cls()


@dataclass(frozen=True)
class SideCheckResult:
    side: BraceletSide
    connection: ConnectionState
    reason: CheckReason
    received_emg: bool = False
    valid_samples: bool = False
    signal_healthy: bool = False
    rate_stable: bool = False

    @property
    def ready_for_collection(self) -> bool:
        return (
            self.connection is ConnectionState.CONNECTED
            and self.reason is CheckReason.HEALTHY
            and self.received_emg
            and self.valid_samples
            and self.signal_healthy
            and self.rate_stable
        )


@dataclass(frozen=True)
class DeviceCheckResult:
    left: SideCheckResult
    right: SideCheckResult
    checking: bool
    message: str
    simulate: bool = False

    @property
    def collection_ready(self) -> bool:
        return self.left.ready_for_collection or self.right.ready_for_collection

    @property
    def signal_status(self) -> str:
        if self.checking:
            return "检测中"
        if self.collection_ready:
            return "正常"
        return "请调整佩戴"

    @property
    def stream_status(self) -> str:
        if self.checking:
            return "检测中"
        return "稳定" if self.collection_ready else "不稳定"


SourceFactory = Callable[[DeviceDescriptor, BraceletSide, QObject], object]


class DeviceCheckService(QObject):
    """Enumerate, observe, and assess zero to two reliably assigned devices."""

    result_changed = Signal(object)
    emg_packets_received = Signal(str, list)

    def __init__(
        self,
        *,
        provider: DeviceProvider | None = None,
        source_factory: SourceFactory | None = None,
        thresholds: DeviceCheckThresholds | None = None,
        simulate: bool = False,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.provider = provider or SerialDeviceProvider()
        self.source_factory = source_factory or (
            self._create_simulated_source if simulate else self._create_serial_source
        )
        self.thresholds = thresholds or DeviceCheckThresholds()
        self.simulate = bool(simulate)
        self._sources: dict[BraceletSide, object] = {}
        self._signal_handlers: list[Callable[..., object]] = []
        self._descriptors: dict[BraceletSide, DeviceDescriptor] = {}
        self._connected: dict[BraceletSide, bool] = {"left": False, "right": False}
        self._samples: dict[BraceletSide, list[np.ndarray]] = {"left": [], "right": []}
        self._first_emg_at: dict[BraceletSide, float | None] = {"left": None, "right": None}
        self._rates: dict[BraceletSide, list[float]] = {"left": [], "right": []}
        self._unassigned_found = False
        self._checking = False
        self._generation = 0
        self._enumeration_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="DeviceCheck")
        self._enumeration_future: Future[Sequence[DeviceDescriptor]] | None = None
        self._enumeration_generation = 0
        self._enumeration_poll_timer = QTimer(self)
        self._enumeration_poll_timer.setInterval(20)
        self._enumeration_poll_timer.timeout.connect(self._poll_enumeration)
        self._finish_timer = QTimer(self)
        self._finish_timer.setSingleShot(True)
        self._finish_timer.timeout.connect(self._finish_check)
        self.result = self._blank_result(message="尚未开始检查")

    def start(self) -> None:
        """Start a new non-blocking check after releasing any previous sources."""

        self.stop()
        self._generation += 1
        self._checking = True
        self._clear_observations()
        self.result = self._blank_result(message="正在检测手环和信号", checking=True)
        self.result_changed.emit(self.result)
        if self.simulate:
            QTimer.singleShot(0, self._start_simulated_sources)
            return
        self._enumeration_generation = self._generation
        self._enumeration_future = self._enumeration_executor.submit(self.provider.list_devices)
        self._enumeration_poll_timer.start()

    def stop(self) -> None:
        """Stop timers and sources; safe to call before each re-check or close."""

        self._finish_timer.stop()
        self._enumeration_poll_timer.stop()
        self._checking = False
        for source in tuple(self._sources.values()):
            self._stop_source(source)
        self._sources.clear()
        self._signal_handlers.clear()
        self._descriptors.clear()
        self._connected = {"left": False, "right": False}

    def close(self) -> None:
        self.stop()
        self._enumeration_executor.shutdown(wait=False, cancel_futures=True)

    def _clear_observations(self) -> None:
        self._samples = {"left": [], "right": []}
        self._first_emg_at = {"left": None, "right": None}
        self._rates = {"left": [], "right": []}
        self._unassigned_found = False

    def _start_simulated_sources(self) -> None:
        if not self._checking:
            return
        descriptors = (
            DeviceDescriptor(port="simulated-left", side="left"),
            DeviceDescriptor(port="simulated-right", side="right"),
        )
        self._on_devices_found(descriptors)

    def _poll_enumeration(self) -> None:
        future = self._enumeration_future
        if future is None or not future.done():
            return
        self._enumeration_poll_timer.stop()
        self._enumeration_future = None
        if not self._checking or self._enumeration_generation != self._generation:
            return
        try:
            descriptors = tuple(future.result())
        except Exception:  # noqa: BLE001 - provider implementations are external.
            self._on_enumeration_failed("设备枚举失败")
            return
        self._on_devices_found(descriptors)

    def _on_devices_found(self, descriptors: Sequence[DeviceDescriptor]) -> None:
        if not self._checking:
            return
        assigned: dict[BraceletSide, DeviceDescriptor] = {}
        assigned_ports: set[str] = set()
        for descriptor in descriptors:
            if (
                descriptor.side is None
                or descriptor.side in assigned
                or descriptor.port in assigned_ports
            ):
                self._unassigned_found = True
                continue
            assigned[descriptor.side] = descriptor
            assigned_ports.add(descriptor.port)

        if not assigned:
            assigned_provider = isinstance(self.provider, AssignedSerialDeviceProvider)
            message = (
                "未检测到已分配手环，请老师检查连接"
                if assigned_provider and self.provider.assignment_configured
                else None
            )
            self._finish_check(message=message)
            return
        self._descriptors = assigned
        for side, descriptor in assigned.items():
            source = self.source_factory(descriptor, side, self)
            self._sources[side] = source
            self._connect_source(source, side)
            self._start_source(source, descriptor)
        self._finish_timer.start(self.thresholds.observe_duration_ms)

    def _on_enumeration_failed(self, _message: str) -> None:
        if not self._checking:
            return
        self._finish_check(message="未检测到手环，请确认手环已开机并连接电脑")

    def _connect_source(self, source: object, side: BraceletSide) -> None:
        if hasattr(source, "emg_packets"):
            handler = lambda packets, selected_side=side: self._on_emg(selected_side, packets)
            self._signal_handlers.append(handler)
            source.emg_packets.connect(handler)
        if hasattr(source, "stats_updated"):
            handler = lambda stats, selected_side=side: self._on_stats(selected_side, stats)
            self._signal_handlers.append(handler)
            source.stats_updated.connect(handler)
        if hasattr(source, "connected"):
            handler = lambda _port, selected_side=side: self._on_connected(selected_side)
            self._signal_handlers.append(handler)
            source.connected.connect(handler)
        if hasattr(source, "disconnected"):
            handler = lambda selected_side=side: self._on_disconnected(selected_side)
            self._signal_handlers.append(handler)
            source.disconnected.connect(handler)
        if hasattr(source, "error_occurred"):
            handler = lambda _message, selected_side=side: self._on_disconnected(selected_side)
            self._signal_handlers.append(handler)
            source.error_occurred.connect(handler)

    def _start_source(self, source: object, descriptor: DeviceDescriptor) -> None:
        if isinstance(source, SerialSource):
            source.connect_port(descriptor.port, DEFAULT_BAUDRATE)
            return
        if hasattr(source, "start"):
            source.start()
            self._on_connected(descriptor.side)  # Simulated and injected sources have no port signal.

    def _stop_source(self, source: object) -> None:
        if isinstance(source, SerialSource):
            source.disconnect_port()
            source.deleteLater()
        elif hasattr(source, "stop"):
            source.stop()

    @staticmethod
    def _create_serial_source(
        _descriptor: DeviceDescriptor,
        _side: BraceletSide,
        parent: QObject,
    ) -> SerialSource:
        return SerialSource(parent=parent)

    @staticmethod
    def _create_simulated_source(
        _descriptor: DeviceDescriptor,
        side: BraceletSide,
        parent: QObject,
    ) -> SimulatedDevice:
        return SimulatedDevice(config=SimulatorConfig(seed=1 if side == "left" else 2), parent=parent)

    def _on_connected(self, side: BraceletSide | None) -> None:
        if side is not None:
            self._connected[side] = True

    def _on_disconnected(self, side: BraceletSide) -> None:
        self._connected[side] = False
        if self._checking:
            self._publish_progress()
        else:
            self.result = self._build_result(checking=False, message=None)
            self.result_changed.emit(self.result)

    def _on_emg(self, side: BraceletSide, packets: Sequence[object]) -> None:
        values = [np.asarray(getattr(packet, "values_uv", ()), dtype=np.float64) for packet in packets]
        if values and self._first_emg_at[side] is None:
            self._first_emg_at[side] = monotonic()
        self._samples[side].extend(values)
        self.emg_packets_received.emit(side, list(packets))
        if self._checking:
            self._publish_progress()

    def _on_stats(self, side: BraceletSide, stats: object) -> None:
        if isinstance(stats, dict) and "emg_rate_sps" in stats:
            try:
                self._rates[side].append(float(stats["emg_rate_sps"]))
            except (TypeError, ValueError):
                pass

    def _publish_progress(self) -> None:
        self.result = self._build_result(checking=True, message="正在检测手环和信号")
        self.result_changed.emit(self.result)

    def _finish_check(self, message: str | None = None) -> None:
        if not self._checking:
            return
        self._checking = False
        self._finish_timer.stop()
        self.result = self._build_result(checking=False, message=message)
        self.result_changed.emit(self.result)

    def _build_result(self, *, checking: bool, message: str | None) -> DeviceCheckResult:
        left = self.assess_side("left", checking=checking)
        right = self.assess_side("right", checking=checking)
        return DeviceCheckResult(
            left=left,
            right=right,
            checking=checking,
            message=message or self._message_for(left, right, checking),
            simulate=self.simulate,
        )

    def _blank_result(self, *, message: str, checking: bool = False) -> DeviceCheckResult:
        return DeviceCheckResult(
            left=SideCheckResult("left", ConnectionState.CHECKING if checking else ConnectionState.DISCONNECTED, CheckReason.NO_DEVICE),
            right=SideCheckResult("right", ConnectionState.CHECKING if checking else ConnectionState.DISCONNECTED, CheckReason.NO_DEVICE),
            checking=checking,
            message=message,
            simulate=self.simulate,
        )

    def assess_side(self, side: BraceletSide, *, checking: bool = False) -> SideCheckResult:
        descriptor = self._descriptors.get(side)
        if descriptor is None:
            if self._unassigned_found:
                return SideCheckResult(side, ConnectionState.UNASSIGNED, CheckReason.SIDE_UNASSIGNED)
            return SideCheckResult(side, ConnectionState.DISCONNECTED, CheckReason.NO_DEVICE)
        if checking and not self._connected[side]:
            return SideCheckResult(side, ConnectionState.CHECKING, CheckReason.NO_EMG_DATA)
        if not self._connected[side]:
            return SideCheckResult(side, ConnectionState.DISCONNECTED, CheckReason.NO_EMG_DATA)
        samples = self._sample_matrix(side)
        if samples.size == 0:
            return SideCheckResult(side, ConnectionState.CONNECTED, CheckReason.NO_EMG_DATA)
        if samples.ndim != 2 or samples.shape[1] != EMG_CHANNELS or not np.isfinite(samples).all():
            return SideCheckResult(side, ConnectionState.CONNECTED, CheckReason.INVALID_VALUES, received_emg=True)
        if samples.shape[0] < self.thresholds.min_samples:
            return SideCheckResult(side, ConnectionState.CONNECTED, CheckReason.INSUFFICIENT_SAMPLES, received_emg=True)
        if np.max(np.abs(samples)) > self.thresholds.max_abs_value_uv:
            return SideCheckResult(side, ConnectionState.CONNECTED, CheckReason.ABNORMAL_CHANNEL, received_emg=True)
        channel_std = np.std(samples, axis=0)
        if np.any(channel_std < self.thresholds.flat_channel_std_uv):
            return SideCheckResult(side, ConnectionState.CONNECTED, CheckReason.FLAT_SIGNAL, received_emg=True, valid_samples=True)
        if not self._rate_is_stable(side):
            return SideCheckResult(
                side,
                ConnectionState.CONNECTED,
                CheckReason.UNSTABLE_RATE,
                received_emg=True,
                valid_samples=True,
                signal_healthy=True,
            )
        return SideCheckResult(
            side,
            ConnectionState.CONNECTED,
            CheckReason.HEALTHY,
            received_emg=True,
            valid_samples=True,
            signal_healthy=True,
            rate_stable=True,
        )

    def _sample_matrix(self, side: BraceletSide) -> np.ndarray:
        samples = self._samples[side]
        if not samples:
            return np.empty((0, EMG_CHANNELS), dtype=np.float64)
        try:
            return np.vstack(samples)
        except ValueError:
            return np.asarray(samples, dtype=np.float64)

    def _rate_is_stable(self, side: BraceletSide) -> bool:
        rates = np.asarray(self._rates[side], dtype=np.float64)
        if rates.size == 0:
            first_emg_at = self._first_emg_at[side]
            elapsed = monotonic() - first_emg_at if first_emg_at is not None else 0.0
            if elapsed <= 0.0:
                return False
            rates = np.asarray([self._sample_matrix(side).shape[0] / elapsed], dtype=np.float64)
        if not np.isfinite(rates).all():
            return False
        return bool(
            np.all((rates >= self.thresholds.min_rate_sps) & (rates <= self.thresholds.max_rate_sps))
            and np.ptp(rates) <= self.thresholds.max_rate_spread_sps
        )

    @staticmethod
    def _message_for(left: SideCheckResult, right: SideCheckResult, checking: bool) -> str:
        if checking:
            return "正在检测手环和信号"
        sides = (left, right)
        if all(result.reason is CheckReason.NO_DEVICE for result in sides):
            return "未检测到手环，请确认手环已开机并连接电脑"
        if any(result.reason is CheckReason.SIDE_UNASSIGNED for result in sides):
            return "无法确认左右手，请老师在教师模式中完成分配"
        if any(result.ready_for_collection for result in sides):
            return "检查完成，已检测到可用于后续课程的手环"
        if any(result.reason is CheckReason.NO_EMG_DATA for result in sides):
            return "手环已连接，但未收到有效肌电数据，请确认设备状态"
        return "基础信号检查未通过，请调整佩戴后重新检测"
