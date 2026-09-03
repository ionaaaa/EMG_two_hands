import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, Signal
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from emg_live_marker.device.check_service import (
    AssignedSerialDeviceProvider,
    CheckReason,
    ConnectionState,
    DeviceCheckResult,
    DeviceCheckService,
    DeviceCheckThresholds,
    DeviceDescriptor,
    SideCheckResult,
)
from emg_live_marker.paths import resolve_project_paths
from emg_live_marker.ui.student_window import StudentMainWindow


class FakePacket:
    def __init__(self, values: np.ndarray) -> None:
        self.values_uv = values


class FakeProvider:
    def __init__(self, descriptors: list[DeviceDescriptor]) -> None:
        self.descriptors = descriptors

    def list_devices(self) -> list[DeviceDescriptor]:
        return self.descriptors


class FakeSource(QObject):
    emg_packets = Signal(list)
    stats_updated = Signal(dict)
    connected = Signal(str)
    disconnected = Signal()
    error_occurred = Signal(str)

    def __init__(self, packets: list[FakePacket], rates: list[float], parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.packets = packets
        self.rates = rates
        self.started = 0
        self.stopped = 0
        self.running = False

    def start(self) -> None:
        self.started += 1
        self.running = True
        self.connected.emit("fake")
        if self.packets:
            self.emg_packets.emit(self.packets)
        for rate in self.rates:
            self.stats_updated.emit({"emg_rate_sps": rate})

    def stop(self) -> None:
        self.stopped += 1
        self.running = False
        self.disconnected.emit()

    def is_running(self) -> bool:
        return self.running


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


@pytest.fixture
def thresholds() -> DeviceCheckThresholds:
    return DeviceCheckThresholds(observe_duration_ms=250, min_samples=10)


def healthy_packets(count: int = 60) -> list[FakePacket]:
    rng = np.random.default_rng(7)
    return [FakePacket(rng.normal(0.0, 20.0, size=8)) for _ in range(count)]


def run_check(
    app: QApplication,
    thresholds: DeviceCheckThresholds,
    descriptors: list[DeviceDescriptor],
    *,
    packets_by_side: dict[str, list[FakePacket]] | None = None,
    rates_by_side: dict[str, list[float]] | None = None,
) -> tuple[DeviceCheckService, list[FakeSource]]:
    sources: list[FakeSource] = []
    packets_by_side = packets_by_side or {}
    rates_by_side = rates_by_side or {}

    def source_factory(_descriptor: DeviceDescriptor, side: str, parent: QObject) -> FakeSource:
        source = FakeSource(packets_by_side.get(side, healthy_packets()), rates_by_side.get(side, [250.0]), parent)
        sources.append(source)
        return source

    service = DeviceCheckService(
        provider=FakeProvider(descriptors),
        source_factory=source_factory,
        thresholds=thresholds,
    )
    service.start()
    QTest.qWait(thresholds.observe_duration_ms + 150)
    app.processEvents()
    return service, sources


def run_assigned_check(
    app: QApplication,
    thresholds: DeviceCheckThresholds,
    provider: AssignedSerialDeviceProvider,
) -> tuple[DeviceCheckService, list[FakeSource]]:
    sources: list[FakeSource] = []

    def source_factory(_descriptor: DeviceDescriptor, side: str, parent: QObject) -> FakeSource:
        source = FakeSource(healthy_packets(), [250.0], parent)
        sources.append(source)
        return source

    service = DeviceCheckService(
        provider=provider,
        source_factory=source_factory,
        thresholds=thresholds,
    )
    service.start()
    QTest.qWait(thresholds.observe_duration_ms + 150)
    app.processEvents()
    return service, sources


def test_no_device_returns_clear_no_device_result(app, thresholds) -> None:
    service, _ = run_check(app, thresholds, [])
    try:
        assert service.result.left.reason is CheckReason.NO_DEVICE
        assert "未检测到手环" in service.result.message
    finally:
        service.close()


@pytest.mark.parametrize("side", ["left", "right"])
def test_single_assigned_device_marks_the_correct_side(app, thresholds, side) -> None:
    service, _ = run_check(app, thresholds, [DeviceDescriptor(f"fake-{side}", side=side)])
    try:
        checked = service.result.left if side == "left" else service.result.right
        missing = service.result.right if side == "left" else service.result.left
        assert checked.reason is CheckReason.HEALTHY
        assert missing.reason is CheckReason.NO_DEVICE
        assert service.result.collection_ready is True
    finally:
        service.close()


def test_two_assigned_devices_are_checked_independently(app, thresholds) -> None:
    service, _ = run_check(
        app,
        thresholds,
        [DeviceDescriptor("fake-left", "left"), DeviceDescriptor("fake-right", "right")],
    )
    try:
        assert service.result.left.reason is CheckReason.HEALTHY
        assert service.result.right.reason is CheckReason.HEALTHY
    finally:
        service.close()


def test_unassigned_devices_are_not_silently_mapped(app, thresholds) -> None:
    service, sources = run_check(
        app,
        thresholds,
        [DeviceDescriptor("fake-one"), DeviceDescriptor("fake-two")],
    )
    try:
        assert not sources
        assert service.result.left.connection is ConnectionState.UNASSIGNED
        assert "无法确认左右手" in service.result.message
        assert service.result.collection_ready is False
    finally:
        service.close()


def test_assigned_provider_keeps_left_right_when_enumeration_order_changes() -> None:
    provider = AssignedSerialDeviceProvider(
        "COM6",
        "COM7",
        port_lister=lambda: ["COM13", "COM7", "COM12", "COM6"],
    )
    assert provider.list_devices() == (
        DeviceDescriptor("COM6", "left"),
        DeviceDescriptor("COM7", "right"),
    )


def test_assigned_provider_ignores_unassigned_ports_and_allows_one_ready_side(app, thresholds) -> None:
    provider = AssignedSerialDeviceProvider(
        "COM6",
        "COM7",
        port_lister=lambda: ["COM12", "COM6", "COM13"],
    )
    service, sources = run_assigned_check(app, thresholds, provider)
    try:
        assert [source for source in sources]
        assert service.result.left.reason is CheckReason.HEALTHY
        assert service.result.right.reason is CheckReason.NO_DEVICE
        assert service.result.collection_ready is True
        assert set(service._descriptors) == {"left"}
    finally:
        service.close()


def test_missing_assigned_ports_shows_teacher_facing_message(app, thresholds) -> None:
    provider = AssignedSerialDeviceProvider(
        "COM6",
        "COM7",
        port_lister=lambda: ["COM12", "COM13"],
    )
    service, sources = run_assigned_check(app, thresholds, provider)
    try:
        assert not sources
        assert service.result.left.reason is CheckReason.NO_DEVICE
        assert service.result.right.reason is CheckReason.NO_DEVICE
        assert service.result.message == "未检测到已分配手环，请老师检查连接"
    finally:
        service.close()


def test_duplicate_port_is_not_opened_for_both_sides(app, thresholds) -> None:
    service, sources = run_check(
        app,
        thresholds,
        [DeviceDescriptor("fake-shared", "left"), DeviceDescriptor("fake-shared", "right")],
    )
    try:
        assert len(sources) == 1
        assert service.result.left.reason is CheckReason.HEALTHY
        assert service.result.right.connection is ConnectionState.UNASSIGNED
    finally:
        service.close()


@pytest.mark.parametrize(
    ("packets", "rates", "reason"),
    [
        ([], [250.0], CheckReason.NO_EMG_DATA),
        (healthy_packets(5), [250.0], CheckReason.INSUFFICIENT_SAMPLES),
        ([FakePacket(np.zeros(8)) for _ in range(60)], [250.0], CheckReason.FLAT_SIGNAL),
        ([FakePacket(np.full(8, np.nan)) for _ in range(60)], [250.0], CheckReason.INVALID_VALUES),
        (healthy_packets(), [120.0, 260.0], CheckReason.UNSTABLE_RATE),
    ],
)
def test_signal_and_stream_failures_have_reason_codes(app, thresholds, packets, rates, reason) -> None:
    service, _ = run_check(
        app,
        thresholds,
        [DeviceDescriptor("fake-left", "left")],
        packets_by_side={"left": packets},
        rates_by_side={"left": rates},
    )
    try:
        assert service.result.left.reason is reason
        assert service.result.collection_ready is False
    finally:
        service.close()


def test_recheck_stops_previous_sources(app, thresholds) -> None:
    service, sources = run_check(app, thresholds, [DeviceDescriptor("fake-left", "left")])
    try:
        first_source = sources[0]
        service.start()
        QTest.qWait(50)
        app.processEvents()
        assert first_source.stopped >= 1
    finally:
        service.close()


def test_disconnection_after_completion_updates_result(app, thresholds) -> None:
    service, sources = run_check(app, thresholds, [DeviceDescriptor("fake-left", "left")])
    try:
        sources[0].disconnected.emit()
        app.processEvents()
        assert service.result.left.connection is ConnectionState.DISCONNECTED
        assert service.result.collection_ready is False
    finally:
        service.close()


@pytest.mark.parametrize("reason", [CheckReason.FLAT_SIGNAL, CheckReason.UNSTABLE_RATE])
def test_connected_quality_warning_is_observable_but_not_collection_ready(reason) -> None:
    result = SideCheckResult(
        "left",
        ConnectionState.CONNECTED,
        reason,
        received_emg=True,
        valid_samples=True,
        signal_healthy=reason is CheckReason.UNSTABLE_RATE,
    )
    assert result.observation_available is True
    assert result.ready_for_collection is False


def test_disconnected_side_is_not_available_for_observation() -> None:
    result = SideCheckResult(
        "left",
        ConnectionState.DISCONNECTED,
        CheckReason.NO_EMG_DATA,
        received_emg=True,
        valid_samples=True,
    )
    assert result.observation_available is False
    assert result.ready_for_collection is False


def test_observation_keeps_verified_two_hand_sources_and_recovers_only_right(app, thresholds) -> None:
    service, sources = run_check(
        app,
        thresholds,
        [DeviceDescriptor("COM6", "left"), DeviceDescriptor("COM7", "right")],
    )
    try:
        left_source, right_source = sources
        service.ensure_checked_sources_running()
        assert len(sources) == 2
        assert left_source.started == 1 and right_source.started == 1

        right_source.disconnected.emit()
        QTest.qWait(150)
        app.processEvents()
        assert len(sources) == 3
        assert left_source.started == 1 and left_source.stopped == 0
        assert right_source.stopped >= 1
        assert sources[-1].started == 1
        assert service.result.left.ready_for_collection is True
        assert service.result.right.ready_for_collection is True
    finally:
        service.close()


def test_observation_recovery_never_uses_unassigned_ports(app, thresholds) -> None:
    available_ports = ["COM6", "COM7"]
    provider = AssignedSerialDeviceProvider(
        "COM6", "COM7",
        port_lister=lambda: list(available_ports),
    )
    service, sources = run_assigned_check(app, thresholds, provider)
    try:
        assert len(sources) == 2
        sources[1].disconnected.emit()
        available_ports[:] = ["COM6", "COM12", "COM13"]
        app.processEvents()
        service.ensure_checked_sources_running()
        assert len(sources) == 2
        assert service.result.left.ready_for_collection is True
        assert service.result.right.ready_for_collection is False
    finally:
        service.close()


def test_simulation_result_is_explicitly_marked(app, thresholds) -> None:
    service = DeviceCheckService(thresholds=thresholds, simulate=True)
    try:
        service.start()
        QTest.qWait(thresholds.observe_duration_ms + 150)
        app.processEvents()
        assert service.result.simulate is True
    finally:
        service.close()


def healthy_result() -> DeviceCheckResult:
    healthy = SideCheckResult(
        "left",
        ConnectionState.CONNECTED,
        CheckReason.HEALTHY,
        received_emg=True,
        valid_samples=True,
        signal_healthy=True,
        rate_stable=True,
    )
    missing = SideCheckResult("right", ConnectionState.DISCONNECTED, CheckReason.NO_DEVICE)
    return DeviceCheckResult(healthy, missing, checking=False, message="检查完成")


def test_collection_navigation_is_gated_until_device_check_passes(app, thresholds) -> None:
    service = DeviceCheckService(provider=FakeProvider([]), thresholds=thresholds)
    window = StudentMainWindow(paths=resolve_project_paths(), device_check_service=service)
    try:
        collection_entry = window.course_entries[3]
        window.open_course_page(collection_entry)
        assert window._stack.currentWidget().objectName() == "student-collection-gate-page"

        service.result_changed.emit(healthy_result())
        app.processEvents()
        window.open_course_page(collection_entry)
        assert window._stack.currentWidget().objectName() == "student-collection-page"
    finally:
        window.close()


def test_closing_student_window_releases_device_check_resources(app, thresholds) -> None:
    service, sources = run_check(app, thresholds, [DeviceDescriptor("fake-left", "left")])
    window = StudentMainWindow(paths=resolve_project_paths(), device_check_service=service)
    window.close()
    app.processEvents()

    assert sources[0].stopped >= 1
