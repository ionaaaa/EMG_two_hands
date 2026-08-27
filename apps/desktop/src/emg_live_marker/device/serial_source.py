"""Threaded serial source for the EMG wristband."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from PySide6.QtCore import QThread, Signal

from emg_live_marker.config import DEFAULT_BAUDRATE
from emg_live_marker.device.protocol import EmgPacket, ImuPacket, PacketParser

try:
    import serial
    from serial.tools import list_ports
except ImportError:  # pragma: no cover - pyserial is a project dependency.
    serial = None
    list_ports = None


@dataclass(frozen=True)
class SerialSourceConfig:
    port: str
    baudrate: int = DEFAULT_BAUDRATE
    timeout: float = 0.01
    read_size: int = 4096


def list_serial_ports() -> list[str]:
    if list_ports is None:
        return []
    return [port.device for port in list_ports.comports()]


class SerialSource(QThread):
    emg_packets = Signal(list)
    imu_packets = Signal(list)
    raw_bytes = Signal(bytes)
    stats_updated = Signal(dict)
    connected = Signal(str)
    disconnected = Signal()
    error_occurred = Signal(str)

    def __init__(self, parent=None, timeout: float = 0.01, read_size: int = 4096) -> None:
        super().__init__(parent)
        self._timeout = float(timeout)
        self._read_size = int(read_size)
        self._config: SerialSourceConfig | None = None
        self._serial = None
        self._stop_requested = False

    def connect_port(self, port: str, baudrate: int = DEFAULT_BAUDRATE) -> None:
        if self.isRunning():
            self.disconnect_port()
        self._config = SerialSourceConfig(
            port=port,
            baudrate=int(baudrate),
            timeout=self._timeout,
            read_size=self._read_size,
        )
        self._stop_requested = False
        self.start()

    def disconnect_port(self) -> None:
        self._stop_requested = True
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass
        if self.isRunning() and QThread.currentThread() is not self:
            self.wait(1500)

    def is_running(self) -> bool:
        return self.isRunning()

    def run(self) -> None:
        if serial is None:
            self.error_occurred.emit("pyserial is not installed.")
            self.disconnected.emit()
            return
        if self._config is None:
            self.error_occurred.emit("No serial port selected.")
            self.disconnected.emit()
            return

        parser = PacketParser()
        last_stats_t = perf_counter()
        last_aa_count = 0
        last_bb_count = 0

        try:
            self._serial = serial.Serial(
                self._config.port,
                baudrate=self._config.baudrate,
                timeout=self._config.timeout,
            )
            self.connected.emit(self._config.port)

            while not self._stop_requested:
                try:
                    data = self._serial.read(self._config.read_size)
                except Exception as exc:
                    if not self._stop_requested:
                        self.error_occurred.emit(f"Serial read failed: {exc}")
                    break

                if data:
                    self.raw_bytes.emit(bytes(data))
                    packets = parser.feed(bytes(data))
                    if packets:
                        emg_packets = [packet for packet in packets if isinstance(packet, EmgPacket)]
                        imu_packets = [packet for packet in packets if isinstance(packet, ImuPacket)]
                        if emg_packets:
                            self.emg_packets.emit(emg_packets)
                        if imu_packets:
                            self.imu_packets.emit(imu_packets)

                now = perf_counter()
                elapsed = now - last_stats_t
                if elapsed >= 1.0:
                    stats = parser.stats
                    aa_count = int(stats.aa_count)
                    bb_count = int(stats.bb_count)
                    self.stats_updated.emit(
                        {
                            "emg_rate_sps": (aa_count - last_aa_count) / elapsed,
                            "imu_rate_sps": (bb_count - last_bb_count) / elapsed,
                            "aa_count": aa_count,
                            "bb_count": bb_count,
                            "aa_lost_count": int(stats.aa_lost_count),
                            "bb_lost_count": int(stats.bb_lost_count),
                            "global_lost_count": int(stats.global_lost_count),
                            "bad_header_count": int(stats.bad_header_count),
                            "bad_type_count": int(stats.bad_type_count),
                            "resync_count": int(stats.resync_count),
                        }
                    )
                    last_stats_t = now
                    last_aa_count = aa_count
                    last_bb_count = bb_count
        except Exception as exc:
            if not self._stop_requested:
                self.error_occurred.emit(f"Serial connection failed: {exc}")
        finally:
            if self._serial is not None:
                try:
                    self._serial.close()
                except Exception:
                    pass
                self._serial = None
            self._stop_requested = False
            self.disconnected.emit()
