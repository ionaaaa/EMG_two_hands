"""Qt-based simulated EMG/IMU data source."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal

from emg_live_marker.device.protocol import (
    EMG_CHANNELS,
    EMG_FS,
    IMU_FS,
    EmgPacket,
    ImuPacket,
    build_aa_packet,
    build_bb_packet,
)


@dataclass(frozen=True)
class SimulatorConfig:
    seed: int = 1
    interval_ms: int = 40
    emg_fs: float = EMG_FS
    imu_fs: float = IMU_FS


class SimulatedDevice(QObject):
    """Generate batched EMG and IMU packets on the Qt event loop."""

    emg_packets = Signal(list)
    imu_packets = Signal(list)
    stats_updated = Signal(dict)

    def __init__(self, config: SimulatorConfig | None = None, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.config = config or SimulatorConfig()
        self._rng = np.random.default_rng(self.config.seed)
        self._timer = QTimer(self)
        self._timer.setInterval(self.config.interval_ms)
        self._timer.timeout.connect(self._on_timeout)
        self._emg_sample_index = 0
        self._imu_sample_index = 0
        self._imu_phase = 0.0
        self._imu_accumulator = 0.0
        self._aa_count = 0
        self._bb_count = 0
        self._burst_until_t = -1.0
        self._next_burst_t = self._random_next_burst_time(0.0)
        self._burst_amplitude = np.zeros(EMG_CHANNELS, dtype=np.float64)

    def start(self) -> None:
        if not self._timer.isActive():
            self._timer.start()

    def stop(self) -> None:
        self._timer.stop()

    def is_running(self) -> bool:
        return self._timer.isActive()

    def _on_timeout(self) -> None:
        emg_packets = self._generate_emg_batch()
        imu_packets = self._generate_imu_batch()
        if emg_packets:
            self.emg_packets.emit(emg_packets)
        if imu_packets:
            self.imu_packets.emit(imu_packets)
        self.stats_updated.emit(
            {
                "aa_count": self._aa_count,
                "bb_count": self._bb_count,
                "aa_lost_count": 0,
                "bb_lost_count": 0,
                "global_lost_count": 0,
            }
        )

    def _generate_emg_batch(self) -> list[EmgPacket]:
        sample_count = max(1, int(round(self.config.emg_fs * self.config.interval_ms / 1000.0)))
        packets: list[EmgPacket] = []
        for _ in range(sample_count):
            sample_index = self._emg_sample_index
            t = sample_index / self.config.emg_fs
            self._update_burst_state(t)
            values = self._rng.normal(0.0, 20.0, size=EMG_CHANNELS)
            if t < self._burst_until_t:
                carrier = abs(np.sin(2.0 * np.pi * 45.0 * t))
                values += self._burst_amplitude * carrier
                values += self._rng.normal(0.0, 0.12 * self._burst_amplitude)

            raw_values = np.clip(np.rint(values), -(1 << 23), (1 << 23) - 1).astype(int)
            seq = sample_index & 0xFF
            packets.append(
                EmgPacket(
                    seq=seq,
                    sample_index=sample_index,
                    t=t,
                    values_uv=raw_values.astype(np.float64),
                    raw_packet=build_aa_packet(seq, raw_values.tolist()),
                )
            )
            self._emg_sample_index += 1
            self._aa_count += 1
        return packets

    def _generate_imu_batch(self) -> list[ImuPacket]:
        self._imu_accumulator += self.config.imu_fs * self.config.interval_ms / 1000.0
        sample_count = int(self._imu_accumulator)
        self._imu_accumulator -= sample_count
        packets: list[ImuPacket] = []
        for _ in range(sample_count):
            sample_index = self._imu_sample_index
            t = sample_index / self.config.imu_fs
            gyro = np.array(
                [
                    0.08 * np.sin(2.0 * np.pi * 0.55 * t),
                    0.05 * np.sin(2.0 * np.pi * 0.35 * t + 0.7),
                    0.04 * np.cos(2.0 * np.pi * 0.25 * t),
                ],
                dtype=np.float64,
            )
            acc = np.array(
                [
                    0.25 * np.sin(2.0 * np.pi * 0.45 * t),
                    0.18 * np.cos(2.0 * np.pi * 0.30 * t),
                    9.81 + 0.12 * np.sin(2.0 * np.pi * 0.20 * t),
                ],
                dtype=np.float64,
            )
            gyro += self._rng.normal(0.0, 0.01, size=3)
            acc += self._rng.normal(0.0, 0.035, size=3)
            gyro_raw = np.clip(np.rint(gyro / 0.0012), -(1 << 15), (1 << 15) - 1).astype(int)
            acc_raw = np.clip(np.rint(acc / 0.0005978), -(1 << 15), (1 << 15) - 1).astype(int)
            seq = sample_index & 0xFF
            packets.append(
                ImuPacket(
                    seq=seq,
                    sample_index=sample_index,
                    t=t,
                    gyro_rad_s=gyro,
                    acc_m_s2=acc,
                    raw_packet=build_bb_packet(seq, gyro_raw.tolist(), acc_raw.tolist()),
                )
            )
            self._imu_sample_index += 1
            self._bb_count += 1
        return packets

    def _update_burst_state(self, t: float) -> None:
        if t < self._next_burst_t:
            return

        duration = float(self._rng.uniform(0.3, 1.0))
        base_amplitude = float(self._rng.uniform(100.0, 500.0))
        channel_scale = self._rng.uniform(0.75, 1.25, size=EMG_CHANNELS)
        self._burst_amplitude = base_amplitude * channel_scale
        self._burst_until_t = t + duration
        self._next_burst_t = self._random_next_burst_time(self._burst_until_t)

    def _random_next_burst_time(self, after_t: float) -> float:
        return after_t + float(self._rng.uniform(2.0, 4.0))
