"""Fixed-size ring buffers for real-time EMG and IMU streams."""

from __future__ import annotations

from math import ceil

import numpy as np

from emg_live_marker.device.protocol import EMG_CHANNELS, EMG_FS, IMU_FS, EmgPacket, ImuPacket


def _stable_sort_and_deduplicate(
    t: np.ndarray,
    data_arrays: tuple[np.ndarray, ...],
    sample_index: np.ndarray,
) -> tuple[np.ndarray, tuple[np.ndarray, ...], np.ndarray]:
    valid = np.isfinite(t) & (sample_index >= 0)
    t = t[valid]
    sample_index = sample_index[valid]
    data_arrays = tuple(data[valid] for data in data_arrays)
    if sample_index.size == 0:
        return t, data_arrays, sample_index

    order = np.argsort(sample_index, kind="stable")
    t = t[order]
    sample_index = sample_index[order]
    data_arrays = tuple(data[order] for data in data_arrays)

    _, unique_last_indices = np.unique(sample_index[::-1], return_index=True)
    keep = np.sort(sample_index.size - 1 - unique_last_indices)
    t = t[keep]
    sample_index = sample_index[keep]
    data_arrays = tuple(data[keep] for data in data_arrays)
    return t, data_arrays, sample_index


class EmgRingBuffer:
    def __init__(self, fs: float = EMG_FS, channels: int = EMG_CHANNELS, seconds: float = 60.0) -> None:
        self.fs = float(fs)
        self.channels = int(channels)
        self.seconds = float(seconds)
        self.capacity = max(1, int(ceil(self.fs * self.seconds)))
        self._t = np.empty(self.capacity, dtype=np.float64)
        self._data = np.empty((self.capacity, self.channels), dtype=np.float64)
        self._sample_index = np.full(self.capacity, -1, dtype=np.int64)
        self._write_pos = 0
        self._count = 0

    def append_packets(self, packets: list[EmgPacket]) -> None:
        if not packets:
            return
        t = np.asarray([packet.t for packet in packets], dtype=np.float64)
        data = np.asarray([packet.values_uv for packet in packets], dtype=np.float64)
        sample_index = np.asarray([packet.sample_index for packet in packets], dtype=np.int64)
        self.append_many(t, data, sample_index)

    def append_many(self, t: np.ndarray, data: np.ndarray, sample_index: np.ndarray) -> None:
        t = np.asarray(t, dtype=np.float64).reshape(-1)
        data = np.asarray(data, dtype=np.float64)
        sample_index = np.asarray(sample_index, dtype=np.int64).reshape(-1)

        if data.ndim != 2 or data.shape[1] != self.channels:
            raise ValueError(f"data must have shape (N, {self.channels})")
        if len(t) != data.shape[0] or len(sample_index) != data.shape[0]:
            raise ValueError("t, data, and sample_index must contain the same number of samples")
        if data.shape[0] == 0:
            return

        if data.shape[0] >= self.capacity:
            self._t[:] = t[-self.capacity:]
            self._data[:] = data[-self.capacity:]
            self._sample_index[:] = sample_index[-self.capacity:]
            self._write_pos = 0
            self._count = self.capacity
            return

        remaining = data.shape[0]
        src_pos = 0
        while remaining:
            chunk = min(remaining, self.capacity - self._write_pos)
            dst = slice(self._write_pos, self._write_pos + chunk)
            src = slice(src_pos, src_pos + chunk)
            self._t[dst] = t[src]
            self._data[dst] = data[src]
            self._sample_index[dst] = sample_index[src]
            self._write_pos = (self._write_pos + chunk) % self.capacity
            self._count = min(self.capacity, self._count + chunk)
            src_pos += chunk
            remaining -= chunk

    def get_window(self, seconds: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        empty = (
            np.empty(0, dtype=np.float64),
            np.empty((0, self.channels), dtype=np.float64),
            np.empty(0, dtype=np.int64),
        )
        if self._count == 0 or seconds <= 0:
            return empty

        idx = self._ordered_indices()
        if idx.size == 0:
            return empty
        t, (data,), sample_index = _stable_sort_and_deduplicate(
            self._t[idx],
            (self._data[idx],),
            self._sample_index[idx],
        )
        if t.size == 0:
            return empty

        cutoff = t[-1] - float(seconds)
        mask = t >= cutoff
        return t[mask].copy(), data[mask].copy(), sample_index[mask].copy()

    def latest_sample_index(self) -> int | None:
        if self._count == 0:
            return None
        idx = self._ordered_indices()
        sample_index = self._sample_index[idx]
        sample_index = sample_index[sample_index >= 0]
        if sample_index.size == 0:
            return None
        return int(sample_index.max())

    def clear(self) -> None:
        self._write_pos = 0
        self._count = 0
        self._sample_index.fill(-1)

    def _ordered_indices(self) -> np.ndarray:
        if self._count == 0:
            return np.array([], dtype=np.int64)
        if self._count < self.capacity:
            return np.arange(self._count, dtype=np.int64)
        return np.concatenate(
            [
                np.arange(self._write_pos, self.capacity, dtype=np.int64),
                np.arange(0, self._write_pos, dtype=np.int64),
            ]
        )


class ImuRingBuffer:
    def __init__(self, fs: float = IMU_FS, seconds: float = 60.0) -> None:
        self.fs = float(fs)
        self.seconds = float(seconds)
        self.capacity = max(1, int(ceil(self.fs * self.seconds)))
        self._t = np.empty(self.capacity, dtype=np.float64)
        self._gyro = np.empty((self.capacity, 3), dtype=np.float64)
        self._acc = np.empty((self.capacity, 3), dtype=np.float64)
        self._sample_index = np.full(self.capacity, -1, dtype=np.int64)
        self._write_pos = 0
        self._count = 0

    def append_packets(self, packets: list[ImuPacket]) -> None:
        if not packets:
            return
        t = np.asarray([packet.t for packet in packets], dtype=np.float64)
        gyro = np.asarray([packet.gyro_rad_s for packet in packets], dtype=np.float64)
        acc = np.asarray([packet.acc_m_s2 for packet in packets], dtype=np.float64)
        sample_index = np.asarray([packet.sample_index for packet in packets], dtype=np.int64)
        self.append_many(t, gyro, acc, sample_index)

    def append_many(
        self,
        t: np.ndarray,
        gyro: np.ndarray,
        acc: np.ndarray,
        sample_index: np.ndarray,
    ) -> None:
        t = np.asarray(t, dtype=np.float64).reshape(-1)
        gyro = np.asarray(gyro, dtype=np.float64)
        acc = np.asarray(acc, dtype=np.float64)
        sample_index = np.asarray(sample_index, dtype=np.int64).reshape(-1)

        if gyro.ndim != 2 or gyro.shape[1] != 3:
            raise ValueError("gyro must have shape (N, 3)")
        if acc.ndim != 2 or acc.shape[1] != 3:
            raise ValueError("acc must have shape (N, 3)")
        if (
            len(t) != gyro.shape[0]
            or len(sample_index) != gyro.shape[0]
            or acc.shape[0] != gyro.shape[0]
        ):
            raise ValueError("t, gyro, acc, and sample_index must contain the same number of samples")
        if gyro.shape[0] == 0:
            return

        if gyro.shape[0] >= self.capacity:
            self._t[:] = t[-self.capacity:]
            self._gyro[:] = gyro[-self.capacity:]
            self._acc[:] = acc[-self.capacity:]
            self._sample_index[:] = sample_index[-self.capacity:]
            self._write_pos = 0
            self._count = self.capacity
            return

        remaining = gyro.shape[0]
        src_pos = 0
        while remaining:
            chunk = min(remaining, self.capacity - self._write_pos)
            dst = slice(self._write_pos, self._write_pos + chunk)
            src = slice(src_pos, src_pos + chunk)
            self._t[dst] = t[src]
            self._gyro[dst] = gyro[src]
            self._acc[dst] = acc[src]
            self._sample_index[dst] = sample_index[src]
            self._write_pos = (self._write_pos + chunk) % self.capacity
            self._count = min(self.capacity, self._count + chunk)
            src_pos += chunk
            remaining -= chunk

    def get_window(self, seconds: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        empty = (
            np.empty(0, dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
            np.empty(0, dtype=np.int64),
        )
        if self._count == 0 or seconds <= 0:
            return empty

        idx = self._ordered_indices()
        if idx.size == 0:
            return empty
        t, (gyro, acc), sample_index = _stable_sort_and_deduplicate(
            self._t[idx],
            (self._gyro[idx], self._acc[idx]),
            self._sample_index[idx],
        )
        if t.size == 0:
            return empty

        cutoff = t[-1] - float(seconds)
        mask = t >= cutoff
        return t[mask].copy(), gyro[mask].copy(), acc[mask].copy(), sample_index[mask].copy()

    def latest_sample_index(self) -> int | None:
        if self._count == 0:
            return None
        idx = self._ordered_indices()
        sample_index = self._sample_index[idx]
        sample_index = sample_index[sample_index >= 0]
        if sample_index.size == 0:
            return None
        return int(sample_index.max())

    def clear(self) -> None:
        self._write_pos = 0
        self._count = 0
        self._sample_index.fill(-1)

    def _ordered_indices(self) -> np.ndarray:
        if self._count == 0:
            return np.array([], dtype=np.int64)
        if self._count < self.capacity:
            return np.arange(self._count, dtype=np.int64)
        return np.concatenate(
            [
                np.arange(self._write_pos, self.capacity, dtype=np.int64),
                np.arange(0, self._write_pos, dtype=np.int64),
            ]
        )

