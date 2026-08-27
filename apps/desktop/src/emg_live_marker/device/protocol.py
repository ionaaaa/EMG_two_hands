"""Packet protocol support for the EMG wristband."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HEADER = bytes([0xD2, 0xD2, 0xD2])
PACKET_LEN = 29
TYPE_EMG = 0xAA
TYPE_IMU = 0xBB
EMG_FS = 250.0
IMU_FS = 104.0
EMG_CHANNELS = 8
PAYLOAD_LEN = 24


@dataclass(frozen=True)
class EmgPacket:
    seq: int
    sample_index: int
    t: float
    values_uv: np.ndarray
    raw_packet: bytes


@dataclass(frozen=True)
class ImuPacket:
    seq: int
    sample_index: int
    t: float
    gyro_rad_s: np.ndarray
    acc_m_s2: np.ndarray
    raw_packet: bytes


@dataclass
class ParserStats:
    aa_count: int = 0
    bb_count: int = 0
    bad_header_count: int = 0
    bad_type_count: int = 0
    aa_lost_count: int = 0
    bb_lost_count: int = 0
    global_lost_count: int = 0
    resync_count: int = 0


ParsedPacket = EmgPacket | ImuPacket


def int24_be_signed(b0: int, b1: int, b2: int) -> int:
    value = ((b0 & 0xFF) << 16) | ((b1 & 0xFF) << 8) | (b2 & 0xFF)
    if value & 0x800000:
        value -= 1 << 24
    return value


def int16_be_signed(b0: int, b1: int) -> int:
    value = ((b0 & 0xFF) << 8) | (b1 & 0xFF)
    if value & 0x8000:
        value -= 1 << 16
    return value


def _encode_int24_be_signed(value: int) -> bytes:
    if not -(1 << 23) <= value <= (1 << 23) - 1:
        raise ValueError(f"int24 value out of range: {value}")
    unsigned = value & 0xFFFFFF
    return bytes([(unsigned >> 16) & 0xFF, (unsigned >> 8) & 0xFF, unsigned & 0xFF])


def _encode_int16_be_signed(value: int) -> bytes:
    if not -(1 << 15) <= value <= (1 << 15) - 1:
        raise ValueError(f"int16 value out of range: {value}")
    unsigned = value & 0xFFFF
    return bytes([(unsigned >> 8) & 0xFF, unsigned & 0xFF])


def build_aa_packet(seq: int, values: list[int]) -> bytes:
    if len(values) != EMG_CHANNELS:
        raise ValueError("AA packet requires exactly 8 channel values")

    payload = b"".join(_encode_int24_be_signed(int(value)) for value in values)
    return HEADER + bytes([TYPE_EMG, seq & 0xFF]) + payload


def build_bb_packet(seq: int, gyro_raw: list[int], acc_raw: list[int]) -> bytes:
    if len(gyro_raw) != 3:
        raise ValueError("BB packet requires exactly 3 gyro values")
    if len(acc_raw) != 3:
        raise ValueError("BB packet requires exactly 3 accelerometer values")

    sensor_payload = b"".join(
        _encode_int16_be_signed(int(value)) for value in [*gyro_raw, *acc_raw]
    )
    payload = b"\x00\x00" + sensor_payload
    payload = payload.ljust(PAYLOAD_LEN, b"\x00")
    return HEADER + bytes([TYPE_IMU, seq & 0xFF]) + payload


class PacketParser:
    """Incremental parser for the 29-byte EMG/IMU packet stream."""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self.stats = ParserStats()
        self._last_global_seq: int | None = None
        self._emg_sample_index = 0
        self._imu_sample_index = 0

    def feed(self, data: bytes) -> list[ParsedPacket]:
        if data:
            self._buffer.extend(data)

        packets: list[ParsedPacket] = []
        while self._sync_to_header():
            if len(self._buffer) < PACKET_LEN:
                break

            packet_type = self._buffer[3]
            if packet_type not in (TYPE_EMG, TYPE_IMU):
                self.stats.bad_type_count += 1
                self.stats.resync_count += 1
                del self._buffer[0]
                continue

            raw_packet = bytes(self._buffer[:PACKET_LEN])
            del self._buffer[:PACKET_LEN]

            if packet_type == TYPE_EMG:
                packets.append(self._parse_aa(raw_packet))
            else:
                packets.append(self._parse_bb(raw_packet))

        return packets

    def _sync_to_header(self) -> bool:
        if len(self._buffer) < len(HEADER):
            return False

        header_index = self._buffer.find(HEADER)
        if header_index == 0:
            return True

        if header_index > 0:
            del self._buffer[:header_index]
            self.stats.bad_header_count += header_index
            self.stats.resync_count += 1
            return True

        keep = self._header_prefix_suffix_len()
        drop_count = len(self._buffer) - keep
        if drop_count > 0:
            del self._buffer[:drop_count]
            self.stats.bad_header_count += drop_count
            self.stats.resync_count += 1
        return False

    def _header_prefix_suffix_len(self) -> int:
        max_keep = min(len(HEADER) - 1, len(self._buffer))
        for keep in range(max_keep, 0, -1):
            if bytes(self._buffer[-keep:]) == HEADER[:keep]:
                return keep
        return 0

    def _parse_aa(self, packet: bytes) -> EmgPacket:
        seq = packet[4]
        self._update_lost_count(seq, packet_type=TYPE_EMG)
        payload = packet[5:29]
        values = [
            int24_be_signed(payload[i], payload[i + 1], payload[i + 2])
            for i in range(0, PAYLOAD_LEN, 3)
        ]
        sample_index = self._emg_sample_index
        self._emg_sample_index += 1
        self.stats.aa_count += 1
        return EmgPacket(
            seq=seq,
            sample_index=sample_index,
            t=sample_index / EMG_FS,
            values_uv=np.asarray(values, dtype=np.float64),
            raw_packet=packet,
        )

    def _parse_bb(self, packet: bytes) -> ImuPacket:
        seq = packet[4]
        self._update_lost_count(seq, packet_type=TYPE_IMU)
        payload = packet[5:29]
        raw_values = [
            int16_be_signed(payload[i], payload[i + 1])
            for i in range(2, 14, 2)
        ]
        sample_index = self._imu_sample_index
        self._imu_sample_index += 1
        self.stats.bb_count += 1
        gyro_raw = np.asarray(raw_values[:3], dtype=np.float64)
        acc_raw = np.asarray(raw_values[3:], dtype=np.float64)
        return ImuPacket(
            seq=seq,
            sample_index=sample_index,
            t=sample_index / IMU_FS,
            gyro_rad_s=gyro_raw * 0.0012,
            acc_m_s2=acc_raw * 0.0005978,
            raw_packet=packet,
        )

    def _update_lost_count(self, seq: int, packet_type: int) -> None:
        _ = packet_type
        if self._last_global_seq is not None:
            expected = (self._last_global_seq + 1) & 0xFF
            if seq != expected:
                self.stats.global_lost_count += (seq - expected) & 0xFF
        self._last_global_seq = seq
