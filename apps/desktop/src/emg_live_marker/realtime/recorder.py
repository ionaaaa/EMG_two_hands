"""Session recording for EMG, IMU, events, and raw serial bytes."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from queue import Queue
from threading import Thread
from time import monotonic
from typing import BinaryIO, TextIO

from emg_live_marker.config import DEFAULT_BAUDRATE
from emg_live_marker.device.protocol import EMG_CHANNELS, EMG_FS, IMU_FS, EmgPacket, ImuPacket
from emg_live_marker.paths import resolve_project_paths


@dataclass(frozen=True)
class RecordingTarget:
    path: Path


class SessionRecorder:
    def __init__(self) -> None:
        self.session_dir: Path | None = None
        self._emg_file: TextIO | None = None
        self._imu_file: TextIO | None = None
        self._events_file: TextIO | None = None
        self._raw_file: BinaryIO | None = None
        self._emg_writer: csv.writer | None = None
        self._imu_writer: csv.writer | None = None
        self._events_writer: csv.writer | None = None
        self._write_queue: Queue | None = None
        self._writer_thread: Thread | None = None
        self._session_start_monotonic = 0.0
        self._collection_mode = False
        self._writer_error: Exception | None = None
        self._first_emg_sample_index: int | None = None
        self._first_imu_sample_index: int | None = None

    @property
    def is_recording(self) -> bool:
        return self.session_dir is not None

    def start(
        self,
        root_dir: Path | str | None = None,
        baudrate: int = DEFAULT_BAUDRATE,
        *,
        session_dir: Path | None = None,
        metadata: dict | None = None,
        collection_mode: bool = False,
    ) -> Path:
        if self.is_recording:
            self.stop()

        if session_dir is None:
            root = Path(root_dir) if root_dir is not None else resolve_project_paths().recordings_root
            root.mkdir(parents=True, exist_ok=True)
            session_dir = self._make_session_dir(root)
        session_dir.mkdir(parents=True, exist_ok=False)

        if metadata is None:
            created_at = datetime.now().isoformat(timespec="seconds")
            metadata = {
                "device": "WAVELETECH 8ch EMG Bluetooth Transparent",
                "emg_fs": EMG_FS,
                "imu_fs": IMU_FS,
                "baudrate": int(baudrate),
                "emg_channels": EMG_CHANNELS,
                "emg_unit": "uV",
                "gyro_unit": "rad/s",
                "acc_unit": "m/s^2",
                "created_at": created_at,
                "software": "emg_live_marker",
            }
        (session_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        self._emg_file = (session_dir / "emg.csv").open("w", newline="", encoding="utf-8")
        self._imu_file = (session_dir / "imu.csv").open("w", newline="", encoding="utf-8")
        self._events_file = (session_dir / "events.csv").open("w", newline="", encoding="utf-8")
        self._raw_file = (session_dir / "raw_packets.bin").open("wb")

        self._emg_writer = csv.writer(self._emg_file)
        self._imu_writer = csv.writer(self._imu_file)
        self._events_writer = csv.writer(self._events_file)
        self._collection_mode = bool(collection_mode)
        self._session_start_monotonic = monotonic()
        self._first_emg_sample_index = None
        self._first_imu_sample_index = None
        self._emg_writer.writerow(
            [
                "sample_index",
                "software_time" if self._collection_mode else "t_emg",
                "ch1_uv",
                "ch2_uv",
                "ch3_uv",
                "ch4_uv",
                "ch5_uv",
                "ch6_uv",
                "ch7_uv",
                "ch8_uv",
            ]
        )
        self._imu_writer.writerow(
            [
                "sample_index",
                "software_time" if self._collection_mode else "t_imu",
                "gr_x_rad_s",
                "gr_y_rad_s",
                "gr_z_rad_s",
                "acc_x_m_s2",
                "acc_y_m_s2",
                "acc_z_m_s2",
            ]
        )
        if self._collection_mode:
            self._events_writer.writerow(
                [
                    "trial_id",
                    "subject_id",
                    "session_id",
                    "gesture",
                    "gesture_name",
                    "phase",
                    "software_time",
                    "sample_index",
                    "note",
                ]
            )
        else:
            self._events_writer.writerow(
                ["event_id", "t_emg", "emg_sample_index", "label", "note", "created_at_wall_time"]
            )
        self._write_queue = Queue()
        self._writer_error = None
        self._writer_thread = Thread(target=self._writer_loop, name="SessionRecorderWriter", daemon=True)
        self._writer_thread.start()
        self.session_dir = session_dir
        return session_dir

    def stop(self) -> None:
        queue = self._write_queue
        thread = self._writer_thread
        if queue is not None and thread is not None:
            queue.put(("stop", None))
            thread.join()

        files = [self._emg_file, self._imu_file, self._events_file, self._raw_file]
        for file_obj in files:
            if file_obj is None:
                continue
            try:
                file_obj.flush()
            finally:
                file_obj.close()

        self.session_dir = None
        self._emg_file = None
        self._imu_file = None
        self._events_file = None
        self._raw_file = None
        self._emg_writer = None
        self._imu_writer = None
        self._events_writer = None
        self._write_queue = None
        self._writer_thread = None
        self._collection_mode = False
        self._session_start_monotonic = 0.0
        self._first_emg_sample_index = None
        self._first_imu_sample_index = None

    def write_emg_packets(self, packets: list[EmgPacket]) -> None:
        if not packets or not self.is_recording or self._emg_writer is None:
            return
        rows = []
        if self._collection_mode and self._first_emg_sample_index is None:
            self._first_emg_sample_index = packets[0].sample_index
        for packet in packets:
            if self._collection_mode:
                assert self._first_emg_sample_index is not None
                t_value = (packet.sample_index - self._first_emg_sample_index) / EMG_FS
            else:
                t_value = packet.t
            rows.append(
                [
                    packet.sample_index,
                    f"{max(0.0, t_value):.6f}",
                    *[f"{float(value):.6f}" for value in packet.values_uv],
                ]
            )
        self._enqueue("emg_rows", rows)

    def write_imu_packets(self, packets: list[ImuPacket]) -> None:
        if not packets or not self.is_recording or self._imu_writer is None:
            return
        rows = []
        if self._collection_mode and self._first_imu_sample_index is None:
            self._first_imu_sample_index = packets[0].sample_index
        for packet in packets:
            if self._collection_mode:
                assert self._first_imu_sample_index is not None
                t_value = (packet.sample_index - self._first_imu_sample_index) / IMU_FS
            else:
                t_value = packet.t
            rows.append(
                [
                    packet.sample_index,
                    f"{max(0.0, t_value):.6f}",
                    *[f"{float(value):.9f}" for value in packet.gyro_rad_s],
                    *[f"{float(value):.9f}" for value in packet.acc_m_s2],
                ]
            )
        self._enqueue("imu_rows", rows)

    def write_event(self, event: object) -> None:
        if not self.is_recording or self._events_writer is None:
            return
        if self._collection_mode:
            row = [
                "",
                "",
                "",
                "",
                "",
                getattr(event, "label"),
                f"{self.software_time():.6f}",
                getattr(event, "emg_sample_index"),
                getattr(event, "note"),
            ]
        else:
            row = [
                getattr(event, "event_id"),
                f"{float(getattr(event, 't_emg')):.6f}",
                getattr(event, "emg_sample_index"),
                getattr(event, "label"),
                getattr(event, "note"),
                getattr(event, "created_at_wall_time"),
            ]
        self._enqueue("event_row", row)

    def write_collection_event(
        self,
        *,
        trial_id: str,
        subject_id: str,
        session_id: str,
        gesture: str,
        phase: str,
        sample_index: int | None,
        gesture_name: str = "",
        software_time: float | None = None,
        note: str = "",
    ) -> None:
        if not self.is_recording or self._events_writer is None:
            return
        row = [
            trial_id,
            subject_id,
            session_id,
            gesture,
            gesture_name,
            phase,
            f"{(self.software_time() if software_time is None else software_time):.6f}",
            "" if sample_index is None else int(sample_index),
            note,
        ]
        self._enqueue("event_row", row)

    def write_raw_bytes(self, data: bytes) -> None:
        if not self.is_recording or self._raw_file is None or not data:
            return
        self._enqueue("raw_bytes", bytes(data))

    def software_time(self) -> float:
        if self._session_start_monotonic <= 0.0:
            return 0.0
        return monotonic() - self._session_start_monotonic

    def _enqueue(self, kind: str, payload: object) -> None:
        if self._writer_error is not None:
            raise RuntimeError(f"Recorder writer failed: {self._writer_error}") from self._writer_error
        if self._write_queue is None:
            return
        self._write_queue.put((kind, payload))

    def _writer_loop(self) -> None:
        assert self._write_queue is not None
        try:
            while True:
                kind, payload = self._write_queue.get()
                if kind == "stop":
                    break
                if kind == "emg_rows" and self._emg_writer is not None:
                    self._emg_writer.writerows(payload)
                elif kind == "imu_rows" and self._imu_writer is not None:
                    self._imu_writer.writerows(payload)
                elif kind == "event_row" and self._events_writer is not None:
                    self._events_writer.writerow(payload)
                elif kind == "raw_bytes" and self._raw_file is not None:
                    self._raw_file.write(payload)
        except Exception as exc:  # pragma: no cover - retained for UI error reporting.
            self._writer_error = exc

    def _make_session_dir(self, root: Path) -> Path:
        base_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        candidate = root / base_name
        suffix = 1
        while candidate.exists():
            candidate = root / f"{base_name}_{suffix:02d}"
            suffix += 1
        return candidate
