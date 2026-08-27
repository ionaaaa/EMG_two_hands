import csv
import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

from emg_live_marker.device.protocol import EmgPacket, ImuPacket
from emg_live_marker.realtime.collection import GESTURE_DISPLAY_NAMES, build_trial_list
from emg_live_marker.realtime.recorder import SessionRecorder
from emg_live_marker.ui.main_window import MainWindow


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_build_trial_list_generates_three_fixed_gestures_without_randomization():
    trials = build_trial_list(2, randomize=False)

    assert [trial.trial_id for trial in trials] == ["0001", "0002", "0003", "0004", "0005", "0006"]
    assert [trial.gesture for trial in trials] == [
        "fist",
        "fist",
        "finger_spread",
        "finger_spread",
        "thumb_index_pinch",
        "thumb_index_pinch",
    ]
    assert {trial.status for trial in trials} == {"pending"}


def test_build_trial_list_uses_trials_per_gesture_for_total_count():
    trials = build_trial_list(5, randomize=False)

    assert len(trials) == 15
    assert [trial.gesture for trial in trials[:5]] == ["fist"] * 5
    assert [trial.gesture for trial in trials[5:10]] == ["finger_spread"] * 5
    assert [trial.gesture for trial in trials[10:]] == ["thumb_index_pinch"] * 5


def test_collection_recorder_writes_dataset_headers_metadata_and_events(tmp_path):
    recorder = SessionRecorder()
    session_dir = tmp_path / "dataset" / "subject_01" / "session_001"
    metadata = {
        "subject_id": "subject_01",
        "session_id": "session_001",
        "gestures": ["fist", "finger_spread", "thumb_index_pinch"],
        "gesture_display_names": GESTURE_DISPLAY_NAMES,
        "trials_per_gesture": 1,
        "trial_duration_s": 3.0,
        "rest_before_s": 0.5,
        "gesture_hold_s": 1.5,
        "rest_after_s": 1.0,
        "emg_fs": 250.0,
        "imu_fs": 104.0,
        "channels": 8,
        "device_mode": "single_bracelet",
        "sync_method": "software_time",
        "created_at": "2026-07-08T12:00:00",
    }

    recorder.start(session_dir=session_dir, metadata=metadata, collection_mode=True)
    recorder.write_emg_packets(
        [
            EmgPacket(
                seq=1,
                sample_index=10,
                t=0.04,
                values_uv=np.arange(8, dtype=np.float64),
                raw_packet=b"abc",
            )
        ]
    )
    recorder.write_imu_packets(
        [
            ImuPacket(
                seq=2,
                sample_index=4,
                t=0.038,
                gyro_rad_s=np.array([1.0, 2.0, 3.0]),
                acc_m_s2=np.array([4.0, 5.0, 6.0]),
                raw_packet=b"def",
            )
        ]
    )
    recorder.write_collection_event(
        trial_id="0001",
        subject_id="subject_01",
        session_id="session_001",
        gesture="fist",
        gesture_name="全力握拳",
        phase="trial_start",
        software_time=12.0,
        sample_index=3000,
    )
    recorder.write_raw_bytes(b"raw")
    recorder.stop()

    assert json.loads((session_dir / "metadata.json").read_text(encoding="utf-8")) == metadata

    with (session_dir / "emg.csv").open(newline="", encoding="utf-8") as file_obj:
        emg_rows = list(csv.reader(file_obj))
    assert emg_rows[0] == [
        "sample_index",
        "software_time",
        "ch1_uv",
        "ch2_uv",
        "ch3_uv",
        "ch4_uv",
        "ch5_uv",
        "ch6_uv",
        "ch7_uv",
        "ch8_uv",
    ]
    assert emg_rows[1][0] == "10"

    with (session_dir / "imu.csv").open(newline="", encoding="utf-8") as file_obj:
        imu_rows = list(csv.reader(file_obj))
    assert imu_rows[0] == [
        "sample_index",
        "software_time",
        "gr_x_rad_s",
        "gr_y_rad_s",
        "gr_z_rad_s",
        "acc_x_m_s2",
        "acc_y_m_s2",
        "acc_z_m_s2",
    ]

    with (session_dir / "events.csv").open(newline="", encoding="utf-8") as file_obj:
        event_rows = list(csv.reader(file_obj))
    assert event_rows[0] == [
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
    assert event_rows[1] == [
        "0001",
        "subject_01",
        "session_001",
        "fist",
        "全力握拳",
        "trial_start",
        "12.000000",
        "3000",
        "",
    ]
    assert (session_dir / "raw_packets.bin").read_bytes() == b"raw"


def test_collection_recorder_uses_sample_index_for_emg_software_time(tmp_path):
    recorder = SessionRecorder()
    session_dir = tmp_path / "dataset" / "subject_01" / "session_001"
    recorder.start(session_dir=session_dir, metadata={}, collection_mode=True)

    first_batch = [
        EmgPacket(
            seq=index,
            sample_index=100 + index,
            t=50.0 + index,
            values_uv=np.full(8, index, dtype=np.float64),
            raw_packet=b"",
        )
        for index in range(2)
    ]
    second_batch = [
        EmgPacket(
            seq=2,
            sample_index=102,
            t=100.0,
            values_uv=np.full(8, 2.0, dtype=np.float64),
            raw_packet=b"",
        )
    ]

    recorder.write_emg_packets(first_batch)
    recorder.write_emg_packets(second_batch)
    recorder.stop()

    with (session_dir / "emg.csv").open(newline="", encoding="utf-8") as file_obj:
        rows = list(csv.DictReader(file_obj))

    assert [int(row["sample_index"]) for row in rows] == [100, 101, 102]
    times = [float(row["software_time"]) for row in rows]
    assert times == [0.0, 0.004, 0.008]
    assert all(next_time > time for time, next_time in zip(times, times[1:]))


def test_collection_recorder_uses_sample_index_for_imu_software_time(tmp_path):
    recorder = SessionRecorder()
    session_dir = tmp_path / "dataset" / "subject_01" / "session_001"
    recorder.start(session_dir=session_dir, metadata={}, collection_mode=True)

    packets = [
        ImuPacket(
            seq=index,
            sample_index=50 + index,
            t=20.0 + index,
            gyro_rad_s=np.full(3, index, dtype=np.float64),
            acc_m_s2=np.full(3, index, dtype=np.float64),
            raw_packet=b"",
        )
        for index in range(3)
    ]

    recorder.write_imu_packets(packets[:1])
    recorder.write_imu_packets(packets[1:])
    recorder.stop()

    with (session_dir / "imu.csv").open(newline="", encoding="utf-8") as file_obj:
        rows = list(csv.DictReader(file_obj))

    assert [int(row["sample_index"]) for row in rows] == [50, 51, 52]
    times = [float(row["software_time"]) for row in rows]
    assert all(next_time > time for time, next_time in zip(times, times[1:]))
    assert np.allclose(np.diff(times), [1.0 / 104.0, 1.0 / 104.0], atol=1e-6)


def test_non_collection_recording_keeps_packet_t_for_saved_time(tmp_path):
    recorder = SessionRecorder()
    session_dir = tmp_path / "recordings" / "manual"
    recorder.start(session_dir=session_dir, metadata={}, collection_mode=False)

    packets = [
        EmgPacket(
            seq=index,
            sample_index=100 + index,
            t=42.0 + index,
            values_uv=np.full(8, index, dtype=np.float64),
            raw_packet=b"",
        )
        for index in range(2)
    ]

    recorder.write_emg_packets(packets)
    recorder.stop()

    with (session_dir / "emg.csv").open(newline="", encoding="utf-8") as file_obj:
        rows = list(csv.DictReader(file_obj))

    assert "t_emg" in rows[0]
    assert [float(row["t_emg"]) for row in rows] == [42.0, 43.0]


def test_start_collection_generates_trial_list_and_dataset_session(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _app()
    window = MainWindow(simulate=False)
    monkeypatch.setattr(window, "_collection_preflight_ok", lambda: True)

    window._subject_id_edit.setText("subject_01")
    window._session_id_edit.setText("session_001")
    window._trials_per_gesture_spin.setValue(1)
    window._randomize_order_checkbox.setChecked(False)

    window._start_collection()

    assert window._collection_active is True
    assert [trial.gesture for trial in window._collect_trials] == [
        "fist",
        "finger_spread",
        "thumb_index_pinch",
    ]
    assert window._recorder.is_recording is True
    metadata_path = tmp_path / "dataset" / "subject_01" / "session_001" / "metadata.json"
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["gestures"] == ["fist", "finger_spread", "thumb_index_pinch"]
    assert metadata["gesture_display_names"] == GESTURE_DISPLAY_NAMES

    window._stop_collection()
    assert window._recorder.is_recording is False
    window.close()


def test_collection_trial_flow_writes_four_phase_events(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _app()
    window = MainWindow(simulate=False)
    monkeypatch.setattr(window, "_collection_preflight_ok", lambda: True)
    window._raw_emg_buffer.append_many(
        np.array([0.0]),
        np.ones((1, 8), dtype=np.float64),
        np.array([123], dtype=np.int64),
    )
    window._subject_id_edit.setText("subject_01")
    window._session_id_edit.setText("session_001")
    window._trials_per_gesture_spin.setValue(1)
    window._randomize_order_checkbox.setChecked(False)

    window._start_collection()
    window._collection_timer.stop()
    window._gesture_start()
    window._collection_timer.stop()
    window._gesture_end()
    window._collection_timer.stop()
    window._trial_end()
    window._collection_timer.stop()
    window._stop_collection()

    events_path = tmp_path / "dataset" / "subject_01" / "session_001" / "events.csv"
    with events_path.open(newline="", encoding="utf-8") as file_obj:
        rows = list(csv.DictReader(file_obj))

    assert [row["phase"] for row in rows] == [
        "trial_start",
        "gesture_start",
        "gesture_end",
        "trial_end",
    ]
    assert {row["trial_id"] for row in rows} == {"0001"}
    assert {row["gesture"] for row in rows} == {"fist"}
    assert {row["gesture_name"] for row in rows} == {"全力握拳"}
    assert {row["sample_index"] for row in rows} == {"123"}
    window.close()


def test_collection_phase_timing_uses_ui_durations_and_updates_trial_duration(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _app()
    window = MainWindow(simulate=False)
    monkeypatch.setattr(window, "_collection_preflight_ok", lambda: True)
    scheduled: list[tuple[int, str]] = []

    def fake_schedule(delay_ms: int, step) -> None:
        scheduled.append((delay_ms, step.__name__))
        window._collection_step = step

    monkeypatch.setattr(window, "_schedule_collection_step", fake_schedule)
    window._subject_id_edit.setText("subject_01")
    window._session_id_edit.setText("session_001")
    window._trials_per_gesture_spin.setValue(1)
    window._randomize_order_checkbox.setChecked(False)
    window._trial_duration_spin.setValue(3.0)
    window._rest_before_spin.setValue(2.0)
    window._hold_duration_spin.setValue(2.0)
    window._rest_after_spin.setValue(2.0)

    window._start_collection()
    window._gesture_start()
    window._gesture_end()

    assert window._trial_duration_spin.value() == 6.0
    assert scheduled[:3] == [
        (2000, "_gesture_start"),
        (2000, "_gesture_end"),
        (2000, "_trial_end"),
    ]

    window._stop_collection()
    window.close()


def test_collection_pause_resume_preserves_current_phase_remaining_time(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _app()
    window = MainWindow(simulate=False)
    monkeypatch.setattr(window, "_collection_preflight_ok", lambda: True)
    window._subject_id_edit.setText("subject_01")
    window._session_id_edit.setText("session_001")
    window._trials_per_gesture_spin.setValue(1)
    window._randomize_order_checkbox.setChecked(False)
    window._rest_before_spin.setValue(2.0)
    window._hold_duration_spin.setValue(2.0)
    window._rest_after_spin.setValue(2.0)

    window._start_collection()
    original_step = window._collection_step
    window._pause_collection()
    remaining_ms = window._collection_remaining_ms
    resumed: list[tuple[int, str]] = []

    def fake_schedule(delay_ms: int, step) -> None:
        resumed.append((delay_ms, step.__name__))

    monkeypatch.setattr(window, "_schedule_collection_step", fake_schedule)
    window._pause_collection()

    assert original_step is not None
    assert 1 <= remaining_ms <= 2000
    assert resumed == [(remaining_ms, "_gesture_start")]

    window._stop_collection()
    window.close()
