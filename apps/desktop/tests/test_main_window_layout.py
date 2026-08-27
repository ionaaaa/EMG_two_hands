import os
import re

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QApplication, QGroupBox, QToolBar, QWidget

from emg_live_marker.device.protocol import EmgPacket
from emg_live_marker.paths import ProjectPaths
from emg_live_marker.ui import main_window as main_window_module
from emg_live_marker.ui.main_window import MainWindow
from emg_live_marker.ui.style import APP_QSS


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _is_descendant(widget: QWidget, ancestor: QWidget) -> bool:
    parent = widget.parentWidget()
    while parent is not None:
        if parent is ancestor:
            return True
        parent = parent.parentWidget()
    return False


def test_main_window_has_no_top_toolbar_and_uses_light_theme():
    _app()
    window = MainWindow(simulate=False)

    assert window.findChildren(QToolBar) == []
    assert window.styleSheet() == APP_QSS

    window.close()


def test_device_and_recording_controls_are_in_left_sidebar():
    _app()
    window = MainWindow(simulate=False)
    sidebar = window._left_sidebar

    assert window._left_sidebar_scroll.widget() is sidebar
    assert window._left_sidebar_scroll.widgetResizable() is False
    assert window._left_sidebar_scroll.minimumWidth() == 360
    assert window._left_sidebar_scroll.maximumWidth() == 360
    assert sidebar.minimumWidth() == 340
    assert sidebar.maximumWidth() == 340
    assert window._left_sidebar_scroll.verticalScrollBarPolicy().name == "ScrollBarAlwaysOn"
    for widget in [
        window._port_combo,
        window._refresh_ports_button,
        window._connect_button,
        window._disconnect_button,
        window._simulate_checkbox,
        window._start_recording_button,
        window._stop_recording_button,
        window._open_recording_folder_button,
    ]:
        assert _is_descendant(widget, sidebar)

    window.close()


def test_device_controls_are_vertically_separated():
    app = _app()
    window = MainWindow(simulate=False)
    window.show()
    app.processEvents()

    widgets = [
        window._port_combo,
        window._refresh_ports_button,
        window._connect_button,
        window._simulate_checkbox,
    ]
    tops = [widget.mapTo(window, QPoint(0, 0)).y() for widget in widgets]
    bottoms = [top + widget.height() for top, widget in zip(tops, widgets)]

    assert bottoms[0] + 4 <= tops[1]
    assert bottoms[1] + 4 <= tops[2]
    assert bottoms[2] + 4 <= tops[3]
    assert window._port_combo.minimumWidth() >= 180
    assert window._refresh_ports_button.minimumHeight() >= 30
    assert window._connect_button.minimumHeight() >= 30
    assert window._disconnect_button.minimumHeight() >= 30

    window.close()


def test_left_sidebar_groups_do_not_overlap():
    app = _app()
    window = MainWindow(simulate=False)
    window.show()
    app.processEvents()

    direct_groups = [
        group
        for group in window._left_sidebar.findChildren(
            QGroupBox,
            options=Qt.FindChildOption.FindDirectChildrenOnly,
        )
    ]
    group_titles = [group.title() for group in direct_groups]
    assert group_titles == ["Device", "Recording", "Display", "Signal"]

    tops = [group.mapTo(window, QPoint(0, 0)).y() for group in direct_groups]
    bottoms = [top + group.height() for top, group in zip(tops, direct_groups)]
    for previous_bottom, next_top in zip(bottoms, tops[1:]):
        assert previous_bottom <= next_top

    device_group = direct_groups[0]
    simulate_top = window._simulate_checkbox.mapTo(window, QPoint(0, 0)).y()
    simulate_bottom = simulate_top + window._simulate_checkbox.height()
    device_top = device_group.mapTo(window, QPoint(0, 0)).y()
    assert simulate_bottom <= device_top + device_group.height()

    window.close()


def test_game_decoder_prefers_effie_dual_model_path(monkeypatch, tmp_path):
    _app()
    effie_model = tmp_path / "models" / "effie_real_full_v2_continue" / "gesture_classifier.ts"
    effie_model.parent.mkdir(parents=True)
    effie_model.write_bytes(b"dummy")
    calibration_model = tmp_path / "models" / "calibration_game_model" / "gesture_classifier.ts"
    calibration_model.parent.mkdir(parents=True)
    calibration_model.write_bytes(b"dummy")
    fallback_model = tmp_path / "models" / "gesture_classifier.pt"
    paths = ProjectPaths(
        project_root=tmp_path,
        dataset_root=tmp_path / "dataset",
        recordings_root=tmp_path / "recordings",
        artifacts_root=tmp_path,
    )
    window = MainWindow(simulate=False, paths=paths)

    assert window._game_model_path_edit.text() == str(effie_model)
    window.close()


def test_game_decoder_runtime_parameter_controls_update_decoder():
    _app()
    window = MainWindow(simulate=False)

    window._game_confidence_threshold_spin.setValue(0.55)
    window._game_smoothing_frames_spin.setValue(7)
    window._game_change_confirm_frames_spin.setValue(3)

    assert window._game_decoder.confidence_threshold == 0.55
    assert window._right_game_decoder.confidence_threshold == 0.55
    assert window._game_bridge.confidence_threshold == 0.55
    assert window._game_decoder.smoothing_frames == 7
    assert window._right_game_decoder.smoothing_frames == 7
    assert window._game_decoder.change_confirmations == 3
    assert window._right_game_decoder.change_confirmations == 3
    window.close()


def test_cross_session_model_warning_is_displayed(tmp_path):
    _app()
    model_dir = tmp_path / "models" / "emg2pose_gesture_v1"
    model_dir.mkdir(parents=True)
    (model_dir / "train_report.json").write_text(
        '{"holdout_session": "session_005", "val_split": "session", "training_args": {}}',
        encoding="utf-8",
    )
    window = MainWindow(simulate=False)

    window._update_game_model_warning(str(model_dir / "gesture_classifier.ts"))

    assert window._game_warning_label.text()
    window.close()


def test_collect_mode_is_in_right_sidebar_not_left_sidebar():
    _app()
    window = MainWindow(simulate=False)

    collect_widgets = [
        window._subject_id_edit,
        window._session_id_edit,
        window._trials_per_gesture_spin,
        window._trial_duration_spin,
        window._start_collection_button,
        window._pause_collection_button,
        window._stop_collection_button,
        window._repeat_trial_button,
        window._collection_progress_label,
        window._collection_prompt_label,
    ]

    assert window._right_sidebar.minimumWidth() == 380
    assert window._right_sidebar.maximumWidth() == 380
    assert 360 <= window._right_sidebar.width() <= 400
    assert window._collect_group.minimumHeight() == 500
    assert window._collect_group.maximumHeight() > 100000
    assert window._right_sidebar.layout().count() == 1
    assert window._right_sidebar.layout().itemAt(0).widget() is window._collect_group
    for widget in collect_widgets:
        assert _is_descendant(widget, window._right_sidebar)
        assert not _is_descendant(widget, window._left_sidebar)

    window.close()


def test_collect_prompt_uses_large_readable_text():
    _app()
    window = MainWindow(simulate=False)

    match = re.search(r"font-size:\s*(\d+)px", window._collection_prompt_label.styleSheet())

    assert match is not None
    assert 32 <= int(match.group(1)) <= 48
    assert window._collection_prompt_label.minimumHeight() >= 88
    assert window._collection_trial_label.text() == "Trial 0 / 0"
    assert window._collection_gesture_label.text() == "Gesture: -"
    assert window._collection_phase_label.text() == "Phase: IDLE"

    window.close()


def test_event_markers_panel_is_removed_from_right_sidebar():
    _app()
    window = MainWindow(simulate=False)

    right_group_titles = {group.title() for group in window._right_sidebar.findChildren(QGroupBox)}
    assert right_group_titles == {"Collect Mode"}
    assert "Event Markers" not in right_group_titles
    assert not hasattr(window, "_event_panel")
    assert not hasattr(window, "_event_markers_group")

    window.close()


def test_game_decoder_controls_are_fixed_at_window_bottom():
    _app()
    window = MainWindow(simulate=False)

    assert not _is_descendant(window._game_decoder_group, window._left_sidebar)
    for widget in [
        window._game_model_path_edit,
        window._load_game_model_button,
        window._enable_game_control_checkbox,
        window._game_api_status_label,
        window._game_current_gesture_label,
        window._game_confidence_label,
        window._send_fist_button,
        window._send_open_palm_button,
        window._send_pinch_button,
        window._send_rest_button,
    ]:
        assert _is_descendant(widget, window._game_decoder_group)
        assert not _is_descendant(widget, window._left_sidebar)

    root_layout = window.centralWidget().layout()
    assert root_layout.itemAt(0).widget() is window._top_workspace
    assert root_layout.itemAt(1).widget() is window._game_decoder_group
    assert window._game_decoder_group.minimumHeight() == 235
    assert window._game_decoder_group.maximumHeight() == 235
    assert window._game_mode_label.text() == "Mode: Demo Mode"
    window.close()


def test_game_decoder_send_test_buttons_use_game_bridge(monkeypatch):
    _app()
    window = MainWindow(simulate=False)
    sent: list[tuple[str, float]] = []
    monkeypatch.setattr(
        window._game_bridge,
        "send_gesture",
        lambda gesture, confidence: sent.append((gesture, confidence)),
    )

    window._send_fist_button.click()
    window._send_open_palm_button.click()
    window._send_pinch_button.click()
    window._send_rest_button.click()

    assert sent == [
        ("fist", 0.95),
        ("open-palm", 0.95),
        ("pinch", 0.95),
        ("rest", 0.95),
    ]
    window.close()


def test_start_collection_disables_game_control(tmp_path, monkeypatch):
    _app()
    paths = ProjectPaths(
        project_root=tmp_path,
        dataset_root=tmp_path / "dataset",
        recordings_root=tmp_path / "recordings",
        artifacts_root=tmp_path / "artifacts",
    )
    window = MainWindow(simulate=False, paths=paths)
    monkeypatch.setattr(window, "_collection_preflight_ok", lambda: True)
    window._subject_id_edit.setText("subject_01")
    window._session_id_edit.setText("session_001")
    window._trials_per_gesture_spin.setValue(1)
    window._randomize_order_checkbox.setChecked(False)

    window._enable_game_control_checkbox.setChecked(True)
    assert window._game_decoder.enabled is True

    window._start_collection()

    assert window._enable_game_control_checkbox.isChecked() is False
    assert window._game_decoder.enabled is False

    window._stop_collection()
    window.close()


def test_waveform_timer_runs_at_20fps_without_frequency_panel():
    _app()
    window = MainWindow(simulate=False)

    assert window._plot_timer.interval() == 50
    assert not hasattr(window, "_spectrum_timer")
    assert not hasattr(window, "_spectrum_view")
    assert not hasattr(window, "_spectrum_panel")
    assert not hasattr(window, "_spectrum_mode_combo")
    assert not hasattr(window, "_spectrum_source_combo")

    window.close()


def test_collection_start_keeps_waveform_available_without_spectrum(tmp_path, monkeypatch):
    _app()
    paths = ProjectPaths(
        project_root=tmp_path,
        dataset_root=tmp_path / "dataset",
        recordings_root=tmp_path / "recordings",
        artifacts_root=tmp_path / "artifacts",
    )
    window = MainWindow(simulate=False, paths=paths)
    monkeypatch.setattr(window, "_collection_preflight_ok", lambda: True)
    window._subject_id_edit.setText("subject_01")
    window._session_id_edit.setText("session_001")
    window._trials_per_gesture_spin.setValue(1)
    window._randomize_order_checkbox.setChecked(False)

    window._start_collection()

    assert window._collection_trial_label.text() == "Trial 1 / 3"
    assert window._collection_phase_label.text() == "Phase: READY"
    assert window._waveform_view is not None

    window._stop_collection()
    window.close()


def test_main_display_contains_waveform_without_frequency_panel():
    app = _app()
    window = MainWindow(simulate=False)
    window.resize(1400, 900)
    window.show()
    app.processEvents()

    assert window._waveform_panel.parentWidget() is window._main_display
    assert window._main_display.layout().count() == 1
    assert "Waveform" in window._waveform_panel.title()
    assert window._main_display.parentWidget() is window._top_workspace
    assert window._game_decoder_group.y() > window._top_workspace.y()

    window.close()


def test_notch_combo_exposes_harmonic_options():
    _app()
    window = MainWindow(simulate=False)

    options = [window._notch_combo.itemText(index) for index in range(window._notch_combo.count())]

    assert options == ["Off", "50Hz", "50+100Hz", "60Hz", "60+120Hz"]

    window.close()


def test_notch_change_clears_processed_buffers():
    _app()
    window = MainWindow(simulate=False)
    t = np.arange(128, dtype=np.float64) / 250.0
    data = np.ones((128, 8), dtype=np.float32)
    sample_index = np.arange(128, dtype=np.int64)

    window._filtered_emg_buffer.append_many(t, data, sample_index)

    window._set_notch("50+100Hz")
    _t, filtered, _sample_index = window._filtered_emg_buffer.get_window(10.0)

    assert window._stream_processor.config.notch_freq == (50.0, 100.0)
    assert filtered.shape[0] == 0

    window.close()


def test_status_bar_displays_global_lost_count():
    _app()
    window = MainWindow(simulate=False)

    window._on_stats_updated({"global_lost_count": 7, "aa_count": 10, "bb_count": 4})

    assert window._global_lost_label.text() == "Global lost: 7"
    assert window._aa_count_label.text() == "AA packets: 10"
    assert window._bb_count_label.text() == "BB packets: 4"

    window.close()


def test_on_emg_packets_writes_processed_filtered_data(monkeypatch):
    _app()
    window = MainWindow(simulate=False)
    packets = [
        EmgPacket(
            seq=index,
            sample_index=index,
            t=index / 250.0,
            values_uv=np.full(8, float(index), dtype=np.float64),
            raw_packet=b"",
        )
        for index in range(4)
    ]

    class FakeProcessor:
        def process_block(self, raw_uv: np.ndarray) -> dict[str, np.ndarray]:
            return {
                "raw": raw_uv,
                "filtered": raw_uv + 10.0,
                "rectified": np.abs(raw_uv + 10.0),
                "rms": raw_uv + 20.0,
            }

    monkeypatch.setattr(window, "_stream_processor", FakeProcessor())

    window._on_emg_packets(packets)
    _t, filtered, _sample_index = window._filtered_emg_buffer.get_window(10.0)

    np.testing.assert_allclose(filtered, np.vstack([packet.values_uv for packet in packets]) + 10.0)

    window.close()


def test_right_emg_packets_use_independent_buffers(monkeypatch):
    _app()
    window = MainWindow(simulate=False)
    packets = [
        EmgPacket(
            seq=index,
            sample_index=index,
            t=index / 250.0,
            values_uv=np.full(8, float(index + 1), dtype=np.float64),
            raw_packet=b"",
        )
        for index in range(4)
    ]

    class FakeProcessor:
        def process_block(self, raw_uv: np.ndarray) -> dict[str, np.ndarray]:
            return {
                "raw": raw_uv,
                "filtered": raw_uv + 100.0,
                "rectified": raw_uv + 200.0,
                "rms": raw_uv + 300.0,
            }

    window._runtimes["right"].stream_processor = FakeProcessor()

    window._on_emg_packets_for_side("right", packets)
    _left_t, left_raw, _left_sample_index = window._raw_emg_buffer.get_window(10.0)
    _right_t, right_raw, _right_sample_index = window._right_raw_emg_buffer.get_window(10.0)
    _right_filtered_t, right_filtered, _right_filtered_sample_index = (
        window._right_filtered_emg_buffer.get_window(10.0)
    )

    assert left_raw.shape[0] == 0
    np.testing.assert_allclose(right_raw, np.vstack([packet.values_uv for packet in packets]))
    np.testing.assert_allclose(right_filtered, right_raw + 100.0)
    window.close()


def test_load_game_model_sets_shared_predictor_on_both_decoders(monkeypatch):
    _app()
    window = MainWindow(simulate=False)

    class FakePredictor:
        model_type = "demo"
        signal_type = "filtered"
        normalization_loaded = False
        window_samples = 4
        model_info: dict[str, object] = {}

        def predict_window(self, window: np.ndarray) -> dict[str, object]:
            _ = window
            probs = {"rest": 1.0}
            return {"gesture": "rest", "confidence": 1.0, "probs": probs}

    predictor = FakePredictor()
    monkeypatch.setattr(main_window_module, "load_model", lambda path: predictor)

    window._game_model_path_edit.setText("dummy.ts")
    window._load_game_model()

    assert window._game_decoder.predictor is predictor
    assert window._right_game_decoder.predictor is predictor
    assert window._game_decoder._inference_lock is window._right_game_decoder._inference_lock
    window.close()
