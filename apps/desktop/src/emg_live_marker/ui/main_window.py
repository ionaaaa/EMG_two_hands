"""Main desktop window for live EMG display and recording."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from importlib import resources
from pathlib import Path
from threading import Lock
from time import monotonic

import numpy as np
from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from emg_live_marker.config import DEFAULT_BAUDRATE
from emg_live_marker.device.protocol import EMG_CHANNELS, EMG_FS, IMU_FS
from emg_live_marker.device.serial_source import SerialSource, list_serial_ports
from emg_live_marker.device.simulator import SimulatedDevice, SimulatorConfig
from emg_live_marker.ml.dual_game_mapper import BraceletSide, DualGameMapper, NoopKeySink
from emg_live_marker.ml.game_bridge import GameBridge
from emg_live_marker.ml.gesture_model import load_model
from emg_live_marker.ml.realtime_decoder import RealtimeGestureDecoder
from emg_live_marker.realtime.collection import (
    COLLECTION_GESTURES,
    GESTURE_DISPLAY_NAMES,
    CollectionTrial,
    build_trial_list,
    gesture_display_name,
)
from emg_live_marker.realtime.gesture_server import GestureServer
from emg_live_marker.realtime.recorder import SessionRecorder
from emg_live_marker.realtime.ring_buffer import EmgRingBuffer, ImuRingBuffer
from emg_live_marker.realtime.stream_processor import StreamingEMGProcessor
from emg_live_marker.ui.style import APP_QSS
from emg_live_marker.ui.waveform_view import MultiChannelWaveformView

NOTCH_OPTIONS: dict[str, tuple[float, ...]] = {
    "Off": (),
    "50Hz": (50.0,),
    "50+100Hz": (50.0, 100.0),
    "60Hz": (60.0,),
    "60+120Hz": (60.0, 120.0),
}
DEFAULT_EFFIE_GAME_MODEL_PATH = Path("models") / "effie_real_full_v2_continue" / "gesture_classifier.ts"
CALIBRATION_MODEL_PATH = Path("models") / "calibration_game_model" / "gesture_classifier.ts"
FALLBACK_GAME_MODEL_PATH = Path("models") / "gesture_classifier.pt"
CROSS_SESSION_WARNING = (
    "Cross-session accuracy may be low. For best game demo, train a calibration model after wearing the bracelet.\n"
    "跨 session 准确率可能较低。演示前建议重新采集校准数据并训练当天模型。"
)


def default_game_model_path() -> Path:
    if DEFAULT_EFFIE_GAME_MODEL_PATH.exists():
        return DEFAULT_EFFIE_GAME_MODEL_PATH
    if CALIBRATION_MODEL_PATH.exists():
        return CALIBRATION_MODEL_PATH
    return FALLBACK_GAME_MODEL_PATH


@dataclass
class BraceletRuntime:
    side: BraceletSide
    raw_emg_buffer: EmgRingBuffer
    filtered_emg_buffer: EmgRingBuffer
    rectified_emg_buffer: EmgRingBuffer
    rms_emg_buffer: EmgRingBuffer
    imu_buffer: ImuRingBuffer
    stream_processor: StreamingEMGProcessor
    decoder: RealtimeGestureDecoder | None = None
    simulator: SimulatedDevice | None = None
    serial_source: SerialSource | None = None
    connected: bool = False
    current_gesture: str = "rest"
    confidence: float = 0.0
    probs: dict[str, float] | None = None
    aa_count: int = 0
    bb_count: int = 0
    aa_lost_count: int = 0
    bb_lost_count: int = 0
    global_lost_count: int = 0
    bad_header_count: int = 0
    bad_type_count: int = 0
    resync_count: int = 0
    emg_rate_sps: float = 0.0
    imu_rate_sps: float = 0.0


class MainWindow(QMainWindow):
    def __init__(
        self,
        simulate: bool = True,
        port: str | None = None,
        baudrate: int = DEFAULT_BAUDRATE,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("emg_live_marker")
        self.resize(1400, 900)

        self._initial_port = port
        self._baudrate = int(baudrate)
        self._recorder = SessionRecorder()
        self._last_recording_dir: Path | None = None
        self._runtimes: dict[BraceletSide, BraceletRuntime] = {
            "left": BraceletRuntime(
                side="left",
                raw_emg_buffer=EmgRingBuffer(seconds=90.0),
                filtered_emg_buffer=EmgRingBuffer(seconds=90.0),
                rectified_emg_buffer=EmgRingBuffer(seconds=90.0),
                rms_emg_buffer=EmgRingBuffer(seconds=90.0),
                imu_buffer=ImuRingBuffer(seconds=90.0),
                stream_processor=StreamingEMGProcessor(),
                probs={},
            ),
            "right": BraceletRuntime(
                side="right",
                raw_emg_buffer=EmgRingBuffer(seconds=90.0),
                filtered_emg_buffer=EmgRingBuffer(seconds=90.0),
                rectified_emg_buffer=EmgRingBuffer(seconds=90.0),
                rms_emg_buffer=EmgRingBuffer(seconds=90.0),
                imu_buffer=ImuRingBuffer(seconds=90.0),
                stream_processor=StreamingEMGProcessor(),
                probs={},
            ),
        }
        self._shared_inference_lock = Lock()
        # 实时手势已通过SSE直接驱动网页游戏，不再注入键盘按键（避免Space等键误触发网页startGame重置）
        self._dual_game_mapper = DualGameMapper(key_sink=NoopKeySink())
        left_runtime = self._runtimes["left"]
        right_runtime = self._runtimes["right"]
        self._simulator: SimulatedDevice | None = None
        self._serial_source: SerialSource | None = None
        self._right_simulator: SimulatedDevice | None = None
        self._right_serial_source: SerialSource | None = None
        self._raw_emg_buffer = left_runtime.raw_emg_buffer
        self._filtered_emg_buffer = left_runtime.filtered_emg_buffer
        self._rectified_emg_buffer = left_runtime.rectified_emg_buffer
        self._rms_emg_buffer = left_runtime.rms_emg_buffer
        self._emg_buffer = self._raw_emg_buffer
        self._imu_buffer = left_runtime.imu_buffer
        self._stream_processor = left_runtime.stream_processor
        self._right_raw_emg_buffer = right_runtime.raw_emg_buffer
        self._right_filtered_emg_buffer = right_runtime.filtered_emg_buffer
        self._right_rectified_emg_buffer = right_runtime.rectified_emg_buffer
        self._right_rms_emg_buffer = right_runtime.rms_emg_buffer
        self._right_imu_buffer = right_runtime.imu_buffer
        self._right_stream_processor = right_runtime.stream_processor
        self._recording = False
        self._aa_count = 0
        self._bb_count = 0
        self._aa_lost_count = 0
        self._bb_lost_count = 0
        self._global_lost_count = 0
        self._bad_header_count = 0
        self._bad_type_count = 0
        self._resync_count = 0
        self._emg_rate_sps = 0.0
        self._imu_rate_sps = 0.0
        self._last_rate_counts = (0, 0)
        self._last_side_rate_counts: dict[BraceletSide, tuple[int, int]] = {
            "left": (0, 0),
            "right": (0, 0),
        }
        self._last_global_lost_for_rate = 0
        self._global_lost_delta_per_sec = 0
        self._display_seconds = 10.0
        self._display_mode = "Raw"
        self._game_bridge = GameBridge(confidence_threshold=0.70)
        left_runtime.decoder = RealtimeGestureDecoder(
            self._filtered_emg_buffer,
            raw_emg_buffer=self._raw_emg_buffer,
            filtered_emg_buffer=self._filtered_emg_buffer,
            confidence_threshold=0.70,
            send_to_game_bridge=False,
            parent=self,
        )
        right_runtime.decoder = RealtimeGestureDecoder(
            self._right_filtered_emg_buffer,
            raw_emg_buffer=self._right_raw_emg_buffer,
            filtered_emg_buffer=self._right_filtered_emg_buffer,
            confidence_threshold=0.70,
            send_to_game_bridge=False,
            parent=self,
        )
        self._game_decoder = left_runtime.decoder
        self._right_game_decoder = right_runtime.decoder
        self._collect_trials: list[CollectionTrial] = []
        self._current_trial_index = -1
        self._current_trial: CollectionTrial | None = None
        self._last_completed_trial: CollectionTrial | None = None
        self._collection_active = False
        self._collection_paused = False
        self._collection_step = None
        self._collection_phase = "IDLE"
        self._collection_phase_before_pause = "IDLE"
        self._collection_step_deadline = 0.0
        self._collection_remaining_ms = 0
        self._collection_rest_before_s = 0.5
        self._collection_hold_s = 1.5
        self._collection_rest_after_s = 1.0
        self._collection_trial_duration_s = 3.0
        self._session_id_auto_generated = True

        self._build_actions()
        self._build_central_widget()
        self._build_status_bar()
        self._gesture_server = GestureServer(host="127.0.0.1", port=8766)
        try:
            self._gesture_server.start()
            self.statusBar().showMessage("Gesture SSE server on http://127.0.0.1:8766/events", 3000)
        except OSError as exc:
            self.statusBar().showMessage(f"Gesture SSE server failed to start: {exc}", 6000)
        self._apply_light_theme()
        self._refresh_ports()
        if port:
            if self._port_combo.findText(port) < 0:
                self._port_combo.addItem(port)
            self._port_combo.setCurrentText(port)
        if self._right_port_combo.count() == 0:
            self._right_port_combo.addItem("COM5")

        self._plot_timer = QTimer(self)
        self._plot_timer.setInterval(50)
        self._plot_timer.timeout.connect(self._refresh_waveform)
        self._plot_timer.start()

        self._status_timer = QTimer(self)
        self._status_timer.setInterval(1000)
        self._status_timer.timeout.connect(self._refresh_status_rates)
        self._status_timer.start()

        self._collection_timer = QTimer(self)
        self._collection_timer.setSingleShot(True)
        self._collection_timer.timeout.connect(self._run_collection_step)

        self._simulate_checkbox.setChecked(simulate)
        self._apply_simulate_ui_state()
        self._apply_right_simulate_ui_state()
        self._stop_recording_button.setEnabled(False)
        self._disconnect_button.setEnabled(False)
        self._disconnect_right_button.setEnabled(False)
        self._update_waveform_visibility()
        self._set_collection_button_state(active=False)
        if simulate or port:
            self._connect()

    def closeEvent(self, event) -> None:  # noqa: N802
        self._game_decoder.close()
        self._right_game_decoder.close()
        self._dual_game_mapper.release_all()
        self._game_bridge.close()
        self._gesture_server.stop()
        if self._collection_active:
            self._stop_collection()
        else:
            self._stop_recording()
        self._disconnect_all()
        super().closeEvent(event)

    def _build_actions(self) -> None:
        self._port_combo = QComboBox()
        self._port_combo.setEditable(True)
        self._right_port_combo = QComboBox()
        self._right_port_combo.setEditable(True)
        self._refresh_ports_button = QPushButton("Refresh Ports")
        self._connect_button = QPushButton("Connect Left")
        self._disconnect_button = QPushButton("Disconnect Left")
        self._connect_right_button = QPushButton("Connect Right")
        self._disconnect_right_button = QPushButton("Disconnect Right")
        self._simulate_checkbox = QCheckBox("Simulate Left")
        self._simulate_right_checkbox = QCheckBox("Simulate Right")
        self._start_recording_button = QPushButton("Start Recording")
        self._stop_recording_button = QPushButton("Stop Recording")
        self._open_recording_folder_button = QPushButton("Open Recording Folder")
        self._start_collection_button = QPushButton("Start Collection")
        self._pause_collection_button = QPushButton("Pause Collection")
        self._stop_collection_button = QPushButton("Stop Collection")
        self._repeat_trial_button = QPushButton("Repeat Trial")
        self._load_game_model_button = QPushButton("Load Model")
        self._browse_game_model_button = QPushButton("...")
        self._enable_game_control_checkbox = QCheckBox("Enable Game Control")
        self._send_fist_button = QPushButton("Fist")
        self._send_open_palm_button = QPushButton("Open Palm")
        self._send_pinch_button = QPushButton("Pinch")
        self._send_rest_button = QPushButton("Rest")

        self._refresh_ports_button.clicked.connect(self._refresh_ports)
        self._connect_button.clicked.connect(self._connect)
        self._disconnect_button.clicked.connect(self._disconnect)
        self._connect_right_button.clicked.connect(self._connect_right)
        self._disconnect_right_button.clicked.connect(self._disconnect_right)
        self._simulate_checkbox.toggled.connect(self._on_simulate_toggled)
        self._simulate_right_checkbox.toggled.connect(self._on_right_simulate_toggled)
        self._start_recording_button.clicked.connect(self._start_recording)
        self._stop_recording_button.clicked.connect(self._stop_recording)
        self._open_recording_folder_button.clicked.connect(self._open_recording_folder)
        self._start_collection_button.clicked.connect(self._start_collection)
        self._pause_collection_button.clicked.connect(self._pause_collection)
        self._stop_collection_button.clicked.connect(self._stop_collection)
        self._repeat_trial_button.clicked.connect(self._repeat_trial)
        self._load_game_model_button.clicked.connect(self._load_game_model)
        self._browse_game_model_button.clicked.connect(self._browse_game_model)
        self._enable_game_control_checkbox.toggled.connect(self._set_game_control_enabled)
        self._send_fist_button.clicked.connect(lambda: self._send_test_game_gesture("fist"))
        self._send_open_palm_button.clicked.connect(lambda: self._send_test_game_gesture("open-palm"))
        self._send_pinch_button.clicked.connect(lambda: self._send_test_game_gesture("pinch"))
        self._send_rest_button.clicked.connect(lambda: self._send_test_game_gesture("rest"))
        self._game_decoder.gesture_changed.connect(self._on_game_gesture_changed)
        self._right_game_decoder.gesture_changed.connect(
            lambda gesture, confidence, probs: self._on_game_gesture_changed_for_side(
                "right",
                gesture,
                confidence,
                probs,
            )
        )

    def _build_central_widget(self) -> None:
        self._waveform_view = MultiChannelWaveformView(display_seconds=self._display_seconds)
        self._right_waveform_view = MultiChannelWaveformView(display_seconds=self._display_seconds)

        self._left_sidebar = self._build_left_sidebar()
        self._right_sidebar = self._build_right_sidebar()
        self._main_display = QWidget()
        self._main_display.setObjectName("MainDisplay")
        self._top_workspace = QWidget()
        self._top_workspace.setObjectName("TopWorkspace")

        display_layout = QVBoxLayout(self._main_display)
        display_layout.setContentsMargins(10, 10, 10, 6)
        display_layout.setSpacing(8)
        self._waveform_panel = self._build_waveform_panel()
        display_layout.addWidget(self._waveform_panel, 1)

        top_layout = QHBoxLayout(self._top_workspace)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(0)
        self._left_sidebar_scroll = self._wrap_sidebar_in_scroll_area(
            self._left_sidebar,
            object_name="LeftSidebarScroll",
            width=360,
        )
        top_layout.addWidget(self._left_sidebar_scroll)
        top_layout.addWidget(self._main_display, 1)
        top_layout.addWidget(self._right_sidebar)

        root = QWidget()
        root.setObjectName("MainRoot")
        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._top_workspace, 1)
        self._game_decoder_group = self._build_game_decoder_group()
        layout.addWidget(self._game_decoder_group, 0)
        self.setCentralWidget(root)

    def _wrap_sidebar_in_scroll_area(self, widget: QWidget, *, object_name: str, width: int) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setObjectName(object_name)
        scroll.setWidget(widget)
        scroll.setWidgetResizable(False)
        scroll.setFixedWidth(width)
        scroll.setMinimumWidth(width)
        scroll.setMaximumWidth(width)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        widget.adjustSize()
        return scroll

    def _build_left_sidebar(self) -> QFrame:
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(340)
        sidebar.setMinimumWidth(340)
        sidebar.setMaximumWidth(340)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        layout.addWidget(self._build_logo_widget())
        layout.addWidget(self._build_device_group())
        layout.addWidget(self._build_recording_group())
        layout.addWidget(self._build_display_group())
        layout.addWidget(self._build_signal_group())
        layout.addStretch(1)
        sidebar.setMinimumHeight(layout.sizeHint().height())
        return sidebar

    def _build_right_sidebar(self) -> QFrame:
        sidebar = QFrame()
        sidebar.setObjectName("RightSidebar")
        sidebar.setFixedWidth(380)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self._collect_group = self._build_collect_group()
        self._collect_group.setObjectName("CollectGroup")
        self._collect_group.setMinimumHeight(500)
        layout.addWidget(self._collect_group, 1)
        return sidebar

    def _build_logo_widget(self) -> QWidget:
        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 4)
        logo_label = self._create_logo_label()
        layout.addWidget(logo_label, alignment=Qt.AlignmentFlag.AlignHCenter)
        return wrapper

    def _build_device_group(self) -> QGroupBox:
        box = QGroupBox("Device")
        box.setMinimumHeight(260)
        box.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(12, 16, 12, 12)
        layout.setSpacing(8)

        for combo in [self._port_combo, self._right_port_combo]:
            combo.setMinimumWidth(180)
            combo.setMinimumHeight(32)
        self._refresh_ports_button.setMinimumHeight(32)
        for button in [
            self._connect_button,
            self._disconnect_button,
            self._connect_right_button,
            self._disconnect_right_button,
        ]:
            button.setMinimumHeight(32)

        port_row = QWidget()
        port_row.setMinimumHeight(50)
        port_layout = QHBoxLayout(port_row)
        port_layout.setContentsMargins(0, 0, 0, 0)
        port_layout.setSpacing(8)
        port_label = QLabel("Port")
        port_label.setMinimumWidth(42)
        port_layout.addWidget(port_label)
        port_layout.addWidget(self._port_combo, 1)
        layout.addWidget(port_row)

        right_port_row = QWidget()
        right_port_row.setMinimumHeight(50)
        right_port_layout = QHBoxLayout(right_port_row)
        right_port_layout.setContentsMargins(0, 0, 0, 0)
        right_port_layout.setSpacing(8)
        right_port_label = QLabel("Right")
        right_port_label.setMinimumWidth(42)
        right_port_layout.addWidget(right_port_label)
        right_port_layout.addWidget(self._right_port_combo, 1)
        layout.addWidget(right_port_row)

        layout.addWidget(self._refresh_ports_button)

        connection_row = QWidget()
        connection_row.setMinimumHeight(54)
        connection_layout = QHBoxLayout(connection_row)
        connection_layout.setContentsMargins(0, 8, 0, 8)
        connection_layout.setSpacing(8)
        connection_layout.addWidget(self._connect_button)
        connection_layout.addWidget(self._disconnect_button)
        layout.addWidget(connection_row)

        right_connection_row = QWidget()
        right_connection_row.setMinimumHeight(54)
        right_connection_layout = QHBoxLayout(right_connection_row)
        right_connection_layout.setContentsMargins(0, 8, 0, 8)
        right_connection_layout.setSpacing(8)
        right_connection_layout.addWidget(self._connect_right_button)
        right_connection_layout.addWidget(self._disconnect_right_button)
        layout.addWidget(right_connection_row)
        layout.addWidget(self._simulate_checkbox)
        layout.addWidget(self._simulate_right_checkbox)
        return box

    def _build_recording_group(self) -> QGroupBox:
        box = QGroupBox("Recording")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(12, 16, 12, 12)
        layout.setSpacing(8)
        layout.addWidget(self._start_recording_button)
        layout.addWidget(self._stop_recording_button)
        layout.addWidget(self._open_recording_folder_button)
        return box

    def _build_game_decoder_group(self) -> QGroupBox:
        box = QGroupBox("Game Decoder")
        box.setObjectName("GameDecoderBottom")
        layout = QGridLayout(box)
        layout.setContentsMargins(8, 14, 8, 8)
        layout.setHorizontalSpacing(14)
        layout.setVerticalSpacing(6)
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setContentsMargins(0, 0, 0, 0)
        form.setVerticalSpacing(4)
        self._game_model_path_edit = QLineEdit(str(default_game_model_path()))
        path_row = QWidget()
        path_layout = QHBoxLayout(path_row)
        path_layout.setContentsMargins(0, 0, 0, 0)
        path_layout.addWidget(self._game_model_path_edit)
        path_layout.addWidget(self._browse_game_model_button)
        form.addRow("Model path", path_row)
        self._game_confidence_threshold_spin = QDoubleSpinBox()
        self._game_confidence_threshold_spin.setRange(0.0, 1.0)
        self._game_confidence_threshold_spin.setDecimals(2)
        self._game_confidence_threshold_spin.setSingleStep(0.05)
        self._game_confidence_threshold_spin.setValue(0.70)
        self._game_smoothing_frames_spin = QSpinBox()
        self._game_smoothing_frames_spin.setRange(1, 25)
        self._game_smoothing_frames_spin.setValue(5)
        self._game_change_confirm_frames_spin = QSpinBox()
        self._game_change_confirm_frames_spin.setRange(1, 10)
        self._game_change_confirm_frames_spin.setValue(2)
        self._game_pinch_threshold_spin = QDoubleSpinBox()
        self._game_pinch_threshold_spin.setRange(0.0, 1.0)
        self._game_pinch_threshold_spin.setDecimals(2)
        self._game_pinch_threshold_spin.setSingleStep(0.05)
        self._game_pinch_threshold_spin.setValue(0.80)
        self._game_pinch_boost_spin = QDoubleSpinBox()
        self._game_pinch_boost_spin.setRange(0.0, 1.0)
        self._game_pinch_boost_spin.setDecimals(2)
        self._game_pinch_boost_spin.setSingleStep(0.05)
        self._game_pinch_boost_spin.setValue(0.00)
        self._game_pinch_margin_spin = QDoubleSpinBox()
        self._game_pinch_margin_spin.setRange(0.0, 1.0)
        self._game_pinch_margin_spin.setDecimals(2)
        self._game_pinch_margin_spin.setSingleStep(0.05)
        self._game_pinch_margin_spin.setValue(0.10)
        self._game_confidence_threshold_spin.valueChanged.connect(self._set_game_confidence_threshold)
        self._game_smoothing_frames_spin.valueChanged.connect(self._set_game_smoothing_frames)
        self._game_change_confirm_frames_spin.valueChanged.connect(self._set_game_change_confirm_frames)
        self._game_pinch_threshold_spin.valueChanged.connect(self._set_game_pinch_params)
        self._game_pinch_boost_spin.valueChanged.connect(self._set_game_pinch_params)
        self._game_pinch_margin_spin.valueChanged.connect(self._set_game_pinch_params)
        form.addRow("confidence_threshold", self._game_confidence_threshold_spin)
        form.addRow("smoothing_frames", self._game_smoothing_frames_spin)
        form.addRow("change_confirm_frames", self._game_change_confirm_frames_spin)
        form.addRow("pinch_threshold", self._game_pinch_threshold_spin)
        form.addRow("pinch_boost", self._game_pinch_boost_spin)
        form.addRow("pinch_margin", self._game_pinch_margin_spin)
        layout.addLayout(form, 0, 0, 3, 1)

        control_column = QWidget()
        control_layout = QVBoxLayout(control_column)
        control_layout.setContentsMargins(0, 0, 0, 0)
        control_layout.setSpacing(6)
        control_layout.addWidget(self._load_game_model_button)
        control_layout.addWidget(self._enable_game_control_checkbox)

        button_grid = QGridLayout()
        button_grid.setContentsMargins(0, 0, 0, 0)
        button_grid.setHorizontalSpacing(6)
        button_grid.setVerticalSpacing(4)
        button_grid.addWidget(self._send_fist_button, 0, 0)
        button_grid.addWidget(self._send_open_palm_button, 0, 1)
        button_grid.addWidget(self._send_pinch_button, 1, 0)
        button_grid.addWidget(self._send_rest_button, 1, 1)
        control_layout.addLayout(button_grid)
        control_layout.addStretch(1)
        layout.addWidget(control_column, 0, 1, 3, 1)

        self._game_mode_label = QLabel("Mode: Demo Mode")
        self._game_model_type_label = QLabel("Model type: demo")
        self._game_signal_type_label = QLabel("Signal: filtered")
        self._game_normalization_label = QLabel("Normalization: no")
        self._game_api_status_label = QLabel("API status: idle")
        self._game_current_gesture_label = QLabel("Left Gesture: rest")
        self._game_confidence_label = QLabel("Left Confidence: 0%")
        self._game_probs_label = QLabel("Left Probs: rest 0% | fist 0% | open-palm 0% | pinch 0%")
        self._game_probs_label.setWordWrap(True)
        self._right_game_current_gesture_label = QLabel("Right Gesture: rest")
        self._right_game_confidence_label = QLabel("Right Confidence: 0%")
        self._right_game_probs_label = QLabel("Right Probs: rest 0% | fist 0% | open-palm 0% | pinch 0%")
        self._right_game_probs_label.setWordWrap(True)
        self._game_warning_label = QLabel("")
        self._game_warning_label.setWordWrap(True)
        self._game_warning_label.setStyleSheet("color: #b45309;")
        status_column = QWidget()
        status_layout = QGridLayout(status_column)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setHorizontalSpacing(12)
        status_layout.setVerticalSpacing(4)
        status_layout.addWidget(self._game_mode_label, 0, 0)
        status_layout.addWidget(self._game_model_type_label, 0, 1)
        status_layout.addWidget(self._game_signal_type_label, 1, 0)
        status_layout.addWidget(self._game_normalization_label, 1, 1)
        status_layout.addWidget(self._game_api_status_label, 2, 0)
        status_layout.addWidget(self._game_current_gesture_label, 2, 1)
        status_layout.addWidget(self._game_confidence_label, 3, 0)
        status_layout.addWidget(self._right_game_current_gesture_label, 3, 1)
        status_layout.addWidget(self._right_game_confidence_label, 4, 0)
        status_layout.addWidget(self._game_probs_label, 5, 0, 1, 2)
        status_layout.addWidget(self._right_game_probs_label, 6, 0, 1, 2)
        status_layout.addWidget(self._game_warning_label, 7, 0, 1, 2)
        layout.addWidget(status_column, 0, 2, 3, 1)
        layout.setColumnStretch(0, 3)
        layout.setColumnStretch(1, 2)
        layout.setColumnStretch(2, 3)
        box.setFixedHeight(235)
        return box

    def _build_collect_group(self) -> QGroupBox:
        box = QGroupBox("Collect Mode")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(6, 14, 6, 6)
        layout.setSpacing(4)

        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setContentsMargins(0, 0, 0, 0)
        form.setVerticalSpacing(3)
        self._subject_id_edit = QLineEdit("subject_01")
        self._session_id_edit = QLineEdit(self._next_session_id("subject_01"))

        self._trials_per_gesture_spin = QSpinBox()
        self._trials_per_gesture_spin.setRange(1, 10000)
        self._trials_per_gesture_spin.setValue(100)

        self._trial_duration_spin = QDoubleSpinBox()
        self._trial_duration_spin.setRange(0.1, 60.0)
        self._trial_duration_spin.setDecimals(1)
        self._trial_duration_spin.setSingleStep(0.1)
        self._trial_duration_spin.setValue(3.0)

        self._rest_before_spin = QDoubleSpinBox()
        self._rest_before_spin.setRange(0.0, 60.0)
        self._rest_before_spin.setDecimals(1)
        self._rest_before_spin.setSingleStep(0.1)
        self._rest_before_spin.setValue(0.5)

        self._hold_duration_spin = QDoubleSpinBox()
        self._hold_duration_spin.setRange(0.1, 60.0)
        self._hold_duration_spin.setDecimals(1)
        self._hold_duration_spin.setSingleStep(0.1)
        self._hold_duration_spin.setValue(1.5)

        self._rest_after_spin = QDoubleSpinBox()
        self._rest_after_spin.setRange(0.0, 60.0)
        self._rest_after_spin.setDecimals(1)
        self._rest_after_spin.setSingleStep(0.1)
        self._rest_after_spin.setValue(1.0)

        self._randomize_order_checkbox = QCheckBox("Randomize order")
        self._randomize_order_checkbox.setChecked(True)

        form.addRow("Subject ID", self._subject_id_edit)
        form.addRow("Session ID", self._session_id_edit)
        form.addRow("Trials per gesture", self._trials_per_gesture_spin)
        form.addRow("Trial duration", self._trial_duration_spin)
        form.addRow("Rest before", self._rest_before_spin)
        form.addRow("Hold duration", self._hold_duration_spin)
        form.addRow("Rest after", self._rest_after_spin)
        form.addRow(self._randomize_order_checkbox)
        layout.addLayout(form)

        button_grid = QGridLayout()
        button_grid.setContentsMargins(0, 0, 0, 0)
        button_grid.setHorizontalSpacing(6)
        button_grid.setVerticalSpacing(4)
        button_grid.addWidget(self._start_collection_button, 0, 0)
        button_grid.addWidget(self._pause_collection_button, 0, 1)
        button_grid.addWidget(self._stop_collection_button, 1, 0)
        button_grid.addWidget(self._repeat_trial_button, 1, 1)
        layout.addLayout(button_grid)

        self._collection_progress_label = QLabel("Trials: 0/0")
        self._collection_trial_label = QLabel("Trial 0 / 0")
        self._collection_gesture_label = QLabel("Gesture: -")
        self._collection_phase_label = QLabel("Phase: IDLE")
        for label in [
            self._collection_progress_label,
            self._collection_trial_label,
            self._collection_gesture_label,
            self._collection_phase_label,
        ]:
            label.setStyleSheet("font-size: 18px; font-weight: 600;")
        self._collection_prompt_label = QLabel("Current Prompt")
        self._collection_prompt_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._collection_prompt_label.setWordWrap(True)
        self._collection_prompt_label.setMinimumHeight(92)
        self._collection_prompt_label.setStyleSheet("font-size: 34px; font-weight: 800;")
        layout.addWidget(self._collection_progress_label)
        layout.addWidget(self._collection_trial_label)
        layout.addWidget(self._collection_gesture_label)
        layout.addWidget(self._collection_phase_label)
        layout.addWidget(self._collection_prompt_label)

        self._subject_id_edit.textEdited.connect(self._on_subject_id_edited)
        self._session_id_edit.textEdited.connect(self._on_session_id_edited)
        return box

    def _build_display_group(self) -> QGroupBox:
        box = QGroupBox("Display")
        form = QFormLayout(box)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setContentsMargins(12, 16, 12, 12)
        form.setVerticalSpacing(8)
        self._window_combo = QComboBox()
        self._window_combo.addItems(["5s", "10s", "30s", "60s"])
        self._window_combo.setCurrentText("10s")
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["Raw", "Filtered", "Rectified", "RMS"])
        self._y_range_combo = QComboBox()
        self._y_range_combo.addItems(
            ["Auto", "Auto robust", "+/-250 uV", "+/-500 uV", "+/-1000 uV", "+/-2000 uV", "+/-5000 uV"]
        )

        form.addRow("Display window", self._window_combo)
        form.addRow("Display mode", self._mode_combo)
        form.addRow("Y range", self._y_range_combo)

        self._window_combo.currentTextChanged.connect(self._set_display_window)
        self._mode_combo.currentTextChanged.connect(self._set_display_mode)
        self._y_range_combo.currentTextChanged.connect(self._waveform_view.set_y_range_mode)
        self._y_range_combo.currentTextChanged.connect(self._right_waveform_view.set_y_range_mode)
        return box

    def _build_signal_group(self) -> QGroupBox:
        box = QGroupBox("Signal")
        form = QFormLayout(box)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setContentsMargins(12, 16, 12, 12)
        form.setVerticalSpacing(8)
        self._notch_combo = QComboBox()
        self._notch_combo.addItems(list(NOTCH_OPTIONS))

        form.addRow("Notch", self._notch_combo)

        self._notch_combo.currentTextChanged.connect(self._set_notch)
        return box

    def _build_waveform_panel(self) -> QGroupBox:
        box = QGroupBox("Waveform")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(8, 14, 8, 8)
        self._left_waveform_label = QLabel("Left bracelet")
        self._right_waveform_label = QLabel("Right bracelet")
        layout.addWidget(self._left_waveform_label)
        layout.addWidget(self._waveform_view)
        layout.addWidget(self._right_waveform_label)
        layout.addWidget(self._right_waveform_view)
        return box

    def _create_logo_label(self) -> QLabel:
        label = QLabel()
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pixmap = QPixmap()
        logo_data: bytes | None = None
        try:
            assets = resources.files("emg_live_marker").joinpath("assets")
            for logo_name in ("omni_logo.jpg", "logo.jpg"):
                try:
                    logo_data = assets.joinpath(logo_name).read_bytes()
                    break
                except FileNotFoundError:
                    continue
        except FileNotFoundError:
            logo_data = None
        if logo_data is not None and pixmap.loadFromData(logo_data):
            label.setPixmap(pixmap.scaledToWidth(220, Qt.TransformationMode.SmoothTransformation))
        else:
            label.setText("全域智能 Omni-Intelligence")
            label.setWordWrap(True)
        return label

    def _build_status_bar(self) -> None:
        status = QStatusBar()
        status.setObjectName("MainStatusBar")
        self.setStatusBar(status)
        self._emg_rate_label = QLabel("EMG rate: 0 SPS")
        self._imu_rate_label = QLabel("IMU rate: 0 SPS")
        self._aa_count_label = QLabel("AA packets: 0")
        self._bb_count_label = QLabel("BB packets: 0")
        self._global_lost_label = QLabel("Global lost: 0")
        self._recording_label = QLabel("Recording OFF")
        for label in [
            self._emg_rate_label,
            self._imu_rate_label,
            self._aa_count_label,
            self._bb_count_label,
            self._global_lost_label,
            self._recording_label,
        ]:
            status.addPermanentWidget(label)

    def _apply_light_theme(self) -> None:
        self.setStyleSheet(APP_QSS)
        self._refresh_left_sidebar_size()

    def _refresh_left_sidebar_size(self) -> None:
        if not hasattr(self, "_left_sidebar"):
            return
        layout = self._left_sidebar.layout()
        if layout is None:
            return
        for index in range(layout.count()):
            widget = layout.itemAt(index).widget()
            if isinstance(widget, QGroupBox):
                widget.setMinimumHeight(max(widget.minimumHeight(), widget.sizeHint().height()))
        layout.activate()
        required_height = max(layout.minimumSize().height(), self._left_sidebar.sizeHint().height())
        self._left_sidebar.setMinimumHeight(required_height)
        self._left_sidebar.resize(self._left_sidebar.width(), required_height)

    def _refresh_ports(self) -> None:
        current_left = self._port_combo.currentText().strip()
        current_right = self._right_port_combo.currentText().strip()
        ports = list_serial_ports()
        for combo, current, fallback in [
            (self._port_combo, current_left, "COM4"),
            (self._right_port_combo, current_right, "COM5"),
        ]:
            combo.clear()
            combo.addItems(ports)
            if current and combo.findText(current) < 0:
                combo.addItem(current)
            if combo.count() == 0:
                combo.addItem(fallback)
            if current:
                combo.setCurrentText(current)

    def _connect(self) -> None:
        self._reset_data_streams(clear_raw=True)
        if self._simulate_checkbox.isChecked():
            self._stop_serial_source_for_side("left")
            self._start_simulator_for_side("left")
            self._set_connected_ui_for_side("left", True)
            return

        self._stop_simulator_for_side("left")
        port = self._port_combo.currentText().strip()
        if not port:
            self.statusBar().showMessage("Select a serial port before connecting.", 3000)
            return

        self._start_serial_source_for_side("left", port)
        self._set_connected_ui_for_side("left", True)

    def _disconnect(self) -> None:
        self._stop_simulator_for_side("left")
        self._stop_serial_source_for_side("left")
        self._set_connected_ui_for_side("left", False)
        self._dual_game_mapper.release_side("left")

    def _connect_right(self) -> None:
        self._reset_data_streams_for_side("right", clear_raw=True)
        if self._simulate_right_checkbox.isChecked():
            self._stop_serial_source_for_side("right")
            self._start_simulator_for_side("right")
            self._set_connected_ui_for_side("right", True)
            return

        self._stop_simulator_for_side("right")
        port = self._right_port_combo.currentText().strip()
        if not port:
            self.statusBar().showMessage("Select a right serial port before connecting.", 3000)
            return

        self._start_serial_source_for_side("right", port)
        self._set_connected_ui_for_side("right", True)

    def _disconnect_right(self) -> None:
        self._stop_simulator_for_side("right")
        self._stop_serial_source_for_side("right")
        self._set_connected_ui_for_side("right", False)
        self._dual_game_mapper.release_side("right")

    def _disconnect_all(self) -> None:
        self._disconnect()
        self._disconnect_right()

    def _on_simulate_toggled(self) -> None:
        self._apply_simulate_ui_state()
        self._disconnect()

    def _on_right_simulate_toggled(self) -> None:
        self._apply_right_simulate_ui_state()
        self._disconnect_right()

    def _apply_simulate_ui_state(self) -> None:
        simulate = self._simulate_checkbox.isChecked()
        self._port_combo.setEnabled(not simulate)
        self._refresh_ports_button.setEnabled(
            (not simulate) or (not self._simulate_right_checkbox.isChecked())
        )

    def _apply_right_simulate_ui_state(self) -> None:
        simulate = self._simulate_right_checkbox.isChecked()
        self._right_port_combo.setEnabled(not simulate)
        self._refresh_ports_button.setEnabled(
            (not simulate) or (not self._simulate_checkbox.isChecked())
        )

    def _start_simulator(self) -> None:
        self._start_simulator_for_side("left")

    def _stop_simulator(self) -> None:
        self._stop_simulator_for_side("left")

    def _start_serial_source(self, port: str) -> None:
        self._start_serial_source_for_side("left", port)

    def _stop_serial_source(self) -> None:
        self._stop_serial_source_for_side("left")

    def _start_simulator_for_side(self, side: BraceletSide) -> None:
        runtime = self._runtimes[side]
        if runtime.simulator is None:
            seed = 1 if side == "left" else 2
            runtime.simulator = SimulatedDevice(config=SimulatorConfig(seed=seed), parent=self)
            runtime.simulator.emg_packets.connect(
                lambda packets, side=side: self._on_emg_packets_for_side(side, packets)
            )
            runtime.simulator.imu_packets.connect(
                lambda packets, side=side: self._on_imu_packets_for_side(side, packets)
            )
            runtime.simulator.stats_updated.connect(
                lambda stats, side=side: self._on_stats_updated_for_side(side, stats)
            )
        runtime.simulator.start()
        runtime.connected = True
        self._sync_source_aliases()
        self.statusBar().showMessage(f"{side.title()} simulator connected.", 2000)

    def _stop_simulator_for_side(self, side: BraceletSide) -> None:
        runtime = self._runtimes[side]
        if runtime.simulator is not None:
            runtime.simulator.stop()
        runtime.connected = bool(runtime.serial_source is not None and runtime.serial_source.is_running())
        self._sync_source_aliases()

    def _start_serial_source_for_side(self, side: BraceletSide, port: str) -> None:
        self._stop_serial_source_for_side(side)
        runtime = self._runtimes[side]
        runtime.serial_source = SerialSource(parent=self)
        runtime.serial_source.emg_packets.connect(
            lambda packets, side=side: self._on_emg_packets_for_side(side, packets)
        )
        runtime.serial_source.imu_packets.connect(
            lambda packets, side=side: self._on_imu_packets_for_side(side, packets)
        )
        runtime.serial_source.raw_bytes.connect(
            lambda data, side=side: self._on_raw_bytes_for_side(side, data)
        )
        runtime.serial_source.stats_updated.connect(
            lambda stats, side=side: self._on_stats_updated_for_side(side, stats)
        )
        runtime.serial_source.connected.connect(
            lambda port, side=side: self._on_serial_connected_for_side(side, port)
        )
        runtime.serial_source.disconnected.connect(
            lambda side=side: self._on_serial_disconnected_for_side(side)
        )
        runtime.serial_source.error_occurred.connect(
            lambda message, side=side: self._on_serial_error_for_side(side, message)
        )
        runtime.serial_source.connect_port(port, self._baudrate)
        self._sync_source_aliases()

    def _stop_serial_source_for_side(self, side: BraceletSide) -> None:
        runtime = self._runtimes[side]
        if runtime.serial_source is None:
            return
        source = runtime.serial_source
        runtime.serial_source = None
        runtime.connected = bool(runtime.simulator is not None and runtime.simulator.is_running())
        source.disconnect_port()
        source.deleteLater()
        self._sync_source_aliases()

    def _on_serial_connected(self, port: str) -> None:
        self._on_serial_connected_for_side("left", port)

    def _on_serial_disconnected(self) -> None:
        self._on_serial_disconnected_for_side("left")

    def _on_serial_error(self, message: str) -> None:
        self._on_serial_error_for_side("left", message)

    def _on_serial_connected_for_side(self, side: BraceletSide, port: str) -> None:
        runtime = self._runtimes[side]
        runtime.connected = True
        self.statusBar().showMessage(
            f"{side.title()} connected to {port} at {self._baudrate} baud.",
            3000,
        )
        self._set_connected_ui_for_side(side, True)

    def _on_serial_disconnected_for_side(self, side: BraceletSide) -> None:
        runtime = self._runtimes[side]
        runtime.serial_source = None
        simulate_checked = (
            self._simulate_checkbox.isChecked()
            if side == "left"
            else self._simulate_right_checkbox.isChecked()
        )
        runtime.connected = bool(runtime.simulator is not None and runtime.simulator.is_running())
        if not simulate_checked:
            self._set_connected_ui_for_side(side, False)
        self._dual_game_mapper.release_side(side)
        self.statusBar().showMessage(f"{side.title()} serial disconnected.", 3000)
        self._sync_source_aliases()

    def _on_serial_error_for_side(self, side: BraceletSide, message: str) -> None:
        self.statusBar().showMessage(f"{side.title()}: {message}", 6000)
        self._set_connected_ui_for_side(side, False)
        self._dual_game_mapper.release_side(side)

    def _set_connected_ui(self, connected: bool) -> None:
        self._set_connected_ui_for_side("left", connected)

    def _set_connected_ui_for_side(self, side: BraceletSide, connected: bool) -> None:
        runtime = self._runtimes[side]
        runtime.connected = bool(connected)
        if side == "left":
            self._connect_button.setEnabled(not connected)
            self._disconnect_button.setEnabled(connected)
            self._simulate_checkbox.setEnabled(not connected)
            self._port_combo.setEnabled((not connected) and (not self._simulate_checkbox.isChecked()))
        else:
            self._connect_right_button.setEnabled(not connected)
            self._disconnect_right_button.setEnabled(connected)
            self._simulate_right_checkbox.setEnabled(not connected)
            self._right_port_combo.setEnabled(
                (not connected) and (not self._simulate_right_checkbox.isChecked())
            )
        self._refresh_ports_button.setEnabled(
            (not self._runtimes["left"].connected and not self._simulate_checkbox.isChecked())
            or (
                not self._runtimes["right"].connected
                and not self._simulate_right_checkbox.isChecked()
            )
        )
        self._update_waveform_visibility()
        self._publish_hand_status(side)

    def _sync_source_aliases(self) -> None:
        self._simulator = self._runtimes["left"].simulator
        self._serial_source = self._runtimes["left"].serial_source
        self._right_simulator = self._runtimes["right"].simulator
        self._right_serial_source = self._runtimes["right"].serial_source

    def _on_emg_packets(self, packets: list) -> None:
        self._runtimes["left"].stream_processor = self._stream_processor
        self._on_emg_packets_for_side("left", packets)

    def _on_emg_packets_for_side(self, side: BraceletSide, packets: list) -> None:
        if not packets:
            return
        runtime = self._runtimes[side]
        t = np.asarray([packet.t for packet in packets], dtype=np.float64)
        sample_index = np.asarray([packet.sample_index for packet in packets], dtype=np.int64)
        raw_uv = np.vstack([packet.values_uv for packet in packets]).astype(np.float32)

        runtime.raw_emg_buffer.append_many(t, raw_uv, sample_index)
        processed = runtime.stream_processor.process_block(raw_uv)
        runtime.filtered_emg_buffer.append_many(t, processed["filtered"], sample_index)
        runtime.rectified_emg_buffer.append_many(t, processed["rectified"], sample_index)
        runtime.rms_emg_buffer.append_many(t, processed["rms"], sample_index)

        runtime.aa_count += len(packets)
        self._sync_aggregate_stats()
        if side == "left" and self._recorder.is_recording:
            self._recorder.write_emg_packets(packets)
            if self._simulate_checkbox.isChecked():
                self._write_packet_raw_bytes(packets)

    def _on_imu_packets(self, packets: list) -> None:
        self._on_imu_packets_for_side("left", packets)

    def _on_imu_packets_for_side(self, side: BraceletSide, packets: list) -> None:
        runtime = self._runtimes[side]
        runtime.imu_buffer.append_packets(packets)
        runtime.bb_count += len(packets)
        self._sync_aggregate_stats()
        if side == "left" and self._recorder.is_recording:
            self._recorder.write_imu_packets(packets)
            if self._simulate_checkbox.isChecked():
                self._write_packet_raw_bytes(packets)

    def _on_raw_bytes(self, data: bytes) -> None:
        self._on_raw_bytes_for_side("left", data)

    def _on_raw_bytes_for_side(self, side: BraceletSide, data: bytes) -> None:
        if side != "left":
            return
        if self._recorder.is_recording:
            self._recorder.write_raw_bytes(data)

    def _write_packet_raw_bytes(self, packets: list) -> None:
        for packet in packets:
            raw_packet = getattr(packet, "raw_packet", b"")
            if raw_packet:
                self._recorder.write_raw_bytes(raw_packet)

    def _on_stats_updated(self, stats: dict) -> None:
        self._on_stats_updated_for_side("left", stats)

    def _on_stats_updated_for_side(self, side: BraceletSide, stats: dict) -> None:
        runtime = self._runtimes[side]
        runtime.aa_count = int(stats.get("aa_count", runtime.aa_count))
        runtime.bb_count = int(stats.get("bb_count", runtime.bb_count))
        runtime.aa_lost_count = int(stats.get("aa_lost_count", runtime.aa_lost_count))
        runtime.bb_lost_count = int(stats.get("bb_lost_count", runtime.bb_lost_count))
        runtime.global_lost_count = int(stats.get("global_lost_count", runtime.global_lost_count))
        runtime.bad_header_count = int(stats.get("bad_header_count", runtime.bad_header_count))
        runtime.bad_type_count = int(stats.get("bad_type_count", runtime.bad_type_count))
        runtime.resync_count = int(stats.get("resync_count", runtime.resync_count))
        if "emg_rate_sps" in stats:
            runtime.emg_rate_sps = float(stats["emg_rate_sps"])
        if "imu_rate_sps" in stats:
            runtime.imu_rate_sps = float(stats["imu_rate_sps"])
        self._sync_aggregate_stats()
        self._refresh_status_counts()

    def _sync_aggregate_stats(self) -> None:
        self._aa_count = sum(runtime.aa_count for runtime in self._runtimes.values())
        self._bb_count = sum(runtime.bb_count for runtime in self._runtimes.values())
        self._aa_lost_count = sum(runtime.aa_lost_count for runtime in self._runtimes.values())
        self._bb_lost_count = sum(runtime.bb_lost_count for runtime in self._runtimes.values())
        self._global_lost_count = sum(runtime.global_lost_count for runtime in self._runtimes.values())
        self._bad_header_count = sum(runtime.bad_header_count for runtime in self._runtimes.values())
        self._bad_type_count = sum(runtime.bad_type_count for runtime in self._runtimes.values())
        self._resync_count = sum(runtime.resync_count for runtime in self._runtimes.values())
        self._emg_rate_sps = sum(runtime.emg_rate_sps for runtime in self._runtimes.values())
        self._imu_rate_sps = sum(runtime.imu_rate_sps for runtime in self._runtimes.values())

    def _refresh_waveform(self) -> None:
        left_buffer = self._buffer_for_display_mode("left")
        t, data, sample_index = left_buffer.get_window(self._display_seconds)
        if data.size != 0:
            self._waveform_view.update_data(t, data, sample_index=sample_index)
        right_buffer = self._buffer_for_display_mode("right")
        right_t, right_data, right_sample_index = right_buffer.get_window(self._display_seconds)
        if right_data.size != 0:
            self._right_waveform_view.update_data(
                right_t,
                right_data,
                sample_index=right_sample_index,
            )

    def _refresh_status_rates(self) -> None:
        for side, runtime in self._runtimes.items():
            last_aa, last_bb = self._last_side_rate_counts[side]
            runtime.emg_rate_sps = float(runtime.aa_count - last_aa)
            runtime.imu_rate_sps = float(runtime.bb_count - last_bb)
            self._last_side_rate_counts[side] = (runtime.aa_count, runtime.bb_count)
        self._sync_aggregate_stats()
        self._last_rate_counts = (self._aa_count, self._bb_count)
        self._global_lost_delta_per_sec = self._global_lost_count - self._last_global_lost_for_rate
        self._last_global_lost_for_rate = self._global_lost_count
        self._refresh_status_counts()

    def _refresh_status_counts(self) -> None:
        self._emg_rate_label.setText(f"EMG rate: {self._emg_rate_sps:.0f} SPS")
        self._imu_rate_label.setText(f"IMU rate: {self._imu_rate_sps:.0f} SPS")
        self._aa_count_label.setText(f"AA packets: {self._aa_count}")
        self._bb_count_label.setText(f"BB packets: {self._bb_count}")
        self._global_lost_label.setText(f"Global lost: {self._global_lost_count}")
        self._recording_label.setText("Recording ON" if self._recording else "Recording OFF")

    def _set_display_window(self, value: str) -> None:
        self._display_seconds = float(value.removesuffix("s"))
        self._waveform_view.set_display_seconds(self._display_seconds)
        self._right_waveform_view.set_display_seconds(self._display_seconds)

    def _set_display_mode(self, value: str) -> None:
        self._display_mode = value

    def _set_notch(self, value: str) -> None:
        notch_freqs = NOTCH_OPTIONS.get(value, ())
        notch = notch_freqs or None
        for runtime in self._runtimes.values():
            runtime.stream_processor.update_config(notch_freq=notch)
        self._clear_processed_buffers()

    def _browse_game_model(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self,
            "Select gesture model",
            str(Path("models").resolve()),
            "Gesture model (*.ts *.pt *.pth);;All files (*)",
        )
        if path:
            self._game_model_path_edit.setText(path)

    def _load_game_model(self) -> None:
        path = self._game_model_path_edit.text().strip()
        if not path:
            for runtime in self._runtimes.values():
                if runtime.decoder is not None:
                    runtime.decoder.use_demo_mode()
            self._game_mode_label.setText("Mode: Demo Mode")
            self._update_game_model_info_labels()
            self._update_game_model_warning("")
            return
        try:
            predictor = load_model(path)
            for runtime in self._runtimes.values():
                if runtime.decoder is not None:
                    runtime.decoder.set_predictor(predictor, self._shared_inference_lock)
        except Exception as exc:
            for runtime in self._runtimes.values():
                if runtime.decoder is not None:
                    runtime.decoder.use_demo_mode()
            self._game_mode_label.setText("Mode: Demo Mode")
            self._update_game_model_info_labels()
            self._update_game_model_warning("")
            self._game_api_status_label.setText(f"API status: model load failed: {exc}")
            return
        self._game_mode_label.setText("Mode: Model")
        self._update_game_model_info_labels()
        self._update_game_model_warning(path)
        self._game_api_status_label.setText("API status: model loaded")

    def _set_game_confidence_threshold(self, value: float) -> None:
        for runtime in self._runtimes.values():
            if runtime.decoder is not None:
                runtime.decoder.set_confidence_threshold(float(value))
        self._game_bridge.confidence_threshold = float(value)

    def _set_game_smoothing_frames(self, value: int) -> None:
        for runtime in self._runtimes.values():
            if runtime.decoder is not None:
                runtime.decoder.set_smoothing_frames(int(value))

    def _set_game_change_confirm_frames(self, value: int) -> None:
        for runtime in self._runtimes.values():
            if runtime.decoder is not None:
                runtime.decoder.set_change_confirmations(int(value))

    def _set_game_pinch_params(self, *_args) -> None:
        params = {
            "pinch_threshold": float(self._game_pinch_threshold_spin.value()),
            "pinch_boost": float(self._game_pinch_boost_spin.value()),
            "pinch_margin": float(self._game_pinch_margin_spin.value()),
        }
        for runtime in self._runtimes.values():
            if runtime.decoder is not None:
                runtime.decoder.set_pinch_params(**params)

    def _set_game_control_enabled(self, enabled: bool) -> None:
        if enabled:
            if self._collection_active:
                self._enable_game_control_checkbox.setChecked(False)
                self._game_api_status_label.setText("API status: disabled during collection")
                return
            self._game_decoder.start()
            self._right_game_decoder.start()
            mode = "Demo Mode" if self._game_decoder.demo_mode else "Model"
            self._game_mode_label.setText(f"Mode: {mode}")
            self._update_game_model_info_labels()
            self._game_api_status_label.setText("API status: enabled")
            self._publish_hand_status("left")
            self._publish_hand_status("right")
            return
        self._game_decoder.stop()
        self._right_game_decoder.stop()
        self._dual_game_mapper.release_all()
        self._game_api_status_label.setText("API status: disabled")
        self._publish_hand_status("left")
        self._publish_hand_status("right")

    def _send_test_game_gesture(self, gesture: str) -> None:
        self._game_bridge.send_gesture(gesture, 0.95)
        self._game_current_gesture_label.setText(f"Left Gesture: {gesture}")
        self._game_confidence_label.setText("Left Confidence: 95%")
        self._game_probs_label.setText(f"Left Probs: test send {gesture}")
        self._game_api_status_label.setText(f"API status: sent {gesture}")

    def _on_game_gesture_changed(self, gesture: str, confidence: float, probs: dict) -> None:
        self._on_game_gesture_changed_for_side("left", gesture, confidence, probs)

    def _on_game_gesture_changed_for_side(
        self,
        side: BraceletSide,
        gesture: str,
        confidence: float,
        probs: dict,
    ) -> None:
        runtime = self._runtimes[side]
        runtime.current_gesture = str(gesture)
        runtime.confidence = float(confidence)
        runtime.probs = dict(probs)
        ordered = ["rest", "fist", "open-palm", "pinch"]
        prob_text = " | ".join(f"{label} {float(probs.get(label, 0.0)) * 100:.0f}%" for label in ordered)
        if side == "left":
            self._game_current_gesture_label.setText(f"Left Gesture: {gesture}")
            self._game_confidence_label.setText(f"Left Confidence: {confidence * 100:.0f}%")
            self._game_probs_label.setText(f"Left Probs: {prob_text}")
        else:
            self._right_game_current_gesture_label.setText(f"Right Gesture: {gesture}")
            self._right_game_confidence_label.setText(f"Right Confidence: {confidence * 100:.0f}%")
            self._right_game_probs_label.setText(f"Right Probs: {prob_text}")
        if self._enable_game_control_checkbox.isChecked():
            self._dual_game_mapper.update(
                self._runtimes["left"].current_gesture,
                self._runtimes["right"].current_gesture,
                enabled=True,
            )
        self._game_api_status_label.setText(f"API status: {self._game_bridge.last_status}")
        self._gesture_server.publish(
            "gesture",
            {
                "hand": side,
                "gesture": str(gesture),
                "confidence": float(confidence),
                "probs": {label: float(probs.get(label, 0.0)) for label in ordered},
                "game_control": self._enable_game_control_checkbox.isChecked(),
                "model_type": getattr(runtime.decoder, "model_type", "demo"),
                "source": "emg_live_marker",
                "connected": runtime.connected,
            },
        )

    def _publish_hand_status(self, side: BraceletSide) -> None:
        runtime = self._runtimes[side]
        game_control = (
            self._enable_game_control_checkbox.isChecked()
            and runtime.decoder is not None
            and runtime.decoder.enabled
        )
        self._gesture_server.publish(
            "hand_status",
            {
                "hand": side,
                "connected": runtime.connected,
                "game_control": game_control,
            },
        )

    def _update_game_model_info_labels(self) -> None:
        if not hasattr(self, "_game_model_type_label"):
            return
        model_type = getattr(self._game_decoder, "model_type", "demo")
        signal_type = getattr(self._game_decoder, "signal_type", "filtered")
        normalization = "yes" if getattr(self._game_decoder, "normalization_loaded", False) else "no"
        self._game_model_type_label.setText(f"Model type: {model_type}")
        self._game_signal_type_label.setText(f"Signal: {signal_type}")
        self._game_normalization_label.setText(f"Normalization: {normalization}")

    def _update_game_model_warning(self, path_text: str) -> None:
        if not hasattr(self, "_game_warning_label"):
            return
        self._game_warning_label.setText(CROSS_SESSION_WARNING if self._is_cross_session_model(path_text) else "")

    def _is_cross_session_model(self, path_text: str) -> bool:
        if not path_text:
            return False
        model_path = Path(path_text)
        model_dir = model_path if model_path.is_dir() else model_path.parent
        if "calibration_game_model" in {part.lower() for part in model_path.parts}:
            return False
        report_path = model_dir / "train_report.json"
        if report_path.exists():
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                report = {}
            preset = report.get("training_args", {}).get("preset")
            if preset == "calibration":
                return False
            if report.get("holdout_session") or report.get("val_split") == "session":
                return True
        lower_parts = {part.lower() for part in model_path.parts}
        return "emg2pose_gesture_cv" in lower_parts or any(part.startswith("fold_session") for part in lower_parts)

    def _start_recording(self) -> None:
        if self._recorder.is_recording:
            return
        try:
            session_dir = self._recorder.start(baudrate=self._baudrate)
        except Exception as exc:
            self.statusBar().showMessage(f"Recording start failed: {exc}", 6000)
            return
        self._last_recording_dir = session_dir
        self._recording = True
        self._start_recording_button.setEnabled(False)
        self._stop_recording_button.setEnabled(True)
        self._refresh_status_counts()
        self.statusBar().showMessage(f"Recording to {session_dir}", 5000)

    def _stop_recording(self) -> None:
        if self._recorder.is_recording:
            try:
                self._recorder.stop()
            except Exception as exc:
                self.statusBar().showMessage(f"Recording stop failed: {exc}", 6000)
        self._recording = False
        self._start_recording_button.setEnabled(True)
        self._stop_recording_button.setEnabled(False)
        self._refresh_status_counts()

    def _start_collection(self) -> None:
        if self._collection_active:
            return
        if not self._collection_preflight_ok():
            return

        subject_id = self._subject_id_edit.text().strip()
        session_id = self._session_id_edit.text().strip()
        if not self._valid_path_component(subject_id) or not self._valid_path_component(session_id):
            QMessageBox.warning(
                self,
                "Collection check failed",
                "Subject ID and Session ID must be non-empty folder names without path separators.",
            )
            return
        if self._recorder.is_recording:
            QMessageBox.warning(self, "Collection check failed", "Stop the current recording first.")
            return

        self._collection_rest_before_s = float(self._rest_before_spin.value())
        self._collection_hold_s = float(self._hold_duration_spin.value())
        self._collection_rest_after_s = float(self._rest_after_spin.value())
        self._collection_trial_duration_s = (
            self._collection_rest_before_s
            + self._collection_hold_s
            + self._collection_rest_after_s
        )
        if abs(self._collection_trial_duration_s - self._trial_duration_spin.value()) > 0.001:
            self._trial_duration_spin.setValue(self._collection_trial_duration_s)

        session_dir = Path("dataset") / subject_id / session_id
        if session_dir.exists():
            QMessageBox.warning(
                self,
                "Collection check failed",
                f"Session directory already exists:\n{session_dir}",
            )
            return

        trials_per_gesture = int(self._trials_per_gesture_spin.value())
        self._collect_trials = build_trial_list(
            trials_per_gesture,
            randomize=self._randomize_order_checkbox.isChecked(),
        )
        metadata = {
            "subject_id": subject_id,
            "session_id": session_id,
            "gestures": list(COLLECTION_GESTURES),
            "gesture_display_names": dict(GESTURE_DISPLAY_NAMES),
            "trials_per_gesture": trials_per_gesture,
            "trial_duration_s": self._collection_trial_duration_s,
            "rest_before_s": self._collection_rest_before_s,
            "gesture_hold_s": self._collection_hold_s,
            "rest_after_s": self._collection_rest_after_s,
            "emg_fs": EMG_FS,
            "imu_fs": IMU_FS,
            "channels": EMG_CHANNELS,
            "device_mode": "single_bracelet",
            "sync_method": "software_time",
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        try:
            recording_dir = self._recorder.start(
                session_dir=session_dir,
                metadata=metadata,
                collection_mode=True,
                baudrate=self._baudrate,
            )
        except Exception as exc:
            QMessageBox.warning(self, "Collection start failed", str(exc))
            return

        self._last_recording_dir = recording_dir
        self._recording = True
        self._collection_active = True
        self._collection_paused = False
        self._current_trial_index = -1
        self._current_trial = None
        self._last_completed_trial = None
        self._display_seconds = 10.0
        self._window_combo.setCurrentText("10s")
        self._waveform_view.set_display_seconds(10.0)
        if self._enable_game_control_checkbox.isChecked():
            self._enable_game_control_checkbox.setChecked(False)
        self._set_collection_button_state(active=True)
        self._refresh_collection_progress()
        self._refresh_status_counts()
        self.statusBar().showMessage(f"Collecting to {recording_dir}", 5000)
        self._start_next_trial()

    def _pause_collection(self) -> None:
        if not self._collection_active:
            return
        self._collection_paused = not self._collection_paused
        self._pause_collection_button.setText(
            "Resume Collection" if self._collection_paused else "Pause Collection"
        )
        if self._collection_paused:
            if self._collection_timer.isActive():
                self._collection_remaining_ms = max(1, self._collection_timer.remainingTime())
            else:
                self._collection_remaining_ms = 1
            self._collection_timer.stop()
            self._collection_phase_before_pause = self._collection_phase
            self._set_collection_phase("PAUSED")
            return

        if self._current_trial is not None and self._collection_step is not None:
            self._set_collection_phase(self._collection_phase_before_pause)
            self._schedule_collection_step(self._collection_remaining_ms, self._collection_step)
        elif self._current_trial is None:
            self._start_next_trial()

    def _stop_collection(self) -> None:
        was_active = self._collection_active
        self._collection_timer.stop()
        self._collection_active = False
        if was_active and self._current_trial is not None:
            self._current_trial.status = "bad"
            if self._current_trial.start_time is not None:
                self._finish_current_trial(note="stopped")
            else:
                self._last_completed_trial = self._current_trial
                self._current_trial = None

        self._collection_paused = False
        self._current_trial = None
        self._collection_step = None
        self._set_collection_button_state(active=False)
        self._set_collection_phase("IDLE")
        self._collection_prompt_label.setText("已停止")
        self._stop_recording()
        self._refresh_collection_progress()

    def _repeat_trial(self) -> None:
        if not self._collect_trials:
            return

        target = self._current_trial or self._last_completed_trial
        if target is None:
            return

        target.status = "bad"
        insert_at = min(self._current_trial_index + 1, len(self._collect_trials))
        repeated_trial = CollectionTrial(
            trial_id=self._next_collection_trial_id(),
            gesture=target.gesture,
        )
        self._collect_trials.insert(insert_at, repeated_trial)

        if self._current_trial is not None:
            self._collection_timer.stop()
            self._finish_current_trial(note="repeated", force_bad=True)
        else:
            self._write_collection_event_for_trial(target, phase="repeat_trial", note="repeated")
            if self._collection_active:
                self._collection_timer.stop()
                self._collection_prompt_label.setText(f"重采：{gesture_display_name(target.gesture)}")
                self._schedule_collection_step(300, self._start_next_trial)

        self._refresh_collection_progress()

    def _start_next_trial(self) -> None:
        if not self._collection_active:
            return
        if self._collection_paused:
            self._collection_prompt_label.setText("暂停")
            return

        self._current_trial_index += 1
        if self._current_trial_index >= len(self._collect_trials):
            self._collection_active = False
            self._collection_prompt_label.setText("采集完成")
            self._set_collection_phase("DONE")
            self._set_collection_button_state(active=False)
            self._stop_recording()
            self._refresh_collection_progress()
            return

        self._current_trial = self._collect_trials[self._current_trial_index]
        self._current_trial.status = "pending"
        self._set_collection_phase("READY")
        self._set_collection_prompt("准备")
        self._refresh_collection_progress()
        self._begin_current_trial()

    def _begin_current_trial(self) -> None:
        if not self._collection_active or self._current_trial is None:
            return
        software_time = self._write_collection_event("trial_start")
        self._current_trial.start_time = software_time
        self._set_collection_phase("READY")
        self._set_collection_prompt("准备")
        rest_before_ms = int(round(self._collection_rest_before_s * 1000.0))
        if rest_before_ms <= 0:
            self._gesture_start()
            return
        self._schedule_collection_step(rest_before_ms, self._gesture_start)

    def _gesture_start(self) -> None:
        if self._current_trial is None:
            return
        self._write_collection_event("gesture_start")
        self._set_collection_phase("HOLD")
        self._set_collection_prompt("保持")
        self._schedule_collection_step(
            int(round(self._collection_hold_s * 1000.0)),
            self._gesture_end,
        )

    def _gesture_end(self) -> None:
        self._write_collection_event("gesture_end")
        self._set_collection_phase("RELAX")
        self._collection_prompt_label.setText("放松")
        self._schedule_collection_step(
            int(round(self._collection_rest_after_s * 1000.0)),
            self._trial_end,
        )

    def _trial_end(self) -> None:
        self._finish_current_trial()

    def _finish_current_trial(self, note: str = "", force_bad: bool = False) -> None:
        if self._current_trial is None:
            return
        software_time = self._write_collection_event("trial_end", note=note)
        self._current_trial.end_time = software_time
        if force_bad:
            self._current_trial.status = "bad"
        elif self._current_trial.status != "bad":
            self._current_trial.status = "done"
        self._last_completed_trial = self._current_trial
        self._collection_prompt_label.setText(f"已保存 {self._current_trial.trial_id}")
        self._current_trial = None
        self._refresh_collection_progress()
        if self._collection_active:
            self._schedule_collection_step(400, self._start_next_trial)

    def _write_collection_event(self, phase: str, note: str = "") -> float:
        if self._current_trial is None:
            return self._recorder.software_time()
        return self._write_collection_event_for_trial(self._current_trial, phase=phase, note=note)

    def _write_collection_event_for_trial(
        self,
        trial: CollectionTrial,
        *,
        phase: str,
        note: str = "",
    ) -> float:
        software_time = self._recorder.software_time()
        self._recorder.write_collection_event(
            trial_id=trial.trial_id,
            subject_id=self._subject_id_edit.text().strip(),
            session_id=self._session_id_edit.text().strip(),
            gesture=trial.gesture,
            gesture_name=gesture_display_name(trial.gesture),
            phase=phase,
            software_time=software_time,
            sample_index=self._raw_emg_buffer.latest_sample_index(),
            note=note,
        )
        return software_time

    def _schedule_collection_step(self, delay_ms: int, step) -> None:
        self._collection_step = step
        delay_ms = max(1, int(delay_ms))
        self._collection_step_deadline = monotonic() + delay_ms / 1000.0
        self._collection_timer.start(delay_ms)

    def _run_collection_step(self) -> None:
        step = self._collection_step
        self._collection_step = None
        if step is not None:
            step()

    def _set_collection_prompt(self, action: str) -> None:
        gesture_name = "-"
        if self._current_trial is not None:
            gesture_name = gesture_display_name(self._current_trial.gesture)
        self._collection_prompt_label.setText(f"{action}：{gesture_name}")

    def _set_collection_phase(self, phase: str) -> None:
        self._collection_phase = phase
        if hasattr(self, "_collection_phase_label"):
            self._collection_phase_label.setText(f"Phase: {phase}")
        self._refresh_collection_progress()

    def _collection_preflight_ok(self) -> bool:
        failures = []
        left_runtime = self._runtimes["left"]
        if not self._device_is_connected():
            failures.append("Connect the serial device before starting collection.")
        if not (230.0 <= left_runtime.emg_rate_sps <= 270.0):
            failures.append(
                f"EMG rate must be 230-270 SPS; current rate is {left_runtime.emg_rate_sps:.0f} SPS."
            )
        if self._global_lost_delta_per_sec > 2:
            failures.append("Global lost is increasing quickly; wait for a stable stream.")
        if self._raw_emg_buffer.latest_sample_index() is None:
            failures.append("EMG buffer is empty; wait until samples are visible.")
        if failures:
            QMessageBox.warning(self, "Collection check failed", "\n".join(failures))
            return False
        return True

    def _device_is_connected(self) -> bool:
        if self._simulator is not None and self._simulator.is_running():
            return True
        return self._disconnect_button.isEnabled()

    def _set_collection_button_state(self, active: bool) -> None:
        self._start_collection_button.setEnabled(not active)
        self._pause_collection_button.setEnabled(active)
        self._stop_collection_button.setEnabled(active)
        self._repeat_trial_button.setEnabled(active)
        self._pause_collection_button.setText("Pause Collection")
        self._start_recording_button.setEnabled((not active) and (not self._recorder.is_recording))
        self._stop_recording_button.setEnabled((not active) and self._recorder.is_recording)
        for widget in [
            self._subject_id_edit,
            self._session_id_edit,
            self._trials_per_gesture_spin,
            self._trial_duration_spin,
            self._rest_before_spin,
            self._hold_duration_spin,
            self._rest_after_spin,
            self._randomize_order_checkbox,
        ]:
            widget.setEnabled(not active)

    def _refresh_collection_progress(self) -> None:
        if not hasattr(self, "_collection_progress_label"):
            return
        done_count = sum(1 for trial in self._collect_trials if trial.status == "done")
        total_count = len(self._collect_trials)
        self._collection_progress_label.setText(f"Trials: {done_count}/{total_count}")
        current_number = self._current_trial_index + 1 if self._current_trial is not None else done_count
        current_number = max(0, min(current_number, total_count))
        self._collection_trial_label.setText(f"Trial {current_number} / {total_count}")
        if self._current_trial is None:
            gesture_text = "-"
        else:
            gesture_text = gesture_display_name(self._current_trial.gesture)
        self._collection_gesture_label.setText(f"Gesture: {gesture_text}")
        self._collection_phase_label.setText(f"Phase: {self._collection_phase}")

    def _next_collection_trial_id(self) -> str:
        max_id = 0
        for trial in self._collect_trials:
            try:
                max_id = max(max_id, int(trial.trial_id))
            except ValueError:
                continue
        return f"{max_id + 1:04d}"

    def _next_session_id(self, subject_id: str) -> str:
        subject_dir = Path("dataset") / subject_id
        max_id = 0
        if subject_dir.exists():
            for path in subject_dir.iterdir():
                if not path.is_dir() or not path.name.startswith("session_"):
                    continue
                try:
                    max_id = max(max_id, int(path.name.removeprefix("session_")))
                except ValueError:
                    continue
        return f"session_{max_id + 1:03d}"

    def _on_subject_id_edited(self, subject_id: str) -> None:
        if self._session_id_auto_generated:
            self._session_id_edit.setText(self._next_session_id(subject_id.strip() or "subject_01"))

    def _on_session_id_edited(self, *_args) -> None:
        self._session_id_auto_generated = False

    @staticmethod
    def _valid_path_component(value: str) -> bool:
        if not value:
            return False
        invalid_chars = set('<>:"/\\|?*')
        return not any(char in invalid_chars for char in value)

    def _open_recording_folder(self) -> None:
        path = self._last_recording_dir or Path("recordings")
        path.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))

    def _buffer_for_display_mode(self, side: BraceletSide = "left") -> EmgRingBuffer:
        runtime = self._runtimes[side]
        mode = self._display_mode.lower()
        if mode == "filtered":
            return runtime.filtered_emg_buffer
        if mode == "rectified":
            return runtime.rectified_emg_buffer
        if mode == "rms":
            return runtime.rms_emg_buffer
        return runtime.raw_emg_buffer

    def _clear_processed_buffers(self) -> None:
        for side in ("left", "right"):
            self._clear_processed_buffers_for_side(side)

    def _clear_processed_buffers_for_side(self, side: BraceletSide) -> None:
        runtime = self._runtimes[side]
        runtime.filtered_emg_buffer.clear()
        runtime.rectified_emg_buffer.clear()
        runtime.rms_emg_buffer.clear()

    def _reset_data_streams(self, clear_raw: bool = False) -> None:
        self._reset_data_streams_for_side("left", clear_raw=clear_raw)

    def _reset_data_streams_for_side(self, side: BraceletSide, clear_raw: bool = False) -> None:
        runtime = self._runtimes[side]
        runtime.stream_processor.reset()
        self._clear_processed_buffers_for_side(side)
        if clear_raw:
            runtime.raw_emg_buffer.clear()
            runtime.imu_buffer.clear()
            if side == "left":
                self._waveform_view.clear()
            else:
                self._right_waveform_view.clear()

    def _update_waveform_visibility(self) -> None:
        if not hasattr(self, "_right_waveform_view"):
            return
        left_visible = self._runtimes["left"].connected or self._raw_emg_buffer.latest_sample_index() is not None
        right_visible = (
            self._runtimes["right"].connected
            or self._right_raw_emg_buffer.latest_sample_index() is not None
        )
        self._left_waveform_label.setVisible(left_visible or not right_visible)
        self._waveform_view.setVisible(left_visible or not right_visible)
        self._right_waveform_label.setVisible(right_visible)
        self._right_waveform_view.setVisible(right_visible)
