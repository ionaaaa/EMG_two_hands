"""Right-side classroom-management dock for the existing teacher MainWindow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDockWidget,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from emg_live_marker.realtime.teacher_classroom import TeacherClassroomService


class TeacherClassroomDock(QDockWidget):
    def __init__(
        self,
        service: TeacherClassroomService,
        main_window: Any,
    ) -> None:
        super().__init__("课堂管理", main_window)
        self.setObjectName("teacher-classroom-dock")
        self.service = service
        self.main_window = main_window
        self.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.LeftDockWidgetArea
        )
        self.setMinimumWidth(430)
        tabs = QTabWidget()
        tabs.addTab(self._build_settings_tab(), "课堂设置")
        tabs.addTab(self._build_sessions_tab(), "学生会话")
        tabs.addTab(self._build_models_tab(), "个人模型")
        tabs.addTab(self._build_results_tab(), "比赛结果")
        tabs.addTab(self._build_diagnostics_tab(), "设备诊断")
        self.setWidget(tabs)
        self.refresh_all()

    def _build_settings_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        form = QGridLayout()
        form.addWidget(QLabel("标准模型版本"), 0, 0)
        self.standard_model_combo = QComboBox()
        form.addWidget(self.standard_model_combo, 0, 1)
        form.addWidget(QLabel("每类采集次数"), 1, 0)
        self.trials_spin = QSpinBox()
        self.trials_spin.setRange(5, 100)
        form.addWidget(self.trials_spin, 1, 1)
        self.personal_training_checkbox = QCheckBox("允许学生训练个人模型")
        form.addWidget(self.personal_training_checkbox, 2, 0, 1, 2)
        self.password_checkbox = QCheckBox("启用教师密码接口（本轮仅保存开关）")
        form.addWidget(self.password_checkbox, 3, 0, 1, 2)
        layout.addLayout(form)
        self.save_settings_button = QPushButton("保存课堂设置")
        self.save_settings_button.clicked.connect(self._save_settings)
        layout.addWidget(self.save_settings_button)
        self.settings_message = QLabel("")
        self.settings_message.setWordWrap(True)
        layout.addWidget(self.settings_message)

        device_status = QGridLayout()
        device_status.addWidget(QLabel("左手环"), 0, 0)
        self.left_device_status_label = QLabel()
        device_status.addWidget(self.left_device_status_label, 0, 1)
        device_status.addWidget(QLabel("右手环"), 1, 0)
        self.right_device_status_label = QLabel()
        device_status.addWidget(self.right_device_status_label, 1, 1)
        layout.addLayout(device_status)
        self.ports_message = QLabel("")
        self.ports_message.setWordWrap(True)
        layout.addWidget(self.ports_message)
        self.refresh_ports_button = QPushButton("刷新端口")
        self.refresh_ports_button.clicked.connect(self.refresh_ports)
        layout.addWidget(self.refresh_ports_button)

        self.next_group_button = QPushButton("准备下一组学生")
        self.next_group_button.clicked.connect(self._prepare_next_group)
        layout.addWidget(self.next_group_button)
        layout.addStretch(1)
        return page

    def _build_sessions_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        self.sessions_table = QTableWidget(0, 8)
        self.sessions_table.setHorizontalHeaderLabels(
            ["匿名编号", "Session", "手", "状态", "有效次数", "Invalid", "Repeated", "重采"]
        )
        self.sessions_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(self.sessions_table)
        buttons = QHBoxLayout()
        refresh = QPushButton("刷新")
        recollect = QPushButton("标记重新采集")
        delete = QPushButton("删除本次采集")
        refresh.clicked.connect(self.refresh_sessions)
        recollect.clicked.connect(self._mark_recollect)
        delete.clicked.connect(self._delete_session)
        buttons.addWidget(refresh)
        buttons.addWidget(recollect)
        buttons.addWidget(delete)
        layout.addLayout(buttons)
        self.sessions_message = QLabel("")
        self.sessions_message.setWordWrap(True)
        layout.addWidget(self.sessions_message)
        return page

    def _build_models_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        self.models_table = QTableWidget(0, 4)
        self.models_table.setHorizontalHeaderLabels(["匿名编号", "模型目录", "验证准确率", "训练时间"])
        self.models_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(self.models_table)
        buttons = QHBoxLayout()
        refresh = QPushButton("刷新")
        delete = QPushButton("删除个人模型")
        refresh.clicked.connect(self.refresh_personal_models)
        delete.clicked.connect(self._delete_personal_model)
        buttons.addWidget(refresh)
        buttons.addWidget(delete)
        layout.addLayout(buttons)
        self.models_message = QLabel("")
        layout.addWidget(self.models_message)
        return page

    def _build_results_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        group_row = QHBoxLayout()
        group_row.addWidget(QLabel("按组查看"))
        self.result_group_combo = QComboBox()
        self.result_group_combo.currentTextChanged.connect(self.refresh_results_table)
        group_row.addWidget(self.result_group_combo)
        layout.addLayout(group_row)
        self.results_table = QTableWidget(0, 6)
        self.results_table.setHorizontalHeaderLabels(
            ["匿名编号", "模式", "得分", "正确率", "最大连击", "结果"]
        )
        layout.addWidget(self.results_table)
        export = QPushButton("导出 CSV")
        export.clicked.connect(self._export_results)
        layout.addWidget(export)
        self.results_message = QLabel("")
        layout.addWidget(self.results_message)
        return page

    def _build_diagnostics_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        self.diagnostics_label = QLabel("")
        self.diagnostics_label.setWordWrap(True)
        layout.addWidget(self.diagnostics_label)
        layout.addStretch(1)
        self._diagnostic_timer = QTimer(self)
        self._diagnostic_timer.setInterval(1000)
        self._diagnostic_timer.timeout.connect(self.refresh_diagnostics)
        self._diagnostic_timer.start()
        return page

    def refresh_all(self) -> None:
        self.refresh_settings()
        self.refresh_ports()
        self.refresh_sessions()
        self.refresh_personal_models()
        self.refresh_results()
        self.refresh_diagnostics()

    def refresh_settings(self) -> None:
        models = self.service.scan_standard_models()
        current = self.service.configured_standard_model_path
        self.standard_model_combo.clear()
        for model in models:
            self.standard_model_combo.addItem(model.version, model.path)
        if current is not None:
            index = self.standard_model_combo.findData(current)
            if index >= 0:
                self.standard_model_combo.setCurrentIndex(index)
        self.trials_spin.setValue(self.service.settings.trials_per_action)
        self.personal_training_checkbox.setChecked(self.service.settings.personal_training_enabled)
        self.password_checkbox.setChecked(self.service.settings.teacher_password_enabled)

    def refresh_ports(self) -> None:
        self.main_window._refresh_ports()
        self.ports_message.setText("端口候选已刷新，请在主窗口 Left / Right 中选择并连接手环。")

    def sync_port_candidates(self, _ports: list[str]) -> None:
        """Refresh the read-only mirror after MainWindow's existing port scan."""

        self.sync_device_status()

    def sync_device_status(self) -> None:
        """Display MainWindow's sole port-selection and connection state."""

        self.left_device_status_label.setText(
            self._device_status_text("left", self.main_window._port_combo.currentText())
        )
        self.right_device_status_label.setText(
            self._device_status_text("right", self.main_window._right_port_combo.currentText())
        )

    def _device_status_text(self, side: str, port: str) -> str:
        selected_port = str(port).strip() or "未选择端口"
        connected = bool(self.main_window._runtimes[side].connected)
        return f"{selected_port} · {'已连接' if connected else '未连接'}"

    def refresh_sessions(self) -> None:
        sessions = self.service.scan_sessions()
        self.sessions_table.setRowCount(len(sessions))
        for row, session in enumerate(sessions):
            counts = "/".join(str(session.valid_counts[name]) for name in ("fist", "finger_spread", "thumb_index_pinch"))
            values = (
                session.student_id, session.session_id, session.hand, session.status, counts,
                str(session.invalid_count), str(session.repeated_count), "是" if session.recollect_requested else "否",
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, session.path)
                self.sessions_table.setItem(row, column, item)

    def refresh_personal_models(self) -> None:
        models = self.service.scan_personal_models()
        self.models_table.setRowCount(len(models))
        for row, model in enumerate(models):
            accuracy = "暂无" if model.validation_accuracy is None else f"{model.validation_accuracy:.1%}"
            for column, value in enumerate((model.student_id, model.path.name, accuracy, model.trained_at)):
                item = QTableWidgetItem(str(value))
                item.setData(Qt.ItemDataRole.UserRole, model.path)
                self.models_table.setItem(row, column, item)

    def refresh_results(self) -> None:
        records = self.service.scan_competition_results()
        current = self.result_group_combo.currentText()
        groups = ["全部"] + sorted({str(record["student_id"]) for record in records})
        self.result_group_combo.blockSignals(True)
        self.result_group_combo.clear()
        self.result_group_combo.addItems(groups)
        self.result_group_combo.setCurrentText(current if current in groups else "全部")
        self.result_group_combo.blockSignals(False)
        self.refresh_results_table()

    def refresh_results_table(self, _group: str = "") -> None:
        group = self.result_group_combo.currentText()
        records = [
            record for record in self.service.scan_competition_results()
            if group in {"", "全部"} or record["student_id"] == group
        ]
        self.results_table.setRowCount(len(records))
        for row, record in enumerate(records):
            values = (
                record["student_id"], record["mode"], record["score"],
                f"{float(record['accuracy']):.1%}", record["max_combo"], record["outcome"],
            )
            for column, value in enumerate(values):
                self.results_table.setItem(row, column, QTableWidgetItem(str(value)))

    def refresh_diagnostics(self) -> None:
        diagnostics = self.service.device_diagnostics(self.main_window._runtimes)
        lines = []
        for side, values in diagnostics.items():
            name = "左手" if side == "left" else "右手"
            lines.append(
                f"{name}：EMG {values['emg_rate_sps']:.0f} SPS，IMU {values['imu_rate_sps']:.0f} SPS，"
                f"丢包 {values['global_lost_count']}，坏头/坏类型 {values['bad_header_count']}/{values['bad_type_count']}，"
                f"重同步 {values['resync_count']}"
            )
        self.diagnostics_label.setText("\n".join(lines))

    def _save_settings(self) -> None:
        model_path = self.standard_model_combo.currentData()
        if model_path is None:
            self.settings_message.setText("没有可用的完整标准模型。")
            return
        _saved, message = self.service.save_settings(
            standard_model_path=model_path,
            trials_per_action=self.trials_spin.value(),
            personal_training_enabled=self.personal_training_checkbox.isChecked(),
            teacher_password_enabled=self.password_checkbox.isChecked(),
        )
        self.settings_message.setText(message)

    def _selected_path(self, table: QTableWidget) -> Path | None:
        row = table.currentRow()
        if row < 0 or table.item(row, 0) is None:
            return None
        value = table.item(row, 0).data(Qt.ItemDataRole.UserRole)
        return Path(value) if value else None

    def _mark_recollect(self) -> None:
        path = self._selected_path(self.sessions_table)
        if path is None:
            self.sessions_message.setText("请先选择一个采集会话。")
            return
        _ok, message = self.service.mark_recollect(path)
        self.sessions_message.setText(message)
        self.refresh_sessions()

    def _delete_session(self) -> None:
        path = self._selected_path(self.sessions_table)
        if path is None:
            self.sessions_message.setText("请先选择一个采集会话。")
            return
        first = QMessageBox.question(self, "确认删除", "确定删除选中的本次采集吗？旧数据不会自动备份。")
        if first != QMessageBox.StandardButton.Yes:
            return
        second = QMessageBox.question(self, "再次确认", "再次确认：删除后无法从程序内恢复，是否继续？")
        if second != QMessageBox.StandardButton.Yes:
            return
        _ok, message = self.service.delete_session(path, confirmed=True)
        self.sessions_message.setText(message)
        self.refresh_sessions()

    def _delete_personal_model(self) -> None:
        path = self._selected_path(self.models_table)
        if path is None:
            self.models_message.setText("请先选择个人模型。")
            return
        answer = QMessageBox.question(self, "确认删除", "确定删除该个人模型吗？标准模型不会被删除。")
        if answer != QMessageBox.StandardButton.Yes:
            return
        _ok, message = self.service.delete_personal_model(path, confirmed=True)
        self.models_message.setText(message)
        self.refresh_personal_models()

    def _export_results(self) -> None:
        path, _selected = QFileDialog.getSaveFileName(self, "导出比赛 CSV", "competition_results.csv", "CSV (*.csv)")
        if not path:
            return
        group = self.result_group_combo.currentText()
        _ok, message = self.service.export_competition_csv(
            path, student_id="" if group == "全部" else group
        )
        self.results_message.setText(message)

    def _prepare_next_group(self) -> None:
        _ok, message = self.service.prepare_next_group()
        self.settings_message.setText(message)
        self.refresh_all()
