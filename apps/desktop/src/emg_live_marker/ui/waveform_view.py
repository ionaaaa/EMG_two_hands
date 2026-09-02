"""Multi-channel real-time waveform display."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QVBoxLayout, QWidget

from emg_live_marker.config import DEFAULT_DISPLAY_OUTLIER_UV
from emg_live_marker.device.protocol import EMG_FS
from emg_live_marker.realtime.signal_processing import Y_RANGE_OPTIONS, normalize_y_range_option

DISPLAY_OUTLIER_UV = DEFAULT_DISPLAY_OUTLIER_UV
WAVEFORM_LEFT_AXIS_WIDTH = 60


@dataclass(frozen=True)
class EventDisplay:
    t_emg: float
    label: str


def sanitize_plot_data(
    t: np.ndarray,
    y: np.ndarray,
    sample_index: np.ndarray | None = None,
    fs: float = EMG_FS,
    display_outlier_uv: float = DISPLAY_OUTLIER_UV,
) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if t.size == 0 or y.size == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    n = min(t.shape[0], y.shape[0])
    t = t[:n]
    y = y[:n]
    if sample_index is not None:
        sample_index = np.asarray(sample_index, dtype=np.int64).reshape(-1)[:n]

    valid = np.isfinite(t) & np.isfinite(y)
    valid &= np.abs(y) <= float(display_outlier_uv)
    if sample_index is not None:
        valid &= sample_index >= 0
    t = t[valid]
    y = y[valid]
    if sample_index is not None:
        sample_index = sample_index[valid]
    if t.size < 2:
        return t, y

    if sample_index is not None:
        order = np.argsort(sample_index, kind="stable")
    else:
        order = np.argsort(t, kind="stable")
    t = t[order]
    y = y[order]
    if sample_index is not None:
        sample_index = sample_index[order]

    if sample_index is not None:
        keep = np.concatenate([[True], np.diff(sample_index) > 0])
    else:
        keep = np.concatenate([[True], np.diff(t) > 0])
    t = t[keep]
    y = y[keep]
    if sample_index is not None:
        sample_index = sample_index[keep]
    if t.size < 2:
        return t, y

    expected_dt = 1.0 / float(fs)
    if sample_index is not None:
        gaps = np.where(np.diff(sample_index) > 1)[0]
    else:
        gaps = np.where(np.diff(t) > 3.0 * expected_dt)[0]
    if gaps.size == 0:
        return t, y

    new_t: list[float] = []
    new_y: list[float] = []
    last = 0
    for gap_pos in gaps:
        new_t.extend(t[last : gap_pos + 1])
        new_y.extend(y[last : gap_pos + 1])
        new_t.append(float(t[gap_pos] + expected_dt))
        new_y.append(np.nan)
        last = gap_pos + 1
    new_t.extend(t[last:])
    new_y.extend(y[last:])
    return np.asarray(new_t, dtype=np.float64), np.asarray(new_y, dtype=np.float64)


class MultiChannelWaveformView(QWidget):
    def __init__(
        self,
        display_seconds: float = 10.0,
        channels: int = 8,
        fs: float = EMG_FS,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.display_seconds = float(display_seconds)
        self.channels = int(channels)
        self.fs = float(fs)
        self.display_outlier_uv = DISPLAY_OUTLIER_UV
        self._y_range_uv: float | None = None
        self._auto_robust = False
        self._plots: list[pg.PlotItem] = []
        self._curves: list[pg.PlotDataItem] = []
        self._event_items: list[list[tuple[pg.PlotItem, pg.GraphicsObject]]] = []

        pg.setConfigOptions(antialias=False)
        self._graphics = pg.GraphicsLayoutWidget()
        self._graphics.setBackground("k")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._graphics)
        self._build_plots()

    def set_display_seconds(self, seconds: float) -> None:
        self.display_seconds = float(seconds)

    def set_y_range_mode(self, mode: str) -> None:
        mode = normalize_y_range_option(mode)
        self._auto_robust = mode == "Auto robust"
        self._y_range_uv = Y_RANGE_OPTIONS[mode]
        for plot in self._plots:
            if self._y_range_uv is None and not self._auto_robust:
                plot.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
            else:
                plot.enableAutoRange(axis=pg.ViewBox.YAxis, enable=False)
                if self._y_range_uv is not None:
                    plot.setYRange(-self._y_range_uv, self._y_range_uv, padding=0.02)

    def update_data(
        self,
        t: np.ndarray,
        data: np.ndarray,
        sample_index: np.ndarray | None = None,
    ) -> None:
        if t is None or data is None:
            return
        t = np.asarray(t, dtype=np.float64).reshape(-1)
        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 2 or data.shape[1] != self.channels:
            return
        if t.shape[0] != data.shape[0]:
            n = min(t.shape[0], data.shape[0])
            t = t[:n]
            data = data[:n]
            if sample_index is not None:
                sample_index = np.asarray(sample_index, dtype=np.int64).reshape(-1)[:n]

        sanitized: list[tuple[np.ndarray, np.ndarray]] = []
        for channel, curve in enumerate(self._curves):
            x, y = sanitize_plot_data(
                t,
                data[:, channel],
                sample_index=sample_index,
                fs=self.fs,
                display_outlier_uv=self.display_outlier_uv,
            )
            curve.setData(x, y, connect="finite")
            sanitized.append((x, y))

        if t.size:
            right = float(np.nanmax(t))
            left = max(0.0, right - self.display_seconds)
            for plot in self._plots:
                plot.setXRange(left, right if right > left else left + self.display_seconds, padding=0)

        if self._auto_robust:
            self._apply_robust_ranges(sanitized)

    def add_event_marker(self, event: object) -> None:
        t_emg = float(getattr(event, "t_emg"))
        label = str(getattr(event, "label"))
        marker_items: list[tuple[pg.PlotItem, pg.GraphicsObject]] = []
        pen = pg.mkPen("y", width=1, style=Qt.PenStyle.DashLine)
        for index, plot in enumerate(self._plots):
            line = pg.InfiniteLine(pos=t_emg, angle=90, movable=False, pen=pen)
            plot.addItem(line)
            marker_items.append((plot, line))
            if index == 0:
                text = pg.TextItem(text=label, color="y", anchor=(0, 1))
                text.setPos(t_emg, plot.viewRange()[1][1])
                plot.addItem(text)
                marker_items.append((plot, text))
        self._event_items.append(marker_items)

    def clear_event_markers(self) -> None:
        for marker_group in self._event_items:
            for plot, item in marker_group:
                plot.removeItem(item)
        self._event_items.clear()

    def clear(self) -> None:
        for curve in self._curves:
            curve.setData([], [])
        self.clear_event_markers()

    def _build_plots(self) -> None:
        first_plot: pg.PlotItem | None = None
        palette = ["#2f80ed", "#27ae60", "#f2994a", "#9b51e0", "#00a8a8", "#eb5757", "#56ccf2", "#b8860b"]
        for channel in range(self.channels):
            plot = self._graphics.addPlot(row=channel, col=0)
            self._style_plot(plot)
            plot.setLabel("left", f"CH{channel + 1}", units="uV")
            if first_plot is None:
                first_plot = plot
            else:
                plot.setXLink(first_plot)
            if channel < self.channels - 1:
                plot.hideAxis("bottom")
            else:
                plot.setLabel("bottom", "Time", units="s")
            curve = plot.plot([], [], pen=pg.mkPen(palette[channel % len(palette)], width=1.2))
            try:
                curve.setDownsampling(auto=False)
            except AttributeError:
                pass
            try:
                curve.setClipToView(True)
            except AttributeError:
                pass
            self._plots.append(plot)
            self._curves.append(curve)
        self.set_y_range_mode("Auto")

    def _style_plot(self, plot: pg.PlotItem) -> None:
        plot.getViewBox().setBackgroundColor("k")
        plot.showGrid(x=True, y=True, alpha=0.16)
        plot.setMenuEnabled(False)
        plot.setMouseEnabled(x=False, y=False)
        for axis_name in ("left", "bottom"):
            axis = plot.getAxis(axis_name)
            axis.setPen(pg.mkPen("#7a7a7a"))
            axis.setTextPen(pg.mkPen("#c8c8c8"))
        plot.getAxis("left").setWidth(WAVEFORM_LEFT_AXIS_WIDTH)
        plot.layout.setColumnFixedWidth(0, WAVEFORM_LEFT_AXIS_WIDTH)

    def _apply_robust_ranges(self, sanitized: list[tuple[np.ndarray, np.ndarray]]) -> None:
        for plot, (_x, y) in zip(self._plots, sanitized, strict=False):
            finite = y[np.isfinite(y)]
            if finite.size < 2:
                continue
            lo, hi = np.nanpercentile(finite, [1, 99])
            if not np.isfinite(lo) or not np.isfinite(hi):
                continue
            margin = max(10.0, 0.2 * float(hi - lo))
            plot.setYRange(float(lo - margin), float(hi + margin), padding=0)


WaveformView = MultiChannelWaveformView
