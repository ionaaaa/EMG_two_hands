import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from emg_live_marker.ui.waveform_view import MultiChannelWaveformView, WAVEFORM_LEFT_AXIS_WIDTH


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_waveform_left_axes_use_equal_fixed_width():
    app = _app()
    view = MultiChannelWaveformView()
    view.resize(900, 600)
    view.show()
    app.processEvents()

    for plot in view._plots:
        assert plot.getAxis("left").width() == WAVEFORM_LEFT_AXIS_WIDTH
    view.close()


def test_only_bottom_waveform_plot_shows_time_axis_label():
    _app()
    view = MultiChannelWaveformView()

    for plot in view._plots[:-1]:
        assert not plot.getAxis("bottom").isVisible()
    assert view._plots[-1].getAxis("bottom").labelText == "Time"

    view.close()
