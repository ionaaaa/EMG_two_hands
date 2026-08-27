import numpy as np

from emg_live_marker.realtime.processing import RealtimeEMGProcessor


def test_display_modes_do_not_crash_and_keep_shape():
    rng = np.random.default_rng(123)
    data = rng.normal(size=(250, 8))
    processor = RealtimeEMGProcessor()

    for mode in ["raw", "filtered", "rectified", "rms"]:
        output = processor.process_for_display(data, mode)
        assert output.shape == data.shape


def test_rectified_and_rms_are_non_negative():
    rng = np.random.default_rng(123)
    data = rng.normal(size=(250, 8))
    processor = RealtimeEMGProcessor()

    rectified = processor.process_for_display(data, "rectified")
    rms = processor.process_for_display(data, "rms")

    assert np.all(rectified >= 0)
    assert np.all(rms >= 0)


def test_short_data_degrades_without_shape_change():
    data = np.array([[1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0]])
    processor = RealtimeEMGProcessor()

    for mode in ["raw", "filtered", "rectified", "rms"]:
        output = processor.process_for_display(data, mode)
        assert output.shape == data.shape

