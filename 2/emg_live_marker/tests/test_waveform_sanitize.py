import numpy as np

from emg_live_marker.ui.waveform_view import DISPLAY_OUTLIER_UV, sanitize_plot_data


def test_non_increasing_time_is_sorted_or_removed():
    t = np.array([0.0, 0.008, 0.004, 0.012])
    y = np.array([1.0, 3.0, 2.0, 4.0])

    x, out_y = sanitize_plot_data(t, y)

    assert x.shape == out_y.shape
    assert np.all(np.diff(x) > 0)


def test_nan_and_inf_are_removed():
    t = np.array([0.0, 0.004, 0.008, 0.012])
    y = np.array([1.0, np.nan, np.inf, 4.0])

    x, out_y = sanitize_plot_data(t, y)

    assert x.shape == out_y.shape
    assert np.all(np.isfinite(out_y))
    np.testing.assert_array_equal(out_y, np.array([1.0, 4.0]))


def test_display_outliers_are_removed():
    t = np.array([0.0, 0.004, 0.008])
    y = np.array([1.0, DISPLAY_OUTLIER_UV + 1.0, 2.0])

    x, out_y = sanitize_plot_data(t, y)

    assert x.shape == out_y.shape
    np.testing.assert_array_equal(out_y, np.array([1.0, 2.0]))


def test_sample_index_gap_inserts_nan_break():
    t = np.array([0.0, 0.004, 0.012])
    y = np.array([1.0, 2.0, 3.0])
    sample_index = np.array([0, 1, 3])

    x, out_y = sanitize_plot_data(t, y, sample_index=sample_index, fs=250.0)

    assert x.shape == out_y.shape
    assert np.isnan(out_y[2])
    np.testing.assert_array_equal(out_y[[0, 1, 3]], np.array([1.0, 2.0, 3.0]))


def test_output_shapes_match_after_cleanup():
    t = np.array([0.0, 0.004, 0.008])
    y = np.array([1.0, np.nan, 3.0])

    x, out_y = sanitize_plot_data(t, y)

    assert x.shape == out_y.shape

