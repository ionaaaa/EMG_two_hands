import numpy as np

from emg_live_marker.realtime.ring_buffer import EmgRingBuffer


def test_append_then_get_window_shape_is_correct():
    buffer = EmgRingBuffer(fs=250.0, channels=8, seconds=1.0)
    t = np.arange(10, dtype=np.float64) / 250.0
    data = np.ones((10, 8), dtype=np.float64)
    sample_index = np.arange(10)

    buffer.append_many(t, data, sample_index)
    out_t, out_data, out_sample_index = buffer.get_window(1.0)

    assert out_t.shape == (10,)
    assert out_data.shape == (10, 8)
    assert out_sample_index.shape == (10,)


def test_over_capacity_returns_time_ordered_samples():
    buffer = EmgRingBuffer(fs=5.0, channels=8, seconds=1.0)
    t = np.arange(8, dtype=np.float64)
    data = np.tile(np.arange(8, dtype=np.float64).reshape(-1, 1), (1, 8))
    sample_index = np.arange(8)

    buffer.append_many(t, data, sample_index)
    out_t, out_data, out_sample_index = buffer.get_window(100.0)

    np.testing.assert_array_equal(out_t, np.array([3, 4, 5, 6, 7], dtype=np.float64))
    np.testing.assert_array_equal(out_sample_index, np.array([3, 4, 5, 6, 7]))
    np.testing.assert_array_equal(out_data[:, 0], np.array([3, 4, 5, 6, 7], dtype=np.float64))


def test_latest_sample_index_is_correct():
    buffer = EmgRingBuffer(fs=5.0, channels=8, seconds=1.0)
    assert buffer.latest_sample_index() is None

    buffer.append_many(
        np.arange(7, dtype=np.float64),
        np.zeros((7, 8), dtype=np.float64),
        np.arange(10, 17),
    )

    assert buffer.latest_sample_index() == 16


def test_empty_window_does_not_error():
    buffer = EmgRingBuffer()
    t, data, sample_index = buffer.get_window(1.0)

    assert t.shape == (0,)
    assert data.shape == (0, 8)
    assert sample_index.shape == (0,)


def test_get_window_returns_increasing_sample_index_before_wrap():
    buffer = EmgRingBuffer(fs=10.0, channels=8, seconds=1.0)
    buffer.append_many(
        np.array([0.0, 0.1, 0.2]),
        np.zeros((3, 8), dtype=np.float64),
        np.array([0, 1, 2]),
    )

    _t, _data, sample_index = buffer.get_window(1.0)

    assert np.all(np.diff(sample_index) > 0)


def test_get_window_returns_increasing_sample_index_after_wrap():
    buffer = EmgRingBuffer(fs=5.0, channels=8, seconds=1.0)
    buffer.append_many(
        np.arange(8, dtype=np.float64) / 5.0,
        np.zeros((8, 8), dtype=np.float64),
        np.arange(8),
    )

    _t, _data, sample_index = buffer.get_window(10.0)

    np.testing.assert_array_equal(sample_index, np.array([3, 4, 5, 6, 7]))
    assert np.all(np.diff(sample_index) > 0)


def test_duplicate_sample_index_keeps_last_value():
    buffer = EmgRingBuffer(fs=10.0, channels=8, seconds=1.0)
    data = np.zeros((4, 8), dtype=np.float64)
    data[:, 0] = [0, 1, 99, 2]
    buffer.append_many(
        np.array([0.0, 0.1, 0.1, 0.2]),
        data,
        np.array([0, 1, 1, 2]),
    )

    _t, out_data, sample_index = buffer.get_window(1.0)

    np.testing.assert_array_equal(sample_index, np.array([0, 1, 2]))
    assert out_data[1, 0] == 99


def test_uninitialized_data_is_not_returned():
    buffer = EmgRingBuffer(fs=100.0, channels=8, seconds=1.0)
    buffer.append_many(
        np.array([0.0, 0.01]),
        np.zeros((2, 8), dtype=np.float64),
        np.array([0, 1]),
    )

    t, data, sample_index = buffer.get_window(10.0)

    assert t.shape == (2,)
    assert data.shape == (2, 8)
    np.testing.assert_array_equal(sample_index, np.array([0, 1]))

