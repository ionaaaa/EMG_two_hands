import inspect

import numpy as np

from emg_live_marker.config import DEFAULT_BANDPASS_HIGH_HZ
from emg_live_marker.realtime.stream_processor import StreamingEMGProcessor


def test_process_block_shapes_for_small_block():
    processor = StreamingEMGProcessor()
    data = np.ones((10, 8), dtype=np.float64)

    out = processor.process_block(data)

    assert set(out) == {"raw", "filtered", "rectified", "rms"}
    for value in out.values():
        assert value.shape == (10, 8)


def test_default_bandpass_high_is_90hz():
    processor = StreamingEMGProcessor()

    assert DEFAULT_BANDPASS_HIGH_HZ == 90.0
    assert processor.config.lowpass == 90.0


def test_process_block_single_sample_does_not_crash():
    processor = StreamingEMGProcessor()
    out = processor.process_block(np.ones((1, 8), dtype=np.float64))

    for value in out.values():
        assert value.shape == (1, 8)


def test_continuous_calls_keep_valid_shapes():
    processor = StreamingEMGProcessor()
    rng = np.random.default_rng(123)

    for _ in range(5):
        out = processor.process_block(rng.normal(size=(7, 8)))
        assert out["filtered"].shape == (7, 8)


def test_rectified_and_rms_are_non_negative():
    processor = StreamingEMGProcessor()
    rng = np.random.default_rng(123)
    out = processor.process_block(rng.normal(size=(30, 8)))

    assert np.all(out["rectified"] >= 0)
    assert np.all(out["rms"] >= 0)


def test_update_config_notch_off_and_on():
    processor = StreamingEMGProcessor()
    data = np.ones((12, 8), dtype=np.float64)

    processor.update_config(notch_freq=None)
    assert processor.process_block(data)["filtered"].shape == (12, 8)

    processor.update_config(notch_freq=50.0)
    assert processor.process_block(data)["filtered"].shape == (12, 8)


def test_update_config_supports_dual_notch_options():
    processor = StreamingEMGProcessor()

    processor.update_config(notch_freq=(50.0, 100.0))
    assert processor.config.notch_freq == (50.0, 100.0)
    assert len(processor.notch_sos) == 2
    assert processor.process_block(np.ones((12, 8), dtype=np.float64))["filtered"].shape == (12, 8)

    processor.update_config(notch_freq=(60.0, 120.0))
    assert processor.config.notch_freq == (60.0, 120.0)
    assert len(processor.notch_sos) == 2
    assert processor.process_block(np.ones((12, 8), dtype=np.float64))["filtered"].shape == (12, 8)


def test_dual_notch_suppresses_50hz_and_100hz_components():
    fs = 250.0
    t = np.arange(2500, dtype=np.float64) / fs
    signal = 200.0 * np.sin(2.0 * np.pi * 50.0 * t) + 200.0 * np.sin(2.0 * np.pi * 100.0 * t)
    data = np.repeat(signal[:, None], 8, axis=1)

    no_notch = StreamingEMGProcessor()
    dual_notch = StreamingEMGProcessor()
    dual_notch.update_config(notch_freq=(50.0, 100.0))

    baseline = no_notch.process_block(data)["filtered"][500:, 0]
    filtered = dual_notch.process_block(data)["filtered"][500:, 0]
    freqs = np.fft.rfftfreq(baseline.size, d=1.0 / fs)
    baseline_amp = np.abs(np.fft.rfft(baseline))
    filtered_amp = np.abs(np.fft.rfft(filtered))

    idx_50 = int(np.argmin(np.abs(freqs - 50.0)))
    idx_100 = int(np.argmin(np.abs(freqs - 100.0)))
    assert filtered_amp[idx_50] < baseline_amp[idx_50] * 0.05
    assert filtered_amp[idx_100] < baseline_amp[idx_100] * 0.05


def test_streaming_processor_does_not_use_offline_filtering():
    source = inspect.getsource(StreamingEMGProcessor).lower()
    assert "filtfilt" not in source
