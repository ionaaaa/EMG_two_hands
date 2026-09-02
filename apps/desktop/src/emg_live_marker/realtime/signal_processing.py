"""Shared, non-UI signal-processing choices for teacher and student modes."""

from __future__ import annotations

from emg_live_marker.realtime.stream_processor import NotchSpec


NOTCH_OPTIONS: dict[str, tuple[float, ...]] = {
    "Off": (),
    "50Hz": (50.0,),
    "50+100Hz": (50.0, 100.0),
    "60Hz": (60.0,),
    "60+120Hz": (60.0, 120.0),
}


def normalize_notch_option(value: object) -> str:
    """Return a supported persisted notch choice, defaulting safely to Off."""

    option = str(value).strip()
    return option if option in NOTCH_OPTIONS else "Off"


def notch_spec_from_option(value: object) -> NotchSpec:
    """Map a teacher-facing option to the processor's notch configuration."""

    frequencies = NOTCH_OPTIONS[normalize_notch_option(value)]
    if not frequencies:
        return None
    return frequencies[0] if len(frequencies) == 1 else frequencies
