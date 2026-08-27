from datetime import datetime

from emg_live_marker.run_naming import build_run_id


def test_build_run_id_includes_model_split_mode_timestamp_and_seed():
    run_id = build_run_id(
        model="EffiE",
        split="cross_session_cv",
        mode="freeze_backbone",
        seed=42,
        timestamp=datetime(2026, 7, 10, 14, 30, 5),
    )

    assert run_id == "effie__cross-session-cv__freeze-backbone__2026-07-10T14-30-05__seed-42"
