"""External EffiE adapter entry points.

This module intentionally re-exports the project adapter instead of copying the
upstream repository into the main package.
"""

from emg_live_marker.ml.effie_adapter import (  # noqa: F401
    MODEL_TYPE,
    EffieGestureNet,
    EffieGesturePredictor,
    load_effie_checkpoint,
    load_effie_finetuned_model,
    replace_classifier,
    set_trainable_mode,
)

