"""Stable, descriptive identifiers for newly generated training runs."""

from __future__ import annotations

import re
from datetime import datetime


def build_run_id(
    *,
    model: str,
    split: str,
    mode: str,
    seed: int,
    timestamp: datetime | None = None,
) -> str:
    """Create ``model__split__mode__timestamp__seed-N`` for a training run."""

    created_at = timestamp or datetime.now()
    return "__".join(
        (
            _slug(model),
            _slug(split),
            _slug(mode),
            created_at.strftime("%Y-%m-%dT%H-%M-%S"),
            f"seed-{int(seed)}",
        )
    )


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if not normalized:
        raise ValueError("Run ID components must contain at least one letter or number.")
    return normalized
