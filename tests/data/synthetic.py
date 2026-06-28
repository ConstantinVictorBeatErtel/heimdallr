"""Synthetic NDWS samples for unit tests."""

from __future__ import annotations

import numpy as np

from wildfire.data.constants import (
    FIRE_UNCERTAIN,
    FIRE_YES,
    INPUT_FEATURES,
    OUTPUT_FEATURES,
    PATCH_SIZE,
)


def make_synthetic_record(
    *,
    fire_center: tuple[int, int] = (32, 32),
    uncertain_pixels: int = 0,
) -> dict[str, np.ndarray]:
    """Create one fake 64x64 example with realistic value ranges."""
    size = PATCH_SIZE
    yy, xx = np.mgrid[0:size, 0:size]

    # Previous-day fire: small hotspot at the center.
    prev_fire = np.zeros((size, size), dtype=np.float32)
    cy, cx = fire_center
    prev_fire[cy - 2 : cy + 3, cx - 2 : cx + 3] = FIRE_YES

    # Next-day target: fire spread one pixel east.
    fire_mask = np.zeros((size, size), dtype=np.float32)
    fire_mask[cy - 2 : cy + 3, cx - 1 : cx + 4] = FIRE_YES

    if uncertain_pixels > 0:
        fire_mask[0:uncertain_pixels, 0] = FIRE_UNCERTAIN

    record: dict[str, np.ndarray] = {
        "PrevFireMask": prev_fire,
        "elevation": np.full((size, size), 800.0, dtype=np.float32),
        "th": np.full((size, size), 180.0, dtype=np.float32),
        "vs": np.full((size, size), 4.0, dtype=np.float32),
        "tmmn": np.full((size, size), 280.0, dtype=np.float32),
        "tmmx": np.full((size, size), 295.0, dtype=np.float32),
        "sph": np.full((size, size), 0.007, dtype=np.float32),
        "pr": np.full((size, size), 1.0, dtype=np.float32),
        "pdsi": np.full((size, size), -0.5, dtype=np.float32),
        "NDVI": np.full((size, size), 5000.0, dtype=np.float32),
        "population": np.full((size, size), 10.0, dtype=np.float32),
        "erc": np.full((size, size), 40.0, dtype=np.float32),
    }

    # Touch coordinates so linters do not flag yy/xx as unused.
    record["elevation"] = record["elevation"] + (yy % 3) * 0.01 + (xx % 5) * 0.01
    record[OUTPUT_FEATURES[0]] = fire_mask

    missing = set(INPUT_FEATURES + OUTPUT_FEATURES) - set(record)
    if missing:
        msg = f"Synthetic record missing keys: {sorted(missing)}"
        raise ValueError(msg)

    return record


def make_synthetic_batch(num_samples: int = 4) -> list[dict[str, np.ndarray]]:
    """Create several synthetic records with varied fire positions."""
    records: list[dict[str, np.ndarray]] = []
    for idx in range(num_samples):
        center = (28 + idx, 30 + idx)
        uncertain = idx  # first sample has no uncertain pixels
        records.append(
            make_synthetic_record(fire_center=center, uncertain_pixels=uncertain)
        )
    return records
