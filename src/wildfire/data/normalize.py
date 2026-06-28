"""Input normalization and fire-mask handling for NDWS samples."""

from __future__ import annotations

import numpy as np

from wildfire.data.constants import (
    DATA_STATS,
    FIRE_UNCERTAIN,
    INPUT_FEATURES,
    OUTPUT_FEATURES,
    ChannelStats,
)


def clip_and_rescale_channel(
    values: np.ndarray,
    stats: ChannelStats,
    *,
    feature_name: str,
) -> np.ndarray:
    """Clip outliers, then map most channels to roughly [-1, 1].

    We follow the original dataset preprocessing:
    1. Clip to training-set min/max to remove extreme sensor outliers.
    2. Min-max rescale to [-1, 1] so all continuous channels share a similar scale.

    PrevFireMask is special: it is only clipped, not rescaled, because it encodes
    discrete classes (-1 uncertain, 0 no fire, 1 fire).
    """
    clipped = np.clip(values, stats.min_clip, stats.max_clip)
    if feature_name == "PrevFireMask":
        return clipped.astype(np.float32)

    span = stats.max_clip - stats.min_clip
    if span == 0:
        return np.zeros_like(clipped, dtype=np.float32)

    rescaled = (clipped - stats.min_clip) / span
    return (rescaled * 2.0 - 1.0).astype(np.float32)


def standardize_channel(values: np.ndarray, stats: ChannelStats) -> np.ndarray:
    """Standardize a channel with training-set mean and std (z-score).

    Used as an alternative to min-max rescaling. The paper's reference code uses
    clip + min-max to [-1, 1]; we expose z-score here for experiments.
    """
    clipped = np.clip(values, stats.min_clip, stats.max_clip)
    if stats.std == 0:
        return np.zeros_like(clipped, dtype=np.float32)
    return ((clipped - stats.mean) / stats.std).astype(np.float32)


def normalize_inputs(
    raw_features: dict[str, np.ndarray],
    *,
    method: str = "minmax",
) -> np.ndarray:
    """Build normalized input tensor with shape (12, H, W)."""
    channels: list[np.ndarray] = []
    for name in INPUT_FEATURES:
        stats = DATA_STATS[name]
        raw = raw_features[name].astype(np.float32)
        if method == "minmax":
            channels.append(clip_and_rescale_channel(raw, stats, feature_name=name))
        elif method == "zscore":
            if name == "PrevFireMask":
                channels.append(clip_and_rescale_channel(raw, stats, feature_name=name))
            else:
                channels.append(standardize_channel(raw, stats))
        else:
            msg = f"Unknown normalization method: {method!r}"
            raise ValueError(msg)

    return np.stack(channels, axis=0)


def normalize_target(raw_target: np.ndarray) -> np.ndarray:
    """Return the fire mask unchanged except for dtype casting.

    Values remain:
      0  -> no fire
      1  -> fire
     -1  -> uncertain (cloud / missing MODIS label)

    We keep -1 in the target tensor and expose a separate valid_mask so the
    training loss can ignore uncertain pixels, matching the original paper.
    """
    return raw_target.astype(np.float32)


def target_valid_mask(target: np.ndarray) -> np.ndarray:
    """True where the next-day label is known (not uncertain)."""
    return target != FIRE_UNCERTAIN


def parse_raw_record(
    raw_features: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Split a parsed TFRecord dict into raw inputs and raw target arrays."""
    inputs = np.stack([raw_features[name] for name in INPUT_FEATURES], axis=0)
    target = raw_features[OUTPUT_FEATURES[0]]
    return inputs.astype(np.float32), target.astype(np.float32)
