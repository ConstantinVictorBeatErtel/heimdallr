"""Constants for the Next Day Wildfire Spread (NDWS) dataset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

PATCH_SIZE: Final[int] = 64

# Channel order matches the original dataset / Google reference implementation.
INPUT_FEATURES: Final[list[str]] = [
    "PrevFireMask",
    "elevation",
    "th",  # wind direction (degrees)
    "vs",  # wind speed (m/s)
    "tmmn",  # min temperature (K)
    "tmmx",  # max temperature (K)
    "sph",  # specific humidity
    "pr",  # precipitation (mm)
    "pdsi",  # drought index
    "NDVI",  # vegetation index
    "population",
    "erc",  # energy release component
]

OUTPUT_FEATURES: Final[list[str]] = ["FireMask"]

PREV_FIRE_MASK_INDEX: Final[int] = INPUT_FEATURES.index("PrevFireMask")

# Fire-mask class values stored in the TFRecords.
FIRE_NO: Final[float] = 0.0
FIRE_YES: Final[float] = 1.0
FIRE_UNCERTAIN: Final[float] = -1.0


@dataclass(frozen=True)
class ChannelStats:
    """Per-channel clip bounds and training-set mean/std from the paper."""

    min_clip: float
    max_clip: float
    mean: float
    std: float


# Stats published with the dataset (computed on the training split only).
# Format: clip to [min_clip, max_clip], then standardize with mean/std.
DATA_STATS: Final[dict[str, ChannelStats]] = {
    "PrevFireMask": ChannelStats(-1.0, 1.0, 0.0, 1.0),
    "elevation": ChannelStats(0.0, 3141.0, 657.3, 649.0),
    "th": ChannelStats(0.0, 360.0, 190.3, 72.6),
    "vs": ChannelStats(0.0, 10.02, 3.85, 1.41),
    "tmmn": ChannelStats(253.15, 298.95, 281.1, 8.98),
    "tmmx": ChannelStats(253.15, 315.09, 295.2, 9.82),
    "sph": ChannelStats(0.0, 1.0, 0.0072, 0.0043),
    "pr": ChannelStats(0.0, 44.53, 1.74, 4.48),
    "pdsi": ChannelStats(-6.13, 7.88, -0.005, 2.68),
    "NDVI": ChannelStats(-9821.0, 9996.0, 5157.6, 2466.7),
    "population": ChannelStats(0.0, 2534.06, 25.53, 154.72),
    "erc": ChannelStats(0.0, 106.25, 37.33, 20.85),
    "FireMask": ChannelStats(-1.0, 1.0, 0.0, 1.0),
}

# Official filename prefixes shipped with the Kaggle / Google release.
SPLIT_FILE_PREFIXES: Final[dict[str, str]] = {
    "train": "next_day_wildfire_spread_train",
    "val": "next_day_wildfire_spread_eval",
    "test": "next_day_wildfire_spread_test",
}
