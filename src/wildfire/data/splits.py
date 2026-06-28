"""Train/validation/test split helpers without cross-split leakage."""

from __future__ import annotations

from pathlib import Path

from wildfire.data.constants import SPLIT_FILE_PREFIXES


def find_split_files(data_dir: Path, split: str) -> list[Path]:
    """Return TFRecord shards for an official dataset split.

    The NDWS authors already split data by calendar week (8:1:1 train/val/test)
    with a one-day buffer between weeks. Re-splitting random patches would leak
    the same fire event across splits, so we always use their files:

      train -> next_day_wildfire_spread_train*
      val   -> next_day_wildfire_spread_eval*
      test  -> next_day_wildfire_spread_test*
    """
    if split not in SPLIT_FILE_PREFIXES:
        allowed = ", ".join(sorted(SPLIT_FILE_PREFIXES))
        msg = f"Unknown split {split!r}. Expected one of: {allowed}"
        raise ValueError(msg)

    prefix = SPLIT_FILE_PREFIXES[split]
    paths = sorted(data_dir.glob(f"{prefix}*"))
    if not paths:
        msg = (
            f"No files matching {prefix}* found in {data_dir}. "
            "Download the official NDWS TFRecord release (see README)."
        )
        raise FileNotFoundError(msg)
    return paths


def official_splits(data_dir: Path) -> dict[str, list[Path]]:
    """Return all official train/val/test TFRecord paths."""
    return {name: find_split_files(data_dir, name) for name in SPLIT_FILE_PREFIXES}
