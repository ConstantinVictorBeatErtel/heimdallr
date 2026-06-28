"""PyTorch Dataset and DataLoader helpers for NDWS."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from wildfire.data.normalize import (
    normalize_inputs,
    normalize_target,
    target_valid_mask,
)
from wildfire.data.reader import build_record_index, read_record_at
from wildfire.data.splits import find_split_files


@dataclass(frozen=True)
class WildfireBatch:
    """One batch returned by the DataLoader collate function."""

    inputs: torch.Tensor  # (N, 12, 64, 64)
    target: torch.Tensor  # (N, 64, 64)
    valid_mask: torch.Tensor  # (N, 64, 64), True = label is known


def collate_wildfire_batch(samples: list[dict[str, torch.Tensor]]) -> WildfireBatch:
    """Stack individual samples into a batch."""
    return WildfireBatch(
        inputs=torch.stack([sample["inputs"] for sample in samples], dim=0),
        target=torch.stack([sample["target"] for sample in samples], dim=0),
        valid_mask=torch.stack([sample["valid_mask"] for sample in samples], dim=0),
    )


class WildfireSpreadDataset(Dataset):
    """Map-style Dataset over official NDWS TFRecord shards.

    Each item returns:
      - inputs:     float tensor (12, 64, 64), normalized channels
      - target:     float tensor (64, 64) with values in {-1, 0, 1}
      - valid_mask: bool tensor (64, 64), False where target is uncertain (-1)

    Uncertain pixels:
      The paper treats -1 as "unknown label" (cloud cover, missing MODIS).
      We keep -1 in `target` but set valid_mask=False so the loss function
      can ignore those pixels instead of treating them as "no fire".
    """

    def __init__(
        self,
        tfrecord_paths: list[Path],
        *,
        normalize_method: str = "minmax",
        require_prev_fire: bool = False,
        require_known_target: bool = False,
    ) -> None:
        self.tfrecord_paths = [Path(path) for path in tfrecord_paths]
        self.normalize_method = normalize_method
        self.require_prev_fire = require_prev_fire
        self.require_known_target = require_known_target

        self._index = build_record_index(self.tfrecord_paths)
        self._kept_indices = self._compute_kept_indices()

    def _compute_kept_indices(self) -> list[int]:
        """Filter samples the reference pipeline would skip."""
        kept: list[int] = []
        for sample_idx in range(len(self._index)):
            raw = self._load_raw(sample_idx)
            target = normalize_target(raw["FireMask"])

            if self.require_prev_fire:
                prev_fire = raw["PrevFireMask"]
                if not np.any(prev_fire == 1.0):
                    continue

            if self.require_known_target and not np.any(target != -1.0):
                continue

            kept.append(sample_idx)
        return kept

    def _load_raw(self, sample_idx: int) -> dict[str, np.ndarray]:
        file_idx, record_idx = self._index[sample_idx]
        return read_record_at(self.tfrecord_paths, file_idx, record_idx)

    def __len__(self) -> int:
        return len(self._kept_indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        raw = self._load_raw(self._kept_indices[idx])
        inputs = normalize_inputs(raw, method=self.normalize_method)
        target = normalize_target(raw["FireMask"])
        valid = target_valid_mask(target)

        return {
            "inputs": torch.from_numpy(inputs),
            "target": torch.from_numpy(target),
            "valid_mask": torch.from_numpy(valid),
        }


class WildfireArrayDataset(Dataset):
    """In-memory dataset for tests and small numpy/array experiments."""

    def __init__(
        self,
        raw_features: list[dict[str, np.ndarray]],
        *,
        normalize_method: str = "minmax",
    ) -> None:
        self.raw_features = raw_features
        self.normalize_method = normalize_method

    def __len__(self) -> int:
        return len(self.raw_features)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        raw = self.raw_features[idx]
        inputs = normalize_inputs(raw, method=self.normalize_method)
        target = normalize_target(raw["FireMask"])
        valid = target_valid_mask(target)
        return {
            "inputs": torch.from_numpy(inputs),
            "target": torch.from_numpy(target),
            "valid_mask": torch.from_numpy(valid),
        }


def create_dataloader(
    dataset: Dataset,
    *,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader with the project collate function."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_wildfire_batch,
    )


def create_split_dataloader(
    data_dir: Path,
    split: str,
    *,
    batch_size: int = 32,
    shuffle: bool | None = None,
    num_workers: int = 0,
    normalize_method: str = "minmax",
) -> DataLoader:
    """Build a DataLoader for one official split (train/val/test)."""
    paths = find_split_files(data_dir, split)
    dataset = WildfireSpreadDataset(paths, normalize_method=normalize_method)
    if shuffle is None:
        shuffle = split == "train"
    return create_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
