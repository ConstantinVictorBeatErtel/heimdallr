"""Tests for the training loop (seed, AMP-skip on CPU, checkpoint, val loss)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

# Reuse the project's synthetic-record factory so we don't need real TFRecords.
from tests.data.synthetic import make_synthetic_batch
from wildfire.data.dataset import WildfireArrayDataset, create_dataloader
from wildfire.train import TrainConfig, train_model


def _tiny_loaders(batch_size: int = 4):
    """Build small train/val DataLoaders from synthetic 64x64 records."""
    train_ds = WildfireArrayDataset(make_synthetic_batch(8))
    val_ds = WildfireArrayDataset(make_synthetic_batch(4))
    train_loader = create_dataloader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = create_dataloader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def test_train_model_runs_and_saves_checkpoint(tmp_path: Path) -> None:
    """A 2-epoch run should return history and write a best.pt checkpoint."""
    train_loader, val_loader = _tiny_loaders()
    config = TrainConfig(
        epochs=2,
        batch_size=4,
        learning_rate=1e-3,
        loss_name="weighted_bce",
        seed=42,
        checkpoint_dir=tmp_path,
        amp=True,  # should be auto-disabled on CPU
    )

    # Force CPU so the test is deterministic and doesn't depend on MPS state.
    model, history = train_model(
        train_loader, val_loader, config=config, device=torch.device("cpu")
    )

    # History has one entry per epoch for both splits.
    assert len(history["train"]) == config.epochs
    assert len(history["val"]) == config.epochs

    # All losses are finite numbers.
    assert all(torch.isfinite(torch.tensor(v)) for v in history["train"])
    assert all(torch.isfinite(torch.tensor(v)) for v in history["val"])

    # Best checkpoint was written and can be reloaded.
    best = tmp_path / "best.pt"
    assert best.exists()
    ckpt = torch.load(best, weights_only=False)
    assert "model_state" in ckpt
    assert "val_loss" in ckpt
    assert ckpt["epoch"] in (1, 2)


def test_train_model_focal_loss(tmp_path: Path) -> None:
    """The focal loss path should also run end-to-end on CPU."""
    train_loader, val_loader = _tiny_loaders()
    config = TrainConfig(epochs=1, loss_name="focal", checkpoint_dir=tmp_path, seed=0)
    model, history = train_model(
        train_loader, val_loader, config=config, device=torch.device("cpu")
    )
    assert len(history["val"]) == 1
    assert (tmp_path / "best.pt").exists()


def test_seed_is_reproducible_on_cpu(tmp_path: Path) -> None:
    """Two runs with the same seed should produce identical val-loss history.

    We let `train_model` build the model *after* `set_seed` runs, so the model's
    random initialization is identical across runs. (If you construct a model
    yourself and pass it in, its init happens before the seed is set and won't
    be reproducible.)
    """
    cfg = TrainConfig(epochs=1, seed=7, checkpoint_dir=tmp_path)

    def run() -> list[float]:
        train_loader, val_loader = _tiny_loaders()
        # Don't pass a model: train_model builds it after set_seed, so both
        # runs start from identical weights and identical shuffle order.
        _, history = train_model(
            train_loader,
            val_loader,
            config=cfg,
            device=torch.device("cpu"),
        )
        return history["val"]

    first = run()
    second = run()
    assert len(first) == len(second) == 1
    assert first[0] == pytest.approx(second[0], rel=1e-6)
