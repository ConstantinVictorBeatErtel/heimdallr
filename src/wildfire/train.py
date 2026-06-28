"""Training entry point for next-day wildfire spread models.

This module is intentionally small and explicit so a newcomer can read it
top-to-bottom. It wires together the pieces from `model.py`, `losses.py`, and
the data pipeline in `data/dataset.py`:

    DataLoader -> UNet -> loss -> AMP-scaled backward -> optimizer step
                                                       -> validation each epoch
                                                       -> best-model checkpoint

Design choices (explained for a novice):
  * Fixed seed -- so re-running produces the same result. Critical while
    debugging; otherwise you can't tell whether a change helped or whether you
    just got a lucky RNG draw.
  * Train/val split -- we use the dataset's *official* train and eval splits
    (see `data/splits.py`). The validation set is held out: the model never
    trains on it, so val loss tells us if the model is generalizing or just
    memorizing the training data.
  * AMP mixed precision -- runs the forward/backward in float16 where safe,
    which is faster and uses less memory on the GPU. The GradScaler keeps a
    float32 master copy of the scale to avoid underflow. We disable it on CPU.
  * Checkpointing -- we save the best model (lowest val loss) so a crash or a
    late, overfit epoch can't destroy our best checkpoint.
  * Print val loss each epoch -- the cheapest possible monitoring; upgrade to
    TensorBoard later if you want curves.
"""

from __future__ import annotations

import argparse
import os
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from wildfire.data.dataset import WildfireBatch
from wildfire.device import default_device
from wildfire.losses import pick_loss
from wildfire.model import UNet

# Target value that marks an uncertain label (see data/constants.py). Imported
# lazily-ish; kept here as a local constant for readability of the loss call.
_FIRE_UNCERTAIN: float = -1.0


@dataclass
class TrainConfig:
    """All knobs for a training run in one place.

    Keeping these in a dataclass (instead of many function arguments) makes the
    signature readable and lets tests construct a config quickly.
    """

    epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4  # L2 regularization; tiny nudge against overfit.
    loss_name: str = "weighted_bce"  # "weighted_bce" or "focal"
    seed: int = 42
    checkpoint_dir: Path = Path("checkpoints")
    amp: bool = True  # mixed precision; auto-disabled on CPU.
    num_workers: int = 0


def set_seed(seed: int) -> None:
    """Fix every RNG we might touch so a run is reproducible.

    We also set `PYTHONHASHSEED` and toggle cuDNN to deterministic mode. Note
    that some GPU operations are still non-deterministic by default; this makes
    them as deterministic as PyTorch supports without a big speed hit.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN (trade a little speed for reproducibility).
    torch.use_deterministic_algorithms(True, warn_only=True)


def _binary_target(batch: WildfireBatch) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (binary target, valid_mask) as float tensors on the input device.

    The dataset stores -1 for uncertain pixels and keeps them in `target`. Here
    we produce a clean 0/1 target and a mask that is True only on labeled pixels,
    so the loss can ignore uncertain ones.
    """
    target = batch.target
    valid = batch.valid_mask
    # Treat any non-{0,1} value as invalid just to be safe.
    valid = valid & (target != _FIRE_UNCERTAIN)
    return target.to(torch.float32), valid.to(torch.float32)


def _epoch_loss(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: Callable[..., torch.Tensor],
    device: torch.device,
) -> float:
    """Average loss over one pass through `loader` (no gradients)."""
    model.eval()
    total = 0.0
    seen = 0
    with torch.no_grad():
        for batch in loader:
            batch = _batch_to(batch, device)
            target, valid = _binary_target(batch)
            logits = model(batch.inputs)
            loss = loss_fn(logits, target, valid)
            n = target.numel()
            total += loss.item() * n
            seen += n
    return total / max(seen, 1)


def _batch_to(batch: WildfireBatch, device: torch.device) -> WildfireBatch:
    """Move a WildfireBatch's tensors to `device` (the dataclass has no `.to`)."""
    return WildfireBatch(
        inputs=batch.inputs.to(device),
        target=batch.target.to(device),
        valid_mask=batch.valid_mask.to(device),
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[..., torch.Tensor],
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
) -> float:
    """Run one training epoch and return the mean training loss."""
    model.train()
    total = 0.0
    seen = 0

    for batch in loader:
        batch = _batch_to(batch, device)
        target, valid = _binary_target(batch)
        inputs = batch.inputs

        optimizer.zero_grad(set_to_none=True)

        # AMP forward: autocast picks float16 for ops that benefit and keeps
        # float32 where precision matters. The scaler handles the backward.
        # AMP forward: autocast picks float16 for ops that benefit and keeps
        # float32 where precision matters. The scaler handles the backward.
        # We only enable autocast on CUDA; on MPS/CPU `use_amp` is False so the
        # dtype here is ignored (autocast becomes a no-op).
        use_amp = scaler is not None
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=use_amp
        ):
            logits = model(inputs)
            loss = loss_fn(logits, target, valid)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        n = target.numel()
        total += loss.item() * n
        seen += n

    return total / max(seen, 1)


def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: nn.Module | None = None,
    config: TrainConfig | None = None,
    device: torch.device | None = None,
) -> tuple[nn.Module, dict[str, list[float]]]:
    """Train `model` and return it along with a per-epoch history.

    Args:
        train_loader / val_loader: already-built DataLoaders (see data/dataset).
        model: optional pre-built UNet; built if not provided.
        config:  hyperparameters (see TrainConfig).
        device:  where to train. Defaults to the machine's best device.

    Returns:
        (model, history) where history = {"train": [...], "val": [...]}.
    """
    config = config or TrainConfig()
    set_seed(config.seed)

    device = device or default_device()
    model = model if model is not None else UNet()
    model = model.to(device)

    # AdamW: Adam with proper weight-decay decoupling. A solid default for
    # small segmentation models; we keep the default betas.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    loss_fn = pick_loss(config.loss_name)

    # AMP scaler: only meaningful on CUDA. On MPS/CPU we skip autocast scaling
    # to avoid device-dtype pitfalls (MPS has limited float16 support).
    use_amp = config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type) if use_amp else None

    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = config.checkpoint_dir / "best.pt"

    history: dict[str, list[float]] = {"train": [], "val": []}
    best_val = float("inf")

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, scaler
        )
        val_loss = _epoch_loss(model, val_loader, loss_fn, device)

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        # Print each epoch so you can watch val loss come down. The leading
        # epoch number makes it easy to grep the log later.
        print(
            f"epoch {epoch:02d}/{config.epochs} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
        )

        # Save the best model. We track val loss (not train) because the best
        # *generalizing* model is what we want, not the best memorizer.
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_loss": val_loss,
                    "config": config.__dict__,
                },
                best_path,
            )

    print(f"best val_loss={best_val:.4f} -> {best_path}")
    return model, history


def _build_arg_parser() -> argparse.ArgumentParser:
    """CLI for `python -m wildfire.train`."""
    parser = argparse.ArgumentParser(description="Train the wildfire U-Net.")
    parser.add_argument("--data-dir", type=Path, help="Root of the NDWS dataset.")
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--lr", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--loss", type=str, default=TrainConfig.loss_name)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=TrainConfig.checkpoint_dir
    )
    return parser


def main() -> None:
    """CLI entry point: build dataloaders from the official splits and train."""
    from wildfire.data.dataset import create_split_dataloader

    args = _build_arg_parser().parse_args()

    if args.data_dir is None:
        raise SystemExit(
            "Pass --data-dir pointing at the NDWS dataset root "
            "(the folder containing next_day_wildfire_spread_train/eval/test)."
        )

    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        loss_name=args.loss,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
    )

    train_loader = create_split_dataloader(
        args.data_dir, "train", batch_size=config.batch_size, shuffle=True
    )
    val_loader = create_split_dataloader(
        args.data_dir, "val", batch_size=config.batch_size, shuffle=False
    )

    train_model(train_loader, val_loader, config=config)


if __name__ == "__main__":
    main()
