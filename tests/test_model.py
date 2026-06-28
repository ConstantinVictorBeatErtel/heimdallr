"""Tests for the small wildfire U-Net."""

from __future__ import annotations

import pytest
import torch

from wildfire.model import NUM_INPUT_CHANNELS, UNet


def test_unet_output_shape_and_range() -> None:
    """Forward pass should produce (N, 1, 64, 64) logits; sigmoid in [0, 1]."""
    model = UNet()
    x = torch.randn(2, NUM_INPUT_CHANNELS, 64, 64)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (2, 1, 64, 64)

    probs = torch.sigmoid(logits)
    assert probs.min().item() >= 0.0
    assert probs.max().item() <= 1.0


def test_unet_is_small() -> None:
    """Keep the model tiny -- under 1M parameters for a 64x64 task."""
    model = UNet()
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params < 1_000_000, f"UNet has {n_params} params; expected < 1M"


def test_unet_gradient_flows() -> None:
    """A backward pass from a dummy loss should populate .grad on parameters."""
    model = UNet()
    x = torch.randn(1, NUM_INPUT_CHANNELS, 64, 64)
    target = torch.zeros(1, 64, 64)
    logits = model(x).squeeze(1)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    loss.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_unet_rejects_wrong_channels() -> None:
    """A mismatched channel count should raise rather than silently run."""
    model = UNet(in_channels=NUM_INPUT_CHANNELS)
    bad = torch.randn(1, NUM_INPUT_CHANNELS + 1, 64, 64)
    with pytest.raises(RuntimeError):
        model(bad)
