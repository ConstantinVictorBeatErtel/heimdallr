"""Tests for the imbalance-aware loss functions."""

from __future__ import annotations

import math

import torch

from wildfire.losses import focal_loss, pick_loss, weighted_bce_loss


def _make_case(*, fire_pixels: int = 4, uncertain_pixels: int = 3):
    """Build a tiny (1, 1, 4, 4) logits/target/mask example.

    Layout: the first `fire_pixels` pixels are fire (1), the next
    `uncertain_pixels` are uncertain (-1, masked out), and the rest are
    no-fire (0).
    """
    h, w = 4, 4
    total = h * w
    target = torch.zeros(total)
    target[:fire_pixels] = 1.0
    target[fire_pixels : fire_pixels + uncertain_pixels] = -1.0
    valid_mask = target != -1.0

    target = target.reshape(1, h, w)
    valid_mask = valid_mask.reshape(1, h, w)
    logits = torch.zeros(1, 1, h, w)
    return logits, target, valid_mask


def test_losses_are_zero_for_perfect_predictions() -> None:
    """Confidently correct logits should yield near-zero loss."""
    logits, target, valid_mask = _make_case(fire_pixels=4, uncertain_pixels=3)
    # Large positive logits where fire, large negative where no-fire.
    logits = torch.where(target.unsqueeze(1) > 0.5, 20.0, -20.0).float()

    bce = weighted_bce_loss(logits, target, valid_mask)
    focal = focal_loss(logits, target, valid_mask)

    assert bce.item() < 1e-3
    assert focal.item() < 1e-3


def test_loss_ignores_uncertain_pixels() -> None:
    """Changing logits only at uncertain (masked) pixels must not change loss."""
    logits, target, valid_mask = _make_case(fire_pixels=4, uncertain_pixels=3)

    base = weighted_bce_loss(logits, target, valid_mask)
    # Flip the sign of the logits exactly at the masked pixels.
    masked_positions = target == -1.0
    logits2 = logits.clone()
    logits2.squeeze(1)[:, :][masked_positions] = 50.0  # huge, but masked
    flipped = weighted_bce_loss(logits2, target, valid_mask)

    assert torch.allclose(base, flipped)


def test_weighted_bce_upweights_positives() -> None:
    """A high pos_weight should make the loss larger than pos_weight=1 on fire."""
    logits, target, valid_mask = _make_case(fire_pixels=4, uncertain_pixels=0)
    # Logits are wrong for positives (predict no-fire), so positives contribute.
    low = weighted_bce_loss(logits, target, valid_mask, pos_weight=1.0)
    high = weighted_bce_loss(logits, target, valid_mask, pos_weight=10.0)
    assert high > low


def test_focal_gamma_zero_matches_bce_shape() -> None:
    """gamma=0 focal reduces to (alpha-weighted) BCE; gamma>0 shrinks easy loss."""
    logits, target, valid_mask = _make_case(fire_pixels=4, uncertain_pixels=0)

    # With gamma=0 focal reduces to a class-weighted BCE. With alpha=0.5 the
    # weight is 0.5 for *every* pixel, so focal_g0 == 0.5 * plain BCE mean.
    focal_g0 = focal_loss(logits, target, valid_mask, gamma=0.0, alpha=0.5)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits.squeeze(1), target, reduction="mean"
    )
    # Both use -ln(0.5) ~= 0.693 per pixel for zero logits.
    assert math.isclose(focal_g0.item(), 0.5 * bce.item(), rel_tol=1e-4)

    # Increasing gamma should *reduce* the loss for these easy, near-0.5 logits
    # because (1 - p_t)^gamma < 1.
    focal_g2 = focal_loss(logits, target, valid_mask, gamma=2.0, alpha=0.5)
    assert focal_g2 < focal_g0


def test_losses_return_scalar_and_backprop() -> None:
    """Each loss must return a 0-d tensor that can backward into logits."""
    logits = torch.zeros(1, 1, 4, 4, requires_grad=True)
    target = torch.zeros(1, 4, 4)
    target[0, 0, 0] = 1.0
    valid_mask = torch.ones(1, 4, 4, dtype=torch.bool)

    for loss_fn in (weighted_bce_loss, focal_loss):
        logits.grad = None
        loss = loss_fn(logits, target, valid_mask)
        assert loss.dim() == 0
        loss.backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()


def test_pick_loss_dispatch() -> None:
    """pick_loss should map known names and reject unknown ones."""
    assert pick_loss("weighted_bce") is weighted_bce_loss
    assert pick_loss("focal") is focal_loss
    try:
        pick_loss("bogus")
    except ValueError:
        return
    raise AssertionError("pick_loss should have raised for an unknown name")
