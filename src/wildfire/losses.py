"""Loss functions for binary fire-mask prediction under class imbalance.

The dataset target has three values:
    1  -> fire
    0  -> no fire
   -1  -> uncertain (cloud / missing MODIS label)

Every loss here accepts a `valid_mask` (bool, True where the label is known) and
ignores uncertain pixels, so they never push the gradient in either direction.

Two losses are provided because fire pixels are rare (~1-2% of valid pixels):

* `weighted_bce_loss` -- pos_weight-scaled binary cross entropy.
* `focal_loss`        -- focal loss (Lin et al., 2017).

Which should you use? See `pick_loss` and the tradeoff notes below.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F

# Shape convention used throughout:
#   logits:     (N, 1, H, W)  -- raw model output, pre-sigmoid
#   target:     (N, H, W)     -- values in {-1, 0, 1}
#   valid_mask: (N, H, W)     -- bool, True where target is a real label


def _prepare(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flatten and align shapes; cast the mask to a float for multiplication."""
    if logits.dim() == 4 and logits.shape[1] == 1:
        logits = logits.squeeze(1)  # (N, H, W)
    if logits.shape != target.shape:
        msg = f"Shape mismatch: logits {logits.shape} vs target {target.shape}"
        raise ValueError(msg)
    if valid_mask.shape != target.shape:
        msg = f"Shape mismatch: mask {valid_mask.shape} vs target {target.shape}"
        raise ValueError(msg)

    flat_logits = logits.reshape(-1)
    flat_target = target.reshape(-1)
    flat_mask = valid_mask.reshape(-1).to(flat_logits.dtype)
    return flat_logits, flat_target, flat_mask


def weighted_bce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    pos_weight: float | torch.Tensor | None = None,
) -> torch.Tensor:
    """Binary cross-entropy that up-weights the rare positive (fire) class.

    Args:
        logits:      (N, 1, H, W) raw model outputs.
        target:      (N, H, W) with values in {-1, 0, 1}.
        valid_mask:  (N, H, W) bool, True where the label is known.
        pos_weight:  multiplier applied to the *positive* term of BCE. A good
            default is #negatives / #positives, which makes the model pay as
            much attention to each fire pixel as to each non-fire pixel. If
            None we estimate this ratio from the current batch.

    Returns:
        Scalar loss (mean over valid pixels).

    Tradeoff (vs focal):
        + Simple, stable, one knob (pos_weight). Easy to reason about.
        + Re-weights every positive equally -- good when positives are simply
          *few* rather than *hard*.
        - Still lets the model rack up easy "true negative" confidence, so the
          gradient can stay dominated by easy background even after weighting.
    """
    flat_logits, flat_target, flat_mask = _prepare(logits, target, valid_mask)

    if pos_weight is None:
        # Estimate from this batch: #no-fire / #fire over valid pixels.
        positives = (flat_target * flat_mask).sum().clamp_min(1.0)
        negatives = ((1.0 - flat_target) * flat_mask).sum().clamp_min(1.0)
        pos_weight = (negatives / positives).detach()

    if not torch.is_tensor(pos_weight):
        pos_weight = torch.tensor(
            float(pos_weight), device=flat_logits.device, dtype=flat_logits.dtype
        )

    # BCEWithLogitsLoss is numerically stable (it uses the log-sum-exp trick
    # instead of computing sigmoid then log, which avoids log(0)).
    loss = F.binary_cross_entropy_with_logits(
        flat_logits,
        flat_target,
        pos_weight=pos_weight,
        reduction="none",  # we'll mask + mean ourselves
    )
    loss = (loss * flat_mask).sum() / flat_mask.sum().clamp_min(1.0)
    return loss


def focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> torch.Tensor:
    """Focal loss for extreme class imbalance.

    Focal loss = alpha * (1 - p_t)^gamma * -log(p_t),
    where p_t = p  if y == 1 else (1 - p).

    - `gamma` "focuses" the loss: easy, confidently-correct pixels get a small
      (1 - p_t)^gamma factor and contribute little gradient. Hard, misclassified
      pixels keep most of their loss. gamma=0 recovers plain BCE.
    - `alpha` balances the two classes like a pos_weight (alpha on positives,
      1-alpha on negatives). alpha=0.25 is the value from the original paper.

    Tradeoff (vs weighted BCE):
        + Down-weights *easy* examples, so the gradient is dominated by the hard
          cases near the fire boundary -- exactly the pixels we care about.
        + Works well at very high imbalance (e.g. 1:100+).
        - Two knobs (gamma, alpha) instead of one; can be unstable early in
          training when the model is unsure about everything.
        - Slightly more expensive and a bit harder to debug than BCE.

    Recommended: start with `weighted_bce_loss`, and switch to focal only if
    the BCE model struggles to predict any fire at all.
    """
    flat_logits, flat_target, flat_mask = _prepare(logits, target, valid_mask)

    # p_t and the modulating factor (1 - p_t)^gamma, computed in a numerically
    # stable way from the logits (no explicit sigmoid needed for the factor).
    p = torch.sigmoid(flat_logits)
    p_t = torch.where(flat_target > 0.5, p, 1.0 - p)
    modulating = (1.0 - p_t) ** gamma

    # Per-example BCE (no reduction) is still stable here because PyTorch's
    # implementation guards against log(0).
    bce = F.binary_cross_entropy_with_logits(flat_logits, flat_target, reduction="none")

    # Class balance weight: alpha for positives, (1 - alpha) for negatives.
    alpha_t = torch.where(
        flat_target > 0.5,
        torch.full_like(p, alpha),
        torch.full_like(p, 1.0 - alpha),
    )

    loss = alpha_t * modulating * bce
    loss = (loss * flat_mask).sum() / flat_mask.sum().clamp_min(1.0)
    return loss


def pick_loss(name: str) -> Callable[..., torch.Tensor]:
    """Look up a loss function by name for the training script.

    Kept tiny so `train.py` can do `pick_loss(args.loss)` without importing
    a long if/else chain.
    """
    losses = {"weighted_bce": weighted_bce_loss, "focal": focal_loss}
    if name not in losses:
        msg = f"Unknown loss {name!r}. Choose one of: {sorted(losses)}"
        raise ValueError(msg)
    return losses[name]
