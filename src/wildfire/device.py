"""Device helpers for Apple Silicon (MPS) and CPU fallback."""

from __future__ import annotations

import torch


def default_device() -> torch.device:
    """Return the best available device on this machine.

    On Apple Silicon Macs, PyTorch can use the Metal backend (MPS) to run
    models on the GPU. Falls back to CPU when MPS is unavailable.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
