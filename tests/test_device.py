"""Tests for device selection on Apple Silicon."""

from __future__ import annotations

import platform

import torch

from wildfire.device import default_device


def test_default_device_is_cpu_or_mps() -> None:
    device = default_device()
    assert device.type in {"cpu", "mps"}

    if platform.machine() == "arm64" and torch.backends.mps.is_available():
        assert device.type == "mps"
