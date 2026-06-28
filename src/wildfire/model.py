"""A small U-Net for next-day wildfire spread on 64x64 patches.

Input  : (N, 12, 64, 64)  -- the 12 NDWS feature channels
Output : (N, 1, 64, 64)   -- per-pixel fire probability in [0, 1] (sigmoid)

Why U-Net?
-----------
Wildfire spread is a *spatial* problem: whether a pixel burns depends on its
neighbors (wind, fuel, terrain). U-Net is a convolutional encoder-decoder with
"skip connections" that copy high-resolution features from the encoder straight
to the decoder. That lets the model combine fine spatial detail (where the fire
edge is) with coarse context (is a big fire nearby?) -- exactly what we need to
predict a per-pixel mask at the same resolution as the input.

We deliberately keep this *small*. 64x64 is tiny by segmentation standards, so a
big U-Net would overfit and train slowly. Base width is 16 channels, we use 3
downsamples (64 -> 32 -> 16 -> 8) and bilinear upsampling (fewer parameters than
transposed convs). Total parameter count is well under 1M.
"""

from __future__ import annotations

import torch
from torch import nn

# Number of input feature channels in the NDWS dataset (see data/constants.py).
NUM_INPUT_CHANNELS: int = 12


class ConvBlock(nn.Module):
    """Two 3x3 convs, each followed by BatchNorm and ReLU.

    This is the standard U-Net building block. BatchNorm keeps activations on a
    stable scale (helps training and lets us use a larger learning rate); ReLU
    is a cheap nonlinearity. Two convs per block give enough receptive field
    without stacking many layers.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """Small U-Net for 64x64 wildfire spread prediction.

    Args:
        in_channels: Number of input feature channels (default 12 for NDWS).
        base_channels: Width of the first encoder level. The network doubles
            this at each downsampling step. 16 keeps the model small.
        num_classes: Output channels. 1 for a binary fire / no-fire mask.
    """

    def __init__(
        self,
        in_channels: int = NUM_INPUT_CHANNELS,
        base_channels: int = 16,
        num_classes: int = 1,
    ) -> None:
        super().__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8  # bottleneck width

        # --- Encoder: each level halves spatial size via 2x2 max-pool. ---
        self.enc1 = ConvBlock(in_channels, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.enc3 = ConvBlock(c2, c3)
        self.pool = nn.MaxPool2d(kernel_size=2)

        # --- Bottleneck: deepest, smallest spatial map (8x8 for 64 input). ---
        self.bottleneck = ConvBlock(c3, c4)

        # --- Decoder: upsample, concat the matching skip, then refine. ---
        # After concatenation the conv block sees (up_channels + skip_channels).
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3 = ConvBlock(c4 + c3, c3)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = ConvBlock(c3 + c2, c2)

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = ConvBlock(c2 + c1, c1)

        # 1x1 conv acts like a linear classifier per pixel -> 1 output channel.
        self.head = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder path -- keep each level's output for the skip connection.
        e1 = self.enc1(x)  # (N, c1, 64, 64)
        e2 = self.enc2(self.pool(e1))  # (N, c2, 32, 32)
        e3 = self.enc3(self.pool(e2))  # (N, c3, 16, 16)
        b = self.bottleneck(self.pool(e3))  # (N, c4, 8, 8)

        # Decoder path -- upsample, concat skip from the matching encoder level,
        # then run a conv block. Matching spatial sizes is required for concat.
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))  # (N, c3, 16, 16)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))  # (N, c2, 32, 32)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))  # (N, c1, 64, 64)

        # Raw logits (no sigmoid here -- the loss applies sigmoid internally via
        # BCEWithLogitsLoss / focal-with-logits, which is more numerically stable).
        return self.head(d1)  # (N, 1, 64, 64)
