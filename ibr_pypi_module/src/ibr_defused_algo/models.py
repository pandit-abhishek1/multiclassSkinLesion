"""From-scratch IBR algorithms.

Includes IBR5Net, IBR6Net, and DefusedIBR5IBR6.
"""

from __future__ import annotations

from typing import Iterable, List

import torch
from torch import Tensor, nn


class ConvStage(nn.Module):
    """One stage with two conv-bn-relu blocks."""

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class _IBRBase(nn.Module):
    """Shared backbone builder for IBR variants."""

    def __init__(
        self,
        *,
        num_classes: int,
        channels: Iterable[int],
        strides: Iterable[int],
        dropout: float,
    ) -> None:
        super().__init__()

        channel_list: List[int] = list(channels)
        stride_list: List[int] = list(strides)
        if len(channel_list) != len(stride_list):
            raise ValueError("channels and strides must have same length")

        in_channels = 3
        layers = []
        for out_channels, stride in zip(channel_list, stride_list):
            layers.append(ConvStage(in_channels, out_channels, stride))
            in_channels = out_channels

        self.stages = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = channel_list[-1]
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.feature_dim, num_classes),
        )

    def extract_features(self, x: Tensor) -> Tensor:
        x = self.stages(x)
        x = self.pool(x)
        return x.flatten(1)

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.extract_features(x))


class IBR5Net(_IBRBase):
    """IBR5 from-scratch network with exactly 5 stages."""

    def __init__(self, num_classes: int = 7, dropout: float = 0.2) -> None:
        super().__init__(
            num_classes=num_classes,
            channels=[32, 64, 128, 256, 512],
            strides=[2, 2, 2, 2, 2],
            dropout=dropout,
        )


class IBR6Net(_IBRBase):
    """IBR6 from-scratch network with exactly 6 stages."""

    def __init__(self, num_classes: int = 7, dropout: float = 0.2) -> None:
        super().__init__(
            num_classes=num_classes,
            channels=[32, 64, 128, 256, 384, 512],
            strides=[2, 2, 2, 2, 1, 1],
            dropout=dropout,
        )


class DefusedIBR5IBR6(nn.Module):
    """Late-fusion defused model using IBR5 and IBR6 features."""

    def __init__(self, num_classes: int = 7, dropout: float = 0.3) -> None:
        super().__init__()
        self.ibr5 = IBR5Net(num_classes=num_classes, dropout=dropout)
        self.ibr6 = IBR6Net(num_classes=num_classes, dropout=dropout)

        fused_dim = self.ibr5.feature_dim + self.ibr6.feature_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        f5 = self.ibr5.extract_features(x)
        f6 = self.ibr6.extract_features(x)
        fused = torch.cat((f5, f6), dim=1)
        return self.fusion_head(fused)


FusedIBR5IBR6 = DefusedIBR5IBR6
