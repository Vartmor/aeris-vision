"""Lightweight ResNet18 U-Net for binary building localization."""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18


class _DecoderBlock(nn.Module):
    def __init__(self, channels: int, skip_channels: int, out: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels + skip_channels, out, 3, padding=1, bias=False),
            nn.BatchNorm2d(out),
            nn.ReLU(inplace=True),
            nn.Conv2d(out, out, 3, padding=1, bias=False),
            nn.BatchNorm2d(out),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        features = F.interpolate(features, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.layers(torch.cat((features, skip), dim=1))


class LocalizationUNet(nn.Module):
    """Return one unnormalized logit per input pixel."""

    def __init__(self, *, pretrained: bool = False) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        encoder = resnet18(weights=weights)
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.pool = encoder.maxpool
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4
        self.decode3 = _DecoderBlock(512, 256, 128)
        self.decode2 = _DecoderBlock(128, 128, 64)
        self.decode1 = _DecoderBlock(64, 64, 32)
        self.decode0 = _DecoderBlock(32, 64, 16)
        self.head = nn.Conv2d(16, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stem = self.stem(x)
        first = self.layer1(self.pool(stem))
        second = self.layer2(first)
        third = self.layer3(second)
        features = self.decode3(self.layer4(third), third)
        features = self.decode2(features, second)
        features = self.decode1(features, first)
        logits = self.head(self.decode0(features, stem))
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


def bce_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute the two loss terms in FP32, including under CUDA AMP."""
    logits = logits.float()
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probabilities = logits.sigmoid().flatten(1)
    target = target.flatten(1)
    intersection = (probabilities * target).sum(dim=1)
    cardinality = probabilities.sum(dim=1) + target.sum(dim=1)
    dice = (2 * intersection + 1.0) / (cardinality + 1.0)
    return bce + (1 - dice).mean()
