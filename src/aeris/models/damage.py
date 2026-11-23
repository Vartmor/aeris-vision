"""Post-only ResNet18 damage classifier."""

from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from aeris.data.labels import DAMAGE_CLASS_NAMES


def build_damage_model(*, pretrained: bool = False) -> nn.Module:
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(DAMAGE_CLASS_NAMES))
    return model
