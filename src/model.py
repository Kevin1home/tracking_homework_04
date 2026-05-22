"""ResNet18 для бинарной классификации (AI vs Human)."""
import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """ResNet18 с заменённой последней FC-слоем на num_classes выходов."""
    if pretrained:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    else:
        model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model


def load_weights(model: nn.Module, weights_path: str, device: torch.device) -> nn.Module:
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    return model
