"""
classifier.py
ResNet-50 backbone adapted for multi-label chest X-ray classification
(NIH ChestX-ray14 style: 14 findings, sigmoid outputs since a single
X-ray can show multiple conditions at once).
"""

import os
import torch
import torch.nn as nn
from torchvision import models

from python.utils.config_loader import load_config, resolve_path

_cfg = load_config()


class ChestXrayResNet50(nn.Module):
    """ResNet-50 with the final FC layer replaced for multi-label
    classification. Uses sigmoid (not softmax) since findings are not
    mutually exclusive."""

    def __init__(self, num_classes: int = None, pretrained: bool = None):
        super().__init__()
        num_classes = num_classes or _cfg["model"]["num_classes"]
        pretrained = _cfg["model"]["pretrained_imagenet"] if pretrained is None else pretrained

        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = models.resnet50(weights=weights)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)  # raw logits; apply sigmoid outside for probabilities


def load_model(weights_path: str = None, device: str = "cpu") -> ChestXrayResNet50:
    """
    Build the model and, if a fine-tuned checkpoint exists, load it.
    If no checkpoint is found, the model falls back to the ImageNet
    backbone with a randomly-initialized classification head, and a
    warning is printed -- predictions in that case are NOT clinically
    meaningful and are for pipeline-testing purposes only.
    """
    model = ChestXrayResNet50()
    weights_path = weights_path or resolve_path(_cfg["paths"]["trained_weights_file"])

    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"[classifier] Loaded fine-tuned weights from {weights_path}")
    else:
        print(
            "[classifier] WARNING: no fine-tuned checkpoint found at "
            f"{weights_path}. Using ImageNet backbone with an untrained "
            "classification head. Run python/model/train.py on the NIH "
            "ChestX-ray14 dataset before relying on predictions."
        )

    model.to(device)
    model.eval()
    return model


def predict(model: ChestXrayResNet50, input_tensor: torch.Tensor, device: str = "cpu"):
    """Run inference and return a dict of {class_name: probability}."""
    class_names = _cfg["model"]["class_names"]
    model.eval()
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        logits = model(input_tensor)
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
    return {name: float(prob) for name, prob in zip(class_names, probs)}
