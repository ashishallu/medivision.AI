"""
gradcam.py
Grad-CAM implementation: highlights which regions of the chest X-ray
drove the model's prediction for a given finding, so the tool is
explainable rather than a black box.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from python.utils.config_loader import load_config
from python.utils.preprocess import tensor_to_display_array

_cfg = load_config()


class GradCAM:
    """Standard Grad-CAM hooked onto a target convolutional layer
    (default: ResNet-50's layer4, the last conv block before pooling)."""

    def __init__(self, model, target_layer_name: str = None):
        self.model = model
        target_layer_name = target_layer_name or _cfg["model"]["gradcam_target_layer"]
        self.target_layer = dict(self.model.backbone.named_modules())[target_layer_name]

        self.activations = None
        self.gradients = None

        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_index: int) -> np.ndarray:
        """Returns a normalized (0-1) heatmap of shape (H, W) for the
        given class index."""
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output[0, class_index]
        score.backward()

        # Global-average-pool the gradients to get per-channel weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam

    def overlay_on_image(self, input_tensor: torch.Tensor, class_index: int, alpha: float = 0.4):
        """Returns an RGB numpy array with the Grad-CAM heatmap overlaid
        on the (denormalized) original input image."""
        cam = self.generate(input_tensor, class_index)

        input_size = _cfg["model"]["input_size"]
        cam_resized = cv2.resize(cam, (input_size, input_size))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        original = tensor_to_display_array(input_tensor)
        overlay = np.uint8(original * (1 - alpha) + heatmap * alpha)
        return overlay
