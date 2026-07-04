"""
preprocess.py
Image preprocessing helpers shared by training, inference, and the
Streamlit app. Keeps all transform logic in exactly one place so the
transforms used at inference time always match training time.
"""

from PIL import Image
import numpy as np
import torch
from torchvision import transforms

from python.utils.config_loader import load_config

_cfg = load_config()
_INPUT_SIZE = _cfg["model"]["input_size"]

# ImageNet mean/std -- correct normalization to use since the backbone
# starts from ImageNet-pretrained ResNet-50 weights.
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def get_inference_transform():
    """Deterministic transform pipeline used at inference time."""
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # X-rays are single channel
        transforms.Resize((_INPUT_SIZE, _INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def get_training_transform():
    """Transform pipeline used during training, with light augmentation
    that is safe for medical images (no flips that would mirror anatomy
    incorrectly for interpretation, only mild rotation/brightness)."""
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((_INPUT_SIZE, _INPUT_SIZE)),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def load_image_as_tensor(image: "Image.Image") -> torch.Tensor:
    """Takes a PIL image (already opened, e.g. from Streamlit's file
    uploader) and returns a batched tensor ready for the model."""
    transform = get_inference_transform()
    tensor = transform(image)
    return tensor.unsqueeze(0)  # add batch dimension


def pil_from_uploaded_file(uploaded_file) -> "Image.Image":
    """Convert a Streamlit UploadedFile into a PIL Image safely."""
    image = Image.open(uploaded_file)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def tensor_to_display_array(tensor: torch.Tensor) -> np.ndarray:
    """Undo normalization so a tensor can be displayed as a normal image
    (used when overlaying Grad-CAM heatmaps)."""
    tensor = tensor.clone().detach().cpu().squeeze(0)
    mean = torch.tensor(_IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(_IMAGENET_STD).view(3, 1, 1)
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    array = tensor.permute(1, 2, 0).numpy()
    return (array * 255).astype(np.uint8)
