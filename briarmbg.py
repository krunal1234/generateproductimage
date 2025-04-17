import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import transforms

def preprocess_image(image: np.ndarray, target_size=(1024, 1024)) -> torch.Tensor:
    """
    Resize and normalize image to feed into BriaRMBG model.
    """
    h, w = image.shape[:2]
    image_resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

    # Convert to tensor and normalize to [0, 1]
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    image_tensor = transform(image_resized).unsqueeze(0)  # Add batch dimension
    return image_tensor

def postprocess_image(prediction: torch.Tensor, original_shape: tuple) -> np.ndarray:
    """
    Convert model output to a grayscale alpha mask image.
    """
    # Resize back to original shape
    mask = F.interpolate(prediction.unsqueeze(0).unsqueeze(0), size=original_shape, mode='bilinear', align_corners=False)
    mask = mask.squeeze().cpu().detach().numpy()

    # Normalize to 0–255
    mask = (mask * 255).astype(np.uint8)
    return mask
