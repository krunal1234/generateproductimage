import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

def preprocess_image(image: np.ndarray, target_size=(1024, 1024)):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(target_size),
        transforms.ToTensor()
    ])
    tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return tensor

def postprocess_image(mask: torch.Tensor, original_size):
    mask = transforms.Resize(original_size)(mask.unsqueeze(0))
    mask = mask.squeeze().cpu().detach().numpy()
    mask = (mask * 255).astype(np.uint8)
    return mask