import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import torch.nn.functional as F

def preprocess_image(image: np.ndarray, target_size=(1024, 1024)):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(target_size),
        transforms.ToTensor()
    ])
    tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return tensor

def postprocess_image(prediction, orig_shape):
    # Ensure input is 4D (N, C, H, W)
    if prediction.dim() == 2:
        prediction = prediction.unsqueeze(0).unsqueeze(0)
    elif prediction.dim() == 3:
        prediction = prediction.unsqueeze(0)
    
    resized = F.interpolate(prediction, size=orig_shape, mode='bilinear', align_corners=False)
    mask = resized.squeeze().cpu().detach().numpy() * 255
    mask = np.clip(mask, 0, 255).astype(np.uint8)
    return mask