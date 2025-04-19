import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

# Suggested preprocess enhancement
def preprocess_image(image: np.ndarray, size: List[int]) -> torch.Tensor:
    image = cv2.resize(image, (size[1], size[0]), interpolation=cv2.INTER_CUBIC)
    image = image.astype(np.float32) / 255.0
    image = image.transpose((2, 0, 1))  # HWC to CHW
    image = torch.from_numpy(image).unsqueeze(0)
    return image

def postprocess_image(mask: torch.Tensor, original_size):
    mask = transforms.Resize(original_size)(mask.unsqueeze(0))
    mask = mask.squeeze().cpu().detach().numpy()
    mask = (mask * 255).astype(np.uint8)
    return mask