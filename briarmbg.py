import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from torchvision import transforms
import timm
import os

class BriaRMBG(nn.Module):
    def __init__(self):
        super().__init__()

        # Check if the model has already been downloaded, otherwise download it
        model_path = self._get_model_path()

        self.model = RMBGNet()
        state_dict = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # Use CUDA if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _get_model_path(self):
        # Check if model is already downloaded and cached
        cache_dir = os.path.join(os.getenv("CACHE_DIR", "./cache"), "briaai/RMBG-1.4")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Use Hugging Face's hub to download the model if it's not in the cache
        model_path = os.path.join(cache_dir, "model.pth")
        if not os.path.exists(model_path):
            print("Downloading model...")
            hf_hub_download(repo_id="briaai/RMBG-1.4", filename="model.pth", cache_dir=cache_dir)
        
        return model_path

    def forward(self, x):
        return self.model(x)

# Model architecture definition
class RMBGNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Create a backbone using ResNet-18 from timm
        self.backbone = timm.create_model('resnet18', pretrained=False, num_classes=0, features_only=True)

        # Define the decoder part of the network
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Extract features from ResNet backbone
        features = self.backbone(x)
        x = features[-1]

        # Decode the features to get the mask (foreground vs. background)
        x = self.decoder(x)
        return x
