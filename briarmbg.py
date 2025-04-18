# briarmbg.py
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from torchvision import transforms
import timm

class BriaRMBG(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pretrained model from Hugging Face
        model_path = hf_hub_download(repo_id="briaai/RMBG-1.4", filename="model.pth")
        self.model = RMBGNet()
        state_dict = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def forward(self, x):
        return self.model(x)

# Model architecture definition
class RMBGNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('resnet18', pretrained=False, num_classes=0, features_only=True)
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
        features = self.backbone(x)
        x = features[-1]
        x = self.decoder(x)
        return x
