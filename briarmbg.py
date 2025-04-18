import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from torch.nn import functional as F


# Official MattingNetwork structure from BRIAAI repo
class BriaRMBG(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize model structure
        self.model = MattingNetwork()

        # Load pretrained weights
        model_path = hf_hub_download(repo_id="briaai/RMBG-1.4", filename="model.pth")
        state_dict = torch.load(model_path, map_location="cpu")

        self.model.load_state_dict(state_dict)
        self.model.eval()

    def forward(self, x):
        return self.model(x)


# Official architecture (copied/translated from BRIAAI)
class ConvGNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=32):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.gn = nn.GroupNorm(groups, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.gn(self.conv(x)))


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.body = nn.Sequential(
            ConvGNReLU(in_channels, out_channels, 3, stride=2, padding=1),
            ConvGNReLU(out_channels, out_channels, 3, stride=1, padding=1)
        )

    def forward(self, x):
        return self.body(x)


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvGNReLU(in_channels, out_channels, 3, stride=1, padding=1),
            ConvGNReLU(out_channels, out_channels, 3, stride=1, padding=1)
        )

    def forward(self, x):
        return self.body(x)


class MattingNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.inconv = ConvGNReLU(3, 64, 3, stride=1, padding=1)
        self.down1 = DownSample(64, 128)
        self.down2 = DownSample(128, 256)
        self.down3 = DownSample(256, 512)

        self.up3 = UpSample(512, 256)
        self.up2 = UpSample(256, 128)
        self.up1 = UpSample(128, 64)

        self.outconv = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x1 = self.inconv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up3(x4)
        x = self.up2(x + x3)
        x = self.up1(x + x2)

        out = self.outconv(x + x1)
        return torch.sigmoid(out)