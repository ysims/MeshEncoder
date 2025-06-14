import torch.nn as nn
import torch.nn.functional as F

class SemanticHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, num_classes, kernel_size=1)
        )

    def forward(self, x, grid_shape):
        x = self.head(x)  # [B, num_classes, H, W]
        x = F.interpolate(x, size=grid_shape, mode='bilinear', align_corners=False)
        return x