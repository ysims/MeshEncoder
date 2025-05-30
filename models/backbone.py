import torch
import torchvision.models as models

class FeatureBackbone(torch.nn.Module):
    def __init__(self, backbone="mobilenetv3"):
        super().__init__()

        # Load the MobileNetV3 model
        if backbone == "mobilenetv3":
            self.backbone = models.mobilenet_v3_small(pretrained=True).features
            self.output_size = 576
        elif backbone == "efficientnet":
            self.backbone = models.efficientnet_b0(pretrained=True).features
            self.output_size = 1280  # EfficientNet-B0 output size
        else:
            raise ValueError("Unsupported backbone. Choose 'mobilenetv3' or 'efficientnet'.")

    def forward(self, x):
        return self.backbone(x)  # output is feature map
    
    