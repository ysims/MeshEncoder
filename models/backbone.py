import torch
import torchvision.models as models

class FeatureBackbone(torch.nn.Module):
    def __init__(self, img_size, backbone="mobilenetv3"):
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
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)


    def forward(self, x):
        x = self.backbone(x)
        # Upsample the feature map to the desired image size
        x = torch.nn.functional.interpolate(x, size=self.img_size, mode='bilinear', align_corners=False)
        return x 
    
    