import os
import glob
import torch
import yaml
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import time
from utils.transforms import JointTransform

class SoccerSegmentationDataset(Dataset):
    """
    Custom dataset for soccer segmentation task.
    Assumes images are named 'image*.jpg', masks are 'mask*.png', and lens parameters are 'lens*.yaml'.
    folder: Path to the folder containing the images, masks, and lens parameters.
    classes: List of colours corresponding to each class in the segmentation task.
    """
    def __init__(self, folder, classes):
        self.folder = folder
        self.classes = classes

        # Collect all image paths
        self.image_paths = sorted(glob.glob(os.path.join(folder, "image*.jpg")))
        self.mask_paths = [p.replace("image", "mask").replace(".jpg", ".png") for p in self.image_paths]
        self.lens_paths = [p.replace("image", "lens").replace(".jpg", ".yaml") for p in self.image_paths]
        
        self.image_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],  # Standard ImageNet
                        std=[0.229, 0.224, 0.225]),
        ])

        self.mask_transform = T.Compose([
            T.PILToTensor(),
            T.Lambda(lambda x: x[:3, :, :] if x.shape[0] == 4 else x),
        ])

        self.joint_transform = JointTransform(hflip=True, rotation=True)

    def __len__(self):
        return len(self.image_paths)

    def load_lens_params(self, path):
        with open(path, 'r') as f:
            params = yaml.safe_load(f)
        # Convert the lens parameters into a tensor
        centre = torch.tensor(params['centre'], dtype=torch.float32)
        focal_length = torch.tensor([params['focal_length']], dtype=torch.float32)
        k = torch.tensor(params['k'], dtype=torch.float32)
        Hoc = torch.tensor(params['Hoc'], dtype=torch.float32)
        return torch.cat([centre, focal_length, k, Hoc.flatten()])

    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx]).convert("RGB")

        # Load mask
        mask = Image.open(self.mask_paths[idx])

        # Apply transforms
        image = self.image_transform(image)
        mask = self.mask_transform(mask)

        # Apply joint augmentations (augmentation)
        # if self.joint_transform:
        #     image, mask = self.joint_transform(image, mask)
            
        # Convert mask to classes
        mask = mask_to_class_indices(mask, self.classes).long() # Convert to class indices

        # Load lens YAML
        lens = self.load_lens_params(self.lens_paths[idx])
        return image, mask, lens

def mask_to_class_indices(mask: torch.Tensor, classes: list[tuple[int, int, int]]) -> torch.Tensor:
    """
    Fast version: Convert RGB mask (3,H,W) to class index mask (H,W) using vectorized matching.
    """
    c, h, w = mask.shape
    mask = mask.permute(1, 2, 0).reshape(-1, 3)  # [H*W, 3]
    classes_tensor = torch.tensor(classes, dtype=torch.uint8, device=mask.device)  # [C, 3]

    # Compute equality mask: [H*W, C]
    matches = (mask[:, None, :] == classes_tensor[None, :, :]).all(dim=2).float()  # [H*W, C]

    # Get class indices or fallback to 0 (unknown)
    indices = matches.argmax(dim=1)  # If no match, returns 0 (safe default)
    return indices.reshape(h, w)