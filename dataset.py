import os
import glob
import torch
import yaml
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
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
            T.PILToTensor(),  # returns ByteTensor [1, H, W]
            T.Lambda(lambda x: x.squeeze(0)),  # make it [H, W]
            T.Lambda(lambda x: x[:, :, :3]),  # remove alpha channel
        ])

        self.joint_transform = JointTransform(hflip=True, rotation=True)

    def __len__(self):
        return len(self.image_paths)

    def load_lens_params(self, path):
        with open(path, 'r') as f:
            params = yaml.safe_load(f)
        # Add any processing if needed
        return params

    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx]).convert("RGB")

        # Load mask
        mask = Image.open(self.mask_paths[idx])

        # Convert mask to classes
        mask = np.array(mask)

        mask = mask_to_class_indices(mask, self.classes)  # Convert to class indices
        mask = torch.tensor(mask, dtype=torch.long)

        # Load lens YAML
        lens = self.load_lens_params(self.lens_paths[idx])

        # Apply transforms (augmentation happens here)
        image = self.transform(image)
        mask = self.mask_transform(mask)

        # Apply joint augmentations
        if self.joint_transform:
            image, mask = self.joint_transform(image, mask)

        return {
            "image": image,       # Tensor [3, H, W]
            "mask": mask,         # Tensor [H, W]
            "lens": lens,         # dict
        }

def mask_to_class_indices(mask, classes):
    """
    Convert a segmentation mask from pixel colors to class indices.

    Args:
        mask (np.ndarray): The segmentation mask as a NumPy array of shape [H, W, 3].
        classes (list): A list of RGB tuples representing class colors.

    Returns:
        torch.Tensor: A tensor of shape [H, W] with class indices.
    """
    # Create a mapping from RGB tuples to class indices
    color_to_index = {tuple(color): idx for idx, color in enumerate(classes)}

    # Flatten the mask and map colors to indices
    h, w, _ = mask.shape
    mask_flat = mask.reshape(-1, 3)
    indices_flat = torch.tensor([color_to_index[tuple(pixel)] for pixel in mask_flat])

    # Reshape back to [H, W]
    return indices_flat.reshape(h, w)