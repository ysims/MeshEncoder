from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
import numpy as np
import torch 

# Clear the previous logs and make a writer
import shutil
log_dir = "./runs"
shutil.rmtree(log_dir, ignore_errors=True)
writer = SummaryWriter(log_dir=log_dir)

# print full numpy array
np.set_printoptions(threshold=np.inf, linewidth=200, precision=3, suppress=True)

def indices_to_mask(indices: torch.Tensor, classes: list[tuple[int, int, int]]) -> torch.Tensor:
    """
    Convert class index mask (H,W) to RGB mask (3,H,W) using vectorized matching.
    """
    b, h, w = indices.shape
    classes_tensor = torch.tensor(classes, dtype=torch.uint8, device=indices.device)  # [C, 3]
    
    # Create a mask of shape [H*W, 3] by repeating the classes tensor
    mask = classes_tensor[indices.flatten()]  # [B, H*W, 3]
    
    return mask.reshape(b, h, w, 3)  # [B, 3, H, W]

def write_plots(epoch, epoch_loss, precision, recall):
    writer.add_scalar("Loss/train", epoch_loss, epoch)
    for class_name, p in precision.items():
        writer.add_scalar(f"Precision/{class_name}", p, epoch)
    for class_name, r in recall.items():
        writer.add_scalar(f"Recall/{class_name}", r, epoch)

def write_images(epoch, colour_grid, mask, image, seg_grid, classes, i):
    # Unnormalize and convert image to [H, W, C], uint8
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)  # [C, 1, 1]
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)  # [C, 1, 1]
    
    image = image.squeeze(0).cpu().numpy()  # [C, H, W]
    image = (image * std + mean)  # [C, H, W], float32 in [0,1]
    # image = np.clip(image, 0, 1)
    image = np.transpose(image, (1, 2, 0))  # [H, W, C]

    # Unnormalize colour_grid the same way as image
    colour_grid = colour_grid.squeeze(0).cpu().numpy()
    colour_grid = (colour_grid * std + mean) * 255
    colour_grid = np.clip(colour_grid, 0, 255).astype(np.uint8)
    colour_grid = np.transpose(colour_grid, (2, 1, 0))  # [H, W, C]

    # Seg grid
    seg_grid = indices_to_mask(seg_grid, classes)  # [H, W, C]
    seg_grid = seg_grid.squeeze(0).numpy()  # [H, W, C]
    seg_grid = np.transpose(seg_grid, (1, 0, 2))  # [H, W, C]

    # Mask: convert to RGB, [H, W, 3], uint8
    mask = indices_to_mask(mask, classes)  # [H, W, C]
    mask = mask.squeeze(0)  # Remove batch dimension
    mask = np.transpose(mask, (1, 0, 2))  # [H, W, C]
    mask = mask.numpy()  # Scale to [0, 255]

    # Flip the mask vertically to match the image orientation
    # mask = np.flipud(mask) 
    # mask = np.fliplr(mask) 
    colour_grid = np.flipud(colour_grid) 
    colour_grid = np.fliplr(colour_grid) 
    seg_grid = np.flipud(seg_grid)
    seg_grid = np.fliplr(seg_grid)

    # Write to TensorBoard with raw, predicted mask, colour grid next to each other for easy comparison
    writer.add_image(f"{i}/A-Grid", colour_grid, epoch, dataformats='HWC')
    writer.add_image(f"{i}/B-Mask", mask, epoch, dataformats='HWC')
    writer.add_image(f"{i}/C-True Mask", seg_grid, epoch, dataformats='HWC')
    writer.add_image(f"{i}/D-Raw Image", image, epoch, dataformats='HWC')
    