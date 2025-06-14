from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
import numpy as np
import torch 

writer = SummaryWriter(log_dir="./runs")


def onehot_to_mask(onehot: torch.Tensor, classes: list[tuple[int, int, int]]) -> np.ndarray:
    """
    Convert a one-hot encoded mask [num_classes, H, W] to an RGB mask [H, W, 3].
    """
    # onehot: [num_classes, H, W]
    indices = torch.argmax(onehot, dim=0)  # [H, W]
    indices = indices.cpu().numpy()  # [H, W]
    classes_arr = np.array(classes, dtype=np.uint8)  # [num_classes, 3]
    rgb_mask = classes_arr[indices]  # [H, W, 3]
    return rgb_mask

def write_plots(epoch, epoch_loss, precision, recall):
    writer.add_scalar("Loss/train", epoch_loss, epoch)
    for class_name, p in precision.items():
        writer.add_scalar(f"Precision/{class_name}", p, epoch)
    for class_name, r in recall.items():
        writer.add_scalar(f"Recall/{class_name}", r, epoch)

def write_images(epoch, colour_grid, mask, image, classes, i):
    # Unnormalize and convert image to [H, W, C], uint8
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    
    image = image.squeeze(0).cpu().numpy()  # [C, H, W]
    image = (image * std + mean)  # [C, H, W], float32 in [0,1]
    image = np.clip(image, 0, 1)
    image = np.transpose(image, (1, 2, 0))  # [H, W, C]

    # Unnormalize colour_grid the same way as image
    colour_grid = colour_grid.squeeze(0).cpu().numpy()
    colour_grid = (colour_grid * std + mean)
    colour_grid = np.clip(colour_grid, 0, 1)
    colour_grid = np.transpose(colour_grid, (1, 2, 0))  # [H, W, C]

    # Mask: convert to RGB, [H, W, 3], uint8
    print(f"Mask shape: {mask.shape}")
    mask = onehot_to_mask(mask, classes)  # [H, W, C]
    print(f"Mask shape: {mask.shape}")
    # mask = np.clip(mask, 0, 255).astype(np.uint8)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(colour_grid)
    plt.title("Colour Grid")
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(mask)
    plt.title("Segmentation Grid")
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')
    plt.show()