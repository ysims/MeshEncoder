from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
import numpy as np
import torch 

writer = SummaryWriter(log_dir="./runs")


def indices_to_mask(indices: torch.Tensor, classes: list[tuple[int, int, int]]) -> torch.Tensor:
    """
    Convert class index mask (C, H, W) to RGB mask (H, W, 3) using vectorized matching.
    """
    c, h, w = indices.shape  # C is the number of classes
    classes_tensor = torch.tensor(classes, dtype=torch.uint8, device=indices.device)  # [C, 3]

    # Ensure indices is in the shape [H, W] by taking the argmax along the class dimension
    class_indices = indices.argmax(dim=0)  # [H, W]

    # Map class indices to RGB values
    mask = classes_tensor[class_indices.flatten()]  # [H*W, 3]
    mask = mask.view(h, w, 3)  # [H, W, 3]
    return mask

def write_plots(epoch, epoch_loss, precision, recall):
    writer.add_scalar("Loss/train", epoch_loss, epoch)
    for class_name, p in precision.items():
        writer.add_scalar(f"Precision/{class_name}", p, epoch)
    for class_name, r in recall.items():
        writer.add_scalar(f"Recall/{class_name}", r, epoch)

def write_images(epoch, colour_grid, mask, image, classes, i):
    # Unnormalise the image and permute the axes
    colour_grid = colour_grid.permute(0, 3, 2, 1)  # [B, H, W, C]
    colour_grid = colour_grid.squeeze(0)
    colour_grid = colour_grid.cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    colour_grid = (colour_grid * std + mean) * 255
    colour_grid = np.clip(colour_grid, 0, 255).astype(np.uint8)   
    
    # Convert to mask colours and squeeze the batch dimension
    mask = indices_to_mask(mask, classes)
    mask = mask.squeeze(0)
    mask = mask.cpu().numpy()
    mask = mask.reshape(mask.shape[1], mask.shape[0], 3)
    
    # Squeeze the image
    image = image.permute(0, 3, 2, 1)  # [B, H, W, C]
    image = image.squeeze(0)
    image = image.cpu().numpy()

    # fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    # titles = ['Colour Grid', 'Predicted Mask', 'Original Image']
    
    # (200, 300, 3) (200, 300, 3)
    print(colour_grid.shape, mask.shape, image.shape)

    # Visualize the results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(colour_grid)
    plt.title("Colour Grid")
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='gray')
    plt.title("Segmentation Grid")
    plt.axis('off')
    # plt.show()
    plt.subplot(1, 3, 3)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')
    plt.show()