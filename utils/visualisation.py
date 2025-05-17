from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
import numpy as np

writer = SummaryWriter(log_dir="./runs")

def write_plots(epoch, epoch_loss, precision, recall):
    writer.add_scalar("Loss/train", epoch_loss, epoch)
    for class_name, p in precision.items():
        writer.add_scalar(f"Precision/{class_name}", p, epoch)
    for class_name, r in recall.items():
        writer.add_scalar(f"Recall/{class_name}", r, epoch)

def write_images(epoch, colour_grid, predicted_mask, image, i):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    titles = ['Colour Grid', 'Predicted Mask', 'Original Image']

    def process(img_tensor):
        img = img_tensor.cpu().numpy().squeeze()
        if img.ndim == 3:  # [C, H, W]
            img = img.transpose(1, 2, 0)
        return img

    def reverse_norm(img):
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        return np.clip(img, 0, 1)

    colour_grid = process(colour_grid)
    predicted_mask = process(predicted_mask)
    image = process(image)

    colour_grid = reverse_norm(colour_grid)
    image = reverse_norm(image)

    for ax, img, title in zip(axes, [colour_grid, predicted_mask, image], titles):
        if img.ndim == 2 or img.shape[-1] == 1:
            ax.imshow(img.squeeze(), cmap='gray')
        else:
            ax.imshow(img)
        ax.axis('off')
        ax.set_title(title)

    fig.suptitle(f"Epoch {epoch} - Sample {i}", fontsize=14)
    writer.add_figure(f"Comparison {i}", fig, epoch)
    plt.close()