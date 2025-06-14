import argparse
import torch 

def get_args():
    parser = argparse.ArgumentParser(description="Train a segmentation model")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="Weight decay for the optimizer")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training")
    parser.add_argument("--backbone", type=str, default="efficientnet", help="The pretrained model to use as backbone.")

    args = parser.parse_args()

    args.classes = [
        (0, 0, 0), # black background
        (255, 0, 0), # red ball
        (255, 255, 0), # yellow goal
        (0, 0, 255), # blue robot
        (0, 255, 0), # green field
        (255, 255, 255), # white line
    ]

    return args