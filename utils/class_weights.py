import torch

def calculate_class_weights(dataset, num_classes):
    """
    Calculate class weights based on the frequency of classes in the dataset.

    Args:
        dataset (Dataset): The dataset object.
        num_classes (int): The number of classes.

    Returns:
        torch.Tensor: A tensor of class weights.
    """
    class_counts = torch.zeros(num_classes, dtype=torch.long)

    for _, mask, _ in dataset:
        mask = mask
        for c in range(num_classes):
            class_counts[c] += (mask == c).sum()

    # Avoid division by zero
    class_counts = class_counts.float()
    class_weights = 1.0 / (class_counts + 1e-6)  # Inverse frequency
    class_weights /= class_weights.sum()  # Normalize weights

    return class_weights