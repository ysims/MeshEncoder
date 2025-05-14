import torch

class MetricsTracker:
    """
    Tracks precision and recall per class across epochs.

    Args:
        num_classes (int): Number of classes.
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.true_positives = torch.zeros(self.num_classes, dtype=torch.long)
        self.false_positives = torch.zeros(self.num_classes, dtype=torch.long)
        self.false_negatives = torch.zeros(self.num_classes, dtype=torch.long)

    def update(self, outputs, masks):
        """
        Update metrics based on model predictions and ground truth.

        Args:
            outputs (torch.Tensor): Model predictions of shape [N, num_classes] (logits or probabilities).
            masks (torch.Tensor): Ground truth labels of shape [N].
        """
        preds = torch.argmax(outputs, dim=1)

        for c in range(self.num_classes):
            self.true_positives[c] += ((preds == c) & (masks == c)).sum().item()
            self.false_positives[c] += ((preds == c) & (masks != c)).sum().item()
            self.false_negatives[c] += ((preds != c) & (masks == c)).sum().item()

    def compute(self):
        """
        Compute precision and recall for each class.

        Returns:
            dict: A dictionary with precision and recall for each class.
        """
        precision = []
        recall = []

        for c in range(self.num_classes):
            tp = self.true_positives[c].item()
            fp = self.false_positives[c].item()
            fn = self.false_negatives[c].item()

            # Precision: TP / (TP + FP)
            precision_c = tp / (tp + fp) if tp + fp > 0 else 0.0

            # Recall: TP / (TP + FN)
            recall_c = tp / (tp + fn) if tp + fn > 0 else 0.0

            precision.append(precision_c)
            recall.append(recall_c)

        metrics = {
            "precision": {f"class_{c}": precision[c] for c in range(self.num_classes)},
            "recall": {f"class_{c}": recall[c] for c in range(self.num_classes)},
        }

        return metrics