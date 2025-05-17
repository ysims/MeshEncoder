from models.backbone import FeatureBackbone
from models.semantic_head import SemanticHead
from dataset import SoccerSegmentationDataset
from utils.class_weights import calculate_class_weights
from utils.metrics import MetricsTracker
from utils.mesh import create_mesh
from setup import get_args
from utils.visualisation import write_plots, write_images

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# Convert to RGB with a color map
def colourise(mask, palette):
    h, w = mask.shape
    color = torch.zeros(3, h, w, dtype=torch.uint8)
    for cls, rgb in enumerate(palette):
        color[:, mask == cls] = torch.tensor(rgb, dtype=torch.uint8).view(3, 1)
    return color

args = get_args()

# Create dataset and dataloader
dataset = SoccerSegmentationDataset(folder="./real_data", classes=args.classes)
train_size = int(len(dataset) * 0.6)
val_size = len(dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

# Create models - the backbone and the semantic head
backbone = FeatureBackbone(backbone=args.backbone).to(args.device)
semantic_head = SemanticHead(in_channels=backbone.output_size, num_classes=len(args.classes)).to(args.device) 

# Optimiser
optimizer = optim.Adam(list(backbone.parameters()) + list(semantic_head.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)

# Calculate class weights
# class_weights = calculate_class_weights(train_dataset, num_classes=len(args.classes))

# Loss function
# criterion = nn.CrossEntropyLoss(weight=class_weights.to(args.device))
criterion = nn.CrossEntropyLoss()

# Metrics tracker
metrics_tracker = MetricsTracker(num_classes=len(args.classes))

# Training loop
for epoch in range(args.num_epochs):
    backbone.train()
    semantic_head.train()
    
    for i, (images, masks, lens) in enumerate(train_loader):
        start = time.time()
        optimizer.zero_grad()
        
        images = images.to(args.device)
        masks = masks.to(args.device)
        
        # Sample ground plane mesh
        cam_grid, colour_grid, seg_grid = create_mesh(images, masks, lens)
        grid_shape = colour_grid.shape[2:]

        # Backbone
        features = backbone(colour_grid)

        # Semantic head
        outputs = semantic_head(features, grid_shape)

        # Compute loss
        loss = criterion(outputs, seg_grid)

        try:
            loss.backward()
        except Exception as e:
            print(f"Error during backward pass: {e}")
            continue
        optimizer.step()
        
    backbone.eval()
    semantic_head.eval()
    running_loss = 0.0
    metrics_tracker.reset()  # Reset metrics at the start of each epoch
    for i, (images, masks, lens) in enumerate(val_loader):

        images = images.to(args.device)
        masks = masks.to(args.device)

        # Sample ground plane mesh
        cam_grid, colour_grid, seg_grid = create_mesh(images, masks, lens)

        grid_shape = colour_grid.shape[2:]

        # Backbone
        features = backbone(colour_grid)

        # Semantic head
        outputs = semantic_head(features, grid_shape)
        
        # Compute loss
        loss = criterion(outputs, seg_grid)
        running_loss += loss.item()

        # Calculate metrics
        metrics_tracker.update(outputs, seg_grid)

    # Compute metrics
    metrics = metrics_tracker.compute()

    # Write plots and images to TensorBoard
    write_plots(epoch, running_loss/len(train_loader), metrics['precision'], metrics['recall'])
    
    # Get ten images from the val loader
    for i in range(1):
        with torch.no_grad():
            # Get a batch of images and masks
            images, masks, lens = next(iter(val_loader))
            images = images.to(args.device)
            masks = masks.to(args.device)
            lens = lens.to(args.device)
            cam_grid, colour_grid, seg_grid = create_mesh(images, masks, lens)
            grid_shape = colour_grid.shape[2:]
            features = backbone(colour_grid)
            outputs = semantic_head(features, grid_shape)

            # Single batch, flatten the batch dimension
            outputs = outputs.view(len(args.classes), outputs.shape[2], outputs.shape[3])
            colour_grid = colour_grid.view(3, colour_grid.shape[2], colour_grid.shape[3])
            images = images.view(3, images.shape[2], images.shape[3])

            # Convert outputs to colours
            outputs = colourise(outputs.argmax(dim=0).cpu().numpy(), args.classes)

            write_images(epoch, colour_grid, outputs, images, i)

    # Print metrics
    print(f"Epoch [{epoch+1}/{args.num_epochs}], Loss: {running_loss/len(train_loader):.4f}, "
          f"Precision: {metrics['precision']}, Recall: {metrics['recall']}")
    