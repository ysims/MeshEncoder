from models.backbone import FeatureBackbone
from models.semantic_head import SemanticHead
from dataset import SoccerSegmentationDataset
from utils.class_weights import calculate_class_weights
from utils.metrics import MetricsTracker
from utils.mesh import create_mesh
from setup import get_args

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

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
class_weights = calculate_class_weights(train_dataset, num_classes=len(args.classes))

# Loss function
criterion = nn.CrossEntropyLoss(weight=class_weights.to(args.device))

# Metrics tracker
metrics_tracker = MetricsTracker(num_classes=len(args.classes))

# Training loop
for epoch in range(args.num_epochs):
    backbone.train()
    semantic_head.train()
    
    running_loss = 0.0
    metrics_tracker.reset()  # Reset metrics at the start of each epoch
    for i, (images, masks, lens) in enumerate(train_loader):
        optimizer.zero_grad()
        
        images = images.to(args.device)
        masks = masks.to(args.device)
        
        # Sample ground plane mesh
        cam_grid, colour_grid, seg_grid = create_mesh(images, masks, lens)

        # Backbone
        features = backbone(colour_grid)

        # Semantic head
        outputs = semantic_head(features)
        outputs = outputs.permute(0, 2, 3, 1)  # [B, H, W, num_classes]
        outputs = outputs.view(-1, len(args.classes))
        masks = masks.view(-1)  # Flatten the mask tensor 

        # Compute loss
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    for i, (images, masks, lens) in enumerate(val_loader):
        images = images.to(args.device)
        masks = masks.to(args.device)

        # Sample ground plane mesh
        cam_grid, colour_grid, seg_grid = create_mesh(images, masks, lens)

        # Backbone
        features = backbone(colour_grid)

        # Semantic head
        outputs = semantic_head(features)
        outputs = outputs.permute(0, 2, 3, 1)
        outputs = outputs.view(-1, len(args.classes))
        masks = masks.view(-1)
        
        # Compute loss
        loss = criterion(outputs, masks)
        running_loss += loss.item()

        # Calculate metrics
        metrics_tracker.update(outputs, masks)

    # Compute metrics
    metrics = metrics_tracker.compute()

    # Print metrics
    print(f"Epoch [{epoch+1}/{args.num_epochs}], Loss: {running_loss/len(train_loader):.4f}, "
          f"Precision: {metrics['precision']}, Recall: {metrics['recall']}")