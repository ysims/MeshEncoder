import torch

def project_to_image(points_3d, lens):
    """
    Project 3D world points to 2D image pixels using fisheye EQUIDISTANT model.
    """
    Hoc = torch.tensor(lens['Hoc'], dtype=torch.float32)
    Hco = torch.linalg.inv(Hoc)
    R = Hco[:3, :3]
    t = torch.tensor([0.0, 0.0, -Hoc[2, 3]], dtype=torch.float32)  # Only Z translation

    cam_points = (R @ points_3d.T + t.reshape(3, 1)).T

    x, y, z = cam_points[:, 0], cam_points[:, 1], cam_points[:, 2]
    theta = torch.atan2(torch.sqrt(x**2 + y**2), z)

    r = lens['focal_length'] * theta * (1 + lens['k'][0] * theta**2 + lens['k'][1] * theta**4)

    phi = torch.atan2(y, x)
    u = r * torch.cos(phi) + lens['centre'][0]
    v = r * torch.sin(phi) + lens['centre'][1]

    pixels = torch.stack([u, v], dim=-1)

    return pixels, cam_points

def create_mesh(images, masks, lens):
    """
    Create a mesh using batched image and mask tensors.
    
    Args:
        images (torch.Tensor): Batched image tensor of shape (B, C, H, W).
        masks (torch.Tensor): Batched mask tensor of shape (B, H, W) or (B, C, H, W).
        lens (dict): Lens parameters.
    
    Returns:
        dict: A dictionary containing the world grid, camera grid, color grid, segmentation grid, and validity mask.
    """
    batch_size, _, height, width = images.shape

    # Convert normalized lens center to pixel coordinates
    centre_normalized = torch.tensor(lens['centre'], dtype=torch.float32)
    lens['centre'] = torch.tensor([
        centre_normalized[0] + (width / 2),
        centre_normalized[1] + (height / 2)
    ], dtype=torch.float32)

    # Create world grid (2D)
    x_range = torch.tensor([0, 10], dtype=torch.float32)
    y_range = torch.tensor([-6, 6], dtype=torch.float32)
    spacing = 0.02
    xs = torch.arange(x_range[0], x_range[1], spacing, dtype=torch.float32)
    ys = torch.arange(y_range[0], y_range[1], spacing, dtype=torch.float32)
    X, Y = torch.meshgrid(xs, ys, indexing='ij')  # shape (H, W)
    H, W = X.shape
    Z = torch.zeros_like(X)
    world_grid = torch.stack([X, Y, Z], dim=-1)  # shape (H, W, 3)

    # Flatten for projection
    world_points_flat = world_grid.reshape(-1, 3)
    pixels, cam_points = project_to_image(world_points_flat, lens)  # (N, 2), (N, 3)

    u = torch.round(pixels[:, 0]).long()
    v = torch.round(pixels[:, 1]).long()
    valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)

    # Initialize output tensors
    colour_grid = torch.zeros((batch_size, H, W, 3), dtype=torch.uint8)
    cam_grid = torch.zeros((batch_size, H, W, 3), dtype=torch.float32)
    seg_grid = torch.zeros((batch_size, H, W, 3), dtype=torch.uint8)

    for b in range(batch_size):
        # Sample colors from the image
        flat_colour_grid = torch.zeros((H * W, 3), dtype=torch.uint8)
        flat_cam_grid = torch.zeros((H * W, 3), dtype=torch.float32)
        flat_colour_grid[valid] = images[b, :, v[valid], u[valid]].permute(1, 0)
        flat_cam_grid[valid] = cam_points[valid]

        colour_grid[b] = flat_colour_grid.reshape(H, W, 3)
        cam_grid[b] = flat_cam_grid.reshape(H, W, 3)

        # Sample segmentation mask
        if masks.shape[1] == 1:  # Single-channel mask
            flat_seg_grid = torch.zeros((H * W, 1), dtype=torch.uint8)
            flat_seg_grid[valid] = masks[b, 0, v[valid], u[valid]].unsqueeze(-1)
        else:  # Multi-channel mask
            flat_seg_grid = torch.zeros((H * W, 3), dtype=torch.uint8)
            flat_seg_grid[valid] = masks[b, :, v[valid], u[valid]].permute(1, 0)

        seg_grid[b] = flat_seg_grid.reshape(H, W, 3)

    return {
        "cam_grid": cam_grid,
        "colour_grid": colour_grid,
        "seg_grid": seg_grid,
    }