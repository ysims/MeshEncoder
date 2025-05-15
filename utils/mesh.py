import torch

def project_to_image(points_3d, centres, focal_lengths, k_params, Hoc, device):
    """
    Project 3D world points to 2D image pixels using fisheye EQUIDISTANT model.
    
    Args:
        points_3d: (B, N, 3) tensor of 3D points in world coordinates
        centres: (B, 2) tensor of principal points (cx, cy)
        focal_lengths: (B,) tensor of focal lengths
        k_params: (B, 2) tensor of distortion parameters (k1, k2)
        Hoc: (B, 4, 4) tensor of camera-to-world transforms
    
    Returns:
        pixels: (B, N, 2) tensor of projected 2D image coordinates
        cam_points: (B, N, 3) tensor of points in camera coordinates
    """
    B, N, _ = points_3d.shape

    # Invert each Hoc to get Hco
    Hco = torch.linalg.inv(Hoc).to(device)  # (B, 4, 4)
    R = Hco[:, :3, :3].to(device)           # (B, 3, 3)
    t = -Hco[:, :3, 3].to(device)           # (B, 3)

    # Transform points: X_cam = R @ X_world + t
    cam_points = torch.bmm(points_3d, R.transpose(1, 2)) + t.unsqueeze(1)  # (B, N, 3)

    x, y, z = cam_points[..., 0], cam_points[..., 1], cam_points[..., 2]
    theta = torch.atan2(torch.sqrt(x ** 2 + y ** 2), z).to(device)  # (B, N)

    k1 = k_params[:, 0].unsqueeze(1)  # (B, 1)
    k2 = k_params[:, 1].unsqueeze(1)  # (B, 1)
    f = focal_lengths.unsqueeze(1)    # (B, 1)

    r = f * theta * (1 + k1 * theta**2 + k2 * theta**4).to(device)  # (B, N)

    phi = torch.atan2(y, x)  # (B, N)
    cx = centres[:, 0].unsqueeze(1)  # (B, 1)
    cy = centres[:, 1].unsqueeze(1)  # (B, 1)

    u = r * torch.cos(phi) + cx
    v = r * torch.sin(phi) + cy

    pixels = torch.stack([u, v], dim=-1)  # (B, N, 2)

    return pixels, cam_points

    
def create_mesh(images, masks, lens):
    """
    Create a mesh using batched image and mask tensors.

    Args:
        images (torch.Tensor): (B, C, H, W) in [0, 1] range.
        masks (torch.Tensor): (B, 1, H, W) or (B, 3, H, W).
        lens (torch.Tensor): (B, N) lens parameters.

    Returns:
        dict: Dictionary with cam_grid, colour_grid, seg_grid.
    """
    batch_size, _, height, width = images.shape

    device = images.device

    # Extract lens parameters
    centres = lens[:, :2].to(device)
    focal_lengths = lens[:, 2].to(device)
    k_params = lens[:, 3:5].to(device)
    Hoc = lens[:, 5:].view(batch_size, 4, 4).to(device)

    # Shift centres
    centres[:, 0] += width / 2
    centres[:, 1] += height / 2

    # World grid
    x_range = torch.linspace(0, 10, int(10 / 0.02), dtype=torch.float32, device=device)
    y_range = torch.linspace(-6, 6, int(12 / 0.02), dtype=torch.float32, device=device)
    X, Y = torch.meshgrid(x_range, y_range, indexing='ij')
    H, W = X.shape
    Z = torch.zeros_like(X)
    world_grid = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)  # (H*W, 3)
    world_grid = world_grid.unsqueeze(0).repeat(batch_size, 1, 1)  # (B, H*W, 3)

    # Project
    pixels, cam_points = project_to_image(world_grid, centres, focal_lengths, k_params, Hoc, device)  # (B, H*W, 2), (B, H*W, 3)

    u = torch.round(pixels[..., 0]).long()
    v = torch.round(pixels[..., 1]).long()
    valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)

    # Create flat indices
    flat_idx = torch.arange(batch_size, device=device).unsqueeze(1)  # (B, 1)
    batch_idx = flat_idx.expand(-1, H * W)[valid]  # (N_valid,)
    u_idx = u[valid]
    v_idx = v[valid]
    flat_valid_idx = torch.arange(H * W, device=device).expand(batch_size, -1)[valid]

    # Images: (B, C, H, W) → (B, H, W, C)
    images_hw = (images.permute(0, 2, 3, 1) * 255).to(torch.uint8)  # (B, H, W, 3)
    colours = images_hw[batch_idx, v_idx, u_idx]  # (N_valid, 3)

    # cam_points: already (B, H*W, 3)
    cam_values = cam_points[valid]  # (N_valid, 3)

    # Masks: handle 1 or 3 channels
    if masks.shape[1] == 1:
        seg = masks[:, 0]  # (B, H, W)
        seg_values = seg[batch_idx, v_idx, u_idx].unsqueeze(-1).repeat(1, 3).to(torch.uint8)  # (N_valid, 3)
    elif masks.dim() == 3:
        masks = masks.unsqueeze(1)  # Convert (B, H, W) -> (B, 1, H, W)
        masks_hw = masks.permute(0, 2, 3, 1)  # (B, H, W, 3)
        seg_values = masks_hw[batch_idx, v_idx, u_idx].to(torch.uint8)
    else:
        masks_hw = masks.permute(0, 2, 3, 1)  # (B, H, W, 3)
        seg_values = masks_hw[batch_idx, v_idx, u_idx].to(torch.uint8)

    # Initialize outputs
    colour_grid = torch.zeros((batch_size, H * W, 3), dtype=torch.uint8, device=device)
    cam_grid = torch.zeros((batch_size, H * W, 3), dtype=torch.float32, device=device)
    seg_grid = torch.zeros((batch_size, H * W, 3), dtype=torch.uint8, device=device)

    # Set valid values using index_put_
    colour_grid.index_put_((batch_idx, flat_valid_idx), colours, accumulate=False)
    cam_grid.index_put_((batch_idx, flat_valid_idx), cam_values, accumulate=False)
    seg_grid.index_put_((batch_idx, flat_valid_idx), seg_values, accumulate=False)

    # Reshape
    colour_grid = colour_grid.view(batch_size, H, W, 3)
    cam_grid = cam_grid.view(batch_size, H, W, 3)
    seg_grid = seg_grid.view(batch_size, H, W, 3)

    # Permute to (B, 3, H, W)
    colour_grid = colour_grid.permute(0, 3, 1, 2)  # (B, 3, H, W)
    cam_grid = cam_grid.permute(0, 3, 1, 2)          # (B, 3, H, W)
    seg_grid = seg_grid.permute(0, 3, 1, 2)          # (B, 3, H, W)

    return cam_grid, colour_grid, seg_grid
