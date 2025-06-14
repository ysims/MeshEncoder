import torch

def project_to_image(grid, centre, focal, k, Hoc, device):
    """
    Project batched 3D world points to 2D image pixels using a fisheye equidistant model.
    
    Args:
        grid: [B, N, 3] tensor of 3D world points
        centre: [2] tensor with (cx, cy)
        focal: scalar focal length
        k: [2] distortion coefficients
        Hoc: [B, 4, 4] transformation from object to camera
    
    Returns:
        pixels: [B, N, 2] pixel coordinates
        cam_points: [B, N, 3] camera-space 3D points
    """
    B, N, _ = grid.shape
    Hco = torch.linalg.inv(Hoc)  # [B, 4, 4]

    R = Hco[:, :3, :3].to(device)  # [B, 3, 3]
    t = torch.zeros((B, 3, 1)).to(device)  # [B, 3, 1]
    t[:, 2, 0] = -Hoc[:, 2, 3]

    # Transform grid points to camera coordinates
    grid = grid.transpose(1, 2)  # [B, 3, N]
    cam_points = torch.bmm(R, grid) + t  # [B, 3, N]
    cam_points = cam_points.transpose(1, 2)  # [B, N, 3]

    # Apply robot-to-camera rotation
    R_robot_to_camera = torch.tensor([
        [0, -1, 0],
        [0,  0, -1],
        [1,  0, 0],
    ], dtype=torch.float32).expand(B, -1, -1).to(device)  # [B, 3, 3]

    cam_points = torch.bmm(R_robot_to_camera, cam_points.transpose(1, 2)).transpose(1, 2)  # [B, N, 3]

    # Project to image plane
    x, y, z = cam_points[..., 0], cam_points[..., 1], cam_points[..., 2]
    theta = torch.atan2(torch.sqrt(x**2 + y**2), z)
    
    focal = focal.expand_as(theta)  # [B, N]
    k = k.unsqueeze(1).expand(-1, theta.shape[1], -1)  # [B, N, 2]

    r = focal * theta * (1 + k[..., 0] * theta**2 + k[..., 1] * theta**4)  # [B, N]
    phi = torch.atan2(y, x)  # [B, N]
    u = r * torch.cos(phi) + centre[..., 0].unsqueeze(1)  # [B, N]
    v = r * torch.sin(phi) + centre[..., 1].unsqueeze(1)  # [B, N]

    pixels = torch.stack([u, v], dim=-1)  # [B, N, 2]

    return pixels.long(), cam_points  # [B, N, 2], [B, N, 3]
    
def create_mesh(images, masks, lens):
    """
    Create a mesh using batched images and masks tensors.

    Args:
        images (torch.Tensor): (B, C, H, W) in [0, 1] range.
        masks (torch.Tensor): (B, 1, H, W) or (B, 3, H, W).
        lens (torch.Tensor): (B, N) lens parameters.

    Returns:
        dict: Dictionary with cam_grid, colour_grid, seg_grid.
    """
    
    # Switch around axes
    images = images.permute(0, 2, 3, 1)  # [B, H, W, C]
    masks = masks.permute(0, 1, 2)  # [B, H, W]

    B, img_height, img_width, _ = images.shape
    device = images.device

    centre = lens[:,:2]
    centre = torch.tensor(centre, dtype=torch.float32).to(device)
    centre = (centre - 0.5) * torch.tensor([img_width, img_height], dtype=torch.float32).to(device)

    focal_length = lens[:,2].to(device)
    k = lens[:,3:5].to(device)
    Hoc = lens[:,5:].view(B, 4, 4).to(device)

    # Create a ground plane grid
    height = 6
    width = 9
    spacing = 0.03
    xs = torch.linspace(0, 6, int(height / spacing))
    ys = torch.linspace(-width / 2, width / 2, int(width / spacing))
    ys, xs = torch.meshgrid(ys, xs, indexing='ij')
    grid = torch.stack([xs, ys, torch.zeros_like(xs)], dim=-1).reshape(-1, 3)  # [N, 3]
    grid = grid.to(device)  # Move to the same device as images

    # The height and width of the grid in pixels
    grid_width = int(6 / spacing)  # Number of points along the x-axis
    grid_height = int(9 / spacing)  # Number of points along the y-axis

    # Unsqueeze to add batch dimension
    grid = grid.unsqueeze(0).expand(B, -1, -1)  # [1, N, 3]

    # Get the pixels corresponding to the grid points
    # and the grid points in camera coordinates
    pixels, cam_points = project_to_image(grid, centre, focal_length, k, Hoc, device)
    
    # Remove points outside the images
    u = torch.round(pixels[..., 0]).long()
    v = torch.round(pixels[..., 1]).long()
    valid = (u >= 0) & (u < img_width) & (v >= 0) & (v < img_height)
    
    # Create grids for the sampled images, camera points, and sampled seg masks
    colour_grid = torch.zeros((B, grid_height, grid_width, 3), dtype=torch.float32).to(device)
    cam_grid = torch.zeros((B, grid_height, grid_width, 3), dtype=torch.float32).to(device)
    seg_grid = torch.zeros((B, grid_height, grid_width), dtype=torch.long).to(device)
    
    # Keep only valid pixels and cam points
    valid_indices = torch.nonzero(valid, as_tuple=False)
    b_indices = valid_indices[:, 0]
    flat_indices = valid_indices[:, 1]
    
    # Convert pixel coordinates to grid indices
    i_x = flat_indices % grid_width
    i_y = torch.div(flat_indices, grid_width, rounding_mode='floor')

    # Extract u and v coordinates from pixels
    u_valid = u[b_indices, flat_indices]
    v_valid = v[b_indices, flat_indices]
    
    # Use advanced indexing to fill the grids
    colour_grid[b_indices, i_y, i_x] = images[b_indices, v_valid, u_valid]
    seg_grid[b_indices, i_y, i_x] = masks[b_indices, v_valid, u_valid]
    cam_grid[b_indices, i_y, i_x] = cam_points[b_indices, flat_indices]

    # Convert order back to [B, C, H, W]
    colour_grid = colour_grid.permute(0, 3, 1, 2)  # [B, C, H, W]
    cam_grid = cam_grid.permute(0, 3, 1, 2)  # [B, C, H, W]

    return cam_grid, colour_grid, seg_grid
