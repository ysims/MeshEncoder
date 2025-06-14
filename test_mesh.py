import numpy as np
import cv2
import yaml
from matplotlib import pyplot as plt
import torch 
from PIL import Image
import torchvision.transforms as T
from dataset import SoccerSegmentationDataset

classes = [
    (0, 0, 0), # black background
    (255, 0, 0), # red ball
    (255, 255, 0), # yellow goal
    (0, 0, 255), # blue robot
    (0, 255, 0), # green field
    (255, 255, 255), # white line
]

def load_lens_params(path):
    with open(path, 'r') as f:
        params = yaml.safe_load(f)
    # Convert the lens parameters into a tensor
    centre = torch.tensor(params['centre'], dtype=torch.float32)
    focal_length = torch.tensor([params['focal_length']], dtype=torch.float32)
    k = torch.tensor(params['k'], dtype=torch.float32)
    Hoc = torch.tensor(params['Hoc'], dtype=torch.float32)
    return torch.cat([centre, focal_length, k, Hoc.flatten()])

def mask_to_class_indices(mask: torch.Tensor, classes: list[tuple[int, int, int]]) -> torch.Tensor:
    """
    Fast version: Convert RGB mask (3,H,W) to class index mask (H,W) using vectorized matching.
    """
    c, h, w = mask.shape
    mask = mask.permute(1, 2, 0).reshape(-1, 3)  # [H*W, 3]
    classes_tensor = torch.tensor(classes, dtype=torch.uint8, device=mask.device)  # [C, 3]

    # Compute equality mask: [H*W, C]
    matches = (mask[:, None, :] == classes_tensor[None, :, :]).all(dim=2).float()  # [H*W, C]

    # Get class indices or fallback to 0 (unknown)
    indices = matches.argmax(dim=1)  # If no match, returns 0 (safe default)
    return indices.reshape(h, w)
    
def indices_to_mask(indices: torch.Tensor, classes: list[tuple[int, int, int]]) -> torch.Tensor:
    """
    Convert class index mask (H,W) to RGB mask (3,H,W) using vectorized matching.
    """
    b, h, w = indices.shape
    classes_tensor = torch.tensor(classes, dtype=torch.uint8, device=indices.device)  # [C, 3]
    
    # Create a mask of shape [H*W, 3] by repeating the classes tensor
    mask = classes_tensor[indices.flatten()]  # [B, H*W, 3]
    
    return mask.reshape(b, h, w, 3)  # [B, 3, H, W]

def project_to_image(grid, centre, focal, k, Hoc):
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

    R = Hco[:, :3, :3]  # [B, 3, 3]
    t = torch.zeros((B, 3, 1), device=grid.device, dtype=grid.dtype)  # [B, 3, 1]
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
    ], dtype=grid.dtype, device=grid.device).expand(B, -1, -1)  # [B, 3, 3]

    cam_points = torch.bmm(R_robot_to_camera, cam_points.transpose(1, 2)).transpose(1, 2)  # [B, N, 3]

    # Project to image plane
    x, y, z = cam_points[..., 0], cam_points[..., 1], cam_points[..., 2]
    theta = torch.atan2(torch.sqrt(x**2 + y**2), z)

    focal = focal.unsqueeze(1).expand(-1, N)  # [B, N]
    k = k.unsqueeze(1).expand(-1, N, -1)      # [B, N, 2]

    r = focal * theta * (1 + k[..., 0] * theta**2 + k[..., 1] * theta**4)  # [B, N]
    phi = torch.atan2(y, x)  # [B, N]
    u = r * torch.cos(phi) + centre[:, 0].unsqueeze(1)  # [B, N]
    v = r * torch.sin(phi) + centre[:, 1].unsqueeze(1)  # [B, N]

    pixels = torch.stack([u, v], dim=-1)  # [B, N, 2]

    return pixels.long(), cam_points  # [B, N, 2], [B, N, 3]

def create_mesh(image, mask, lens):
    raw_image = image.clone()
    # Switch around axes
    image = image.permute(0, 2, 3, 1)  # [B, H, W, C]
    mask = mask.permute(0, 1, 2)  # [B, H, W]

    B, img_height, img_width, _ = image.shape

    centre = lens[:,:2]
    centre = centre + torch.tensor([img_width / 2, img_height / 2], dtype=centre.dtype, device=centre.device).unsqueeze(0)
    focal_length = lens[:,2]
    k = lens[:,3:5]
    Hoc = lens[:, 5:].view(B, 4, 4)

    # Create a ground plane grid
    height = 8
    width = 6
    spacing = 0.03
    xs = torch.linspace(0, 6, int(height / spacing))
    ys = torch.linspace(-width / 2, width / 2, int(width / spacing))
    ys, xs = torch.meshgrid(ys, xs, indexing='ij')
    grid = torch.stack([xs, ys, torch.zeros_like(xs)], dim=-1).reshape(-1, 3)  # [N, 3]

    # The height and width of the grid in pixels
    grid_width = int(height / spacing)  # Number of points along the x-axis
    grid_height = int(width / spacing)  # Number of points along the y-axis

    # Unsqueeze to add batch dimension
    grid = grid.unsqueeze(0).expand(B, -1, -1)  # [1, N, 3]

    # Get the pixels corresponding to the grid points
    # and the grid points in camera coordinates
    pixels, cam_points = project_to_image(grid, centre, focal_length, k, Hoc)
    
    # Remove points outside the image
    u = torch.round(pixels[..., 0]).long()
    v = torch.round(pixels[..., 1]).long()
    valid = (u >= 0) & (u < img_width) & (v >= 0) & (v < img_height)
    
    # Create grids for the sampled image, camera points, and sampled seg mask
    colour_grid = torch.zeros((B, grid_height, grid_width, 3), dtype=torch.float32)
    cam_grid = torch.zeros((B, grid_height, grid_width, 3), dtype=torch.float32)
    seg_grid = torch.zeros((B, grid_height, grid_width), dtype=torch.long)
    
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
    colour_grid[b_indices, i_y, i_x] = image[b_indices, v_valid, u_valid]
    seg_grid[b_indices, i_y, i_x] = mask[b_indices, v_valid, u_valid]
    cam_grid[b_indices, i_y, i_x] = cam_points[b_indices, flat_indices]

    # Squeeze
    colour_grid = colour_grid.squeeze(0)
    cam_grid = cam_grid.squeeze(0)

    # Convert seg grid to RGB
    seg_grid = indices_to_mask(seg_grid, classes)  # [B, 3, H, W]
    seg_grid = seg_grid.squeeze(0)

    # Unnormalize the colour grid with ImageNet mean and std
    colour_grid_np = colour_grid.numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    colour_grid_np = (colour_grid_np * std + mean) * 255
    colour_grid_np = np.clip(colour_grid_np, 0, 255).astype(np.uint8)   

    # Transpose to match expected orientation (X horizontal, Y vertical)
    colour_grid_np = colour_grid_np.transpose(1, 0, 2)
    seg_grid_np = seg_grid.numpy().transpose(1, 0, 2)
    colour_grid_np = np.flipud(colour_grid_np)
    seg_grid_np = np.flipud(seg_grid_np)
    colour_grid_np = np.fliplr(colour_grid_np)
    seg_grid_np = np.fliplr(seg_grid_np)
    print(colour_grid_np.shape, seg_grid_np.shape)
    # Visualize the results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(colour_grid_np)
    plt.title("Colour Grid")
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(seg_grid_np, cmap='gray')
    plt.title("Segmentation Grid")
    plt.axis('off')
    
    # raw image
    raw_image_np = raw_image.squeeze(0).permute(1, 2, 0).numpy()
    raw_image_np = (raw_image_np * std + mean) * 255
    raw_image_np = np.clip(raw_image_np, 0, 255).astype(np.uint8)
    # raw_image_np = np.flipud(raw_image_np)
    plt.subplot(1, 3, 3)
    plt.imshow(raw_image_np)
    plt.title("Raw Image")
    plt.axis('off')

    plt.show()

    # Grids need to be converted to expected H, W, C format
    cam_grid = cam_grid.permute(1, 0, 2)
    colour_grid = colour_grid.permute(1, 0, 2)
    seg_grid = seg_grid.permute(1, 0, 2)

    return colour_grid, cam_grid

def data_to_tensor(image, mask):
    image_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],  # Standard ImageNet
                    std=[0.229, 0.224, 0.225]),
    ])

    mask_transform = T.Compose([
        T.PILToTensor(),
        T.Lambda(lambda x: x[:3, :, :] if x.shape[0] == 4 else x),
    ])

    image = image_transform(image).to(torch.float32)
    mask = mask_transform(mask)
    mask = mask_to_class_indices(mask, classes).long()  # Convert to class indices

    # Put it in a batch
    image = image.unsqueeze(0)  # [1, C, H, W]
    mask = mask.unsqueeze(0)  # [1, H, W]
    return image, mask

if __name__ == "__main__":
    dataset = SoccerSegmentationDataset(folder="./real_eindhoven", classes=classes)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    for i, (image, mask, lens) in enumerate(dataloader):
        create_mesh(image, mask, lens)
    
    # create_mesh(image, mask, lens[0])

    # image = Image.open("test/image.jpg").convert("RGB")
    # mask = Image.open("test/mask.png")
    # lens = load_lens_params("test/lens.yaml")
    # image, mask = data_to_tensor(image, mask)
    # create_mesh(image, mask, lens)
