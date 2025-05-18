import numpy as np
import cv2
import yaml
from matplotlib import pyplot as plt
import torch 
from PIL import Image
import torchvision.transforms as T


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

def project_to_image(grid, centre, focal, k, Hoc):
    """
    Project 3D world points to 2D image pixels using a fisheye equidistant model.
    """
    Hco = torch.linalg.inv(Hoc)

    R = Hco[:3, :3]
    t = torch.tensor([0, 0, -Hoc[2, 3]], dtype=torch.float32)  # Translation vector

    # Transform grid points to camera coordinates
    cam_points = (R @ grid.T + t.reshape(3, 1)).T

    # Camera to image
    R_robot_to_camera = torch.tensor([
        [0, -1, 0],
        [0,  0, -1],
        [1,  0, 0],
    ], dtype=torch.float32)
    cam_points = (R_robot_to_camera @ cam_points.T).T

    # Calculate pixel coordinates
    x, y, z = cam_points[..., 0], cam_points[..., 1], cam_points[..., 2]

    # Projection model
    theta = torch.atan2(torch.sqrt(x ** 2 + y ** 2), z)
    r = focal * theta * (1 + k[0] * theta**2 + k[1] * theta**4)
    # Calculate pixel coordinates
    phi = torch.atan2(y, x)
    u = r * torch.cos(phi) + centre[0]
    v = r * torch.sin(phi) + centre[1]

    return torch.stack([u, v], axis=-1).long(), cam_points

def create_mesh(image, mask, lens):

    image_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],  # Standard ImageNet
                    std=[0.229, 0.224, 0.225]),
    ])

    mask_transform = T.Compose([
        T.PILToTensor(),
        T.Lambda(lambda x: x[:3, :, :]),
    ])
    image_np = np.array(image)
    mask_np = np.array(mask)
    image_np = torch.from_numpy(image_np).to(torch.uint8)
    mask_np = torch.from_numpy(mask_np).to(torch.uint8)
    print(image_np.shape)
    print(mask_np.shape)
    print(image_np.dtype)
    print(mask_np.dtype)

    # image = cv2.imread(image)
    # mask = cv2.imread(mask)
    image = image_transform(image).to(torch.float32)
    mask = mask_transform(mask).to(torch.uint8)
    print(image.shape)
    print(mask.shape)
    print(image.dtype)
    print(mask.dtype)

    # image = np.array(image)
    # mask = np.array(mask)
    # image = torch.from_numpy(image).to(torch.uint8)
    # mask = torch.from_numpy(mask).to(torch.uint8)
    # # Cut out 4th channel of mask
    # mask = mask[:, :, 0:3]

    # Switch around axes
    image = image.permute(1, 2, 0)  # [H, W, C]
    mask = mask.permute(1, 2, 0)  # [H, W, C]

    img_height, img_width = image.shape[:2]

    classes = [
        (0, 0, 0), # black background
        (255, 0, 0), # red ball
        (0, 255, 255), # yellow goal
        (0, 0, 255), # blue robot
        (0, 255, 0), # green field
        (255, 255, 255), # white line
    ]

    centre = lens[:2]
    centre = np.array([centre[0] + img_width / 2, centre[1] + img_height / 2])
    focal_length = lens[2]
    k = lens[3:5]
    Hoc = lens[5:].view(4, 4)

    # Create a ground plane grid
    height = 6
    width = 9
    spacing = 0.03
    xs = torch.linspace(0, 6, int(height / spacing))
    ys = torch.linspace(-width / 2, width / 2, int(width / spacing))
    ys, xs = torch.meshgrid(ys, xs, indexing='ij')  # Match NumPy's iteration order
    grid = torch.stack([xs, ys, torch.zeros_like(xs)], dim=-1).reshape(-1, 3)  # [N, 3]

    # The height and width of the grid in pixels
    grid_width = int(6 / spacing)  # Number of points along the x-axis
    grid_height = int(9 / spacing)  # Number of points along the y-axis

    # Get the pixels corresponding to the grid points
    # and the grid points in camera coordinates
    pixels, cam_points = project_to_image(grid, centre, focal_length, k, Hoc)
    
    # Remove points outside the image
    u = torch.round(pixels[..., 0]).long()
    v = torch.round(pixels[..., 1]).long()
    valid = (u >= 0) & (u < img_width) & (v >= 0) & (v < img_height)
    
    # Keep only valid pixels and cam points
    pixels = pixels[valid]
    cam_points = cam_points[valid]
    valid_indices = torch.nonzero(valid).squeeze()
    
    # Convert pixel coordinates to grid indices
    i_x = valid_indices % grid_width
    i_y = torch.div(valid_indices, grid_width, rounding_mode='floor')

    # Create grids for the sampled image, camera points, and sampled seg mask
    colour_grid = torch.zeros((grid_height, grid_width, 3), dtype=torch.float32)
    cam_grid = torch.zeros((grid_height, grid_width, 3), dtype=torch.float32)
    seg_grid = torch.zeros((grid_height, grid_width, 3), dtype=torch.uint8)

    # Extract u and v coordinates from pixels
    u, v = pixels[:, 0], pixels[:, 1]

    # Use advanced indexing to fill the grids
    colour_grid[i_y, i_x] = image[v, u]
    seg_grid[i_y, i_x] = mask[v, u]
    cam_grid[i_y, i_x] = cam_points

    colour_grid = colour_grid.numpy()
    cam_grid = cam_grid.numpy()
    seg_grid = seg_grid.numpy()

    # Unnormalize the colour grid with ImageNet mean and std
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    colour_grid = (colour_grid * std + mean) * 255
    colour_grid = np.clip(colour_grid, 0, 255).astype(np.uint8)
    
    

    # Transpose to match expected orientation (X horizontal, Y vertical)
    colour_grid = colour_grid.transpose(1, 0, 2)
    cam_grid = cam_grid.transpose(1, 0, 2)
    seg_grid = seg_grid.transpose(1, 0, 2)
    colour_grid_flipped = np.flipud(colour_grid)
    seg_grid_flipped = np.flipud(seg_grid)

    # Visualize the results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(colour_grid_flipped)
    plt.title("Colour Grid")
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(seg_grid_flipped, cmap='gray')
    plt.title("Segmentation Grid")
    plt.axis('off')
    plt.show()

    return colour_grid, cam_grid

if __name__ == "__main__":
    image = Image.open("test/image.jpg")
    mask = Image.open("test/mask.png")
    lens = load_lens_params("test/lens.yaml")
    create_mesh(image, mask, lens)
