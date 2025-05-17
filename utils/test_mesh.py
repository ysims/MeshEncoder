import numpy as np
import cv2
import yaml
from matplotlib import pyplot as plt
import torch 
from PIL import Image

def load_lens_params_yaml(lens):
    with open(lens, 'r') as f:
        lens = yaml.safe_load(f)

    return {
        'Hoc': np.array(lens['Hoc']),
        'centre': np.array(lens['centre']),
        'focal_length': lens['focal_length'],
        'k': lens['k'],
        'projection': lens['projection']
    }

def load_lens_params_torch(path):
    with open(path, 'r') as f:
        params = yaml.safe_load(f)
    # Convert the lens parameters into a tensor
    centre = torch.tensor(params['centre'], dtype=torch.float32)
    focal_length = torch.tensor([params['focal_length']], dtype=torch.float32)
    k = torch.tensor(params['k'], dtype=torch.float32)
    Hoc = torch.tensor(params['Hoc'], dtype=torch.float32)
    return torch.cat([centre, focal_length, k, Hoc.flatten()])


def valid_pixels(pixels, image_shape):
    h, w = image_shape[:2]
    return (
        (pixels[:, 0] >= 0) & (pixels[:, 0] < w) &
        (pixels[:, 1] >= 0) & (pixels[:, 1] < h)
    )

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
    # t = np.array([0.0, 0.0, -Hoc[2, 3]])
    t = torch.zeros(3)
    t[2] = -Hoc[2, 3]  # Only Z translation
    # t = t.numpy()
    # R = R.numpy()

    Hco_numpy = np.linalg.inv(Hoc)
    R_numpy = Hco_numpy[:3, :3]
    t_numpy = np.zeros(3)
    t_numpy[2] = -Hoc[2, 3]  # Only Z translation

    t = t.numpy()
    R = R.numpy()

    # print("R_numpy", R_numpy)
    # print("R torch", R)
    # print("t_numpy", t_numpy)
    # print("t torch", t)
    

    cam_points = (R @ grid.T + t.reshape(3, 1)).T

    # Convert to OpenCV's camera coordinate system
    R_robot_to_camera = np.array([
        [0, -1, 0],
        [0,  0, -1],
        [1,  0, 0],
    ])
    cam_points = (R_robot_to_camera @ cam_points.T).T

    x, y, z = cam_points[:, 0], cam_points[:, 1], cam_points[:, 2]
    theta = np.arctan2(np.sqrt(x**2 + y**2), z)

    r = focal * theta * (1 + k[0]*theta**2 + k[1]*theta**4)
    phi = np.arctan2(y, x)

    u = r * np.cos(phi) + centre[0]
    v = r * np.sin(phi) + centre[1]

    return np.stack([u, v], axis=-1), cam_points

def create_mesh(image, mask, lens):
    # image = cv2.imread(image)
    # mask = cv2.imread(mask)
    image = np.array(image)
    mask = np.array(mask)
    # Cut out 4th channel of mask
    mask = mask[:, :, 0:3]

    height, width = image.shape[:2]

    classes = [
        (0, 0, 0), # black background
        (255, 0, 0), # red ball
        (0, 255, 255), # yellow goal
        (0, 0, 255), # blue robot
        (0, 255, 0), # green field
        (255, 255, 255), # white line
    ]

    centre = lens[:2]
    centre = np.array([centre[0] + width / 2, centre[1] + height / 2])
    focal_length = lens[2]
    k = lens[3:5]
    Hoc = lens[5:].view(4, 4)

    # Define world grid (XY plane at Z=0)
    height = 6
    width = 9
    spacing = 0.03
    xs = np.linspace(0, 6, int(height / spacing))
    ys = np.linspace(-width / 2, width / 2, int(width / spacing))
    # xs = np.arange(x_range[0], x_range[1], spacing)
    # ys = np.arange(y_range[0], y_range[1], spacing)
    grid = np.array([[x, y, 0] for y in ys for x in xs])

    pixels, cam_points = project_to_image(grid, centre, focal_length, k, Hoc)

    in_front = cam_points[:, 2] > 0
    valid = in_front & valid_pixels(pixels, image.shape)

    pixels_valid = pixels[valid]
    cam_points_valid = cam_points[valid]
    valid_indices = np.where(valid)[0]
    i_y, i_x = np.divmod(valid_indices, len(xs))

    colour_grid = np.zeros((len(ys), len(xs), 3), dtype=np.uint8)
    cam_grid = np.zeros((len(ys), len(xs), 3), dtype=np.float32)
    seg_grid = np.zeros((len(ys), len(xs), 3), dtype=np.uint8)

    for idx, (u, v) in enumerate(pixels_valid.astype(int)):
        colour_grid[i_y[idx], i_x[idx]] = image[v, u]
        seg_grid[i_y[idx], i_x[idx]] = mask[v, u]
        cam_grid[i_y[idx], i_x[idx]] = cam_points_valid[idx]

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
    lens_yaml = load_lens_params_yaml("test/lens.yaml")
    lens = load_lens_params_torch("test/lens.yaml")
    create_mesh(image, mask, lens)
