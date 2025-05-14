import numpy as np
import yaml
import cv2

def load_lens_params(lens_file):
    with open(lens_file, 'r') as f:
        lens = yaml.safe_load(f)
    
    Hoc = np.array(lens['Hoc'])  # 4x4
    centre = np.array(lens['centre'])  # 2D
    focal_length = lens['focal_length']
    k = lens['k']
    projection = lens['projection']

    return {
        'Hoc': Hoc,
        'centre': centre,
        'focal_length': focal_length,
        'k': k,
        'projection': projection
    }

def world_grid(x_range, y_range, spacing=0.05):
    """
    Generate a grid of points (X, Y, 0) on the ground plane.
    """
    xs = np.arange(x_range[0], x_range[1], spacing)
    ys = np.arange(y_range[0], y_range[1], spacing)
    grid = np.array([[x, y, 0] for x in xs for y in ys])
    return grid  # shape (N, 3)

def valid_pixels(pixels, image_shape):
    h, w = image_shape[:2]
    mask = (
        (pixels[:, 0] >= 0) & (pixels[:, 0] < w) &
        (pixels[:, 1] >= 0) & (pixels[:, 1] < h)
    )
    return mask
def project_to_image(points_3d, lens_params):
    """
    Project 3D world points to 2D image pixels using fisheye EQUIDISTANT model.
    """
    Hoc = lens_params['Hoc']
    focal = lens_params['focal_length']
    centre = lens_params['centre']
    k = lens_params['k']
    projection = lens_params['projection']
    
    # World to camera transformation in your coordinate system
    R = Hoc[:3, :3]
    t = np.array([0.0, 0.0, Hoc[2, 3]])  # Only Z translation
    # R = np.eye(3)  # Identity rotation

    cam_points = (R @ points_3d.T + t.reshape(3, 1)).T  # shape (N, 3)

    # Convert to OpenCV's coordinate system
    R_robot_to_camera = np.array([
        [0, -1, 0],  # X_cam = -Y_robot
        [0, 0, -1],  # Y_cam = -Z_robot
        [1, 0, 0],   # Z_cam =  X_robot
    ])
    cam_points = (R_robot_to_camera @ cam_points.T).T  # shape (N, 3)

    # Skip points behind the camera (z <= 0)
    in_front = cam_points[:, 2] > 0
    cam_points = cam_points[in_front]
    x, y, z = cam_points[:, 0], cam_points[:, 1], cam_points[:, 2]
    theta = np.arctan2(np.sqrt(x**2 + y**2), z)

    # EQUIDISTANT fisheye model
    r = focal * theta * (1 + k[0]*theta**2 + k[1]*theta**4)

    phi = np.arctan2(y, x)
    u = r * np.cos(phi) + centre[0]
    v = r * np.sin(phi) + centre[1]

    pixels = np.stack([u, v], axis=-1)
    return pixels, cam_points

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_3d_points(points_3d):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the 3D points
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='b', marker='o')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.show()


def create_mesh(image_path, lens_file):
    image = cv2.imread(image_path)
    lens_params = load_lens_params(lens_file)

    # Image dimensions
    height, width = image.shape[:2]

    # Convert the normalized center coordinates to pixel coordinates
    centre_normalized = lens_params['centre']
    centre_x = centre_normalized[0] + (width / 2)
    centre_y = centre_normalized[1] + (height / 2)

    # Update lens_params['centre'] to be in pixel coordinates
    lens_params['centre'] = np.array([centre_x, centre_y])
    
    print(f"Image shape: {image.shape}")
    print(f"Lens centre (normalized): {centre_normalized}")
    print(f"Lens centre (pixel): {lens_params['centre']}")

    # Draw a larger red circle at the optical center (converted to pixel space)
    cv2.circle(image, tuple(lens_params['centre'].astype(int)), 20, (0, 0, 255), 2)

    # Create a grid of points on the ground in the world coordinate system
    x_range = [0, 6]
    y_range = [-3, 3]
    spacing = 0.2
    xs = np.arange(x_range[0], x_range[1], spacing)
    ys = np.arange(y_range[0], y_range[1], spacing)
    grid = np.array([[x, y, 0] for y in ys for x in xs])

    # Visualize the 3D grid points (before projection)
    visualize_3d_points(grid)

    pixels, cam_points = project_to_image(grid, lens_params)

    # Keep only points in front of camera
    in_front = cam_points[:, 2] > 0
    pixels = pixels[in_front]

    # Keep only pixels inside image bounds
    valid = valid_pixels(pixels, image.shape)
    pixels = pixels[valid]

    # Draw dots on image
    for u, v in pixels.astype(int):
        cv2.circle(image, (u, v), 1, (0, 255, 0), -1)

    # Display image with projected points
    cv2.imshow("Projected Points", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    create_mesh("test/image.jpg", "test/lens.yaml")
    