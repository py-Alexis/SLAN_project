"""
map_utils.py — Map loading and obstacle utilities for path planning.

Adapted for ROS 2 OccupancyGrid input from SLAM Toolbox.
Also retains the original image-based utilities for standalone testing.

OccupancyGrid convention (from SLAM Toolbox / Nav2):
    -1  = unknown
     0  = definitely free
    100 = definitely occupied
    Values in between represent probability of occupancy.

Internal convention:
    occupancy[y, x] = True  → free space
    occupancy[y, x] = False → obstacle
"""

import numpy as np
from scipy.ndimage import distance_transform_edt, binary_erosion


# ---------------------------------------------------------------------------
# ROS 2 OccupancyGrid conversion
# ---------------------------------------------------------------------------

def occupancy_grid_to_array(
    data: list[int],
    width: int,
    height: int,
    resolution: float,
    origin_x: float,
    origin_y: float,
    origin_yaw: float = 0.0,
    unknown_as_free: bool = True,
    occupied_threshold: int = 50,
) -> tuple[np.ndarray, float, tuple[float, float, float]]:
    """
    Convert a ROS 2 OccupancyGrid to a boolean occupancy array.

    Args:
        data: Flat list of occupancy values from OccupancyGrid.data.
        width: Grid width in cells (OccupancyGrid.info.width).
        height: Grid height in cells (OccupancyGrid.info.height).
        resolution: Meters per cell (OccupancyGrid.info.resolution).
        origin_x: X position of cell (0,0) in the map frame (meters).
        origin_y: Y position of cell (0,0) in the map frame (meters).
        origin_yaw: Orientation of the grid in the map frame (radians).
        unknown_as_free: If True, unknown cells (-1) are treated as free.
            This enables planning through unexplored space toward unmapped
            goals. As SLAM discovers obstacles, replanning will handle them.
        occupied_threshold: Cells with value >= this are obstacles (default 50).

    Returns:
        (occupancy, resolution, origin) where:
            occupancy: 2D bool array, shape (height, width). True = free.
            resolution: meters per cell (pass-through).
            origin: (origin_x, origin_y, origin_yaw) tuple.
    """
    grid = np.array(data, dtype=np.int8).reshape((height, width))

    # Build boolean occupancy: True = free, False = obstacle
    if unknown_as_free:
        # Free: value == 0 (known free) OR value == -1 (unknown)
        # Obstacle: value >= occupied_threshold (known occupied)
        occupancy = grid < occupied_threshold  # -1, 0..49 → True; 50..100 → False
    else:
        # Conservative: only known-free cells are traversable
        # Free: 0 <= value < occupied_threshold
        # Obstacle: value == -1 (unknown) OR value >= occupied_threshold
        occupancy = (grid >= 0) & (grid < occupied_threshold)

    return occupancy, resolution, (origin_x, origin_y, origin_yaw)


def world_to_grid(
    x_m: float,
    y_m: float,
    resolution: float,
    origin: tuple[float, float, float],
) -> tuple[int, int]:
    """
    Convert world coordinates (meters) to grid cell indices.

    Args:
        x_m, y_m: Position in the map frame (meters).
        resolution: Grid resolution (meters/cell).
        origin: (origin_x, origin_y, origin_yaw) of the grid.

    Returns:
        (gx, gy): Grid cell indices. Note: occupancy is indexed as [gy, gx].
    """
    ox, oy, _ = origin
    gx = int(round((x_m - ox) / resolution))
    gy = int(round((y_m - oy) / resolution))
    return gx, gy


def grid_to_world(
    gx: int,
    gy: int,
    resolution: float,
    origin: tuple[float, float, float],
) -> tuple[float, float]:
    """
    Convert grid cell indices to world coordinates (meters).

    Args:
        gx, gy: Grid cell indices.
        resolution: Grid resolution (meters/cell).
        origin: (origin_x, origin_y, origin_yaw) of the grid.

    Returns:
        (x_m, y_m): Position in the map frame (meters).
    """
    ox, oy, _ = origin
    x_m = gx * resolution + ox
    y_m = gy * resolution + oy
    return x_m, y_m


def path_grid_to_world(
    path: list[tuple[int, int]],
    resolution: float,
    origin: tuple[float, float, float],
) -> list[tuple[float, float]]:
    """
    Convert an entire grid-coordinate path to world coordinates.

    Args:
        path: List of (gx, gy) grid cell tuples.
        resolution: Grid resolution (meters/cell).
        origin: (origin_x, origin_y, origin_yaw).

    Returns:
        List of (x_m, y_m) world coordinate tuples.
    """
    return [grid_to_world(gx, gy, resolution, origin) for gx, gy in path]


# ---------------------------------------------------------------------------
# Image-based map loading (kept for standalone testing)
# ---------------------------------------------------------------------------

def load_map(image_path: str) -> np.ndarray:
    """
    Load a black-and-white image as a binary occupancy grid.

    Args:
        image_path: Path to the image file (PNG, JPG, etc.)

    Returns:
        occupancy: 2D boolean numpy array where True = free, False = obstacle.
                   Shape is (height, width), indexed as occupancy[y, x].
    """
    import cv2
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    return img > 128


# ---------------------------------------------------------------------------
# Obstacle manipulation
# ---------------------------------------------------------------------------

def dilate_obstacles(occupancy: np.ndarray, radius: int = 3) -> np.ndarray:
    """
    Grow obstacle regions by `radius` pixels to create a safety margin.

    Equivalent to eroding the free-space mask. The robot should not
    get closer than `radius` cells to any wall.

    Args:
        occupancy: Boolean occupancy grid (True = free).
        radius: Number of cells to dilate obstacles by.

    Returns:
        New occupancy grid with dilated obstacles.
    """
    if radius <= 0:
        return occupancy.copy()

    # Create a circular structuring element
    struct = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=bool)
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    struct = (x ** 2 + y ** 2) <= radius ** 2

    # Eroding free space = dilating obstacles
    return binary_erosion(occupancy, structure=struct).astype(bool)


# ---------------------------------------------------------------------------
# Distance transform and gradient
# ---------------------------------------------------------------------------

def compute_distance_transform(occupancy: np.ndarray) -> np.ndarray:
    """
    Compute the Euclidean distance transform of the occupancy grid.

    For every free cell, returns the distance to the nearest obstacle cell.
    Obstacle cells have distance 0.

    Args:
        occupancy: Boolean occupancy grid (True = free).

    Returns:
        dist_transform: Float array, same shape as occupancy.
    """
    return distance_transform_edt(occupancy.astype(np.float64))


def compute_distance_gradient(
    dist_transform: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the gradient of the distance transform.

    The gradient points away from the nearest obstacle — used as the
    repulsive force direction in the Elastic Band algorithm.

    Args:
        dist_transform: The Euclidean distance transform array.

    Returns:
        (grad_x, grad_y): Gradient components.
    """
    grad_y, grad_x = np.gradient(dist_transform)
    return grad_x, grad_y


def is_free(occupancy: np.ndarray, x: float, y: float) -> bool:
    """
    Check if a position is free (not obstacle, within bounds).

    Args:
        occupancy: Boolean occupancy grid.
        x, y: Position in grid coordinates.

    Returns:
        True if free, False if obstacle or out of bounds.
    """
    ix, iy = int(round(x)), int(round(y))
    h, w = occupancy.shape
    if ix < 0 or ix >= w or iy < 0 or iy >= h:
        return False
    return bool(occupancy[iy, ix])
