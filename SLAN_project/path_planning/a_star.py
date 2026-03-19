"""
a_star.py — A* pathfinding on a 2D occupancy grid.

Finds the shortest path between two points on a binary grid
using the A* algorithm with Euclidean distance heuristic.

Key concepts:
- Open set: priority queue (min-heap) of nodes to explore, ordered by f = g + h
- Closed set: nodes already expanded (we won't revisit them)
- g(n): actual cost from start to node n
- h(n): heuristic estimate from node n to goal (Euclidean distance)
- f(n) = g(n) + h(n): estimated total cost through node n

The heuristic is admissible (never overestimates), so A* is guaranteed
to find the optimal path.
"""

import heapq
import math
import numpy as np


def heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
    """
    Euclidean distance heuristic.

    This is admissible (never overestimates the true distance)
    for a grid where we can move in 4 directions with cost 1.

    Args:
        a: (x, y) current position
        b: (x, y) goal position

    Returns:
        Euclidean distance between a and b.
    """
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


# 8-connected grid neighbors: (dx, dy, cost)
# Cardinal moves (cost = 1), diagonal moves (cost = sqrt(2))
NEIGHBORS_8 = [
    (1, 0, 1.0),    # right
    (-1, 0, 1.0),   # left
    (0, 1, 1.0),    # down
    (0, -1, 1.0),   # up
    (1, 1, 1.414),   # down-right
    (-1, 1, 1.414),  # down-left
    (1, -1, 1.414),  # up-right
    (-1, -1, 1.414), # up-left
]


def astar(
    occupancy: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    downscale: float = 1.0,
) -> list[tuple[int, int]] | None:
    """
    Run A* on an occupancy grid to find a path from start to goal.

    Args:
        occupancy: 2D boolean array (True = free, False = obstacle).
                   Indexed as occupancy[y, x].
        start: (x, y) start position in grid coordinates.
        goal: (x, y) goal position in grid coordinates.
        downscale: If < 1.0, run A* on a downscaled grid for speed,
                   then scale the path back up. E.g. 0.5 = half resolution.

    Returns:
        List of (x, y) tuples from start to goal, or None if no path found.
    """
    use_downscale = downscale < 1.0

    if use_downscale:
        # Downscale the occupancy grid for faster A*
        h_orig, w_orig = occupancy.shape
        h_new = max(1, int(h_orig * downscale))
        w_new = max(1, int(w_orig * downscale))

        # Use cv2 for resizing (INTER_AREA preserves obstacle connectivity)
        import cv2
        occ_small = cv2.resize(
            occupancy.astype(np.uint8) * 255,
            (w_new, h_new),
            interpolation=cv2.INTER_AREA,
        )
        occupancy_used = occ_small > 128
        # Scale start and goal to the smaller grid
        start_used = (int(round(start[0] * downscale)), int(round(start[1] * downscale)))
        goal_used = (int(round(goal[0] * downscale)), int(round(goal[1] * downscale)))
    else:
        occupancy_used = occupancy
        start_used = start
        goal_used = goal

    h, w = occupancy_used.shape

    # Validate start and goal
    sx, sy = start_used
    gx, gy = goal_used
    if not (0 <= sx < w and 0 <= sy < h and occupancy_used[sy, sx]):
        return None
    if not (0 <= gx < w and 0 <= gy < h and occupancy_used[gy, gx]):
        return None

    # --- A* algorithm ---

    # Open set: min-heap of (f, counter, (x, y))
    counter = 0
    open_set: list[tuple[float, int, tuple[int, int]]] = []
    heapq.heappush(open_set, (heuristic(start_used, goal_used), counter, start_used))
    counter += 1

    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_score: dict[tuple[int, int], float] = {start_used: 0.0}
    closed_set: set[tuple[int, int]] = set()

    while open_set:
        f, _, current = heapq.heappop(open_set)

        if current in closed_set:
            continue

        if current == goal_used:
            path = _reconstruct_path(came_from, current)
            if use_downscale:
                inv = 1.0 / downscale
                path = [(round(x * inv), round(y * inv)) for x, y in path]
            return path

        closed_set.add(current)

        cx, cy = current
        current_g = g_score[current]

        for dx, dy, cost in NEIGHBORS_8:
            nx, ny = cx + dx, cy + dy

            if nx < 0 or nx >= w or ny < 0 or ny >= h:
                continue

            if not occupancy_used[ny, nx]:
                continue

            # For diagonal moves, check that both cardinal neighbors are free
            if dx != 0 and dy != 0:
                if not occupancy_used[cy, cx + dx] or not occupancy_used[cy + dy, cx]:
                    continue

            neighbor = (nx, ny)

            if neighbor in closed_set:
                continue

            tentative_g = current_g + cost

            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal_used)
                heapq.heappush(open_set, (f_score, counter, neighbor))
                counter += 1

    return None


def _reconstruct_path(
    came_from: dict[tuple[int, int], tuple[int, int]],
    current: tuple[int, int],
) -> list[tuple[int, int]]:
    """
    Reconstruct the path from start to current by following parent pointers.
    """
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path
