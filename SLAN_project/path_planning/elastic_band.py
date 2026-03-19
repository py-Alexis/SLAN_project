"""
elastic_band.py — Elastic Band path optimizer.

Takes a raw A* path and iteratively optimizes it using two forces:
  1. Spring force: pulls each point toward the average of its neighbors
     → tends to shorten and straighten the path
  2. Repulsive force: pushes each point away from nearby obstacles
     → keeps the path safely away from walls

Key concepts:
  - Bubble: each path point has a "bubble radius" = distance to the nearest obstacle.
    The bubble tells us how much room we have to move the point around.
  - Distance transform: precomputed field where each pixel stores its distance
    to the nearest obstacle. Used for both bubble radius and repulsive force direction.
  - Resampling: ensures points stay evenly spaced along the path so forces
    are applied uniformly.
"""

import numpy as np


class ElasticBand:
    """
    Elastic Band path optimizer.

    Usage:
        eb = ElasticBand(path, dist_transform, occupancy)
        eb.optimize()
        path = eb.get_optimized_path()
    """

    def __init__(
        self,
        path: list[tuple[int, int]],
        dist_transform: np.ndarray,
        occupancy: np.ndarray,
        # --- Tunable parameters ---
        min_bubble_radius: float = 3.0,
        max_bubble_radius: float = 20.0,
        spring_weight: float = 0.8,
        repulsive_weight: float = 0.6,
        influence_radius: float = 15.0,
        alpha: float = 0.5,
        min_spacing: float = 3.0,
        max_spacing: float = 6.0,
        max_iterations: int = 80,
        convergence_threshold: float = 0.05,
    ):
        """
        Args:
            path: List of (x, y) points from A* (start -> goal).
            dist_transform: Distance transform array (from map_utils).
            occupancy: Boolean occupancy grid (True = free).
            min_bubble_radius: Minimum bubble radius (clamp).
            max_bubble_radius: Maximum bubble radius (clamp).
            spring_weight: Base weight for the spring (internal) force.
            repulsive_weight: Base weight for the repulsive (obstacle) force.
            influence_radius: Distance within which obstacles exert repulsive force.
            alpha: Step size for applying force displacements.
            min_spacing: Minimum distance between consecutive path points.
            max_spacing: Maximum distance between consecutive path points.
            max_iterations: Maximum optimization iterations.
            convergence_threshold: Stop when total displacement falls below this.
        """
        self.path = np.array(path, dtype=np.float64)
        self.initial_path = self.path.copy()
        self.dist_transform = dist_transform
        self.occupancy = occupancy

        # Precompute the gradient of the distance transform.
        grad_y, grad_x = np.gradient(dist_transform)
        self.grad_x = grad_x
        self.grad_y = grad_y

        # Parameters
        self.min_bubble = min_bubble_radius
        self.max_bubble = max_bubble_radius
        self.spring_weight = spring_weight
        self.repulsive_weight = repulsive_weight
        self.influence_radius = influence_radius
        self.alpha = alpha
        self.min_spacing = min_spacing
        self.max_spacing = max_spacing
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold

        # State
        self.bubbles = np.zeros(len(self.path))
        self.iteration = 0

    def _compute_bubbles(self):
        """Set each point's bubble radius = distance to closest obstacle, clamped."""
        h, w = self.dist_transform.shape
        ix = np.round(self.path[:, 0]).astype(int)
        iy = np.round(self.path[:, 1]).astype(int)
        valid = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
        dists = np.full(len(self.path), 0.0)
        dists[valid] = self.dist_transform[iy[valid], ix[valid]]
        self.bubbles = np.clip(dists, self.min_bubble, self.max_bubble)

    def _resample_path(self):
        """Enforce uniform spacing between consecutive path points."""
        if len(self.path) < 2:
            return

        new_path = [self.path[0]]

        for i in range(1, len(self.path)):
            prev = new_path[-1]
            curr = self.path[i]
            dist = np.linalg.norm(curr - prev)

            if dist < self.min_spacing and i < len(self.path) - 1:
                continue
            elif dist > self.max_spacing:
                n_segments = int(np.ceil(dist / self.max_spacing))
                for j in range(1, n_segments):
                    t = j / n_segments
                    interp = prev + t * (curr - prev)
                    new_path.append(interp)
                new_path.append(curr)
            else:
                new_path.append(curr)

        self.path = np.array(new_path, dtype=np.float64)
        self.bubbles = np.zeros(len(self.path))

    def _compute_all_forces(self) -> np.ndarray:
        """Compute spring + repulsive forces for ALL interior points at once."""
        n = len(self.path)
        forces = np.zeros_like(self.path)

        if n < 3:
            return forces

        interior = slice(1, n - 1)

        # ---- Spring force (vectorized) ----
        avg_neighbors = 0.5 * (self.path[:-2] + self.path[2:])
        displacement_to_avg = avg_neighbors - self.path[interior]

        r = self.bubbles[interior]
        normalized = (r - self.min_bubble) / max(self.max_bubble - self.min_bubble, 1e-6)
        sigmoid = 1.0 / (1.0 + np.exp(-6.0 * (2.0 * normalized - 1.0)))
        w = self.spring_weight * (0.3 + 0.7 * sigmoid)
        spring_forces = w[:, np.newaxis] * displacement_to_avg

        # ---- Repulsive force (vectorized) ----
        h, map_w = self.dist_transform.shape
        ix = np.round(self.path[interior, 0]).astype(int)
        iy = np.round(self.path[interior, 1]).astype(int)

        valid = (ix >= 0) & (ix < map_w) & (iy >= 0) & (iy < h)

        dists = np.full(n - 2, self.influence_radius)
        dists[valid] = self.dist_transform[iy[valid], ix[valid]]

        in_range = dists < self.influence_radius
        strength = np.zeros(n - 2)
        strength[in_range] = ((1.0 - dists[in_range] / self.influence_radius) ** 2)

        gx = np.zeros(n - 2)
        gy = np.zeros(n - 2)
        mask = valid & in_range
        gx[mask] = self.grad_x[iy[mask], ix[mask]]
        gy[mask] = self.grad_y[iy[mask], ix[mask]]

        rep_forces = self.repulsive_weight * strength[:, np.newaxis] * np.column_stack([gx, gy])

        # Remove tangential component of repulsive force
        tangents = self.path[2:] - self.path[:-2]
        tangent_norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        tangent_norms = np.maximum(tangent_norms, 1e-6)
        tangents = tangents / tangent_norms
        proj = np.sum(rep_forces * tangents, axis=1, keepdims=True)
        rep_forces = rep_forces - proj * tangents

        forces[interior] = spring_forces + rep_forces
        return forces

    def optimize(self):
        """Run the full elastic band optimization loop (vectorized)."""
        for iteration in range(self.max_iterations):
            self.iteration = iteration

            self._resample_path()
            self._compute_bubbles()

            n = len(self.path)
            if n < 3:
                break

            forces = self._compute_all_forces()

            displacements = self.alpha * forces
            disp_mags = np.linalg.norm(displacements, axis=1)

            max_disps = self.bubbles * 0.5
            too_big = disp_mags > max_disps
            scale = np.ones(n)
            scale[too_big] = max_disps[too_big] / np.maximum(disp_mags[too_big], 1e-9)
            displacements *= scale[:, np.newaxis]
            disp_mags = np.minimum(disp_mags, max_disps)

            new_positions = self.path + displacements
            h, w = self.occupancy.shape
            nx = np.round(new_positions[:, 0]).astype(int)
            ny = np.round(new_positions[:, 1]).astype(int)
            in_bounds = (nx >= 0) & (nx < w) & (ny >= 0) & (ny < h)

            is_free = np.zeros(n, dtype=bool)
            is_free[in_bounds] = self.occupancy[ny[in_bounds], nx[in_bounds]]

            is_free[0] = False
            is_free[-1] = False

            self.path[is_free] = new_positions[is_free]
            total_displacement = float(disp_mags[is_free].sum())

            if total_displacement < self.convergence_threshold:
                break

    def get_optimized_path(self) -> list[tuple[float, float]]:
        """Return the elastic-band-optimized path."""
        return [(float(p[0]), float(p[1])) for p in self.path]

    def get_bubbles(self) -> tuple[np.ndarray, np.ndarray]:
        """Return path points and their bubble radii."""
        return self.path.copy(), self.bubbles.copy()

    def get_initial_path(self) -> list[tuple[float, float]]:
        """Return the original A* path."""
        return [(float(p[0]), float(p[1])) for p in self.initial_path]

    def is_path_valid(self) -> bool:
        """
        Check if the current optimized path is collision-free.

        Returns:
            True if all path points are in free space, False otherwise.
        """
        h, w = self.occupancy.shape
        ix = np.round(self.path[:, 0]).astype(int)
        iy = np.round(self.path[:, 1]).astype(int)
        in_bounds = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)

        if not np.all(in_bounds):
            return False

        return bool(np.all(self.occupancy[iy[in_bounds], ix[in_bounds]]))
