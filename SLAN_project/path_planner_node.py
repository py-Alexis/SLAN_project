"""
path_planner_node.py — Dynamic A* + Elastic Band path planner for ROS 2.

Subscribes to SLAM Toolbox's /map (OccupancyGrid) and replans paths
dynamically as the map evolves. Designed for navigating to goals in
partially or fully unmapped environments.

Replanning strategy:
    - Event-driven: replans when map changes near the path, when the path
      becomes blocked, or when the robot deviates too far from the path.
    - EB-first: tries Elastic Band re-optimization first (fast, smooth).
      Falls back to full A*+EB only when the path is blocked.
    - Path blending: on replan, the old path near the robot is kept and
      smoothly blended into the new path to avoid discontinuities.

Unknown cells are treated as FREE (optimistic) by default, so the planner
can route through unexplored space toward unmapped goals. As SLAM discovers
new obstacles, replanning handles them automatically.

Topics (relative to node namespace unless noted):
    Subscriptions:
        /map (absolute)           - nav_msgs/OccupancyGrid from SLAM Toolbox
        odom                      - nav_msgs/Odometry (robot pose)
        nav/goal_pose             - geometry_msgs/PoseStamped (target)
        nav/stop                  - std_msgs/Bool (cancel)
    Publications:
        nav/planned_path          - nav_msgs/Path (for pure pursuit)
        nav/path_markers          - visualization_msgs/MarkerArray (RViz debug)
"""

import math
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration
import tf_transformations
from tf2_ros import Buffer, TransformException, TransformListener

from SLAN_project.path_planning.map_utils import (
    occupancy_grid_to_array,
    dilate_obstacles,
    compute_distance_transform,
    world_to_grid,
    grid_to_world,
    path_grid_to_world,
)
from SLAN_project.path_planning.a_star import astar
from SLAN_project.path_planning.elastic_band import ElasticBand


class PlannerState:
    IDLE = "IDLE"
    WAITING_FOR_MAP = "WAITING_FOR_MAP"
    PLANNING = "PLANNING"
    MONITORING = "MONITORING"
    REPLANNING = "REPLANNING"


class PathPlannerNode(Node):
    def __init__(self):
        super().__init__("path_planner_node")

        # ── Parameters ───────────────────────────────────────────────
        self.declare_parameter("obstacle_dilation", 8)
        self.declare_parameter("astar_downscale", 0.4)
        self.declare_parameter("replan_distance_threshold", 0.15)
        self.declare_parameter("map_change_threshold", 0.05)
        self.declare_parameter("enable_replanning", True)
        self.declare_parameter("path_blend_distance", 0.3)
        self.declare_parameter("path_corridor_width", 0.2)  # meters around path to monitor
        self.declare_parameter("unknown_as_free", True)

        # Elastic Band parameters
        self.declare_parameter("eb_min_bubble_radius", 3.0)
        self.declare_parameter("eb_max_bubble_radius", 20.0)
        self.declare_parameter("eb_spring_weight", 0.8)
        self.declare_parameter("eb_repulsive_weight", 0.6)
        self.declare_parameter("eb_influence_radius", 15.0)
        self.declare_parameter("eb_alpha", 0.5)
        self.declare_parameter("eb_min_spacing", 3.0)
        self.declare_parameter("eb_max_spacing", 6.0)
        self.declare_parameter("eb_max_iterations", 80)
        self.declare_parameter("eb_convergence_threshold", 0.05)

        # ── State ────────────────────────────────────────────────────
        self.state = PlannerState.IDLE
        self.goal_world = None          # (x_m, y_m) in map frame
        self.goal_frame = "map"
        self.goal_reverse = False       # reverse mode flag
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.odom_frame = "odom"
        self.has_odom = False

        # Map data (protected by lock)
        self._map_lock = threading.Lock()
        self._occupancy = None          # bool array (True=free)
        self._occupancy_dilated = None
        self._dist_transform = None
        self._map_resolution = 0.05
        self._map_origin = (0.0, 0.0, 0.0)
        self._map_stamp = None
        self._map_frame = "map"
        self._prev_occupancy = None     # for change detection

        # Current path (protected by lock)
        self._path_lock = threading.Lock()
        self._current_path_world = None     # list of (x, y) in meters
        self._current_path_grid = None      # list of (gx, gy) grid cells
        self._current_eb = None             # ElasticBand object for re-optimization

        # Planning thread
        self._plan_event = threading.Event()
        self._plan_thread = threading.Thread(target=self._planning_loop, daemon=True)
        self._plan_request = None       # "full" or "eb_only"
        self._shutdown = False

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ── Publishers ───────────────────────────────────────────────
        self.path_pub = self.create_publisher(Path, "nav/planned_path", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "nav/path_markers", 10)

        # ── Subscribers ──────────────────────────────────────────────
        # /map is absolute (SLAM Toolbox publishes on /map)
        map_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(OccupancyGrid, "/map", self._map_callback, map_qos)
        self.create_subscription(Odometry, "odom", self._odom_callback, 10)
        self.create_subscription(PoseStamped, "nav/goal_pose", self._goal_callback, 10)
        self.create_subscription(Bool, "nav/stop", self._stop_callback, 10)

        # ── Monitoring timer (5 Hz) ─────────────────────────────────
        self.create_timer(0.2, self._monitor_callback)

        # Start planning thread
        self._plan_thread.start()

        self.get_logger().info("PathPlanner node started (unknown_as_free={})".format(
            self.get_parameter("unknown_as_free").value
        ))

    # ==================================================================
    # Callbacks
    # ==================================================================

    def _odom_callback(self, msg: Odometry):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        self.odom_frame = msg.header.frame_id or "odom"
        q = msg.pose.pose.orientation
        _, _, self.robot_yaw = tf_transformations.euler_from_quaternion(
            [q.x, q.y, q.z, q.w]
        )
        self.has_odom = True

    def _goal_callback(self, msg: PoseStamped):
        self.goal_world = (msg.pose.position.x, msg.pose.position.y)
        self.goal_frame = msg.header.frame_id or "map"
        self.goal_reverse = msg.pose.position.z > 0.5

        self.get_logger().info(
            f"New goal: ({self.goal_world[0]:.2f}, {self.goal_world[1]:.2f})"
            f"{' [REVERSE]' if self.goal_reverse else ''}"
        )

        if self._occupancy is not None:
            self.state = PlannerState.PLANNING
            self._request_plan("full")
        else:
            self.state = PlannerState.WAITING_FOR_MAP
            self.get_logger().info("Waiting for /map before planning...")

    def _stop_callback(self, msg: Bool):
        if msg.data:
            self.get_logger().info("Stop received — cancelling planning")
            self.state = PlannerState.IDLE
            self.goal_world = None
            with self._path_lock:
                self._current_path_world = None
                self._current_path_grid = None
                self._current_eb = None
            # Publish empty path to signal stop
            self._publish_empty_path()

    def _map_callback(self, msg: OccupancyGrid):
        """Process new map from SLAM Toolbox."""
        t0 = time.time()

        resolution = msg.info.resolution
        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y
        q = msg.info.origin.orientation
        _, _, origin_yaw = tf_transformations.euler_from_quaternion(
            [q.x, q.y, q.z, q.w]
        )

        unknown_as_free = self.get_parameter("unknown_as_free").value
        dilation = self.get_parameter("obstacle_dilation").value

        occupancy, res, origin = occupancy_grid_to_array(
            list(msg.data),
            msg.info.width,
            msg.info.height,
            resolution,
            origin_x,
            origin_y,
            origin_yaw,
            unknown_as_free=unknown_as_free,
        )

        occupancy_dilated = dilate_obstacles(occupancy, radius=dilation)
        dist_transform = compute_distance_transform(occupancy_dilated)

        with self._map_lock:
            self._prev_occupancy = self._occupancy_dilated
            self._occupancy = occupancy
            self._occupancy_dilated = occupancy_dilated
            self._dist_transform = dist_transform
            self._map_resolution = res
            self._map_origin = origin
            self._map_stamp = msg.header.stamp
            self._map_frame = msg.header.frame_id or "map"

        elapsed = time.time() - t0
        self.get_logger().debug(
            f"Map updated: {msg.info.width}x{msg.info.height}, "
            f"processed in {elapsed:.3f}s"
        )

        # If we were waiting for a map to plan, trigger planning now
        if self.state == PlannerState.WAITING_FOR_MAP and self.goal_world is not None:
            self.state = PlannerState.PLANNING
            self._request_plan("full")

    # ==================================================================
    # Monitoring (runs at 5 Hz)
    # ==================================================================

    def _monitor_callback(self):
        """Check for replan triggers while a path is active."""
        if self.state != PlannerState.MONITORING:
            return
        if not self.get_parameter("enable_replanning").value:
            return
        if not self.has_odom:
            return

        with self._path_lock:
            if self._current_path_world is None:
                return
            path_world = self._current_path_world

        robot_pose = self._get_robot_pose_in_map_frame()
        if robot_pose is None:
            return
        robot_x, robot_y, _ = robot_pose

        # ── Trigger 1: Robot deviated too far from path ──────────────
        replan_dist = self.get_parameter("replan_distance_threshold").value
        min_dist = self._distance_to_path(robot_x, robot_y, path_world)
        if min_dist > replan_dist:
            self.get_logger().info(
                f"Robot deviated {min_dist:.2f}m from path (threshold {replan_dist:.2f}m) — full replan"
            )
            self.state = PlannerState.REPLANNING
            self._request_plan("full")
            return

        # ── Trigger 2: Path blocked (cells on path now occupied) ─────
        with self._map_lock:
            if self._occupancy_dilated is None:
                return
            occ = self._occupancy_dilated
            res = self._map_resolution
            origin = self._map_origin

        blocked = self._check_path_blocked(path_world, occ, res, origin)
        if blocked:
            self.get_logger().info("Path blocked by new obstacle — full replan")
            self.state = PlannerState.REPLANNING
            self._request_plan("full")
            return

        # ── Trigger 3: Significant map change near path corridor ─────
        with self._map_lock:
            prev_occ = self._prev_occupancy
            curr_occ = self._occupancy_dilated

        if prev_occ is not None and prev_occ.shape == curr_occ.shape:
            change_frac = self._check_map_change_near_path(
                path_world, prev_occ, curr_occ, res, origin
            )
            change_thresh = self.get_parameter("map_change_threshold").value
            if change_frac > change_thresh:
                self.get_logger().info(
                    f"Map changed {change_frac:.1%} near path (threshold {change_thresh:.1%}) — EB replan"
                )
                self.state = PlannerState.REPLANNING
                self._request_plan("eb_only")

    # ==================================================================
    # Planning thread
    # ==================================================================

    def _request_plan(self, mode: str):
        """Signal the planning thread. mode = 'full' or 'eb_only'."""
        self._plan_request = mode
        self._plan_event.set()

    def _planning_loop(self):
        """Background thread that runs planning when signaled."""
        while not self._shutdown:
            self._plan_event.wait(timeout=1.0)
            self._plan_event.clear()

            if self._shutdown:
                break

            mode = self._plan_request
            if mode is None:
                continue

            self._plan_request = None

            if self.goal_world is None:
                continue

            try:
                if mode == "eb_only":
                    self._run_eb_only_replan()
                else:
                    self._run_full_plan()
            except Exception as e:
                self.get_logger().error(f"Planning failed: {e}")
                import traceback
                self.get_logger().error(traceback.format_exc())

    def _run_full_plan(self):
        """Full A* + Elastic Band planning from current robot position to goal."""
        t0 = time.time()

        with self._map_lock:
            if self._occupancy_dilated is None:
                self.get_logger().warn("No map available for planning")
                return
            occ_dilated = self._occupancy_dilated.copy()
            dist_transform = self._dist_transform.copy()
            occ_raw = self._occupancy.copy()
            resolution = self._map_resolution
            origin = self._map_origin

        # Convert start (robot position) and goal to grid coordinates in map frame
        robot_pose = self._get_robot_pose_in_map_frame()
        if robot_pose is None:
            self.get_logger().warn("No odom — cannot plan")
            return
        robot_x, robot_y, _ = robot_pose
        start_grid = world_to_grid(robot_x, robot_y, resolution, origin)

        goal_world = self._get_goal_in_map_frame()
        if goal_world is None:
            self.get_logger().warn("Goal is not available in map frame yet")
            return
        goal_grid = world_to_grid(goal_world[0], goal_world[1], resolution, origin)

        h, w = occ_dilated.shape
        sx, sy = start_grid
        gx, gy = goal_grid

        # Validate start — if robot is in dilated obstacle zone, use undilated map check
        if not (0 <= sx < w and 0 <= sy < h):
            self.get_logger().warn(f"Start {start_grid} is outside map bounds")
            return
        if not occ_dilated[sy, sx]:
            # Robot is in dilated zone — try to find nearest free cell
            self.get_logger().warn("Robot is in dilated obstacle zone, searching for nearest free cell...")
            start_grid = self._find_nearest_free(sx, sy, occ_dilated)
            if start_grid is None:
                self.get_logger().error("Cannot find free cell near robot!")
                return

        # Validate goal — if in unknown space (treated as free), that's fine
        if not (0 <= gx < w and 0 <= gy < h):
            self.get_logger().warn(f"Goal {goal_grid} is outside current map bounds")
            # Extend goal to map edge as a best-effort waypoint
            gx = max(0, min(gx, w - 1))
            gy = max(0, min(gy, h - 1))
            goal_grid = (gx, gy)
            self.get_logger().info(f"Clamped goal to map edge: {goal_grid}")

        if not occ_dilated[gy, gx]:
            # Goal is in an obstacle — try to find nearest free cell
            self.get_logger().warn("Goal is in obstacle/dilated zone, searching for nearest free cell...")
            goal_grid = self._find_nearest_free(gx, gy, occ_dilated)
            if goal_grid is None:
                self.get_logger().error("Cannot find free cell near goal!")
                return

        # Run A*
        downscale = self.get_parameter("astar_downscale").value
        self.get_logger().info(
            f"A* planning: {start_grid} → {goal_grid} "
            f"(map {w}x{h}, downscale={downscale})"
        )

        raw_path = astar(occ_dilated, start_grid, goal_grid, downscale=downscale)

        if raw_path is None:
            self.get_logger().warn("A* found no path!")
            # Stay in current state — will retry on next map update
            if self.state == PlannerState.REPLANNING:
                self.state = PlannerState.MONITORING
            else:
                self.state = PlannerState.WAITING_FOR_MAP
            return

        t_astar = time.time() - t0
        self.get_logger().info(f"A* found path with {len(raw_path)} points in {t_astar:.3f}s")

        # Run Elastic Band
        eb_params = self._get_eb_params()
        eb = ElasticBand(raw_path, dist_transform, occ_dilated, **eb_params)
        eb.optimize()

        opt_path_grid = eb.get_optimized_path()
        opt_path_grid_int = [(int(round(x)), int(round(y))) for x, y in opt_path_grid]

        # Convert to world coordinates
        opt_path_world = path_grid_to_world(opt_path_grid_int, resolution, origin)

        t_total = time.time() - t0
        self.get_logger().info(
            f"Full plan complete: {len(opt_path_world)} waypoints in {t_total:.3f}s"
        )

        # Blend with old path if this is a replan
        old_path = None
        with self._path_lock:
            if self.state == PlannerState.REPLANNING and self._current_path_world is not None:
                old_path = self._current_path_world

        if old_path is not None:
            opt_path_world = self._blend_paths(old_path, opt_path_world)

        # Store and publish
        with self._path_lock:
            self._current_path_world = opt_path_world
            self._current_path_grid = opt_path_grid_int
            self._current_eb = eb

        self._publish_path(opt_path_world)
        self._publish_markers(opt_path_world)
        self.state = PlannerState.MONITORING

    def _run_eb_only_replan(self):
        """Re-optimize the current path using only Elastic Band (fast replan)."""
        with self._path_lock:
            if self._current_path_grid is None or self._current_eb is None:
                # No existing path to re-optimize — fall back to full plan
                self.get_logger().info("No existing path for EB replan — falling back to full plan")
                self._run_full_plan()
                return
            old_path_grid = self._current_path_grid
            old_path_world = self._current_path_world

        with self._map_lock:
            if self._occupancy_dilated is None:
                return
            occ_dilated = self._occupancy_dilated.copy()
            dist_transform = self._dist_transform.copy()
            resolution = self._map_resolution
            origin = self._map_origin

        t0 = time.time()

        # Trim path: remove points behind the robot
        robot_pose = self._get_robot_pose_in_map_frame()
        if robot_pose is None:
            self.get_logger().warn("Cannot run EB-only replan without robot pose in map frame")
            return

        trimmed_path = self._trim_path_behind_robot(
            old_path_grid,
            resolution,
            origin,
            robot_pose[0],
            robot_pose[1],
        )
        if len(trimmed_path) < 2:
            self.get_logger().info("Path too short for EB replan — full replan")
            self._run_full_plan()
            return

        # Re-run Elastic Band with updated map data
        eb_params = self._get_eb_params()
        eb = ElasticBand(trimmed_path, dist_transform, occ_dilated, **eb_params)
        eb.optimize()

        # Check if the EB result is valid (no collisions)
        if not eb.is_path_valid():
            self.get_logger().info("EB replan produced invalid path — falling back to full A*+EB")
            self._run_full_plan()
            return

        opt_path_grid = eb.get_optimized_path()
        opt_path_grid_int = [(int(round(x)), int(round(y))) for x, y in opt_path_grid]
        opt_path_world = path_grid_to_world(opt_path_grid_int, resolution, origin)

        t_eb = time.time() - t0
        self.get_logger().info(f"EB replan: {len(opt_path_world)} waypoints in {t_eb:.3f}s")

        # Blend with old path
        if old_path_world is not None:
            opt_path_world = self._blend_paths(old_path_world, opt_path_world)

        with self._path_lock:
            self._current_path_world = opt_path_world
            self._current_path_grid = opt_path_grid_int
            self._current_eb = eb

        self._publish_path(opt_path_world)
        self._publish_markers(opt_path_world)
        self.state = PlannerState.MONITORING

    # ==================================================================
    # Path utilities
    # ==================================================================

    def _get_eb_params(self) -> dict:
        """Collect Elastic Band parameters from ROS parameters."""
        return {
            "min_bubble_radius": self.get_parameter("eb_min_bubble_radius").value,
            "max_bubble_radius": self.get_parameter("eb_max_bubble_radius").value,
            "spring_weight": self.get_parameter("eb_spring_weight").value,
            "repulsive_weight": self.get_parameter("eb_repulsive_weight").value,
            "influence_radius": self.get_parameter("eb_influence_radius").value,
            "alpha": self.get_parameter("eb_alpha").value,
            "min_spacing": self.get_parameter("eb_min_spacing").value,
            "max_spacing": self.get_parameter("eb_max_spacing").value,
            "max_iterations": self.get_parameter("eb_max_iterations").value,
            "convergence_threshold": self.get_parameter("eb_convergence_threshold").value,
        }

    def _find_nearest_free(
        self, gx: int, gy: int, occ: np.ndarray, max_radius: int = 50
    ) -> tuple[int, int] | None:
        """
        Find the nearest free cell to (gx, gy) via expanding ring search.

        Returns:
            (x, y) of nearest free cell, or None if nothing found within max_radius.
        """
        h, w = occ.shape
        for r in range(1, max_radius + 1):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if abs(dx) != r and abs(dy) != r:
                        continue  # only check perimeter
                    nx, ny = gx + dx, gy + dy
                    if 0 <= nx < w and 0 <= ny < h and occ[ny, nx]:
                        return (nx, ny)
        return None

    def _trim_path_behind_robot(
        self,
        path_grid: list[tuple[int, int]],
        resolution: float,
        origin: tuple[float, float, float],
        robot_x: float,
        robot_y: float,
    ) -> list[tuple[int, int]]:
        """
        Remove path points that the robot has already passed.

        Returns the remaining path starting from the closest point to the robot.
        """
        if len(path_grid) < 2:
            return path_grid

        robot_grid = world_to_grid(robot_x, robot_y, resolution, origin)
        rx, ry = robot_grid

        # Find closest point on path
        min_dist = float("inf")
        closest_idx = 0
        for i, (px, py) in enumerate(path_grid):
            d = (px - rx) ** 2 + (py - ry) ** 2
            if d < min_dist:
                min_dist = d
                closest_idx = i

        # Keep from closest point onward (include robot's approximate position)
        return [(rx, ry)] + path_grid[closest_idx:]

    def _blend_paths(
        self,
        old_path: list[tuple[float, float]],
        new_path: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        """
        Blend old path into new path near the robot for smooth transitions.

        Keeps the old path from the robot's position up to blend_distance ahead,
        then smoothly interpolates to the new path.
        """
        blend_dist = self.get_parameter("path_blend_distance").value
        if len(old_path) < 2 or len(new_path) < 2:
            return new_path

        robot_pose = self._get_robot_pose_in_map_frame()
        if robot_pose is None:
            return new_path
        rx, ry, _ = robot_pose

        # Find closest point on old path to robot
        old_closest_idx = 0
        min_d = float("inf")
        for i, (px, py) in enumerate(old_path):
            d = (px - rx) ** 2 + (py - ry) ** 2
            if d < min_d:
                min_d = d
                old_closest_idx = i

        # Walk along old path from closest point, accumulating distance until blend_dist
        blend_end_idx = old_closest_idx
        accum = 0.0
        for i in range(old_closest_idx, len(old_path) - 1):
            dx = old_path[i + 1][0] - old_path[i][0]
            dy = old_path[i + 1][1] - old_path[i][1]
            accum += math.sqrt(dx * dx + dy * dy)
            blend_end_idx = i + 1
            if accum >= blend_dist:
                break

        # The old segment to keep
        old_segment = old_path[old_closest_idx:blend_end_idx + 1]

        if len(old_segment) == 0:
            return new_path

        # Find best join point on new path near the blend endpoint, preferring heading continuity
        blend_end_x, blend_end_y = old_segment[-1]
        old_heading = self._path_heading(old_segment, len(old_segment) - 1)
        heading_weight = 0.35

        new_start_idx = 0
        best_score = float("inf")
        for i, (px, py) in enumerate(new_path):
            dist2 = (px - blend_end_x) ** 2 + (py - blend_end_y) ** 2
            new_heading = self._path_heading(new_path, i)
            dtheta = abs(self._normalize_angle(new_heading - old_heading))
            score = dist2 + heading_weight * (dtheta ** 2)
            if score < best_score:
                best_score = score
                new_start_idx = i

        # Build smooth transition with cubic Hermite interpolation
        blended = list(old_segment[:-1])

        if new_start_idx < len(new_path):
            end_old = old_segment[-1]
            start_new = new_path[new_start_idx]
            new_heading = self._path_heading(new_path, new_start_idx)

            connector = self._hermite_connector(
                end_old,
                start_new,
                old_heading,
                new_heading,
                n_points=6,
            )

            if len(connector) > 0:
                blended.extend(connector)

            blended.extend(new_path[new_start_idx:])
        else:
            blended.extend(new_path)

        return blended

    @staticmethod
    def _path_heading(path: list[tuple[float, float]], idx: int) -> float:
        """Estimate local heading of a polyline at index idx."""
        if len(path) < 2:
            return 0.0

        if idx <= 0:
            p0 = path[0]
            p1 = path[1]
        elif idx >= len(path) - 1:
            p0 = path[-2]
            p1 = path[-1]
        else:
            p0 = path[idx - 1]
            p1 = path[idx + 1]

        return math.atan2(p1[1] - p0[1], p1[0] - p0[0])

    @staticmethod
    def _hermite_connector(
        p0: tuple[float, float],
        p1: tuple[float, float],
        heading0: float,
        heading1: float,
        n_points: int = 6,
    ) -> list[tuple[float, float]]:
        """Generate interior points of a cubic Hermite connector between p0 and p1."""
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 1e-4 or n_points <= 0:
            return []

        tangent_scale = 0.45 * dist
        m0 = (math.cos(heading0) * tangent_scale, math.sin(heading0) * tangent_scale)
        m1 = (math.cos(heading1) * tangent_scale, math.sin(heading1) * tangent_scale)

        points = []
        for i in range(1, n_points + 1):
            t = i / (n_points + 1)
            t2 = t * t
            t3 = t2 * t

            h00 = 2.0 * t3 - 3.0 * t2 + 1.0
            h10 = t3 - 2.0 * t2 + t
            h01 = -2.0 * t3 + 3.0 * t2
            h11 = t3 - t2

            x = h00 * p0[0] + h10 * m0[0] + h01 * p1[0] + h11 * m1[0]
            y = h00 * p0[1] + h10 * m0[1] + h01 * p1[1] + h11 * m1[1]
            points.append((x, y))

        return points

    @staticmethod
    def _distance_to_path(
        x: float, y: float, path: list[tuple[float, float]]
    ) -> float:
        """Min distance from point (x,y) to the polyline path."""
        min_d = float("inf")
        for i in range(len(path) - 1):
            d = _point_to_segment_dist(
                x, y, path[i][0], path[i][1], path[i + 1][0], path[i + 1][1]
            )
            min_d = min(min_d, d)
        if len(path) == 1:
            dx = x - path[0][0]
            dy = y - path[0][1]
            min_d = math.sqrt(dx * dx + dy * dy)
        return min_d

    def _check_path_blocked(
        self,
        path_world: list[tuple[float, float]],
        occ: np.ndarray,
        resolution: float,
        origin: tuple[float, float, float],
    ) -> bool:
        """Check if any cell along the current path is now an obstacle."""
        h, w = occ.shape
        for px, py in path_world:
            gx, gy = world_to_grid(px, py, resolution, origin)
            if 0 <= gx < w and 0 <= gy < h:
                if not occ[gy, gx]:
                    return True
        return False

    def _check_map_change_near_path(
        self,
        path_world: list[tuple[float, float]],
        prev_occ: np.ndarray,
        curr_occ: np.ndarray,
        resolution: float,
        origin: tuple[float, float, float],
    ) -> float:
        """
        Compute the fraction of cells near the path that changed between map updates.

        Returns a float in [0, 1].
        """
        corridor_m = self.get_parameter("path_corridor_width").value
        corridor_cells = max(1, int(corridor_m / resolution))

        h, w = curr_occ.shape
        checked = 0
        changed = 0

        # Sample every few path points to avoid checking too many cells
        step = max(1, len(path_world) // 50)
        for idx in range(0, len(path_world), step):
            px, py = path_world[idx]
            gx, gy = world_to_grid(px, py, resolution, origin)

            for dx in range(-corridor_cells, corridor_cells + 1):
                for dy in range(-corridor_cells, corridor_cells + 1):
                    nx, ny = gx + dx, gy + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        checked += 1
                        if prev_occ[ny, nx] != curr_occ[ny, nx]:
                            changed += 1

        if checked == 0:
            return 0.0
        return changed / checked

    # ==================================================================
    # Publishing
    # ==================================================================

    def _publish_path(self, path_world: list[tuple[float, float]]):
        """Publish nav_msgs/Path for pure pursuit to follow."""
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._map_frame

        for x, y in path_world:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = 1.0 if self.goal_reverse else 0.0
            # Orientation: face next waypoint (or keep identity)
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)

        self.path_pub.publish(msg)

    def _publish_empty_path(self):
        """Publish an empty path to signal no active plan."""
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._map_frame
        self.path_pub.publish(msg)

    def _publish_markers(self, path_world: list[tuple[float, float]]):
        """Publish visualization markers for RViz."""
        ma = MarkerArray()

        # Path line strip
        line = Marker()
        line.header.stamp = self.get_clock().now().to_msg()
        line.header.frame_id = self._map_frame
        line.ns = "planned_path"
        line.id = 0
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.scale.x = 0.02  # line width
        line.color.r = 0.0
        line.color.g = 1.0
        line.color.b = 0.0
        line.color.a = 0.8
        line.lifetime = Duration(sec=0, nanosec=0)  # persistent

        from geometry_msgs.msg import Point
        for x, y in path_world:
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = 0.01
            line.points.append(p)

        ma.markers.append(line)

        # Goal marker
        if self.goal_world is not None:
            goal_marker = Marker()
            goal_marker.header = line.header
            goal_marker.ns = "planned_path"
            goal_marker.id = 1
            goal_marker.type = Marker.SPHERE
            goal_marker.action = Marker.ADD
            goal_marker.pose.position.x = float(self.goal_world[0])
            goal_marker.pose.position.y = float(self.goal_world[1])
            goal_marker.pose.position.z = 0.05
            goal_marker.pose.orientation.w = 1.0
            goal_marker.scale.x = 0.08
            goal_marker.scale.y = 0.08
            goal_marker.scale.z = 0.08
            goal_marker.color.r = 1.0
            goal_marker.color.g = 0.0
            goal_marker.color.b = 0.0
            goal_marker.color.a = 1.0
            goal_marker.lifetime = Duration(sec=0, nanosec=0)
            ma.markers.append(goal_marker)

        self.marker_pub.publish(ma)

    # ==================================================================
    # Cleanup
    # ==================================================================

    def destroy_node(self):
        self._shutdown = True
        self._plan_event.set()
        self._plan_thread.join(timeout=2.0)
        super().destroy_node()

    def _get_robot_pose_in_map_frame(self) -> tuple[float, float, float] | None:
        """
        Return the robot pose in the current map frame.

        The path is published in the map frame, while odometry usually arrives
        in the odom frame. Using odom directly causes the exact offset you saw.
        """
        if not self.has_odom:
            return None

        map_frame = self._map_frame or "map"
        source_frame = self.odom_frame or "odom"
        if map_frame == source_frame:
            return self.robot_x, self.robot_y, self.robot_yaw

        try:
            transform = self.tf_buffer.lookup_transform(
                map_frame,
                source_frame,
                rclpy.time.Time(),
            )
        except TransformException as exc:
            self.get_logger().debug(
                f"Waiting for TF {map_frame} <- {source_frame}: {exc}"
            )
            return None

        tx = transform.transform.translation.x
        ty = transform.transform.translation.y
        q = transform.transform.rotation
        _, _, tf_yaw = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])

        cos_yaw = math.cos(tf_yaw)
        sin_yaw = math.sin(tf_yaw)
        map_x = tx + cos_yaw * self.robot_x - sin_yaw * self.robot_y
        map_y = ty + sin_yaw * self.robot_x + cos_yaw * self.robot_y
        map_yaw = _normalize_angle(tf_yaw + self.robot_yaw)
        return map_x, map_y, map_yaw

    def _get_goal_in_map_frame(self) -> tuple[float, float] | None:
        """Resolve the current goal into the current map frame."""
        if self.goal_world is None:
            return None

        map_frame = self._map_frame or "map"
        goal_frame = self.goal_frame or map_frame
        if goal_frame == map_frame:
            return self.goal_world

        try:
            transform = self.tf_buffer.lookup_transform(
                map_frame,
                goal_frame,
                rclpy.time.Time(),
            )
        except TransformException as exc:
            self.get_logger().debug(
                f"Waiting for TF {map_frame} <- {goal_frame}: {exc}"
            )
            return None

        tx = transform.transform.translation.x
        ty = transform.transform.translation.y
        q = transform.transform.rotation
        _, _, tf_yaw = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])

        gx, gy = self.goal_world
        cos_yaw = math.cos(tf_yaw)
        sin_yaw = math.sin(tf_yaw)
        map_x = tx + cos_yaw * gx - sin_yaw * gy
        map_y = ty + sin_yaw * gx + cos_yaw * gy
        return map_x, map_y


# ======================================================================
# Geometry helpers
# ======================================================================

def _point_to_segment_dist(
    px: float, py: float,
    ax: float, ay: float,
    bx: float, by: float,
) -> float:
    """Distance from point (px,py) to line segment (ax,ay)-(bx,by)."""
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.sqrt((px - ax) ** 2 + (py - ay) ** 2)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    return math.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)


def _normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


# ======================================================================
# Entry point
# ======================================================================

def main(args=None):
    rclpy.init(args=args)
    node = PathPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
