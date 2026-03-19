"""
pure_pursuit_node.py — Pure Pursuit path tracking controller for ROS 2.

Follows a nav_msgs/Path published by the path_planner_node using the
Pure Pursuit algorithm. Outputs velocity commands to cmd_vel_diff_count.

Pure Pursuit algorithm:
    1. Find the lookahead point on the path at distance L_d from the robot
    2. Compute the curvature κ = 2 * sin(α) / L_d, where α is the angle
       to the lookahead point in the robot frame
    3. Desired angular velocity ω = v * κ
    4. Linear velocity v from a trapezoidal profile (accel/decel)
       with some reduction on high-curvature stretches

Input:
    - nav/planned_path (nav_msgs/Path) from path_planner_node
    - odom (nav_msgs/Odometry) for robot pose
    - nav/stop (std_msgs/Bool) for emergency stop
    - nav/velocity_rates (std_msgs/Float32MultiArray) from obstacle detection

Output:
    - cmd_vel_diff_count (geometry_msgs/Twist) to motors (same as go_to_point)
    - nav/target_reached (std_msgs/Bool) when goal reached

Topics:
    Subscriptions:
        nav/planned_path       - nav_msgs/Path (path to follow)
        odom                   - nav_msgs/Odometry (robot pose)
        nav/stop               - std_msgs/Bool (cancel)
        nav/velocity_rates     - std_msgs/Float32MultiArray (obstacle modulation)

    Publications:
        cmd_vel_diff_count     - geometry_msgs/Twist (velocity commands)
        nav/target_reached     - std_msgs/Bool (goal reached signal)
"""

import math
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Twist, Point, Quaternion
from std_msgs.msg import Bool, Float32MultiArray
import tf_transformations
from tf2_ros import Buffer, TransformException, TransformListener


class PurePursuitNode(Node):
    def __init__(self):
        super().__init__("pure_pursuit_node")

        # ── Parameters ───────────────────────────────────────────────
        self.declare_parameter("lookahead_distance", 0.15)
        self.declare_parameter("min_lookahead", 0.08)
        self.declare_parameter("max_lookahead", 0.4)
        self.declare_parameter("adaptive_lookahead", True)

        self.declare_parameter("max_linear_speed", 2.0)
        self.declare_parameter("max_angular_speed", 3.0)
        self.declare_parameter("linear_acceleration", 0.4)
        self.declare_parameter("angular_acceleration", 8.0)

        self.declare_parameter("goal_tolerance", 0.03)
        self.declare_parameter("kp_angular", 2.0)
        self.declare_parameter("curve_slow_factor", 0.7)  # slow down on curves
        self.declare_parameter("initial_alignment_enabled", True)
        self.declare_parameter("initial_alignment_tolerance", 0.12)
        self.declare_parameter("initial_alignment_preview_distance", 0.25)

        # ── State ────────────────────────────────────────────────────
        self.path_lock = threading.Lock()
        self.current_path = None            # list of (x, y, z) from Path.poses
        self.closest_path_idx = 0

        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.odom_frame = "odom"
        self.has_odom = False

        self.current_linear_vel = 0.0
        self.current_angular_vel = 0.0

        # Velocity rate modulation from obstacle detection
        self.linear_rate = 1.0
        self.angular_rate = 1.0

        self.stop_requested = False
        self.path_reverse = False  # if goal was sent with reverse flag
        self.path_frame = "map"
        self.initial_alignment_required = False

        # ── Publishers and Subscribers ───────────────────────────────
        self.cmd_vel_pub = self.create_publisher(Twist, "cmd_vel_diff_count", 10)
        self.target_reached_pub = self.create_publisher(Bool, "nav/target_reached", 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(Path, "nav/planned_path", self._path_callback, 10)
        self.create_subscription(Odometry, "odom", self._odom_callback, 10)
        self.create_subscription(Bool, "nav/stop", self._stop_callback, 10)
        self.create_subscription(
            Float32MultiArray, "nav/velocity_rates", self._velocity_rates_callback, 10
        )

        # ── Control timer (20 Hz) ────────────────────────────────────
        self.create_timer(0.05, self._control_loop)

        self.get_logger().info("PurePursuit node started")

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

    def _path_callback(self, msg: Path):
        """Receive new planned path."""
        if len(msg.poses) == 0:
            # Empty path → stop
            with self.path_lock:
                self.current_path = None
                self.closest_path_idx = 0
                self.path_frame = "map"
                self.initial_alignment_required = False
            self.get_logger().info("Empty path received — stopping")
            self._publish_velocity(0.0, 0.0)
            return

        path = []
        for pose in msg.poses:
            x = pose.pose.position.x
            y = pose.pose.position.y
            z = pose.pose.position.z
            path.append((x, y, z))

        self.stop_requested = False
        self.path_frame = msg.header.frame_id or "map"
        self.path_reverse = path[-1][2] > 0.5 if path else False

        with self.path_lock:
            if self.current_path is None:
                # First path → start at closest point
                self.current_path = path
                self.closest_path_idx = 0
                self.initial_alignment_required = True
                self.get_logger().info(f"New path: {len(path)} waypoints")
            else:
                # Transition to new path — re-index at closest point
                old_path = self.current_path
                self.current_path = path
                self._reindex_closest_point()
                self.get_logger().info(
                    f"Path updated: {len(old_path)} → {len(path)} waypoints"
                )

    def _stop_callback(self, msg: Bool):
        if msg.data:
            self.get_logger().info("Stop requested")
            self.stop_requested = True
            with self.path_lock:
                self.current_path = None
            self._publish_velocity(0.0, 0.0)
            reached_msg = Bool()
            reached_msg.data = False
            self.target_reached_pub.publish(reached_msg)

    def _velocity_rates_callback(self, msg: Float32MultiArray):
        """Update velocity modulation rates from obstacle detection."""
        if len(msg.data) >= 2:
            self.linear_rate = msg.data[0]
            self.angular_rate = msg.data[1]

    # ==================================================================
    # Main control loop (20 Hz)
    # ==================================================================

    def _control_loop(self):
        if not self.has_odom:
            return

        if self.stop_requested:
            self._publish_velocity(0.0, 0.0)
            return

        robot_pose = self._get_robot_pose_in_path_frame()
        if robot_pose is None:
            self._publish_velocity(0.0, 0.0)
            return
        robot_x, robot_y, robot_yaw = robot_pose

        with self.path_lock:
            if self.current_path is None:
                self._publish_velocity(0.0, 0.0)
                return
            path = list(self.current_path)

        # Re-index closest point on current path
        self._reindex_closest_point_internal(path, robot_x, robot_y)

        # If we've reached the goal, stop
        goal_tol = self.get_parameter("goal_tolerance").value
        if self._distance_to_point(robot_x, robot_y, path[-1]) < goal_tol:
            self.get_logger().info("Goal reached!")
            self._publish_velocity(0.0, 0.0)
            reached_msg = Bool()
            reached_msg.data = True
            self.target_reached_pub.publish(reached_msg)
            self.initial_alignment_required = False
            return

        if self._handle_initial_alignment(path, robot_x, robot_y, robot_yaw):
            return

        # Find lookahead point
        lookahead_pt = self._find_lookahead_point(path)
        if lookahead_pt is None:
            # Path is behind us or empty — stop
            self._publish_velocity(0.0, 0.0)
            return

        lx, ly = lookahead_pt

        # Transform lookahead to robot frame
        dx = lx - robot_x
        dy = ly - robot_y
        rx = dx * math.cos(robot_yaw) + dy * math.sin(robot_yaw)
        ry = -dx * math.sin(robot_yaw) + dy * math.cos(robot_yaw)

        # Pure pursuit curvature
        ld = math.sqrt(rx * rx + ry * ry)
        if ld < 1e-6:
            curvature = 0.0
        else:
            # α = angle to lookahead in robot frame
            alpha = math.atan2(ry, rx)
            curvature = 2.0 * math.sin(alpha) / ld

        # Linear velocity: trapezoidal profile with end-of-path deceleration
        remaining_dist = self._remaining_path_distance(path, robot_x, robot_y)
        max_lin = self.get_parameter("max_linear_speed").value
        lin_accel = self.get_parameter("linear_acceleration").value
        desired_lin = self._calculate_desired_speed(remaining_dist, max_lin, lin_accel)

        # Reduce speed on high-curvature segments
        curve_factor = self.get_parameter("curve_slow_factor").value
        curve_reduction = curve_factor ** min(abs(curvature), 1.0)
        desired_lin = desired_lin * (0.5 + 0.5 * curve_reduction)

        # Apply acceleration profile
        dt = 0.05  # 20 Hz timer
        new_lin = self._apply_accel_profile(
            self.current_linear_vel,
            desired_lin,
            lin_accel,
            dt,
        )

        # Angular velocity from pure pursuit
        max_ang = self.get_parameter("max_angular_speed").value
        desired_ang = new_lin * curvature
        desired_ang = max(-max_ang, min(max_ang, desired_ang))

        ang_accel = self.get_parameter("angular_acceleration").value
        new_ang = self._apply_accel_profile(
            self.current_angular_vel,
            desired_ang,
            ang_accel,
            dt,
        )

        # For reverse mode, negate linear velocity
        if self.path_reverse:
            new_lin = -new_lin

        self.current_linear_vel = new_lin
        self.current_angular_vel = new_ang

        self._publish_velocity(new_lin, new_ang)

    # ==================================================================
    # Path following utilities
    # ==================================================================

    def _reindex_closest_point(self):
        """Re-find the closest point on the current path (with lock)."""
        if self.current_path is None or len(self.current_path) == 0:
            self.closest_path_idx = 0
            return

        robot_pose = self._get_robot_pose_in_path_frame()
        if robot_pose is None:
            return
        robot_x, robot_y, _ = robot_pose

        path = self.current_path
        min_d = float("inf")
        closest_idx = 0
        for i, (px, py, _) in enumerate(path):
            d = (px - robot_x) ** 2 + (py - robot_y) ** 2
            if d < min_d:
                min_d = d
                closest_idx = i

        self.closest_path_idx = closest_idx

    def _reindex_closest_point_internal(self, path, robot_x: float, robot_y: float):
        """Re-find the closest point on a given path (no lock)."""
        if len(path) == 0:
            self.closest_path_idx = 0
            return

        min_d = float("inf")
        closest_idx = 0
        for i, (px, py, _) in enumerate(path):
            d = (px - robot_x) ** 2 + (py - robot_y) ** 2
            if d < min_d:
                min_d = d
                closest_idx = i

        self.closest_path_idx = closest_idx

    def _find_lookahead_point(self, path) -> tuple[float, float] | None:
        """
        Find the lookahead point on the path.

        Adaptive lookahead: varies with speed. Returns the point at distance L_d
        ahead of the robot along the path.
        """
        if len(path) == 0:
            return None

        # Adaptive lookahead
        if self.get_parameter("adaptive_lookahead").value:
            min_ld = self.get_parameter("min_lookahead").value
            max_ld = self.get_parameter("max_lookahead").value
            base_ld = self.get_parameter("lookahead_distance").value
            # Vary between min and max based on speed (0 → min_ld, max_speed → max_ld)
            speed_ratio = min(1.0, abs(self.current_linear_vel) / 2.0)
            ld = min_ld + speed_ratio * (max_ld - min_ld)
        else:
            ld = self.get_parameter("lookahead_distance").value

        # Start from closest point or further along path
        start_idx = max(0, self.closest_path_idx)

        # Walk along path accumulating distance
        accum_dist = 0.0
        for i in range(start_idx, len(path) - 1):
            px1, py1, _ = path[i]
            px2, py2, _ = path[i + 1]
            seg_len = math.sqrt((px2 - px1) ** 2 + (py2 - py1) ** 2)

            if accum_dist + seg_len >= ld:
                # Interpolate within this segment
                t = (ld - accum_dist) / seg_len if seg_len > 0 else 0.0
                t = max(0.0, min(1.0, t))
                lx = px1 + t * (px2 - px1)
                ly = py1 + t * (py2 - py1)
                return (lx, ly)

            accum_dist += seg_len

        # If we've walked the entire remaining path, return the goal
        if len(path) > 0:
            return (path[-1][0], path[-1][1])

        return None

    def _remaining_path_distance(self, path, robot_x: float, robot_y: float) -> float:
        """Distance from closest path point to the goal."""
        if len(path) == 0:
            return 0.0

        start_idx = max(0, self.closest_path_idx)
        accum = 0.0

        # Distance from robot to closest point
        px, py, _ = path[start_idx]
        accum += math.sqrt((px - robot_x) ** 2 + (py - robot_y) ** 2)

        # Distance along path from closest point to goal
        for i in range(start_idx, len(path) - 1):
            px1, py1, _ = path[i]
            px2, py2, _ = path[i + 1]
            accum += math.sqrt((px2 - px1) ** 2 + (py2 - py1) ** 2)

        return accum

    def _distance_to_point(self, x: float, y: float, pt: tuple) -> float:
        """Euclidean distance from (x,y) to pt=(x,y,z)."""
        dx = x - pt[0]
        dy = y - pt[1]
        return math.sqrt(dx * dx + dy * dy)

    def _handle_initial_alignment(
        self,
        path,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
    ) -> bool:
        """
        Rotate in place to face the start of the path before moving.

        This prevents the initial looping behavior when the first useful path
        segment starts behind the robot.
        """
        if not self.initial_alignment_required:
            return False
        if not self.get_parameter("initial_alignment_enabled").value:
            self.initial_alignment_required = False
            return False

        target = self._get_alignment_target(path)
        if target is None:
            self.initial_alignment_required = False
            return False

        desired_heading = math.atan2(target[1] - robot_y, target[0] - robot_x)
        if self.path_reverse:
            desired_heading = self._normalize_angle(desired_heading + math.pi)

        heading_error = self._normalize_angle(desired_heading - robot_yaw)
        tolerance = self.get_parameter("initial_alignment_tolerance").value
        if abs(heading_error) <= tolerance:
            self.initial_alignment_required = False
            self.current_angular_vel = 0.0
            return False

        kp_angular = self.get_parameter("kp_angular").value
        max_ang = self.get_parameter("max_angular_speed").value
        ang_accel = self.get_parameter("angular_acceleration").value
        dt = 0.05

        desired_ang = max(-max_ang, min(max_ang, kp_angular * heading_error))
        new_ang = self._apply_accel_profile(
            self.current_angular_vel,
            desired_ang,
            ang_accel,
            dt,
        )

        self.current_linear_vel = 0.0
        self.current_angular_vel = new_ang
        self._publish_velocity(0.0, new_ang)
        return True

    def _get_alignment_target(self, path) -> tuple[float, float] | None:
        """Pick a point a little ahead on the path to align toward."""
        if len(path) == 0:
            return None

        preview_distance = self.get_parameter("initial_alignment_preview_distance").value
        start_idx = max(0, self.closest_path_idx)
        accum_dist = 0.0

        for i in range(start_idx, len(path) - 1):
            px1, py1, _ = path[i]
            px2, py2, _ = path[i + 1]
            seg_len = math.sqrt((px2 - px1) ** 2 + (py2 - py1) ** 2)

            if accum_dist + seg_len >= preview_distance:
                t = (preview_distance - accum_dist) / seg_len if seg_len > 0 else 0.0
                return (px1 + t * (px2 - px1), py1 + t * (py2 - py1))

            accum_dist += seg_len

        return (path[-1][0], path[-1][1])

    def _get_robot_pose_in_path_frame(self) -> tuple[float, float, float] | None:
        """Transform the robot pose from odom into the current path frame."""
        if not self.has_odom:
            return None

        path_frame = self.path_frame or "map"
        source_frame = self.odom_frame or "odom"
        if path_frame == source_frame:
            return self.robot_x, self.robot_y, self.robot_yaw

        try:
            transform = self.tf_buffer.lookup_transform(
                path_frame,
                source_frame,
                rclpy.time.Time(),
            )
        except TransformException as exc:
            self.get_logger().debug(
                f"Waiting for TF {path_frame} <- {source_frame}: {exc}"
            )
            return None

        tx = transform.transform.translation.x
        ty = transform.transform.translation.y
        q = transform.transform.rotation
        _, _, tf_yaw = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])

        cos_yaw = math.cos(tf_yaw)
        sin_yaw = math.sin(tf_yaw)
        path_x = tx + cos_yaw * self.robot_x - sin_yaw * self.robot_y
        path_y = ty + sin_yaw * self.robot_x + cos_yaw * self.robot_y
        path_yaw = self._normalize_angle(tf_yaw + self.robot_yaw)
        return path_x, path_y, path_yaw

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    # ==================================================================
    # Velocity control
    # ==================================================================

    def _calculate_desired_speed(
        self, remaining_dist: float, max_speed: float, accel: float
    ) -> float:
        """
        Calculate desired linear speed using trapezoidal profile.

        For given remaining distance and acceleration, compute the speed
        that allows smooth ramp-up, hold, and ramp-down without overshoot.
        """
        if remaining_dist <= 0.0:
            return 0.0

        # For a triangular profile: distance = v_max^2 / accel
        # So v_max = sqrt(distance * accel)
        triangular_max = math.sqrt(remaining_dist * accel)

        # Use the smaller of configured max speed and what's possible
        optimal_max = min(max_speed, triangular_max)

        # Ensure minimum speed to maintain motion
        return max(0.05, optimal_max)

    def _apply_accel_profile(
        self, current_vel: float, target_vel: float, max_accel: float, dt: float
    ) -> float:
        """
        Apply acceleration limit to smooth velocity transitions.

        Args:
            current_vel: Current velocity
            target_vel: Desired/target velocity
            max_accel: Max acceleration magnitude (m/s^2 or rad/s^2)
            dt: Time step (seconds)

        Returns:
            New velocity after applying acceleration limit
        """
        vel_error = target_vel - current_vel
        max_vel_change = max_accel * dt

        if abs(vel_error) <= max_vel_change:
            return target_vel
        else:
            return current_vel + math.copysign(max_vel_change, vel_error)

    # ==================================================================
    # Output
    # ==================================================================

    def _publish_velocity(self, linear_x: float, angular_z: float):
        """Publish velocity command with obstacle modulation applied."""
        # Apply velocity rates from obstacle detection
        final_linear = linear_x * self.linear_rate
        final_angular = angular_z * self.angular_rate

        twist = Twist()
        twist.linear.x = float(final_linear)
        twist.angular.z = float(final_angular)
        self.cmd_vel_pub.publish(twist)


# ======================================================================
# Entry point
# ======================================================================

def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
