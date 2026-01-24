#!/usr/bin/env python3
import math
from typing import Optional

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose


def _pose_dist_xy(a: PoseStamped, b: PoseStamped) -> float:
    dx = a.pose.position.x - b.pose.position.x
    dy = a.pose.position.y - b.pose.position.y
    return math.hypot(dx, dy)


class NavGoalPoseBridge(Node):
    """
    Subscribes to a PoseStamped topic (default: /nav/goal_pose) and forwards each pose
    as a NavigateToPose action goal (default action: /navigate_to_pose).

    Useful when some tool publishes PoseStamped goals but you want Nav2 action goals.
    """

    def __init__(self) -> None:
        super().__init__("nav_goal_pose_bridge")

        self.declare_parameter("goal_topic", "/nav/goal_pose")
        self.declare_parameter("action_name", "/navigate_to_pose")
        self.declare_parameter("wait_for_server_sec", 5.0)
        self.declare_parameter("cancel_previous", True)
        self.declare_parameter("min_goal_separation_xy", 0.05)  # meters; ignore near-duplicates
        self.declare_parameter("min_goal_period_sec", 0.2)       # seconds; ignore bursts
        self.declare_parameter("behavior_tree", "")              # optional BT XML path

        self._goal_topic = self.get_parameter("goal_topic").get_parameter_value().string_value
        self._action_name = self.get_parameter("action_name").get_parameter_value().string_value
        self._wait_for_server_sec = float(self.get_parameter("wait_for_server_sec").value)
        self._cancel_previous = bool(self.get_parameter("cancel_previous").value)
        self._min_sep = float(self.get_parameter("min_goal_separation_xy").value)
        self._min_period = float(self.get_parameter("min_goal_period_sec").value)
        self._bt = self.get_parameter("behavior_tree").get_parameter_value().string_value

        # RViz goal tools may publish BEST_EFFORT; RELIABLE subscription can be incompatible (no messages).
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._sub = self.create_subscription(PoseStamped, self._goal_topic, self._on_goal_pose, qos)

        self._client = ActionClient(self, NavigateToPose, self._action_name)

        self._last_goal_msg: Optional[PoseStamped] = None
        self._last_goal_time = self.get_clock().now()

        self._current_goal_handle = None

        self.get_logger().info(
            f"Bridging topic '{self._goal_topic}' -> action '{self._action_name}' "
            f"(cancel_previous={self._cancel_previous})"
        )

    def _on_goal_pose(self, msg: PoseStamped) -> None:
        now = self.get_clock().now()

        self.get_logger().info(
            f"Received goal_pose frame='{msg.header.frame_id}' "
            f"x={msg.pose.position.x:.3f}, y={msg.pose.position.y:.3f}"
        )

        # Basic filtering: ignore bursts / duplicates
        dt = (now - self._last_goal_time).nanoseconds * 1e-9
        if dt < self._min_period:
            self.get_logger().info(f"Ignoring goal (throttle): dt={dt:.3f}s < {self._min_period:.3f}s")
            return
        if self._last_goal_msg is not None:
            try:
                if msg.header.frame_id == self._last_goal_msg.header.frame_id:
                    d = _pose_dist_xy(msg, self._last_goal_msg)
                    if d < self._min_sep:
                        self.get_logger().info(
                            f"Ignoring goal (duplicate): d={d:.3f}m < {self._min_sep:.3f}m"
                        )
                        return
            except Exception:
                pass

        self._last_goal_msg = msg
        self._last_goal_time = now

        self.get_logger().info(f"Waiting for action server '{self._action_name}'...")
        if not self._client.wait_for_server(timeout_sec=self._wait_for_server_sec):
            self.get_logger().error(
                f"Nav2 action server not available: '{self._action_name}'. "
                f"Check `ros2 action list | grep navigate_to_pose` and namespaces."
            )
            return

        # Optionally cancel previous goal
        if self._cancel_previous and self._current_goal_handle is not None:
            try:
                self.get_logger().info("Canceling previous NavigateToPose goal...")
                self._current_goal_handle.cancel_goal_async()
            except Exception as e:
                self.get_logger().warn(f"Failed to cancel previous goal: {e!r}")

        goal = NavigateToPose.Goal()
        goal.pose = msg
        if self._bt:
            goal.behavior_tree = self._bt

        self.get_logger().info(f"Sending NavigateToPose goal to action '{self._action_name}'")
        send_future = self._client.send_goal_async(goal, feedback_callback=self._on_feedback)
        send_future.add_done_callback(self._on_goal_response)

    def _on_feedback(self, feedback_msg) -> None:
        """
        ROS 2 Python action feedback callback payload differs across distros:
        - Sometimes it's a wrapper with `.feedback`
        - Sometimes it's the feedback object itself
        """
        # Keep quiet by default; uncomment if needed.
        # fb = getattr(feedback_msg, "feedback", feedback_msg)
        # self.get_logger().info(f"distance_remaining={getattr(fb, 'distance_remaining', None)}")
        pass

    def _on_goal_response(self, future) -> None:
        try:
            goal_handle = future.result()
        except Exception as e:
            self.get_logger().error(f"Send goal failed: {e!r}")
            return

        if not goal_handle.accepted:
            self.get_logger().error("Goal rejected by NavigateToPose action server.")
            return

        self._current_goal_handle = goal_handle
        self.get_logger().info("Goal accepted.")

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_result)

    def _on_result(self, future) -> None:
        try:
            result = future.result().result
            status = future.result().status
        except Exception as e:
            self.get_logger().error(f"Get result failed: {e!r}")
            return

        self.get_logger().info(f"Goal finished with status={status}, result={result}")
        self._current_goal_handle = None


def main() -> None:
    rclpy.init()
    node = NavGoalPoseBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
