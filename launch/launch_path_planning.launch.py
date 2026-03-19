"""
launch_path_planning.launch.py

Launches the dynamic path planning system with SLAM Toolbox:
  1. SLAM Toolbox (online async) — builds and maintains the map
  2. Path Planner node — subscribes to /map and plans paths dynamically
  3. Pure Pursuit node — follows the planned path with velocity control

Designed for testing the path planning nodes in simulation or on the real robot.
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Get package directories
    slan_project_dir = get_package_share_directory("SLAN_project")
    slam_toolbox_dir = get_package_share_directory("slam_toolbox")

    # Configuration files
    mapper_params = os.path.join(
        slan_project_dir, "config", "mapper_params_online_async.yaml"
    )

    # SLAM Toolbox launch file
    slam_launch = os.path.join(
        slam_toolbox_dir, "launch", "online_async_launch.py"
    )

    # Simulation time parameter
    use_sim_time = LaunchConfiguration("use_sim_time", default="false")

    return LaunchDescription(
        [
            # ── SLAM Toolbox (online async) ──────────────────────────
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(slam_launch),
                launch_arguments={
                    "params_file": mapper_params,
                    "use_sim_time": use_sim_time,
                }.items(),
            ),
            # ── Path Planner Node ────────────────────────────────────
            Node(
                package="SLAN_project",
                executable="pathPlanner",
                name="path_planner",
                # namespace="MainRobot",
                parameters=[
                    {"use_sim_time": use_sim_time},
                    # Path planning algorithm parameters
                    {"obstacle_dilation": 8},
                    {"astar_downscale": 1.0},
                    {"unknown_as_free": True},
                    # Replanning triggers
                    {"replan_distance_threshold": 0.15},
                    {"map_change_threshold": 0.05},
                    {"path_blend_distance": 0.3},
                    # Elastic Band parameters
                    {"eb_spring_weight": 1.5},
                    {"eb_repulsive_weight": 0.6},
                    {"eb_max_iterations": 80},
                ],
                remappings=[
                    # Map subscribes to absolute /map from SLAM Toolbox
                    ("/map", "/map"),
                ],
                output="screen",
            ),
            # ── Pure Pursuit Node ────────────────────────────────────
            Node(
                package="SLAN_project",
                executable="purePursuit",
                name="pure_pursuit",
                # namespace="MainRobot",
                parameters=[
                    {"use_sim_time": use_sim_time},
                    # Pure Pursuit parameters
                    {"lookahead_distance": 0.15},
                    {"min_lookahead": 0.08},
                    {"max_lookahead": 0.4},
                    {"adaptive_lookahead": True},
                    # Velocity limits
                    {"max_linear_speed": 0.5},
                    {"max_angular_speed": 1.0},
                    {"linear_acceleration": 0.4},
                    {"angular_acceleration": 2.0},
                    # Control parameters
                    {"goal_tolerance": 0.03},
                    {"kp_angular": 1.0},
                    {"curve_slow_factor": 0.7},
                ],
                output="screen",
            ),
        ]
    )
