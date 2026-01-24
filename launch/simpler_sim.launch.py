import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    # Define package names
    # Note: We are using resources from top_secret_robot even though this file is in SLAN_project
    robot_package_name = 'top_secret_robot'
    robot_package_share = get_package_share_directory(robot_package_name)
    sim_package_name = 'nantrobot_robot_sim'
    sim_package_share = get_package_share_directory(sim_package_name)
    slan_package_share = get_package_share_directory("SLAN_project")

    # Button config file in the config folder of top_secret_robot
    button_config = os.path.join(
        slan_package_share,
        'config',
        'button.yaml'
    )

    # RViz config file in the config folder of SLAN_project
    rviz_config = os.path.join(
        slan_package_share,
        'config/',
        'mapping.rviz'
    )

    # Start simulation (Gazebo + RViz)
    start_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(sim_package_share, 'launch', 'start_sim.launch.py')
        ]),
        launch_arguments={
            'button_config': button_config,
            'rviz_config': rviz_config,
            "world": os.path.join(slan_package_share,'worlds','Obstacle_3.world')
        }.items()
    )

    # Static transform: world -> odom
    static_tf_robot = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'world', 'odom']
    )

    # Spawn single robot without namespace
    spawn_robot = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(sim_package_share, 'launch', 'spawn_robot.launch.py')
        ]),
        launch_arguments={
            'robot_namespace': '',
            'strategy_file': 'example.json',
            'world_name': 'Obstacle_1'
            
        }.items()
    )

    # SLAM Toolbox and Nav2
    slam_toolbox_test = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(slan_package_share, 'launch', 'slam_toolbox_test.launch.py')
        ])
    )

    nav_goal_pose_bridge = Node(
        package='SLAN_project',
        executable='nav_goal_pose_bridge',
        name='nav_goal_pose_bridge',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'goal_topic': '/nav/goal_pose',
            'action_name': '/navigate_to_pose',
            'cancel_previous': True,
            'wait_for_server_sec': 30.0,
            'min_goal_period_sec': 0.0,
            'min_goal_separation_xy': 0.0,
        }],
    )

    return LaunchDescription([
        start_sim,
        # static_tf_robot,
        spawn_robot,
        slam_toolbox_test,
        nav_goal_pose_bridge,
    ])