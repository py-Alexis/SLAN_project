import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    package_name = 'top_secret_robot'
    sim_package_name = 'nantrobot_robot_sim'
    slan_package_name = 'SLAN_project'

    buttons = Node(
        package='top_secret_robot',
        executable='buttons',
        parameters=[{'use_sim_time': False}],
        output='screen'
    )

    rsp = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory(sim_package_name), 'launch', 'rsp.launch.py'
        )]), launch_arguments={'use_sim_time': 'false', 'use_namespace': ''}.items()
    )

    rplidar = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory(package_name), 'launch', 'rplidar.launch.py'
        )])
    )

    twist_mux_params = os.path.join(get_package_share_directory(package_name), 'config', 'twist_mux.yaml')
    twist_mux = Node(
        package="twist_mux",
        executable="twist_mux",
        parameters=[twist_mux_params, {'use_sim_time': False}],
        remappings=[('cmd_vel_out', 'diff_cont/cmd_vel_unstamped')],
    )

    rviz_config = os.path.join(get_package_share_directory(package_name), 'config', 'main.rviz')
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': False}],
        output='screen'
    )

    obstacle_detection = Node(
        package='top_secret_robot',
        executable='obstacle_detection',
        parameters=[{'use_sim_time': False}],
        output='screen'
    )

    esp_serial_listener = Node(
        package='top_secret_robot',
        executable='esp_serial_listener',
        parameters=[{
            'use_sim_time': False,
            'port': '/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0'
        }],
        output='screen'
    )

    esp_serial_sender = Node(
        package='top_secret_robot',
        executable='esp_serial_sender',
        parameters=[{
            'use_sim_time': False,
            'port': '/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0'
        }],
        output='screen'
    )

    odom_tf_broadcaster = Node(
        package=sim_package_name,
        executable='odom_tf_broadcaster',
        parameters=[{'use_sim_time': False}],
        output='screen'
    )

    joy_linux = Node(
        package='joy_linux',
        executable='joy_linux_node',
        parameters=[{'use_sim_time': False}],
        output='screen'
    )

    teleop_twist_joy = Node(
        package='teleop_twist_joy',
        executable='teleop_node',
        parameters=[{
            'use_sim_time': False,
            'cmd_vel_topic': 'cmd_vel_joy',
            'require_enable_button': True,
            'axis_linear.x': 4,
            'axis_angular.yaw': 0,
            'enable_turbo_button': 4,
            'scale_linear.x': 4.0,
            'scale_angular.yaw': 10.0,
            'scale_linear_turbo.x': 0.7,
            'scale_angular_turbo.yaw': 1.0
        }],
        output='screen'
    )

    camera = Node(
        package='camera_ros',
        executable='camera_node',
        parameters=[{'use_sim_time': False}],
        output='screen'
    )

    lidar_led = Node(
        package='top_secret_robot',
        executable='lidar_led',
        parameters=[{'use_sim_time': False}],
        output='screen'
    )

    table_publisher = Node(
        package='nantrobot_robot_sim',
        executable='table_mesh_publisher',
        parameters=[{'use_sim_time': False}],
        output='screen'
    )

    # SLAM Toolbox
    slam_toolbox_test = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory(slan_package_name), 'launch', 'slam_toolbox_test.launch.py')
        ])
    )

    # Nav Goal Pose Bridge
    nav_goal_pose_bridge = Node(
        package='SLAN_project',
        executable='nav_goal_pose_bridge',
        name='nav_goal_pose_bridge',
        output='screen',
        parameters=[{
            'use_sim_time': False,
            'goal_topic': '/nav/goal_pose',
            'action_name': '/navigate_to_pose',
            'cancel_previous': True,
            'wait_for_server_sec': 30.0,
            'min_goal_period_sec': 0.0,
            'min_goal_separation_xy': 0.0,
        }],
    )

    return LaunchDescription([
        # buttons,
        # rsp,
        rplidar,
        # twist_mux,
        # rviz,
        # obstacle_detection,
        esp_serial_listener,
        esp_serial_sender,
        # odom_tf_broadcaster,
        # joy_linux,
        # teleop_twist_joy,
        # camera,
        # lidar_led,
        # table_publisher,
        # slam_toolbox_test,
        # nav_goal_pose_bridge,
    ])
