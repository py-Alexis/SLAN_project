import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    # Path to the config file in your package
    slan_project_dir = get_package_share_directory('SLAN_project')
    params_file = os.path.join(slan_project_dir, 'config', 'mapper_params_online_async.yaml')

    # Path to the slam_toolbox launch file
    slam_toolbox_dir = get_package_share_directory('slam_toolbox')
    slam_launch_file = os.path.join(slam_toolbox_dir, 'launch', 'online_async_launch.py')

    # Path to the nav2_bringup launch file
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')
    nav2_launch_file = os.path.join(nav2_bringup_dir, 'launch', 'navigation_launch.py')

    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(slam_launch_file),
            launch_arguments={
                'params_file': params_file,
                'use_sim_time': use_sim_time
            }.items()
        ),
        # TimerAction(
        #     period=10.0,
        #     actions=[
        #         IncludeLaunchDescription(
        #             PythonLaunchDescriptionSource(nav2_launch_file),
        #             launch_arguments={
        #                 'use_sim_time': use_sim_time
        #             }.items()
        #         )
        #     ]
        # )
        # TODO : ros2 launch nav2_bringup navigation_launch.py use_sim_time:=true
    ])