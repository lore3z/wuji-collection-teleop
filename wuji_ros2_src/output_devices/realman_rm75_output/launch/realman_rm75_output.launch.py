from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config = str(Path(get_package_share_directory("realman_rm75_output")) /
                 "config" / "realman_rm75_output.yaml")
    return LaunchDescription([Node(
        package="realman_rm75_output",
        executable="realman_rm75_output_node",
        name="realman_rm75_output_node",
        output="screen",
        parameters=[config],
    )])
