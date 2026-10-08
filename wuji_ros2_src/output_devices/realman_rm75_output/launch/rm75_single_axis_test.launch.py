from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config = str(Path(get_package_share_directory("realman_rm75_output")) /
                 "config" / "rm75_single_axis_test.yaml")
    return LaunchDescription([Node(
        package="realman_rm75_output",
        executable="rm75_single_axis_test_node",
        name="rm75_single_axis_test_node",
        output="screen", parameters=[config])])

