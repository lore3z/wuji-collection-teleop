"""Safe standalone launch for single-tracker right-arm PICO input.

This launch starts no robot, dexterous-hand, or camera output nodes.
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    config = str(
        Path(get_package_share_directory("pico_input"))
        / "config"
        / "pico_right_arm.yaml"
    )

    return LaunchDescription([
        Node(
            package="pico_input",
            executable="pico_right_arm_input_node",
            name="pico_right_arm_input_node",
            output="screen",
            parameters=[config],
        ),
    ])
