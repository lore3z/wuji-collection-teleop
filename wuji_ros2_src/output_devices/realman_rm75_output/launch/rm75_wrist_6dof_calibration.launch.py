"""Quiet RViz shadow launch for interactive single-wrist 6DoF calibration."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    rm_share = Path(get_package_share_directory("rm_description"))
    output_share = Path(get_package_share_directory("realman_rm75_output"))
    pico_share = Path(get_package_share_directory("pico_input"))
    robot_description = (rm_share / "urdf" / "rm_75.urdf").read_text()
    pico_config = str(pico_share / "config" / "pico_right_arm.yaml")
    quiet = ["--ros-args", "--log-level", "warn"]

    return LaunchDescription([
        Node(
            package="pico_input",
            executable="pico_right_arm_input_node",
            name="pico_right_arm_input_node",
            parameters=[pico_config],
            arguments=quiet,
            output="screen",
        ),
        Node(
            package="realman_rm75_output",
            executable="rm75_kinematic_shadow",
            name="rm75_kinematic_shadow",
            parameters=[{
                "enable_arm_angle_constraint": False,
                "arm_angle_mode": "fixed",
            }],
            arguments=quiet,
            output="screen",
        ),
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="rm75_shadow_robot_state_publisher",
            parameters=[{"robot_description": robot_description}],
            remappings=[
                ("/joint_states", "/rm75_shadow/q_cmd_joint_states"),
            ],
            arguments=quiet,
            output="screen",
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="rm75_shadow_rviz",
            arguments=[
                "-d", str(output_share / "rviz" /
                          "rm75_kinematic_shadow.rviz"),
                "--ros-args", "--log-level", "warn",
            ],
            output="screen",
        ),
    ])
