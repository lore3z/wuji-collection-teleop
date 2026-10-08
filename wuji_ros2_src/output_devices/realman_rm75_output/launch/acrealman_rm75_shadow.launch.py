"""Read-only RM75 shadow driven by acRealman's absolute TCP target."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    rm_share = Path(get_package_share_directory("rm_description"))
    output_share = Path(get_package_share_directory("realman_rm75_output"))
    robot_description = (rm_share / "urdf" / "rm_75.urdf").read_text()

    return LaunchDescription([
        DeclareLaunchArgument("robot_ip", default_value="192.168.192.19"),
        DeclareLaunchArgument("robot_port", default_value="8080"),
        Node(
            package="realman_rm75_output",
            executable="rm75_kinematic_shadow",
            name="acrealman_rm75_kinematic_shadow",
            output="screen",
            parameters=[{
                "robot_ip": LaunchConfiguration("robot_ip"),
                "robot_port": LaunchConfiguration("robot_port"),
                "input_topic": "/xr_rm/right_rm75/target_pose",
                "input_frame": "rm_base",
                "input_pose_mode": "absolute_rm",
                "arm_angle_mode": "ordinary",
                "enable_arm_angle_constraint": False,
            }],
        ),
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="acrealman_shadow_robot_state_publisher",
            parameters=[{"robot_description": robot_description}],
            remappings=[
                ("/joint_states", "/rm75_shadow/q_cmd_joint_states"),
            ],
            output="screen",
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="acrealman_rm75_shadow_rviz",
            arguments=["-d", str(
                output_share / "rviz" / "rm75_kinematic_shadow.rviz")],
            output="screen",
        ),
    ])
