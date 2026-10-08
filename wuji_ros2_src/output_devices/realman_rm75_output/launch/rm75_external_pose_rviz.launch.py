"""Offline RM75 RViz follower for a generic absolute PoseStamped command."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    rm_share = Path(get_package_share_directory("rm_description"))
    output_share = Path(get_package_share_directory("realman_rm75_output"))
    robot_description = (rm_share / "urdf" / "rm_75.urdf").read_text()

    return LaunchDescription([
        DeclareLaunchArgument(
            "input_topic", default_value="/rm75_sim/target_pose_cmd"),
        DeclareLaunchArgument("rviz", default_value="true"),
        Node(
            package="realman_rm75_output",
            executable="rm75_external_pose_follower",
            name="rm75_external_pose_follower",
            output="screen",
            parameters=[{
                "input_topic": LaunchConfiguration("input_topic"),
                "input_frame": "base_link",
                "input_mode": "absolute",
            }],
        ),
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="rm75_external_robot_state_publisher",
            parameters=[{"robot_description": robot_description}],
            remappings=[("/joint_states", "/rm75_sim/joint_states")],
            output="screen",
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="rm75_external_rviz",
            arguments=["-d", str(
                output_share / "rviz" / "rm75_offline_trajectory.rviz")],
            condition=IfCondition(LaunchConfiguration("rviz")),
            output="screen",
        ),
    ])
