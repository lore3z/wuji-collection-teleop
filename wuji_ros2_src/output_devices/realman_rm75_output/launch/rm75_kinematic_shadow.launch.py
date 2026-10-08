from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    rm_share = Path(get_package_share_directory("rm_description"))
    output_share = Path(get_package_share_directory("realman_rm75_output"))
    pico_share = Path(get_package_share_directory("pico_input"))
    robot_description = (rm_share / "urdf" / "rm_75.urdf").read_text()

    return LaunchDescription([
        DeclareLaunchArgument(
            "enable_arm_angle_constraint", default_value="true"),
        DeclareLaunchArgument(
            "enable_six_axis_monitor", default_value="false"),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(
            str(pico_share / "launch" / "pico_right_arm.launch.py"))),
        Node(
            package="realman_rm75_output",
            executable="rm75_kinematic_shadow",
            name="rm75_kinematic_shadow",
            output="screen",
            parameters=[{
                "enable_arm_angle_constraint": LaunchConfiguration(
                    "enable_arm_angle_constraint"),
            }],
        ),
        Node(
            package="realman_rm75_output",
            executable="rm75_six_axis_monitor",
            name="rm75_six_axis_monitor",
            output="screen",
            condition=IfCondition(LaunchConfiguration("enable_six_axis_monitor")),
        ),
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="rm75_shadow_robot_state_publisher",
            parameters=[{"robot_description": robot_description}],
            remappings=[
                ("/joint_states", "/rm75_shadow/q_cmd_joint_states"),
            ],
            output="screen",
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="rm75_shadow_rviz",
            arguments=["-d", str(
                output_share / "rviz" / "rm75_kinematic_shadow.rviz")],
            output="screen",
        ),
    ])
