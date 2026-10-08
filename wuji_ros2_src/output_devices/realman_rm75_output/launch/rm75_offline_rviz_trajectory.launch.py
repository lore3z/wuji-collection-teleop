from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    rm_share = Path(get_package_share_directory("rm_description"))
    output_share = Path(get_package_share_directory("realman_rm75_output"))
    robot_description = (rm_share / "urdf" / "rm_75.urdf").read_text()

    return LaunchDescription([
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("rate_hz", default_value="60.0"),
        DeclareLaunchArgument("loop_period_sec", default_value="12.0"),
        DeclareLaunchArgument("warmup_sec", default_value="2.0"),
        DeclareLaunchArgument("radius_x_m", default_value="0.08"),
        DeclareLaunchArgument("radius_y_m", default_value="0.06"),
        DeclareLaunchArgument("radius_z_m", default_value="0.04"),
        Node(
            package="realman_rm75_output",
            executable="rm75_offline_trajectory",
            name="rm75_offline_trajectory",
            output="screen",
            parameters=[{
                "rate_hz": ParameterValue(
                    LaunchConfiguration("rate_hz"), value_type=float),
                "loop_period_sec": ParameterValue(
                    LaunchConfiguration("loop_period_sec"), value_type=float),
                "warmup_sec": ParameterValue(
                    LaunchConfiguration("warmup_sec"), value_type=float),
                "radius_x_m": ParameterValue(
                    LaunchConfiguration("radius_x_m"), value_type=float),
                "radius_y_m": ParameterValue(
                    LaunchConfiguration("radius_y_m"), value_type=float),
                "radius_z_m": ParameterValue(
                    LaunchConfiguration("radius_z_m"), value_type=float),
            }],
        ),
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="rm75_offline_robot_state_publisher",
            parameters=[{"robot_description": robot_description}],
            remappings=[
                ("/joint_states", "/rm75_sim/joint_states"),
            ],
            output="screen",
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="rm75_offline_rviz",
            arguments=["-d", str(
                output_share / "rviz" / "rm75_offline_trajectory.rviz")],
            condition=IfCondition(LaunchConfiguration("rviz")),
            output="screen",
        ),
    ])
