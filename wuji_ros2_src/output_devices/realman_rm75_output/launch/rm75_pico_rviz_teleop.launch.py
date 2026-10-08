"""Direct PICO wrist -> RM75 Pinocchio IK -> RViz teleoperation."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.actions import Node


def generate_launch_description():
    rm_share = Path(get_package_share_directory("rm_description"))
    output_share = Path(get_package_share_directory("realman_rm75_output"))
    pico_share = Path(get_package_share_directory("pico_input"))
    robot_description = (rm_share / "urdf" / "rm_75.urdf").read_text()
    pico_config = str(pico_share / "config" / "pico_rm75_right_wrist.yaml")

    # Raw OpenXR PICO axes -> RM base: X=-PICO_Z, Y=-PICO_X, Z=PICO_Y.
    source_to_base = [
        0.0, 0.0, -1.0,
        -1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
    ]
    # Human wrist flat at rebase -> RM75 Link7/tool Z horizontal and forward.
    # The arm stays bent and offline scanning retains roughly 0.18 m upward
    # and 0.21 m downward travel from this neutral pose.
    safe_home_joints = [
        -1.5707963268, 0.4, 0.0, 1.5, 0.0,
        -0.3292036732, 0.1745329252]
    return LaunchDescription([
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("tracker_serial", default_value=""),
        DeclareLaunchArgument("position_scale", default_value="1.0"),
        DeclareLaunchArgument("enable_pose_diagnostics", default_value="true"),
        DeclareLaunchArgument("diagnostic_every_n_frames", default_value="1"),
        Node(
            package="pico_input",
            executable="pico_tracker_pose_publisher",
            name="pico_tracker_pose_publisher",
            output="screen",
            parameters=[pico_config, {
                "tracker_serial": LaunchConfiguration("tracker_serial"),
            }],
        ),
        Node(
            package="realman_rm75_output",
            executable="rm75_external_pose_follower",
            name="rm75_pico_pose_follower",
            output="screen",
            parameters=[{
                "input_topic": "/pico/right_wrist/raw_pose",
                "input_frame": "pico_tracking",
                "input_mode": "relative",
                "source_to_base": source_to_base,
                "position_scale": ParameterValue(
                    LaunchConfiguration("position_scale"), value_type=float),
                "home_joints_rad": safe_home_joints,
                "enable_ik_boundary_fallback": True,
                "boundary_search_iterations": 7,
                "min_boundary_progress_fraction": 0.01,
                "enable_pose_diagnostics": ParameterValue(
                    LaunchConfiguration("enable_pose_diagnostics"),
                    value_type=bool),
                "diagnostic_every_n_frames": ParameterValue(
                    LaunchConfiguration("diagnostic_every_n_frames"),
                    value_type=int),
            }],
        ),
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="rm75_pico_robot_state_publisher",
            parameters=[{"robot_description": robot_description}],
            remappings=[("/joint_states", "/rm75_sim/joint_states")],
            output="screen",
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="rm75_pico_rviz",
            arguments=["-d", str(
                output_share / "rviz" / "rm75_offline_trajectory.rviz")],
            condition=IfCondition(LaunchConfiguration("rviz")),
            output="screen",
        ),
    ])
