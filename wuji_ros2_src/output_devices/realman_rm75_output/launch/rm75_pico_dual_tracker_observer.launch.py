"""Wrist-driven RM75 shadow with optional low-gain elbow assistance."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from realman_rm75_output.urdf_link_prefix import prefix_urdf_link_names


def generate_launch_description():
    rm_share = Path(get_package_share_directory("rm_description"))
    output_share = Path(get_package_share_directory("realman_rm75_output"))
    robot_description = (rm_share / "urdf" / "rm_75.urdf").read_text()
    command_robot_description = prefix_urdf_link_names(
        robot_description,
        "rm75_command_",
        visual_rgba="0.1 0.85 0.25 1",
    )

    # Raw OpenXR PICO axes -> RM base: X=-PICO_Z, Y=-PICO_X, Z=PICO_Y.
    source_to_base = [
        0.0, 0.0, -1.0,
        -1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
    ]
    safe_home_joints = [
        -1.5707963268, 0.4, 0.0, 1.5, 0.0,
        -0.3292036732, 0.1745329252]

    position_scale = ParameterValue(
        LaunchConfiguration("position_scale"), value_type=float)
    diagnostic_rate = ParameterValue(
        LaunchConfiguration("upper_diagnostic_rate_hz"), value_type=float)

    return LaunchDescription([
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument(
            "wrist_serial", default_value="PC2310MLKC190056G"),
        DeclareLaunchArgument(
            "upper_arm_serial", default_value="PC2310MLKC190573G"),
        DeclareLaunchArgument("position_scale", default_value="0.5"),
        DeclareLaunchArgument(
            "command_shadow_y_offset", default_value="0.55",
            description=(
                "RViz-only lateral offset for the limited command model; "
                "set 0.0 to overlay it on the unrestricted shadow")),
        DeclareLaunchArgument(
            "upper_diagnostic_rate_hz", default_value="5.0"),
        DeclareLaunchArgument(
            "elbow_assist_mode", default_value="observe"),
        DeclareLaunchArgument(
            "elbow_assist_weight", default_value="0.02"),
        DeclareLaunchArgument(
            "enable_singularity_adaptive_damping", default_value="false"),
        Node(
            package="pico_input",
            executable="pico_dual_tracker_pose_publisher",
            name="pico_dual_tracker_pose_publisher",
            output="screen",
            parameters=[{
                "wrist_serial": LaunchConfiguration("wrist_serial"),
                "upper_arm_serial": LaunchConfiguration("upper_arm_serial"),
                "publish_rate_hz": 60.0,
            }],
        ),
        Node(
            package="realman_rm75_output",
            executable="rm75_external_pose_follower",
            name="rm75_dual_tracker_wrist_follower",
            output="screen",
            parameters=[{
                "input_topic": "/pico/right_wrist/raw_pose",
                "input_frame": "pico_tracking",
                "input_mode": "relative",
                "source_to_base": source_to_base,
                "position_scale": position_scale,
                "home_joints_rad": safe_home_joints,
                "enable_ik_boundary_fallback": True,
                "boundary_search_iterations": 7,
                "min_boundary_progress_fraction": 0.01,
                # The launcher's stdout is captured to a host-side diagnostic
                # log, so every accepted input can be audited without making
                # the interactive Enter prompt unreadable.
                "enable_pose_diagnostics": True,
                "diagnostic_every_n_frames": 1,
                "elbow_assist_mode": LaunchConfiguration(
                    "elbow_assist_mode"),
                "elbow_assist_weight": ParameterValue(
                    LaunchConfiguration("elbow_assist_weight"),
                    value_type=float),
                "enable_singularity_adaptive_damping": ParameterValue(
                    LaunchConfiguration(
                        "enable_singularity_adaptive_damping"),
                    value_type=bool),
            }],
        ),
        Node(
            package="realman_rm75_output",
            executable="rm75_upper_arm_shadow_observer",
            name="rm75_upper_arm_shadow_observer",
            output="screen",
            parameters=[{
                "upper_arm_topic": "/pico/right_upper_arm/raw_pose",
                "wrist_target_topic": "/rm75_sim/target_pose",
                "input_frame": "pico_tracking",
                "source_to_base": source_to_base,
                "position_scale": position_scale,
                "home_joints_rad": safe_home_joints,
                "diagnostic_rate_hz": diagnostic_rate,
            }],
        ),
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="rm75_dual_tracker_robot_state_publisher",
            parameters=[{"robot_description": robot_description}],
            remappings=[("/joint_states", "/rm75_sim/joint_states")],
            output="screen",
        ),
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            namespace="rm75_real_command",
            name="robot_state_publisher",
            parameters=[{
                # This URDF owns a completely separate set of link frame names.
                # RViz therefore does not need its fragile TF Prefix setting.
                "robot_description": command_robot_description,
                "publish_frequency": 60.0,
            }],
            remappings=[(
                "joint_states",
                "/rm75_real_trial/command_joint_states")],
            output="screen",
        ),
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="rm75_real_command_base_transform",
            arguments=[
                "--x", "0",
                "--y", LaunchConfiguration("command_shadow_y_offset"),
                "--z", "0",
                "--roll", "0", "--pitch", "0", "--yaw", "0",
                "--frame-id", "base_link",
                "--child-frame-id", "rm75_command_base_link",
            ],
            output="screen",
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="rm75_dual_tracker_rviz",
            arguments=[
                "-d",
                str(output_share / "rviz" / "rm75_offline_trajectory.rviz"),
            ],
            condition=IfCondition(LaunchConfiguration("rviz")),
            output="screen",
        ),
    ])
