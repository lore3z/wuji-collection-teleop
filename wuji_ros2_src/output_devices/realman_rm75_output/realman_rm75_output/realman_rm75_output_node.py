"""Read-only RM75 algorithms over WUJI right-arm target poses."""

import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from rclpy.node import Node

from Robotic_Arm.rm_robot_interface import (
    RoboticArm,
    rm_inverse_kinematics_params_t,
    rm_thread_mode_e,
)

from .ik_dry_run import IKDryRunEvaluator, validate_safety_mode
from .elbow_arm_angle_mapper import (
    InvalidElbowDirection,
    RelativeArmAngleMapper,
)
from .pose_mapper import (
    InvalidPose,
    RelativePoseMapper,
    quaternion_wxyz_to_xyzw,
)


class RealManAlgorithmAdapter:
    """Narrow adapter exposing only the approved ordinary algorithm calls."""

    def __init__(self, robot):
        self.robot = robot

    def inverse(self, reference_joints_deg, target_pose_wxyz):
        params = rm_inverse_kinematics_params_t(
            q_in=reference_joints_deg,
            q_pose=target_pose_wxyz,
            flag=0,
        )
        return self.robot.rm_algo_inverse_kinematics(params)

    def inverse_for_arm_angle(self, reference_joints_deg, target_pose_wxyz,
                              arm_angle_deg):
        params = rm_inverse_kinematics_params_t(
            q_in=reference_joints_deg,
            q_pose=target_pose_wxyz,
            flag=0,
        )
        return self.robot.rm_algo_inverse_kinematics_rm75_for_arm_angle(
            params, arm_angle_deg)

    def forward(self, joints_deg):
        return self.robot.rm_algo_forward_kinematics(joints_deg, flag=0)

    def arm_angle(self, joints_deg):
        return self.robot.rm_algo_calculate_arm_angle_from_config_rm75(joints_deg)


class RealManRM75OutputNode(Node):
    def __init__(self):
        super().__init__("realman_rm75_output_node")
        self.declare_parameter("robot_ip", "192.168.1.18")
        self.declare_parameter("robot_port", 8080)
        self.declare_parameter("input_topic", "/right_arm_target_pose")
        self.declare_parameter("input_frame", "world_right")
        self.declare_parameter("elbow_topic", "/right_arm_elbow_direction")
        self.declare_parameter("elbow_frame", "world_right")
        self.declare_parameter("enable_arm_angle_constraint", True)
        self.declare_parameter("arm_angle_scale", 0.25)
        self.declare_parameter("max_arm_angle_step_deg", 5.0)
        self.declare_parameter("dry_run", True)
        self.declare_parameter("enable_motion", False)
        self.declare_parameter("axis_mapping", [1.0, 0.0, 0.0,
                                                 0.0, 1.0, 0.0,
                                                 0.0, 0.0, 1.0])
        self.declare_parameter("position_scale", 1.0)
        self.declare_parameter("translation_only", True)
        self.declare_parameter("rotation_only", False)
        self.declare_parameter("rotation_scale", 1.0)
        self.declare_parameter("max_position_jump_m", 0.10)
        self.declare_parameter("max_rotation_jump_deg", 45.0)
        self.declare_parameter("tracker_timeout_sec", 0.50)
        self.declare_parameter("freeze_timeout_sec", 0.75)
        self.declare_parameter("workspace_min", [-0.80, -0.80, -0.10])
        self.declare_parameter("workspace_max", [0.80, 0.80, 1.20])
        self.declare_parameter("max_joint_jump_deg", 12.0)
        self.declare_parameter("joint_limit_margin_deg", 5.0)
        self.declare_parameter("fk_position_warn_m", 0.003)
        self.declare_parameter("fk_position_fault_m", 0.005)
        self.declare_parameter("max_fk_orientation_error_deg", 1.0)

        self.dry_run = bool(self.get_parameter("dry_run").value)
        self.enable_motion = bool(self.get_parameter("enable_motion").value)
        self.input_frame = str(self.get_parameter("input_frame").value)
        self.elbow_frame = str(self.get_parameter("elbow_frame").value)
        self.enable_arm_angle_constraint = bool(
            self.get_parameter("enable_arm_angle_constraint").value)
        validate_safety_mode(self.dry_run, self.enable_motion)

        self.mapper = RelativePoseMapper(
            axis_mapping=self.get_parameter("axis_mapping").value,
            position_scale=self.get_parameter("position_scale").value,
            max_position_jump_m=self.get_parameter("max_position_jump_m").value,
            max_rotation_jump_rad=math.radians(
                self.get_parameter("max_rotation_jump_deg").value),
            tracker_timeout_sec=self.get_parameter("tracker_timeout_sec").value,
            freeze_timeout_sec=self.get_parameter("freeze_timeout_sec").value,
            translation_only=self.get_parameter("translation_only").value,
            rotation_only=self.get_parameter("rotation_only").value,
            rotation_scale=self.get_parameter("rotation_scale").value,
            workspace_min=self.get_parameter("workspace_min").value,
            workspace_max=self.get_parameter("workspace_max").value,
        )
        self.robot = None
        self.ik_evaluator = None
        self._connected = False
        self._stale_reported = False
        self._message_count = 0
        self._accepted_count = 0
        self._rejection_counts = {}
        self._rate_start = time.monotonic()
        self._last_callback_time = None
        self._max_joint_delta = 0.0
        self._max_arm_angle_delta = 0.0
        self._max_fk_position_error = 0.0
        self._max_fk_orientation_error = 0.0
        self._latest_elbow_direction = None
        self.arm_angle_mapper = None

        self._connect_and_read_initial_state()

        topic = str(self.get_parameter("input_topic").value)
        self.target_publisher = self.create_publisher(
            PoseStamped, "/rm75/dry_run_target_pose", 10)
        self.subscription = self.create_subscription(
            PoseStamped, topic, self._pose_callback, 10)
        elbow_topic = str(self.get_parameter("elbow_topic").value)
        self.elbow_subscription = self.create_subscription(
            Vector3Stamped, elbow_topic, self._elbow_callback, 10)
        self.watchdog = self.create_timer(0.05, self._watchdog_callback)
        self.get_logger().warning(
            "DRY-RUN ONLY: ordinary IK/FK and arm-angle observation; no command path"
        )
        self.get_logger().warning(
            "axis_mapping converts WUJI right-chest deltas to RM75 base; "
            "physical single-axis verification is still required"
        )
        if self.mapper.translation_only:
            self.get_logger().warning(
                "TRANSLATION-ONLY: RM75 target orientation is locked to startup pose")
        if self.mapper.rotation_only:
            self.get_logger().warning(
                "ROTATION-ONLY: RM75 target position is locked; rotation_scale="
                f"{self.mapper.rotation_scale:.3f}")
        self.get_logger().info(
            f"Subscribed to WUJI target topic {topic} in {self.input_frame} frame")
        self.get_logger().info(
            f"Subscribed to WUJI elbow topic {elbow_topic} in {self.elbow_frame} frame; "
            f"arm-angle constraint={self.enable_arm_angle_constraint}")

    def _connect_and_read_initial_state(self):
        ip = str(self.get_parameter("robot_ip").value)
        port = int(self.get_parameter("robot_port").value)
        self.robot = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        handle = self.robot.rm_create_robot_arm(ip, port)
        self.get_logger().info(f"Read-only RM75 connection handle.id={handle.id}")
        if handle.id < 0:
            raise RuntimeError(f"RM75 connection failed: handle.id={handle.id}")
        self._connected = True

        ret, state = self.robot.rm_get_current_arm_state()
        if ret != 0:
            raise RuntimeError(f"RM75 state query failed: return code {ret}")
        pose = np.asarray(state.get("pose", []), dtype=float)
        joints = np.asarray(state.get("joint", []), dtype=float)
        if pose.shape != (6,) or not np.all(np.isfinite(pose)):
            raise RuntimeError(f"RM75 returned invalid pose: {pose}")
        if joints.shape != (7,) or not np.all(np.isfinite(joints)):
            raise RuntimeError(f"RM75 returned invalid joints: {joints}")

        # Official conversion returns [w,x,y,z]; mapper/ROS use [x,y,z,w].
        initial_wxyz = self.robot.rm_algo_euler2quaternion(pose[3:6].tolist())
        initial_xyzw = quaternion_wxyz_to_xyzw(initial_wxyz)
        self.mapper.set_rm_initial_pose(pose[:3], initial_xyzw)

        joint_min = self.robot.rm_algo_get_joint_min_limit()
        joint_max = self.robot.rm_algo_get_joint_max_limit()
        self.ik_evaluator = IKDryRunEvaluator(
            algorithm=RealManAlgorithmAdapter(self.robot),
            initial_joints_deg=joints,
            joint_min_deg=joint_min,
            joint_max_deg=joint_max,
            max_joint_jump_deg=self.get_parameter("max_joint_jump_deg").value,
            joint_limit_margin_deg=self.get_parameter("joint_limit_margin_deg").value,
            max_fk_position_error_m=self.get_parameter(
                "fk_position_warn_m").value,
            max_fk_orientation_error_rad=math.radians(
                self.get_parameter("max_fk_orientation_error_deg").value),
            fk_position_warn_m=self.get_parameter(
                "fk_position_warn_m").value,
            fk_position_fault_m=self.get_parameter(
                "fk_position_fault_m").value,
        )
        self.get_logger().info(
            "RM75 initial pose [x,y,z,rx,ry,rz]=" + self._fmt(pose))
        self.get_logger().info("RM75 initial joints deg=" + self._fmt(joints))
        self.get_logger().info("Algorithm joint minimum deg=" + self._fmt(joint_min))
        self.get_logger().info("Algorithm joint maximum deg=" + self._fmt(joint_max))
        arm_code, arm_angle = self.ik_evaluator.algorithm.arm_angle(joints.tolist())
        if arm_code != 0:
            raise RuntimeError(f"RM75 initial arm-angle calculation failed: {arm_code}")
        self.arm_angle_mapper = RelativeArmAngleMapper(
            rm_initial_angle_deg=arm_angle,
            scale=self.get_parameter("arm_angle_scale").value,
            max_step_deg=self.get_parameter("max_arm_angle_step_deg").value,
        )
        self.get_logger().info(f"RM75 initial arm angle deg={float(arm_angle):.6f}")

    @staticmethod
    def _fmt(values):
        return "[" + ", ".join(f"{float(v):.6f}" for v in values) + "]"

    def _reject(self, reason):
        self._rejection_counts[reason] = self._rejection_counts.get(reason, 0) + 1
        self.get_logger().warning(f"REJECTED dry-run frame: {reason}")

    def _elbow_callback(self, msg: Vector3Stamped):
        if msg.header.frame_id != self.elbow_frame:
            self._reject(f"unexpected elbow frame_id={msg.header.frame_id!r}")
            return
        direction = np.array([msg.vector.x, msg.vector.y, msg.vector.z], dtype=float)
        if direction.shape == (3,) and np.all(np.isfinite(direction)):
            self._latest_elbow_direction = direction

    def _pose_callback(self, msg: PoseStamped):
        callback_time = time.monotonic()
        self._message_count += 1
        if msg.header.frame_id != self.input_frame:
            self._reject(f"unexpected frame_id={msg.header.frame_id!r}")
            return
        position = np.array([msg.pose.position.x, msg.pose.position.y,
                             msg.pose.position.z], dtype=float)
        quaternion = np.array([msg.pose.orientation.x, msg.pose.orientation.y,
                               msg.pose.orientation.z, msg.pose.orientation.w], dtype=float)
        try:
            mapped = self.mapper.process(position, quaternion, callback_time)
        except (InvalidPose, ValueError) as exc:
            self._reject(str(exc))
            return

        target_arm_angle = None
        if self.enable_arm_angle_constraint:
            if self._latest_elbow_direction is None:
                self._reject("waiting for elbow direction")
                return
            try:
                target_arm_angle = self.arm_angle_mapper.process(
                    position, self._latest_elbow_direction)
            except InvalidElbowDirection as exc:
                self._reject(str(exc))
                return

        ik = self.ik_evaluator.evaluate(
            mapped.target_position, mapped.target_quaternion_xyzw,
            target_arm_angle_deg=target_arm_angle)
        if not ik.accepted:
            self._reject(ik.reason)
            return

        self._accepted_count += 1
        self._max_joint_delta = max(self._max_joint_delta, ik.max_joint_delta_deg)
        if np.isfinite(ik.arm_angle_delta_deg):
            self._max_arm_angle_delta = max(
                self._max_arm_angle_delta, ik.arm_angle_delta_deg)
        self._max_fk_position_error = max(
            self._max_fk_position_error, ik.fk_position_error_m)
        self._max_fk_orientation_error = max(
            self._max_fk_orientation_error, ik.fk_orientation_error_rad)

        elapsed = max(callback_time - self._rate_start, 1e-9)
        average_hz = self._message_count / elapsed
        instant_hz = 0.0 if self._last_callback_time is None else (
            1.0 / max(callback_time - self._last_callback_time, 1e-9))
        self._last_callback_time = callback_time
        stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        latency_ms = float("nan") if stamp_sec <= 0.0 else max(
            0.0, (self.get_clock().now().nanoseconds * 1e-9 - stamp_sec) * 1000.0)
        timestamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        self.get_logger().info(
            f"timestamp_ns={timestamp_ns} wuji_target_pos={self._fmt(position)} "
            f"wuji_target_quat_xyzw={self._fmt(quaternion)}")
        self.get_logger().info(
            f"rm_target_pos={self._fmt(mapped.target_position)} "
            f"rm_target_quat_xyzw={self._fmt(mapped.target_quaternion_xyzw)}")
        self.get_logger().info(
            f"ik_code={ik.ik_return_code} joints_deg={self._fmt(ik.target_joints_deg)} "
            f"max_joint_delta_deg={ik.max_joint_delta_deg:.6f} "
            f"target_arm_angle_deg="
            f"{target_arm_angle if target_arm_angle is not None else float('nan'):.6f} "
            f"arm_angle_deg={ik.arm_angle_deg:.6f} "
            f"arm_angle_delta_deg={ik.arm_angle_delta_deg:.6f}")
        self.get_logger().info(
            f"fk_position_error_m={ik.fk_position_error_m:.9f} "
            f"fk_orientation_error_rad={ik.fk_orientation_error_rad:.9f} "
            f"near_joint_limit={ik.near_joint_limit} instant_hz={instant_hz:.2f} "
            f"average_hz={average_hz:.2f} latency_ms={latency_ms:.3f}")

        output = PoseStamped()
        output.header = msg.header
        output.header.frame_id = "rm75_base_dry_run"
        output.pose.position.x, output.pose.position.y, output.pose.position.z = (
            mapped.target_position.tolist())
        output.pose.orientation.x, output.pose.orientation.y, \
            output.pose.orientation.z, output.pose.orientation.w = (
                mapped.target_quaternion_xyzw.tolist())
        self.target_publisher.publish(output)
        self._stale_reported = False

    def _watchdog_callback(self):
        if self.mapper.last_message_time is None:
            return
        if self.mapper.tracker_timed_out() and not self._stale_reported:
            self._stale_reported = True
            self.get_logger().error(
                "Tracker timeout: dry-run target is stale and no update is published")

    def _log_summary(self):
        success_rate = (100.0 * self._accepted_count / self._message_count
                        if self._message_count else 0.0)
        self.get_logger().info(
            f"DRY-RUN SUMMARY frames={self._message_count} accepted={self._accepted_count} "
            f"success_rate={success_rate:.2f}% rejected={self._message_count-self._accepted_count} "
            f"max_joint_delta_deg={self._max_joint_delta:.6f} "
            f"max_arm_angle_delta_deg={self._max_arm_angle_delta:.6f} "
            f"max_fk_position_error_m={self._max_fk_position_error:.9f} "
            f"max_fk_orientation_error_rad={self._max_fk_orientation_error:.9f} "
            f"rejection_reasons={self._rejection_counts}")

    def close(self):
        self._log_summary()
        if self.robot is not None and self._connected:
            ret = self.robot.rm_delete_robot_arm()
            self.get_logger().info(f"Read-only RM75 disconnect return code={ret}")
            self._connected = False

    def destroy_node(self):
        self.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = RealManRM75OutputNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
