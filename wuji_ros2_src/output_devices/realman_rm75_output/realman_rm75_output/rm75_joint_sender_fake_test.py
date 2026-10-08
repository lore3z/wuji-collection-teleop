"""Dual-tracker IK producer plus independent 125 Hz joint sender trials."""

import argparse
import json
import math
import sys
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e

from .elbow_arm_angle_mapper import InvalidElbowDirection, RelativeArmAngleMapper
from .coordinate_frames import wuji_right_chest_to_rm_base_flat
from .ik_dry_run import IKDryRunEvaluator
from .joint_sender import FakeJointBackend, JointSender, LatestJointTarget
from .movej_backend import RealFollowJointBackend
from .pose_mapper import InvalidPose, RelativePoseMapper, quaternion_wxyz_to_xyzw
from .realman_rm75_output_node import RealManAlgorithmAdapter


class RM75JointSenderFakeTest(Node):
    JOINT_VELOCITY_LIMIT_DEG_S = 45.0
    JOINT_ACCELERATION_LIMIT_DEG_S2 = 150.0
    JOINT_LIMIT_MARGIN_DEG = 5.0
    RATE_HZ = 125.0

    def __init__(self, real_motion=False, robot_ip="192.168.1.18",
                 robot_port=8080, publish_shadow=False):
        super().__init__(
            "rm75_kinematic_shadow" if publish_shadow else
            ("rm75_joint_sender_follow_trial" if real_motion
             else "rm75_joint_sender_fake_test"))
        self.declare_parameter("fk_position_warn_m", 0.003)
        self.declare_parameter("fk_position_fault_m", 0.005)
        self.declare_parameter("fk_position_max_consecutive_warnings", 5)
        self.declare_parameter("enable_arm_angle_constraint", True)
        self.declare_parameter("arm_angle_mode", "auto")
        self.declare_parameter("input_topic", "/right_arm_target_pose")
        self.declare_parameter("input_frame", "world_right")
        self.declare_parameter("input_pose_mode", "relative_wuji")
        self.declare_parameter("robot_ip", robot_ip)
        self.declare_parameter("robot_port", robot_port)
        self.real_motion = bool(real_motion)
        self.publish_shadow = bool(publish_shadow)
        self.input_topic = str(self.get_parameter("input_topic").value)
        self.input_frame = str(self.get_parameter("input_frame").value)
        self.input_pose_mode = str(
            self.get_parameter("input_pose_mode").value).strip().lower()
        if self.input_pose_mode not in ("relative_wuji", "absolute_rm"):
            raise ValueError("input_pose_mode must be relative_wuji or absolute_rm")
        self.enable_arm_angle_constraint = bool(
            self.get_parameter("enable_arm_angle_constraint").value)
        requested_arm_angle_mode = str(
            self.get_parameter("arm_angle_mode").value).strip().lower()
        if requested_arm_angle_mode == "auto":
            requested_arm_angle_mode = (
                "tracked" if self.enable_arm_angle_constraint else "ordinary")
        if requested_arm_angle_mode not in ("ordinary", "fixed", "tracked"):
            raise ValueError(
                "arm_angle_mode must be ordinary, fixed, tracked, or auto")
        self.arm_angle_mode = requested_arm_angle_mode
        self.uses_elbow_tracker = self.arm_angle_mode == "tracked"
        self.max_duration = 30.0 if self.real_motion else None
        self.active_started_at = None
        self.phase = "WAITING_ENTER"
        self.start_event = threading.Event()
        self.stop_event = threading.Event()
        self.latest_elbow = None
        self.latest_pose_at = None
        self.latest_elbow_at = None
        self.consecutive_elbow_rejections = 0
        self.consecutive_fk_warnings = 0
        self.fk_warning_count = 0
        self.produced_count = 0
        self.rejected_count = 0
        self.ik_compute_ms = []
        self.previous_ik_target = None
        self.shadow_target_joints = None
        self.shadow_target_position = None
        self.shadow_target_quaternion = None
        self.shadow_fk_pose_wxyz = None
        self.connected = False

        self.robot = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        handle = self.robot.rm_create_robot_arm(
            str(self.get_parameter("robot_ip").value),
            int(self.get_parameter("robot_port").value))
        if handle.id < 0:
            raise RuntimeError(f"RM75 read-only connection failed: {handle.id}")
        self.connected = True
        ret, state = self.robot.rm_get_current_arm_state()
        if ret != 0:
            raise RuntimeError(f"RM75 state query failed: {ret}")
        pose = np.asarray(state.get("pose", []), dtype=float)
        joints = np.asarray(state.get("joint", []), dtype=float)
        if pose.shape != (6,) or joints.shape != (7,):
            raise RuntimeError("RM75 returned invalid initial state")
        initial_q = quaternion_wxyz_to_xyzw(
            self.robot.rm_algo_euler2quaternion(pose[3:6].tolist()))
        self.mapper = RelativePoseMapper(
            wuji_right_chest_to_rm_base_flat(), 1.0,
            0.10, math.radians(45), [-0.8, -0.8, -0.1],
            [0.8, 0.8, 1.2], 0.5, freeze_timeout_sec=60.0,
            translation_only=False, rotation_scale=1.0)
        self.mapper.set_rm_initial_pose(pose[:3], initial_q)
        algorithm = RealManAlgorithmAdapter(self.robot)
        self.initial_arm_angle = float("nan")
        self.arm_mapper = None
        if self.arm_angle_mode in ("fixed", "tracked"):
            arm_code, arm_angle = algorithm.arm_angle(joints.tolist())
            if arm_code != 0:
                raise RuntimeError(f"initial arm-angle failed: {arm_code}")
            self.initial_arm_angle = float(arm_angle)
            self.arm_mapper = self._new_arm_mapper()
        joint_min = np.asarray(
            self.robot.rm_algo_get_joint_min_limit(), dtype=float)
        joint_max = np.asarray(
            self.robot.rm_algo_get_joint_max_limit(), dtype=float)
        sender_joint_min = joint_min + self.JOINT_LIMIT_MARGIN_DEG
        sender_joint_max = joint_max - self.JOINT_LIMIT_MARGIN_DEG
        self.ik = IKDryRunEvaluator(
            algorithm, joints, joint_min, joint_max,
            12.0, self.JOINT_LIMIT_MARGIN_DEG,
            self.get_parameter("fk_position_warn_m").value,
            math.radians(1.0),
            fk_position_warn_m=self.get_parameter(
                "fk_position_warn_m").value,
            fk_position_fault_m=self.get_parameter(
                "fk_position_fault_m").value)
        self.max_consecutive_fk_warnings = int(self.get_parameter(
            "fk_position_max_consecutive_warnings").value)
        if self.max_consecutive_fk_warnings < 1:
            raise ValueError(
                "fk_position_max_consecutive_warnings must be >= 1")

        self.target_buffer = LatestJointTarget()
        self.backend = (RealFollowJointBackend(
                            self.robot, joints,
                            sender_joint_min, sender_joint_max)
                        if self.real_motion else FakeJointBackend())
        self.sender = JointSender(
            self.target_buffer, self.backend, joints,
            rate_hz=self.RATE_HZ,
            max_speed_deg_s=self.JOINT_VELOCITY_LIMIT_DEG_S,
            max_accel_deg_s2=self.JOINT_ACCELERATION_LIMIT_DEG_S2,
            max_step_deg=(self.JOINT_VELOCITY_LIMIT_DEG_S /
                          self.RATE_HZ),
            min_position_deg=sender_joint_min,
            max_position_deg=sender_joint_max)
        self.sender_started = False
        # A real motion API must not be called before explicit operator Enter.
        if not self.real_motion:
            self.sender.start()
            self.sender_started = True

        self.create_subscription(PoseStamped, self.input_topic,
                                 self._pose, 10)
        if self.uses_elbow_tracker:
            self.create_subscription(
                Vector3Stamped, "/right_arm_elbow_direction", self._elbow, 10)
        self.create_timer(0.05, self._watchdog)
        if self.publish_shadow:
            self.shadow_joint_pub = self.create_publisher(
                JointState, "/rm75_shadow/q_target_joint_states", 10)
            self.shadow_command_pub = self.create_publisher(
                JointState, "/rm75_shadow/q_cmd_joint_states", 10)
            self.shadow_target_pose_pub = self.create_publisher(
                PoseStamped, "/rm75_shadow/target_pose", 10)
            self.shadow_fk_pose_pub = self.create_publisher(
                PoseStamped, "/rm75_shadow/fk_pose", 10)
            self.create_timer(1.0 / 60.0, self._publish_shadow)
            self.start_event.set()
        else:
            threading.Thread(target=self._keyboard, daemon=True).start()
        self.get_logger().warning(
            ("REAL FOLLOW=True TELEOP: 125 Hz, RM75 joint limits with "
             "5 deg margin, 45 deg/s, 150 deg/s^2" if self.real_motion else
             "FAKE BACKEND ONLY: ~90 Hz dual-tracker IK producer -> "
             "125 Hz independent JointSender"))
        self.get_logger().info(
            "KINEMATIC SHADOW auto-rebase; Ctrl+C to stop" if
            self.publish_shadow else
            "Press Enter to REBASE+START; move both trackers; "
            "press Enter again to STOP+HOLD")
        self.get_logger().warning(
            "ARM-ANGLE MODE: " + {
                "ordinary": "ORDINARY (wrist only; previous-q IK)",
                "fixed": ("FIXED (wrist only; startup arm angle="
                          f"{self.initial_arm_angle:.3f} deg)"),
                "tracked": "TRACKED (wrist + elbow topic)",
            }[self.arm_angle_mode])

    def _new_arm_mapper(self):
        return RelativeArmAngleMapper(
            self.initial_arm_angle, scale=0.25, max_step_deg=5.0)

    def _keyboard(self):
        sys.stdin.readline()
        self.start_event.set()
        sys.stdin.readline()
        self.stop_event.set()

    def _elbow(self, msg):
        if msg.header.frame_id != "world_right":
            self._fault("unexpected elbow frame")
            return
        value = np.asarray([msg.vector.x, msg.vector.y, msg.vector.z], dtype=float)
        if np.all(np.isfinite(value)):
            self.latest_elbow = value
            self.latest_elbow_at = time.monotonic()

    def _pose(self, msg):
        if self.phase not in ("REBASING", "ACTIVE"):
            return
        started = time.monotonic()
        try:
            if msg.header.frame_id != self.input_frame:
                raise InvalidPose("unexpected target frame")
            if self.uses_elbow_tracker and self.latest_elbow is None:
                return
            position = np.asarray([
                msg.pose.position.x, msg.pose.position.y,
                msg.pose.position.z], dtype=float)
            quaternion = np.asarray([
                msg.pose.orientation.x, msg.pose.orientation.y,
                msg.pose.orientation.z, msg.pose.orientation.w], dtype=float)
            if self.input_pose_mode == "absolute_rm":
                norm = float(np.linalg.norm(quaternion))
                if norm < 1e-9:
                    raise InvalidPose("zero target quaternion")
                target_position = position
                target_quaternion = quaternion / norm
            else:
                mapped = self.mapper.process(position, quaternion, started)
                target_position = mapped.target_position
                target_quaternion = mapped.target_quaternion_xyzw
            if self.arm_angle_mode == "tracked":
                target_arm_angle = self.arm_mapper.process(
                    position, self.latest_elbow)
            elif self.arm_angle_mode == "fixed":
                target_arm_angle = self.initial_arm_angle
            else:
                target_arm_angle = None
            # Producer seed follows the command actually generated by the
            # consumer, never an unconstrained future IK target.
            self.ik.reference_joints = self.sender.current_command()
            result = self.ik.evaluate(
                target_position, target_quaternion,
                target_arm_angle_deg=target_arm_angle,
                joint_continuity_reference_deg=(
                    self.previous_ik_target
                    if self.previous_ik_target is not None
                    else self.sender.current_command()))
            if not result.accepted:
                # Shadow mode is deliberately permissive: every finite IK
                # candidate is visualized even when real-motion guards would
                # reject it. This branch is impossible in real_motion mode.
                if self.publish_shadow:
                    self.rejected_count += 1
                    self.latest_pose_at = started
                    candidate = result.target_joints_deg
                    if (candidate is not None and
                            np.asarray(candidate).shape == (7,) and
                            np.all(np.isfinite(candidate))):
                        self.previous_ik_target = candidate.copy()
                        self._set_shadow_candidate(
                            candidate, target_position, target_quaternion)
                        self.target_buffer.publish(candidate, started)
                        self.produced_count += 1
                    if self.rejected_count == 1 or \
                            self.rejected_count % 90 == 0:
                        self.get_logger().warning(
                            "SHADOW BYPASS (NO ROBOT MOTION): " +
                            result.reason)
                    return
                if result.safety_action == "hold":
                    self.rejected_count += 1
                    self.fk_warning_count += 1
                    self.consecutive_fk_warnings += 1
                    self.latest_pose_at = started
                    if (self.consecutive_fk_warnings >
                            self.max_consecutive_fk_warnings):
                        raise RuntimeError(
                            f"FK residual warning persisted for "
                            f"{self.consecutive_fk_warnings} frames: "
                            f"{result.reason}")
                    if self.consecutive_fk_warnings in (
                            1, self.max_consecutive_fk_warnings):
                        self.get_logger().warning(result.reason)
                    return
                raise RuntimeError(result.reason)
            self.consecutive_fk_warnings = 0
            self.previous_ik_target = result.target_joints_deg.copy()
            if self.publish_shadow:
                self._set_shadow_candidate(
                    result.target_joints_deg, target_position,
                    target_quaternion)
            self.target_buffer.publish(result.target_joints_deg, started)
            self.produced_count += 1
            self.latest_pose_at = started
            if self.uses_elbow_tracker:
                self.consecutive_elbow_rejections = 0
            if self.phase == "REBASING":
                self.phase = "ACTIVE"
                self.active_started_at = time.monotonic()
                self.get_logger().warning(
                    "ACTIVE REAL LIMITED FOLLOW TRIAL" if self.real_motion else
                    ("ACTIVE KINEMATIC SHADOW" if self.publish_shadow else
                     "ACTIVE FAKE producer-consumer test"))
        except InvalidElbowDirection as exc:
            self.rejected_count += 1
            self.consecutive_elbow_rejections += 1
            self.latest_pose_at = started
            if self.consecutive_elbow_rejections > 5:
                self._fault(f"elbow invalid for 6 frames: {exc}")
        except (InvalidPose, ValueError, RuntimeError) as exc:
            self.rejected_count += 1
            self._fault(exc)
        finally:
            self.ik_compute_ms.append((time.monotonic() - started) * 1000.0)

    def _set_shadow_candidate(self, joints_deg, target_position,
                              target_quaternion_xyzw):
        self.shadow_target_joints = np.asarray(joints_deg, dtype=float).copy()
        self.shadow_target_position = np.asarray(
            target_position, dtype=float).copy()
        self.shadow_target_quaternion = np.asarray(
            target_quaternion_xyzw, dtype=float).copy()
        self.shadow_fk_pose_wxyz = np.asarray(
            self.ik.algorithm.forward(
                self.shadow_target_joints.tolist()), dtype=float)

    @staticmethod
    def _joint_state(stamp, joints_deg):
        msg = JointState()
        msg.header.stamp = stamp
        msg.name = [f"joint{index}" for index in range(1, 8)]
        msg.position = np.radians(joints_deg).tolist()
        return msg

    @staticmethod
    def _pose_stamped(stamp, position, quaternion_xyzw):
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = "base_link"
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = \
            np.asarray(position, dtype=float).tolist()
        (msg.pose.orientation.x, msg.pose.orientation.y,
         msg.pose.orientation.z, msg.pose.orientation.w) = \
            np.asarray(quaternion_xyzw, dtype=float).tolist()
        return msg

    def _publish_shadow(self):
        if not self.publish_shadow or self.shadow_target_joints is None:
            return
        stamp = self.get_clock().now().to_msg()
        self.shadow_joint_pub.publish(self._joint_state(
            stamp, self.shadow_target_joints))
        self.shadow_command_pub.publish(self._joint_state(
            stamp, self.sender.current_command()))
        self.shadow_target_pose_pub.publish(self._pose_stamped(
            stamp, self.shadow_target_position,
            self.shadow_target_quaternion))
        fk = self.shadow_fk_pose_wxyz
        self.shadow_fk_pose_pub.publish(self._pose_stamped(
            stamp, fk[:3], quaternion_wxyz_to_xyzw(fk[3:7])))

    def _watchdog(self):
        if self.phase == "WAITING_ENTER" and self.start_event.is_set():
            if not self.sender_started:
                self.sender.start()
                self.sender_started = True
            self.phase = "REBASING"
            if self.input_pose_mode == "relative_wuji":
                self.mapper.reset_tracker_reference()
            if self.uses_elbow_tracker:
                self.arm_mapper = self._new_arm_mapper()
                self.latest_elbow = None
            self.latest_pose_at = None
            self.previous_ik_target = None
            self.get_logger().warning(
                "WAITING FOR ABSOLUTE RM TCP TARGET" if
                self.input_pose_mode == "absolute_rm" else
                ("REBASING; hold both trackers still" if
                 self.uses_elbow_tracker else
                 "REBASING; hold the wrist Tracker still"))
            return
        if self.phase == "ACTIVE" and self.stop_event.is_set():
            self.phase = "STOP"
            self.target_buffer.stop("operator Enter")
            self.get_logger().warning(
                "STOP: rejecting new Tracker targets and holding current q_cmd")
            return
        if self.phase == "ACTIVE":
            now = time.monotonic()
            buffer_state, buffer_reason, _ = self.target_buffer.snapshot()
            if buffer_state == "FAULT":
                self._fault(buffer_reason)
                return
            if (self.max_duration is not None and self.active_started_at is not None
                    and now - self.active_started_at >= self.max_duration):
                self.phase = "STOP"
                self.target_buffer.stop("30 second maximum")
                self.get_logger().warning(
                    "STOP: 30 second maximum; holding current q_cmd")
                return
            elbow_stale = (
                self.uses_elbow_tracker and
                (self.latest_elbow_at is None or
                 now - self.latest_elbow_at > 0.5))
            if (self.latest_pose_at is None or
                    now - self.latest_pose_at > 0.5 or elbow_stale):
                self._fault("PICO input timeout")

    def _fault(self, reason):
        if self.publish_shadow:
            self.get_logger().warning(
                f"SHADOW HOLD (NO ROBOT MOTION): {reason}")
            return
        if self.phase in ("STOP", "FAULT"):
            return
        self.phase = "FAULT"
        self.target_buffer.fault(reason)
        self.get_logger().error(
            f"FAULT: {reason}; sender holds current q_cmd" +
            ("; REAL backend" if self.real_motion else "; no robot backend"))

    @staticmethod
    def _stats_ms(values):
        if not values:
            return {}
        data = np.asarray(values, dtype=float)
        return {
            "mean": float(np.mean(data)),
            "p95": float(np.percentile(data, 95)),
            "p99": float(np.percentile(data, 99)),
            "max": float(np.max(data)),
        }

    def close(self):
        self.target_buffer.stop("node close")
        # Keep emitting the frozen q_cmd briefly before ending high-follow.
        time.sleep(0.10 if self.real_motion else 0.05)
        if self.sender_started:
            self.sender.stop()
        summary = self.sender.metrics()
        summary.update({
            "phase": self.phase,
            "producer_targets": self.produced_count,
            "producer_rejections": self.rejected_count,
            "fk_warning_frames": self.fk_warning_count,
            "ik_compute_ms": self._stats_ms(self.ik_compute_ms),
        })
        if self.real_motion and self.sender_started:
            summary["max_command_offset_deg"] = \
                self.backend.max_observed_offset
            summary["terminal_stop_code"] = (
                self.backend.fault_stop() if self.phase == "FAULT"
                else self.backend.hold_stop())
        self.get_logger().warning(
            "JOINT_SENDER_SUMMARY " + json.dumps(summary, sort_keys=True))
        if self.connected:
            self.robot.rm_delete_robot_arm()
            self.connected = False

    def destroy_node(self):
        self.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RM75JointSenderFakeTest()
    try:
        while rclpy.ok() and node.phase not in ("STOP", "FAULT"):
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        node.target_buffer.stop("KeyboardInterrupt")
        node.phase = "STOP"
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


def real_main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-ip", default="192.168.1.18")
    parser.add_argument("--robot-port", type=int, default=8080)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--enable-motion", action="store_true")
    args, ros_args = parser.parse_known_args(argv)
    if not (args.execute and args.enable_motion):
        print("REFUSED: real motion requires --execute --enable-motion",
              file=sys.stderr)
        return 2
    rclpy.init(args=ros_args)
    node = None
    try:
        node = RM75JointSenderFakeTest(
            True, args.robot_ip, args.robot_port)
        while rclpy.ok() and node.phase not in ("STOP", "FAULT"):
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        if node is not None:
            node.target_buffer.stop("KeyboardInterrupt")
            node.phase = "STOP"
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


def shadow_main(args=None):
    rclpy.init(args=args)
    node = RM75JointSenderFakeTest(publish_shadow=True)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.target_buffer.stop("KeyboardInterrupt")
        node.phase = "STOP"
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
