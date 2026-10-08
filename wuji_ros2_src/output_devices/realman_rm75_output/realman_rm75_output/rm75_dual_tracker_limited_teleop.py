"""First real dual-tracker RM75 trial, hard-limited to +/-1 degree per joint."""

import argparse
import math
import sys
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from rclpy.node import Node
from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e

from .elbow_arm_angle_mapper import InvalidElbowDirection, RelativeArmAngleMapper
from .coordinate_frames import wuji_right_chest_to_rm_base_flat
from .ik_dry_run import IKDryRunEvaluator
from .joint_command_limiter import JointCommandLimiter
from .movej_backend import RealMoveJBackend
from .pose_mapper import InvalidPose, RelativePoseMapper, quaternion_wxyz_to_xyzw
from .readonly_preflight import collect_read_only_audit
from .realman_rm75_output_node import RealManAlgorithmAdapter


class RM75DualTrackerLimitedTeleop(Node):
    RATE_HZ = 50.0
    MAX_DURATION = 10.0
    INPUT_TIMEOUT = 0.5

    def __init__(self, robot_ip, robot_port):
        super().__init__("rm75_dual_tracker_limited_teleop")
        self.period = 1.0 / self.RATE_HZ
        self.phase = "WAITING_ENTER"
        self.start_event = threading.Event()
        self.stop_event = threading.Event()
        self.latest_elbow = None
        self.latest_pose_at = None
        self.latest_elbow_at = None
        self.desired_joints = None
        self.started_at = None
        self.return_started_at = None
        self.connected = False
        self.stop_sent = False
        self.sent_count = 0
        self.max_command_offset = 0.0
        self.max_command_step = 0.0
        self.max_feedback_offset = 0.0
        self.elbow_rejection_count = 0
        self.consecutive_elbow_rejections = 0
        self.last_feedback_joints = None
        self.actual_joints = None
        self.last_command_joints = None

        self.robot = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        handle = self.robot.rm_create_robot_arm(robot_ip, robot_port)
        if handle.id < 0:
            raise RuntimeError(f"RM75 connection failed: handle.id={handle.id}")
        self.connected = True
        audit = collect_read_only_audit(self.robot)
        state = audit["rm_get_current_arm_state"][1]
        controller = audit["rm_get_controller_state"]
        if controller.get("system_error", controller.get("sys_err", 0)) != 0:
            raise RuntimeError("RM75 controller error is nonzero")
        if any(int(value) != 0 for value in
               audit["rm_get_joint_err_flag"].get("err_flag", [])):
            raise RuntimeError("RM75 joint error is nonzero")
        pose = np.asarray(state.get("pose", []), dtype=float)
        joints = np.asarray(state.get("joint", []), dtype=float)
        if pose.shape != (6,) or joints.shape != (7,):
            raise RuntimeError("RM75 returned invalid initial state")
        self.initial_joints = joints.copy()
        self.last_feedback_joints = joints.copy()
        self.actual_joints = joints.copy()
        self.last_command_joints = joints.copy()
        initial_q = quaternion_wxyz_to_xyzw(
            self.robot.rm_algo_euler2quaternion(pose[3:6].tolist()))

        self.mapper = RelativePoseMapper(
            axis_mapping=wuji_right_chest_to_rm_base_flat(),
            position_scale=0.15,
            max_position_jump_m=0.10,
            max_rotation_jump_rad=math.radians(45),
            workspace_min=[-0.8, -0.8, -0.1],
            workspace_max=[0.8, 0.8, 1.2],
            tracker_timeout_sec=self.INPUT_TIMEOUT,
            freeze_timeout_sec=60.0,
            translation_only=False,
            rotation_scale=0.20)
        self.mapper.set_rm_initial_pose(pose[:3], initial_q)
        algorithm = RealManAlgorithmAdapter(self.robot)
        arm_code, initial_arm_angle = algorithm.arm_angle(joints.tolist())
        if arm_code != 0:
            raise RuntimeError(f"initial RM75 arm-angle failed: {arm_code}")
        self.initial_arm_angle = float(initial_arm_angle)
        self.arm_mapper = self._new_arm_mapper()
        self.ik = IKDryRunEvaluator(
            algorithm, joints,
            self.robot.rm_algo_get_joint_min_limit(),
            self.robot.rm_algo_get_joint_max_limit(),
            max_joint_jump_deg=12.0,
            joint_limit_margin_deg=5.0,
            max_fk_position_error_m=0.002,
            max_fk_orientation_error_rad=math.radians(1.0))
        self.limiter = JointCommandLimiter(
            joints, max_offset_deg=1.0, max_speed_deg_s=1.0,
            nominal_period_s=self.period)
        self.backend = RealMoveJBackend(self.robot)

        self.create_subscription(PoseStamped, "/right_arm_target_pose",
                                 self._pose, 10)
        self.create_subscription(Vector3Stamped, "/right_arm_elbow_direction",
                                 self._elbow, 10)
        self.create_timer(self.period, self._tick)
        threading.Thread(target=self._keyboard, daemon=True).start()
        self.get_logger().warning(
            "REAL LIMITED JOINT TRIAL: 50 Hz low-follow, each joint +/-1 deg, "
            "1 deg/s, 10 s maximum")
        self.get_logger().info(
            "Press Enter to REBASE+START; press Enter again to RETURN HOME+STOP")

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
            if self.phase == "ACTIVE":
                self._fault("unexpected elbow frame")
            return
        value = np.asarray([msg.vector.x, msg.vector.y, msg.vector.z], dtype=float)
        if np.all(np.isfinite(value)):
            self.latest_elbow = value
            self.latest_elbow_at = time.monotonic()

    def _pose(self, msg):
        if self.phase not in ("REBASING", "ACTIVE"):
            return
        now = time.monotonic()
        try:
            if msg.header.frame_id != "world_right":
                raise InvalidPose("unexpected target frame")
            if self.latest_elbow is None:
                return
            position = np.asarray([
                msg.pose.position.x, msg.pose.position.y,
                msg.pose.position.z], dtype=float)
            quaternion = np.asarray([
                msg.pose.orientation.x, msg.pose.orientation.y,
                msg.pose.orientation.z, msg.pose.orientation.w], dtype=float)
            mapped = self.mapper.process(position, quaternion, now)
            arm_angle = self.arm_mapper.process(position, self.latest_elbow)
            result = self.ik.evaluate(
                mapped.target_position, mapped.target_quaternion_xyzw,
                target_arm_angle_deg=arm_angle)
            if not result.accepted:
                raise RuntimeError(result.reason)
            self.desired_joints = result.target_joints_deg.copy()
            self.latest_pose_at = now
        except InvalidElbowDirection as exc:
            # The arm-angle mapper has already absorbed the new direction as
            # its basis while holding the preceding target. Keep sending the
            # last bounded command; fault only if the stream stays invalid.
            self.elbow_rejection_count += 1
            self.consecutive_elbow_rejections += 1
            self.latest_pose_at = now
            if self.consecutive_elbow_rejections > 5 and self.phase == "ACTIVE":
                self._fault(
                    f"elbow direction invalid for "
                    f"{self.consecutive_elbow_rejections} consecutive frames: {exc}")
            elif self.consecutive_elbow_rejections == 1:
                self.get_logger().warning(
                    f"Holding last joint target for one rejected elbow frame: {exc}")
        except (InvalidPose, ValueError, RuntimeError) as exc:
            if self.phase == "ACTIVE":
                self._fault(exc)
        else:
            self.consecutive_elbow_rejections = 0

    def _feedback(self):
        ret, state = self.robot.rm_get_current_arm_state()
        if ret != 0:
            raise RuntimeError(f"feedback failed: {ret}")
        joints = np.asarray(state.get("joint", []), dtype=float)
        if joints.shape != (7,) or not np.all(np.isfinite(joints)):
            raise RuntimeError("invalid joint feedback")
        offset = float(np.max(np.abs(joints - self.initial_joints)))
        jump = float(np.max(np.abs(joints - self.last_feedback_joints)))
        self.max_feedback_offset = max(self.max_feedback_offset, offset)
        if offset > 1.20:
            raise RuntimeError(f"actual joint escaped 1.20 deg guard: {offset:.3f}")
        if jump > 0.50:
            raise RuntimeError(f"actual joint feedback jumped {jump:.3f} deg")
        self.last_feedback_joints = joints
        self.actual_joints = joints.copy()

    def _send(self, command):
        code = self.backend.send(command.tolist())
        if code != 0:
            raise RuntimeError(f"movej_canfd return code {code}")
        self.sent_count += 1
        self.max_command_offset = max(
            self.max_command_offset,
            float(np.max(np.abs(command - self.initial_joints))))
        self.max_command_step = max(
            self.max_command_step,
            float(np.max(np.abs(command - self.last_command_joints))))
        self.last_command_joints = command.copy()
        self.ik.reference_joints = command.copy()
        if self.sent_count % 5 == 0:
            self._feedback()

    def _begin_return(self, reason):
        if self.phase != "RETURNING":
            self.phase = "RETURNING"
            self.return_started_at = time.monotonic()
            self.get_logger().warning(f"RETURNING HOME: {reason}")

    def _finish(self, reason):
        if self.stop_sent:
            return
        self.stop_sent = True
        self.phase = "STOPPED"
        code = self.backend.slow_stop()
        self.get_logger().warning(
            f"STOPPED: {reason}; slow_stop={code}; sent={self.sent_count}; "
            f"max_command_offset_deg={self.max_command_offset:.6f}; "
            f"max_command_step_deg={self.max_command_step:.6f}; "
            f"max_feedback_offset_deg={self.max_feedback_offset:.6f}; "
            f"elbow_rejected_frames={self.elbow_rejection_count}")

    def _fault(self, reason):
        if self.stop_sent:
            return
        self.stop_sent = True
        self.phase = "FAULT"
        code = self.backend.emergency_stop()
        self.get_logger().error(
            f"FAULT: {reason}; emergency_stop={code}; sent={self.sent_count}; "
            f"max_command_offset_deg={self.max_command_offset:.6f}; "
            f"max_command_step_deg={self.max_command_step:.6f}; "
            f"max_feedback_offset_deg={self.max_feedback_offset:.6f}; "
            f"elbow_rejected_frames={self.elbow_rejection_count}")

    def _tick(self):
        now = time.monotonic()
        if self.phase == "WAITING_ENTER" and self.start_event.is_set():
            self.phase = "REBASING"
            self.mapper.reset_tracker_reference()
            self.arm_mapper = self._new_arm_mapper()
            self.latest_elbow = None
            self.latest_pose_at = None
            self.desired_joints = None
            self.ik.reference_joints = self.initial_joints.copy()
            self.get_logger().warning("REBASING; hold both trackers still")
            return
        if self.phase == "REBASING" and self.desired_joints is not None:
            self.phase = "ACTIVE"
            self.started_at = now
            self.get_logger().warning("ACTIVE REAL LIMITED JOINT TRIAL")
        if self.phase == "ACTIVE":
            if self.stop_event.is_set():
                self._begin_return("operator Enter")
            elif now - self.started_at >= self.MAX_DURATION:
                self._begin_return("10 second maximum")
            elif (self.latest_pose_at is None or self.latest_elbow_at is None or
                  now - self.latest_pose_at > self.INPUT_TIMEOUT or
                  now - self.latest_elbow_at > self.INPUT_TIMEOUT):
                self._fault("PICO input timeout")
                return
            else:
                try:
                    command = self.limiter.step(self.desired_joints)
                    self._send(command)
                except Exception as exc:
                    self._fault(exc)
                return
        if self.phase == "RETURNING":
            try:
                if now - self.return_started_at > 5.0:
                    raise RuntimeError("return-home exceeded 5 second limit")
                command = self.limiter.step_home()
                self._send(command)
                actual_home = float(np.max(np.abs(
                    self.actual_joints - self.initial_joints))) <= 0.05
                if self.limiter.at_home() and actual_home:
                    self._finish("returned to startup joints")
            except Exception as exc:
                self._fault(exc)

    def destroy_node(self):
        if self.phase in ("ACTIVE", "REBASING", "RETURNING"):
            self._fault("node shutdown before controlled completion")
        if self.connected:
            self.robot.rm_delete_robot_arm()
            self.connected = False
        return super().destroy_node()


def main(argv=None):
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
        node = RM75DualTrackerLimitedTeleop(args.robot_ip, args.robot_port)
        while rclpy.ok() and node.phase not in ("STOPPED", "FAULT"):
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
