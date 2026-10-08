"""Interactive calibrated PICO -> RM75 6DoF teleoperation trial."""

import argparse
import csv
import math
import os
import sys
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e

from .cartesian_pose_limiter import CartesianPoseLimiter
from .ik_dry_run import IKDryRunEvaluator
from .movep_backend import RealMovePBackend
from .pose_mapper import (InvalidPose, RelativePoseMapper,
                          quaternion_angle_rad, quaternion_wxyz_to_xyzw)
from .pose_ema_filter import PoseEMAFilter
from .readonly_preflight import collect_read_only_audit, run_base_x_preflight
from .realman_rm75_output_node import RealManAlgorithmAdapter


class RM756DoFTeleop(Node):
    def __init__(self, ip, port):
        super().__init__("rm75_6dof_teleop")
        self.period, self.timeout = 0.0125, 0.5
        self.phase, self.started_at = "WAITING_ENTER", None
        self.phase_started_at = time.monotonic()
        self.position_error_started_at = None
        self.position_error_soft_limit_m = 0.040
        self.position_error_hard_limit_m = 0.080
        self.position_error_grace_sec = 0.50
        self.position_error_hold_sec = 0.25
        self.start_event, self.stop_event = threading.Event(), threading.Event()
        self.latest, self.latest_at, self.stop_sent = None, None, False
        self.latest_raw_position = None
        self.latest_raw_quaternion = None
        self.latest_source_stamp_sec = None
        self.last_tick_at = None
        self.feedback_count = 0
        trace_name = f"rm75_ema_trace_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        self.trace_path = os.path.join("/tmp", trace_name)
        self.trace_file = open(self.trace_path, "w", newline="")
        self.trace = csv.writer(self.trace_file)
        self.trace.writerow([
            "monotonic_sec", "source_stamp_sec", "actual_tick_dt_sec",
            "ema_dt_sec", "translation_alpha", "rotation_alpha",
            "raw_pico_x", "raw_pico_y", "raw_pico_z",
            "raw_pico_qx", "raw_pico_qy", "raw_pico_qz", "raw_pico_qw",
            "relative_pico_x", "relative_pico_y", "relative_pico_z",
            "relative_pico_qx", "relative_pico_qy",
            "relative_pico_qz", "relative_pico_qw",
            "mapped_relative_x", "mapped_relative_y", "mapped_relative_z",
            "mapped_relative_qx", "mapped_relative_qy",
            "mapped_relative_qz", "mapped_relative_qw",
            "mapped_rm_x", "mapped_rm_y", "mapped_rm_z",
            "mapped_rm_qx", "mapped_rm_qy", "mapped_rm_qz", "mapped_rm_qw",
            "ema_rm_x", "ema_rm_y", "ema_rm_z",
            "ema_rm_qx", "ema_rm_qy", "ema_rm_qz", "ema_rm_qw",
            "sent_rm_x", "sent_rm_y", "sent_rm_z",
            "sent_rm_qx", "sent_rm_qy", "sent_rm_qz", "sent_rm_qw",
            "actual_rm_x", "actual_rm_y", "actual_rm_z",
            "actual_joint_1_deg", "actual_joint_2_deg", "actual_joint_3_deg",
            "actual_joint_4_deg", "actual_joint_5_deg", "actual_joint_6_deg",
            "actual_joint_7_deg",
        ])
        self.robot = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        handle = self.robot.rm_create_robot_arm(ip, port)
        if handle.id < 0:
            raise RuntimeError(f"connection failed: {handle.id}")
        self.connected = True
        audit = collect_read_only_audit(self.robot)
        state = audit["rm_get_current_arm_state"][1]
        if audit["rm_get_controller_state"].get("system_error", 0) != 0:
            raise RuntimeError("controller error is nonzero")
        if any(audit["rm_get_joint_err_flag"].get("err_flag", [])):
            raise RuntimeError("joint error is nonzero")
        preflight = run_base_x_preflight(
            self.robot, state, 0.005, 0.0005)
        if not preflight["passed"]:
            rejected = [point for point in preflight["points"]
                        if not point["accepted"]]
            details = "; ".join(
                f"offset={point['offset_m']*1000:+.1f}mm "
                f"reason={point['reason']} "
                f"joint_delta={point['max_joint_delta_deg']:.3f}deg "
                f"joint_margin={point['minimum_joint_margin_deg']:.3f}deg "
                f"fk_pos={point['fk_position_error_m']*1000:.3f}mm "
                f"fk_rot={math.degrees(point['fk_orientation_error_rad']):.3f}deg "
                f"self_collision={point['local_self_collision']}"
                for point in rejected)
            raise RuntimeError(
                f"startup-neighborhood preflight failed: {details}")
        pose = np.asarray(state["pose"], dtype=float)
        joints = np.asarray(state["joint"], dtype=float)
        self.initial_position = pose[:3].copy()
        self.initial_q = quaternion_wxyz_to_xyzw(
            self.robot.rm_algo_euler2quaternion(pose[3:6].tolist()))
        self.previous_joints = joints.copy()
        matrix = [0., 0., -1., -1., 0., 0., 0., 1., 0.]
        self.mapper = RelativePoseMapper(
            matrix, 1.0, 0.10, math.radians(45),
            # Mapping produces the unconstrained operator request.  The
            # Cartesian limiter below is the single owner of the +/-100 mm
            # command bound; wider mapper bounds avoid rejecting a request
            # before it can be clipped.
            self.initial_position - 0.500, self.initial_position + 0.500,
            self.timeout, translation_only=False, freeze_timeout_sec=60.0,
            rotation_only=False, rotation_scale=0.20)
        self.mapper.set_rm_initial_pose(self.initial_position, self.initial_q)
        self.ema = PoseEMAFilter(0.05, 0.05)
        self.ema.reset(self.initial_position, self.initial_q)
        self.limiter = CartesianPoseLimiter(
            self.initial_position, self.initial_q, 0.100, 0.050,
            math.radians(20.0), 0.2)
        self.ik = IKDryRunEvaluator(
            RealManAlgorithmAdapter(self.robot), joints,
            self.robot.rm_algo_get_joint_min_limit(),
            self.robot.rm_algo_get_joint_max_limit(),
            5.0, 5.0, 0.002, math.radians(1.0))
        baseline = self.ik.evaluate(self.initial_position, self.initial_q)
        if not baseline.accepted:
            raise RuntimeError(f"initial shadow IK failed: {baseline.reason}")
        self.backend = RealMovePBackend(self.robot, follow=False)
        qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                         history=QoSHistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(PoseStamped, "/pico/right_wrist/raw_pose",
                                 self._pose, qos)
        self.create_timer(self.period, self._tick)
        threading.Thread(target=self._keyboard, daemon=True).start()
        self.get_logger().warning(
            "REAL 6DoF: translation scale=1, +/-100 mm, 50 mm/s, "
            "200 mm/s^2; EMA tau=50 ms; 80 Hz; "
            "rotation scale=0.2, 20 deg/s; no time limit")
        self.get_logger().info(
            f"SESSION ZERO position={self.initial_position.tolist()} "
            f"quaternion_xyzw={self.initial_q.tolist()}")
        self.get_logger().info(f"EMA trace CSV: {self.trace_path}")
        self.get_logger().info("Enter START; Enter again RETURN TO ZERO AND STOP")

    def _keyboard(self):
        sys.stdin.readline(); self.start_event.set()
        sys.stdin.readline(); self.stop_event.set()

    def _pose(self, msg):
        now = time.monotonic()
        try:
            if msg.header.frame_id != "pico_tracking":
                raise InvalidPose("unexpected PICO frame")
            raw_position = np.asarray([
                msg.pose.position.x, msg.pose.position.y,
                msg.pose.position.z], dtype=float)
            raw_quaternion = np.asarray([
                msg.pose.orientation.x, msg.pose.orientation.y,
                msg.pose.orientation.z, msg.pose.orientation.w], dtype=float)
            self.latest = self.mapper.process(
                raw_position, raw_quaternion, now)
            self.latest_raw_position = raw_position
            self.latest_raw_quaternion = raw_quaternion
            self.latest_source_stamp_sec = (
                float(msg.header.stamp.sec) +
                float(msg.header.stamp.nanosec) * 1e-9)
            self.latest_at = now
        except Exception as exc:
            if self.phase == "ACTIVE": self._fault(exc)
            else:
                self.mapper.reset_tracker_reference(); self.latest = None

    def _stop(self, reason):
        if self.stop_sent: return
        self.stop_sent, self.phase = True, "STOPPED"
        code = self.backend.slow_stop()
        self.get_logger().warning(f"STOPPED: {reason}; return_code={code}")

    def _begin_return(self, reason):
        if self.phase != "RETURNING":
            self.phase = "RETURNING"
            self.phase_started_at = time.monotonic()
            self.position_error_started_at = None
            self.get_logger().warning(
                f"RETURNING TO SESSION ZERO: {reason}; PICO input ignored")

    def _check_position_tracking_error(self, error_m, now):
        """Allow low-follow lag but reject large or sustained divergence."""
        error_m = float(error_m)
        if error_m > self.position_error_hard_limit_m:
            raise RuntimeError(
                f"target/actual position error {error_m*1000:.1f} mm "
                f"exceeded hard limit {self.position_error_hard_limit_m*1000:.0f} mm")
        if now - self.phase_started_at <= self.position_error_grace_sec:
            self.position_error_started_at = None
            return
        if error_m <= self.position_error_soft_limit_m:
            self.position_error_started_at = None
            return
        if self.position_error_started_at is None:
            self.position_error_started_at = now
            return
        held = now - self.position_error_started_at
        if held >= self.position_error_hold_sec:
            raise RuntimeError(
                f"target/actual position error {error_m*1000:.1f} mm "
                f"stayed above {self.position_error_soft_limit_m*1000:.0f} mm "
                f"for {held:.2f} s")

    def _fault(self, reason):
        if self.stop_sent: return
        self.stop_sent, self.phase = True, "FAULT"
        code = self.backend.emergency_stop()
        self.get_logger().error(f"FAULT: {reason}; return_code={code}")

    def _tick(self):
        now = time.monotonic()
        actual_tick_dt = (self.period if self.last_tick_at is None
                          else now - self.last_tick_at)
        self.last_tick_at = now
        if self.phase == "WAITING_ENTER" and self.start_event.is_set():
            self.phase = "REBASING"; self.mapper.reset_tracker_reference()
            self.latest = None; self.get_logger().info("Rebasing PICO pose")
            return
        if self.phase == "REBASING" and self.latest is not None:
            self.phase, self.started_at = "ACTIVE", now
            self.phase_started_at = now
            self.position_error_started_at = None
            self.ema.reset(self.initial_position, self.initial_q)
            self.limiter.reset(); self.get_logger().warning("ACTIVE 6DoF")
        if self.phase not in ("ACTIVE", "RETURNING"): return
        if self.phase == "ACTIVE" and self.stop_event.is_set():
            self._begin_return("operator Enter")
        if (self.phase == "ACTIVE" and
                (self.latest_at is None or now - self.latest_at > self.timeout)):
            self._fault("PICO timeout"); return
        try:
            requested_position = (self.initial_position if self.phase == "RETURNING"
                                  else self.latest.target_position)
            requested_quaternion = (self.initial_q if self.phase == "RETURNING"
                                    else self.latest.target_quaternion_xyzw)
            ema_position, ema_quaternion = self.ema.step(
                requested_position, requested_quaternion, self.period)
            pos, quat = self.limiter.step(
                ema_position, ema_quaternion,
                self.period)
            ik = self.ik.evaluate(pos, quat)
            if not ik.accepted:
                raise RuntimeError(ik.reason)
            if ik.near_joint_limit and self.phase == "ACTIVE":
                self._begin_return("shadow IK approached joint limit")
                return
            code = self.backend.send_pose(pos, quat, now)
            if code != 0: raise RuntimeError(f"pass-through code {code}")
            ret, state = self.robot.rm_get_current_arm_state()
            if ret != 0: raise RuntimeError(f"feedback code {ret}")
            actual = np.asarray(state["pose"], dtype=float)
            joints = np.asarray(state["joint"], dtype=float)
            if any(str(value) != "0" for value in
                   state.get("err", {}).get("err", [])):
                raise RuntimeError("arm error became nonzero")
            actual_offset = actual[:3] - self.initial_position
            if np.any(np.abs(actual_offset) > 0.102):
                raise RuntimeError("actual XYZ escaped 102 mm guard")
            position_error = float(np.linalg.norm(actual[:3] - pos))
            self._check_position_tracking_error(position_error, now)
            actual_q = quaternion_wxyz_to_xyzw(
                self.robot.rm_algo_euler2quaternion(actual[3:6].tolist()))
            if quaternion_angle_rad(actual_q, quat) > math.radians(15.0):
                raise RuntimeError("target/actual orientation error exceeded 15 deg")
            if np.max(np.abs(joints - self.previous_joints)) > 2.0:
                raise RuntimeError("actual joint jump >2 deg")
            self.previous_joints = joints
            self.feedback_count += 1
            if (self.latest_raw_position is not None and
                    self.latest_raw_quaternion is not None):
                translation_alpha, rotation_alpha = self.ema.alphas(self.period)
                self.trace.writerow([
                    now, self.latest_source_stamp_sec, actual_tick_dt,
                    self.period, translation_alpha, rotation_alpha,
                    *self.latest_raw_position.tolist(),
                    *self.latest_raw_quaternion.tolist(),
                    *self.latest.relative_translation.tolist(),
                    *self.latest.relative_rotation_xyzw.tolist(),
                    *self.latest.mapped_translation.tolist(),
                    *self.latest.mapped_relative_rotation_xyzw.tolist(),
                    *requested_position.tolist(), *requested_quaternion.tolist(),
                    *ema_position.tolist(), *ema_quaternion.tolist(),
                    *pos.tolist(), *quat.tolist(), *actual[:3].tolist(),
                    *joints.tolist(),
                ])
            if self.feedback_count % 80 == 0:
                self.trace_file.flush()
                offset = actual_offset * 1000
                target_angle = math.degrees(quaternion_angle_rad(
                    quat, self.initial_q))
                actual_angle = math.degrees(quaternion_angle_rad(
                    actual_q, self.initial_q))
                self.get_logger().info(
                    f"actual_offset_mm={offset.round(2).tolist()} "
                    f"requested_offset_mm={((requested_position-self.initial_position)*1000).round(2).tolist()} "
                    f"ema_offset_mm={((ema_position-self.initial_position)*1000).round(2).tolist()} "
                    f"sent_offset_mm={((pos-self.initial_position)*1000).round(2).tolist()} "
                    f"tick_dt_ms={actual_tick_dt*1000:.2f} "
                    f"ema_dt_ms={self.period*1000:.2f} "
                    f"tracking_error_mm={position_error*1000:.2f} "
                    f"rotation_deg target={target_angle:.2f} actual={actual_angle:.2f} "
                    f"ik_delta_deg={ik.max_joint_delta_deg:.3f}")
            if (self.phase == "RETURNING" and
                    np.linalg.norm(pos - self.initial_position) <= 0.0001 and
                    quaternion_angle_rad(quat, self.initial_q) <= math.radians(0.1) and
                    np.linalg.norm(actual[:3] - self.initial_position) <= 0.001 and
                    quaternion_angle_rad(actual_q, self.initial_q) <= math.radians(1.0)):
                self._stop("session zero reached")
        except Exception as exc: self._fault(exc)

    def destroy_node(self):
        if self.phase in ("ACTIVE", "RETURNING"): self._stop("shutdown")
        if getattr(self, "connected", False):
            self.robot.rm_delete_robot_arm(); self.connected = False
        if getattr(self, "trace_file", None) is not None:
            self.trace_file.flush()
            self.trace_file.close()
            self.trace_file = None
        return super().destroy_node()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-ip", default="192.168.1.18")
    parser.add_argument("--robot-port", type=int, default=8080)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--enable-motion", action="store_true")
    args, ros_args = parser.parse_known_args(argv)
    if not (args.execute and args.enable_motion):
        print("REFUSED: requires --execute --enable-motion", file=sys.stderr); return 2
    rclpy.init(args=ros_args); node = None
    try:
        node = RM756DoFTeleop(args.robot_ip, args.robot_port); rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        if node is not None: node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()
    return 0
