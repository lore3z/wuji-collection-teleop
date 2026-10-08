"""Interactive ten-second PICO -> RM Base-X real teleoperation trial."""

import argparse
import math
import sys
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e

from .movep_backend import RealMovePBackend
from .pose_mapper import InvalidPose, RelativePoseMapper, quaternion_wxyz_to_xyzw
from .readonly_preflight import collect_read_only_audit, run_base_x_preflight
from .single_axis_target_limiter import SingleAxisTargetLimiter


class RM75SingleAxisTeleop(Node):
    def __init__(self, robot_ip, robot_port):
        super().__init__("rm75_single_axis_teleop")
        self.rate_hz = 20.0
        self.period = 1.0 / self.rate_hz
        self.max_duration = 10.0
        self.tracker_timeout = 0.5
        self.start_requested = threading.Event()
        self.stop_requested = threading.Event()
        self.phase = "WAITING_ENTER"
        self.started_at = None
        self.latest_target = None
        self.latest_pose_time = None
        self.previous_joints = None
        self.connected = False
        self.stop_sent = False
        self.feedback_count = 0

        self.robot = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        handle = self.robot.rm_create_robot_arm(robot_ip, robot_port)
        if handle.id < 0:
            raise RuntimeError(f"RM75 connection failed: handle.id={handle.id}")
        self.connected = True
        audit = collect_read_only_audit(self.robot)
        state = audit["rm_get_current_arm_state"][1]
        if any(str(value) != "0" for value in state.get("err", {}).get("err", [])):
            raise RuntimeError("RM75 arm error is nonzero")
        if audit["rm_get_controller_state"].get(
                "system_error", audit["rm_get_controller_state"].get("sys_err", 0)) != 0:
            raise RuntimeError("RM75 controller error is nonzero")
        if any(int(value) != 0 for value in
               audit["rm_get_joint_err_flag"].get("err_flag", [])):
            raise RuntimeError("RM75 joint error is nonzero")
        preflight = run_base_x_preflight(
            self.robot, state, max_offset_m=0.100, sample_step_m=0.0005)
        if not preflight["passed"]:
            raise RuntimeError("Base-X +/-100 mm preflight failed")

        pose = np.asarray(state["pose"], dtype=float)
        self.initial_position = pose[:3].copy()
        self.initial_quaternion = quaternion_wxyz_to_xyzw(
            self.robot.rm_algo_euler2quaternion(pose[3:6].tolist()))
        self.previous_joints = np.asarray(state["joint"], dtype=float)
        self.mapper = RelativePoseMapper(
            axis_mapping=[0., 0., -1., -1., 0., 0., 0., 1., 0.],
            position_scale=1.0,
            max_position_jump_m=0.10,
            max_rotation_jump_rad=math.radians(45),
            workspace_min=[-0.8, -0.8, -0.1],
            workspace_max=[0.8, 0.8, 1.2],
            tracker_timeout_sec=self.tracker_timeout,
            translation_only=True,
            freeze_timeout_sec=60.0)
        self.mapper.set_rm_initial_pose(self.initial_position,
                                        self.initial_quaternion)
        self.limiter = SingleAxisTargetLimiter(
            self.initial_position, self.initial_quaternion,
            max_offset_m=0.100, max_speed_m_s=0.020,
            max_step_m=0.001)
        self.backend = RealMovePBackend(self.robot, follow=False)

        qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                         history=QoSHistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(PoseStamped, "/pico/right_wrist/raw_pose",
                                 self._pose_callback, qos)
        self.create_timer(self.period, self._tick)
        threading.Thread(target=self._keyboard_loop, daemon=True).start()
        self.get_logger().warning(
            "REAL 1-AXIS TELEOP: scale=1.0, Base X +/-100 mm, "
            "20 mm/s, 10 s maximum")
        self.get_logger().info(
            "Press Enter once to rebase and START; press Enter again to STOP")

    def _keyboard_loop(self):
        sys.stdin.readline()
        self.start_requested.set()
        sys.stdin.readline()
        self.stop_requested.set()

    def _pose_callback(self, msg):
        now = time.monotonic()
        try:
            if msg.header.frame_id != "pico_tracking":
                raise InvalidPose(f"unexpected frame_id={msg.header.frame_id!r}")
            self.latest_target = self.mapper.process(
                [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
                [msg.pose.orientation.x, msg.pose.orientation.y,
                 msg.pose.orientation.z, msg.pose.orientation.w], now)
            self.latest_pose_time = now
        except (InvalidPose, ValueError) as exc:
            if self.phase == "ACTIVE":
                self._fault(exc)
            elif self.phase in ("WAITING_ENTER", "REBASING"):
                self.mapper.reset_tracker_reference()
                self.latest_target = None
                self.latest_pose_time = None

    def _feedback(self, target_x):
        ret, state = self.robot.rm_get_current_arm_state()
        if ret != 0:
            raise RuntimeError(f"state feedback failed: {ret}")
        pose = np.asarray(state.get("pose", []), dtype=float)
        joints = np.asarray(state.get("joint", []), dtype=float)
        if pose.shape != (6,) or joints.shape != (7,):
            raise RuntimeError("invalid state feedback")
        actual_offset = abs(float(pose[0] - self.initial_position[0]))
        if actual_offset > 0.102:
            raise RuntimeError(f"actual X escaped 102 mm guard: {actual_offset}")
        if abs(float(pose[0] - target_x)) > 0.010:
            raise RuntimeError("target/actual X error exceeded 10 mm")
        if np.max(np.abs(joints - self.previous_joints)) > 2.0:
            raise RuntimeError("actual joint feedback jumped by more than 2 deg")
        self.previous_joints = joints
        self.feedback_count += 1
        if self.feedback_count % 20 == 0:
            target_offset = target_x - self.initial_position[0]
            actual_signed_offset = float(pose[0] - self.initial_position[0])
            self.get_logger().info(
                f"target_x_offset_mm={target_offset * 1000:.3f} "
                f"actual_x_offset_mm={actual_signed_offset * 1000:.3f} "
                f"tracking_error_mm={(target_x - pose[0]) * 1000:.3f}")

    def _stop(self, reason):
        if self.stop_sent:
            return
        self.stop_sent = True
        self.phase = "STOPPED"
        code = self.backend.slow_stop()
        self.get_logger().warning(
            f"STOPPED and locked: {reason}; slow-stop return_code={code}")

    def _fault(self, reason):
        if self.stop_sent:
            return
        self.stop_sent = True
        self.phase = "FAULT"
        code = self.backend.emergency_stop()
        self.get_logger().error(
            f"FAULT locked: {reason}; emergency-stop return_code={code}")

    def _tick(self):
        now = time.monotonic()
        if self.phase == "WAITING_ENTER" and self.start_requested.is_set():
            self.phase = "REBASING"
            self.mapper.reset_tracker_reference()
            self.latest_target = None
            self.latest_pose_time = None
            self.get_logger().info("Enter received; rebasing on next PICO frame")
            return
        if self.phase == "REBASING" and self.latest_target is not None:
            self.phase = "ACTIVE"
            self.started_at = now
            self.limiter.reset()
            self.get_logger().warning("ACTIVE: real single-axis teleoperation started")
        if self.phase != "ACTIVE":
            return
        if self.stop_requested.is_set():
            self._stop("operator pressed Enter")
            return
        if now - self.started_at >= self.max_duration:
            self._stop("10 second maximum duration reached")
            return
        if (self.latest_pose_time is None or
                now - self.latest_pose_time > self.tracker_timeout):
            self._fault("PICO timeout")
            return
        try:
            position, quaternion = self.limiter.step(
                self.latest_target.target_position, self.period)
            code = self.backend.send_pose(position, quaternion, now)
            if code != 0:
                raise RuntimeError(f"pose pass-through return code {code}")
            self._feedback(position[0])
        except Exception as exc:
            self._fault(exc)

    def destroy_node(self):
        if self.phase == "ACTIVE":
            self._stop("node shutdown")
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
        node = RM75SingleAxisTeleop(args.robot_ip, args.robot_port)
        rclpy.spin(node)
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
