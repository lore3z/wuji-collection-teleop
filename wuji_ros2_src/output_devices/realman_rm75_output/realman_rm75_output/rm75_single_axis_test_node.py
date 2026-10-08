"""Fake-only RM-X rehearsal. No RM motion API is present in this module."""

import math
import sys
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import Bool
from std_srvs.srv import Trigger
from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e

from .ik_dry_run import validate_safety_mode
from .motion_safety_gate import MotionSafetyGate, MotionState
from .movep_backend import FakeMovePBackend
from .pose_mapper import InvalidPose, RelativePoseMapper, quaternion_wxyz_to_xyzw
from .readonly_preflight import run_base_x_preflight
from .single_axis_target_limiter import SingleAxisTargetLimiter


class RM75SingleAxisTestNode(Node):
    def __init__(self):
        super().__init__("rm75_single_axis_test_node")
        defaults = {
            "robot_ip": "192.168.1.18", "robot_port": 8080,
            "input_topic": "/pico/right_wrist/raw_pose",
            "dry_run": True, "enable_motion": False, "backend": "fake",
            "command_rate_hz": 20.0, "max_offset_m": 0.005,
            "max_linear_speed_m_s": 0.005,
            "max_test_duration_sec": 10.0,
            "preflight_sample_step_m": 0.0005,
            "tracker_timeout_sec": 0.5,
            "axis_mapping": [0., 0., -1., -1., 0., 0., 0., 1., 0.],
            "position_scale": 0.05,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        validate_safety_mode(bool(self.get_parameter("dry_run").value),
                             bool(self.get_parameter("enable_motion").value))
        if self.get_parameter("backend").value != "fake":
            raise RuntimeError("this phase supports backend=fake only")
        self.rate_hz = float(self.get_parameter("command_rate_hz").value)
        self.max_test_duration = float(
            self.get_parameter("max_test_duration_sec").value)
        if self.rate_hz <= 0.0 or self.max_test_duration <= 0.0:
            raise ValueError("rate and test duration must be positive")

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
        pose = np.asarray(state["pose"], dtype=float)
        initial_q = quaternion_wxyz_to_xyzw(
            self.robot.rm_algo_euler2quaternion(pose[3:6].tolist()))

        preflight = run_base_x_preflight(
            self.robot, state,
            max_offset_m=float(self.get_parameter("max_offset_m").value),
            sample_step_m=float(
                self.get_parameter("preflight_sample_step_m").value),
        )
        if not preflight["passed"]:
            rejected = [point for point in preflight["points"]
                        if not point["accepted"]]
            raise RuntimeError(f"Base-X startup preflight failed: {rejected}")
        self.get_logger().info(
            f"Base-X startup preflight PASS: {len(preflight['points'])} points")

        self.mapper = RelativePoseMapper(
            axis_mapping=self.get_parameter("axis_mapping").value,
            position_scale=self.get_parameter("position_scale").value,
            max_position_jump_m=0.10,
            max_rotation_jump_rad=math.radians(45),
            workspace_min=[-0.8, -0.8, -0.1], workspace_max=[0.8, 0.8, 1.2],
            tracker_timeout_sec=self.get_parameter("tracker_timeout_sec").value,
            translation_only=True,
            # Identical poses are also produced by a healthy stationary tracker.
            # During this ten-second rehearsal, liveness is therefore based on
            # message arrival, not on pose value changes.
            freeze_timeout_sec=max(60.0, self.max_test_duration + 1.0))
        self.mapper.set_rm_initial_pose(pose[:3], initial_q)
        self.limiter = SingleAxisTargetLimiter(
            pose[:3], initial_q,
            self.get_parameter("max_offset_m").value,
            self.get_parameter("max_linear_speed_m_s").value,
            max_step_m=(
                float(self.get_parameter("max_linear_speed_m_s").value) /
                self.rate_hz))
        self.gate = MotionSafetyGate()
        self.latest_target = None
        self.latest_pose_time = None
        self.last_tick = time.monotonic()
        self.tick_intervals = []
        self.session_started_at = None
        self.session_finished = False
        self.enter_requested = threading.Event()
        self.fake_pub = self.create_publisher(PoseStamped, "/rm75/fake_movep_target", 10)
        self.backend = FakeMovePBackend(self._publish_fake)
        latest_pose_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(PoseStamped,
                                 str(self.get_parameter("input_topic").value),
                                 self._pose_callback, latest_pose_qos)
        self.create_subscription(Bool, "/rm75/deadman", self._deadman_callback, 10)
        self.create_service(Trigger, "/rm75/reset_fault", self._reset_fault)
        self.create_timer(1.0 / self.rate_hz, self._tick)
        self.get_logger().warning(
            "FAKE BACKEND ONLY: RM X +/-5 mm, 5 mm/s, orientation locked; no motion API")
        if sys.stdin.isatty():
            threading.Thread(target=self._wait_for_enter, daemon=True).start()
            self.get_logger().info("Press Enter to finish and lock this fake session")

    def _wait_for_enter(self):
        try:
            sys.stdin.readline()
            self.enter_requested.set()
        except Exception:
            pass

    def _finish_session(self, reason):
        if self.session_finished:
            return
        self.session_finished = True
        self.gate.set_deadman(False)
        self.limiter.reset()
        self.get_logger().warning(
            f"Fake session finished and locked: {reason}; no further targets allowed")

    def _trip(self, reason):
        self.gate.trip(reason)
        self.get_logger().error(f"FAULT latched: {self.gate.fault_reason}")

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
            if self.gate.state is MotionState.ACTIVE:
                self._trip(exc)
            elif self.gate.state is MotionState.DISARMED:
                self.mapper.reset_tracker_reference()
                self.latest_target = None
                self.latest_pose_time = None

    def _deadman_callback(self, msg):
        if self.session_finished:
            return
        was_active = self.gate.state is MotionState.ACTIVE
        self.gate.set_deadman(msg.data)
        if was_active and not msg.data:
            self.limiter.reset()
            self.get_logger().info("Deadman released: DISARMED, fake output stopped")

    def _reset_fault(self, _request, response):
        try:
            self.gate.reset()
            self.limiter.reset()
            self.mapper.reset_tracker_reference()
            self.latest_target = None
            self.latest_pose_time = None
            response.success, response.message = True, "FAULT reset to DISARMED"
        except RuntimeError as exc:
            response.success, response.message = False, str(exc)
        return response

    def _tick(self):
        now = time.monotonic()
        dt = now - self.last_tick
        self.last_tick = now
        self.tick_intervals.append(dt)
        if self.enter_requested.is_set():
            self._finish_session("operator pressed Enter")
        if self.session_finished:
            return
        fresh = (self.latest_pose_time is not None and
                 now - self.latest_pose_time <=
                 float(self.get_parameter("tracker_timeout_sec").value))
        if self.gate.state is MotionState.ACTIVE and not fresh:
            self._trip("PICO timeout")
        was_active = self.gate.state is MotionState.ACTIVE
        if not self.gate.activate_if_ready(fresh) or self.latest_target is None:
            return
        if not was_active:
            self.session_started_at = now
            self.get_logger().info(
                f"Fake session ACTIVE; automatic finish in {self.max_test_duration:.1f}s")
        if now - self.session_started_at >= self.max_test_duration:
            self._finish_session("maximum test duration reached")
            return
        try:
            position, quaternion = self.limiter.step(
                self.latest_target.target_position, dt)
        except ValueError as exc:
            self._trip(exc)
            return
        self.backend.send_pose(position, quaternion, now)

    def _publish_fake(self, position, quaternion):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "rm75_base_fake"
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = position
        (msg.pose.orientation.x, msg.pose.orientation.y,
         msg.pose.orientation.z, msg.pose.orientation.w) = quaternion
        self.fake_pub.publish(msg)

    def destroy_node(self):
        if self.tick_intervals:
            values = np.asarray(self.tick_intervals)
            self.get_logger().info(
                f"FAKE SUMMARY sent={len(self.backend.sent)} state={self.gate.state.value} "
                f"fault={self.gate.fault_reason!r} mean_period_ms={values.mean()*1000:.3f} "
                f"max_period_ms={values.max()*1000:.3f}")
        if getattr(self, "connected", False):
            ret = self.robot.rm_delete_robot_arm()
            self.get_logger().info(f"Read-only RM75 disconnect return code={ret}")
            self.connected = False
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = RM75SingleAxisTestNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
