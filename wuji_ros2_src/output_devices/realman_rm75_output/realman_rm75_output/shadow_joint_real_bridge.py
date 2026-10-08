"""Safety bridge from the validated RM75 shadow to limited real joint motion."""

import argparse
from collections import OrderedDict, deque
from contextlib import suppress
import json
import math
import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import JointState
from std_msgs.msg import UInt64MultiArray
from std_srvs.srv import Trigger
from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e

from .joint_sender import FakeJointBackend, JointSender, LatestJointTarget
from .latency_trace import (
    PipelineLatencyStatistics,
    parse_shadow_trace,
    stamp_to_ns,
)
from .movej_backend import RealFollowJointBackend


EXPECTED_JOINT_NAMES = [f"joint{index}" for index in range(1, 8)]
DEFAULT_SAFE_HOME_DEG = np.degrees(np.array(
    [-math.pi / 2.0, 0.4, 0.0, 1.5, 0.0,
     -0.3292036732, math.radians(10.0)]))
RM75_MODEL_JOINT_MIN_DEG = np.degrees(np.array(
    [-3.1, -2.268, -3.1, -2.355, -3.1, -2.233, -6.28]))
RM75_MODEL_JOINT_MAX_DEG = np.degrees(np.array(
    [3.1, 2.268, 3.1, 2.355, 3.1, 2.233, 6.28]))


class ShadowJointJump(ValueError):
    """A finite shadow sample that is discontinuous with the prior sample."""

    def __init__(self, step_deg, soft_limit_deg):
        self.step_deg = np.asarray(step_deg, dtype=float)
        if (self.step_deg.shape != (7,) or
                not np.all(np.isfinite(self.step_deg))):
            raise ValueError("shadow joint step must be a finite 7-vector")
        self.joint_index = int(np.argmax(np.abs(self.step_deg)))
        self.joint_name = EXPECTED_JOINT_NAMES[self.joint_index]
        self.magnitude_deg = float(abs(self.step_deg[self.joint_index]))
        self.soft_limit_deg = float(soft_limit_deg)
        super().__init__(
            f"shadow {self.joint_name} jump {self.magnitude_deg:.3f} deg "
            f"exceeds "
            f"{self.soft_limit_deg:.3f} deg")


class ShadowJumpMonitor:
    """Drop isolated discontinuities and fault hard or persistent jumps."""

    def __init__(self, hard_limit_deg=15.0, consecutive_fault_samples=3):
        self.hard_limit = float(hard_limit_deg)
        self.limit = int(consecutive_fault_samples)
        if self.hard_limit <= 0.0 or self.limit < 1:
            raise ValueError("invalid shadow jump monitor limits")
        self.consecutive = 0
        self.rejected_samples = 0
        self.maximum_jump = 0.0
        self.maximum_by_joint = np.zeros(7, dtype=float)

    def accepted(self):
        self.consecutive = 0

    def rejected(self, step_deg):
        step = np.asarray(step_deg, dtype=float)
        if step.shape != (7,) or not np.all(np.isfinite(step)):
            raise ValueError("shadow joint step must be a finite 7-vector")
        absolute_step = np.abs(step)
        joint_index = int(np.argmax(absolute_step))
        magnitude = float(absolute_step[joint_index])
        self.consecutive += 1
        self.rejected_samples += 1
        self.maximum_jump = max(self.maximum_jump, magnitude)
        self.maximum_by_joint = np.maximum(
            self.maximum_by_joint, absolute_step)
        return {
            "fault": (
                magnitude >= self.hard_limit or
                self.consecutive >= self.limit),
            "hard": magnitude >= self.hard_limit,
            "consecutive": self.consecutive,
            "magnitude_deg": magnitude,
            "joint_index": joint_index,
            "joint_name": EXPECTED_JOINT_NAMES[joint_index],
            "step_deg": step.copy(),
        }


def joint_state_radians(message):
    """Validate and reorder an RM75 JointState into seven radians."""
    names = list(message.name)
    positions = list(message.position)
    if len(names) != 7 or len(positions) != 7:
        raise ValueError("shadow JointState must contain seven named joints")
    if set(names) != set(EXPECTED_JOINT_NAMES):
        raise ValueError(
            f"shadow joint names must be {EXPECTED_JOINT_NAMES}, got {names}")
    by_name = dict(zip(names, positions))
    result = np.asarray(
        [by_name[name] for name in EXPECTED_JOINT_NAMES], dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError("shadow joints must be finite")
    return result


def wrapped_joint_error_deg(joints_deg, reference_deg):
    """Shortest signed per-joint error, robust to equivalent +/-360 values."""
    joints = np.asarray(joints_deg, dtype=float)
    reference = np.asarray(reference_deg, dtype=float)
    if (joints.shape != (7,) or reference.shape != (7,) or
            not np.all(np.isfinite(joints)) or
            not np.all(np.isfinite(reference))):
        raise ValueError("joint vectors must contain seven finite values")
    return (joints - reference + 180.0) % 360.0 - 180.0


class StableJointReferenceMonitor:
    """Require consecutive quiet shadow samples near a joint reference."""

    def __init__(self, reference_deg, home_tolerance_deg=0.5,
                 max_step_deg=0.15, required_samples=12):
        self.reference = np.asarray(reference_deg, dtype=float)
        self.home_tolerance = float(home_tolerance_deg)
        self.max_step = float(max_step_deg)
        self.required_samples = int(required_samples)
        if (self.reference.shape != (7,) or
                not np.all(np.isfinite(self.reference)) or
                self.home_tolerance <= 0.0 or self.max_step <= 0.0 or
                self.required_samples < 1):
            raise ValueError("invalid stable joint reference monitor")
        self.reset()

    def reset(self):
        self.previous = None
        self.consecutive = 0
        self.latest_home_error = float("inf")
        self.latest_step = float("inf")

    def observe(self, joints_deg):
        joints = np.asarray(joints_deg, dtype=float)
        if joints.shape != (7,) or not np.all(np.isfinite(joints)):
            raise ValueError("observed joints must be a finite 7-vector")
        home_error = wrapped_joint_error_deg(joints, self.reference)
        maximum_home_error = float(np.max(np.abs(home_error)))
        maximum_step = 0.0
        if self.previous is not None:
            step = wrapped_joint_error_deg(joints, self.previous)
            maximum_step = float(np.max(np.abs(step)))
        self.previous = joints.copy()
        self.latest_home_error = maximum_home_error
        self.latest_step = maximum_step
        if (maximum_home_error <= self.home_tolerance and
                maximum_step <= self.max_step):
            self.consecutive += 1
        else:
            self.consecutive = 0
        return {
            "ready": self.consecutive >= self.required_samples,
            "consecutive": self.consecutive,
            "required_samples": self.required_samples,
            "max_home_error_deg": maximum_home_error,
            "max_step_deg": maximum_step,
        }


class ShadowJointDeltaMapper:
    """Map a shadow displacement onto a scaled, bounded real displacement."""

    def __init__(self, shadow_reference_rad, robot_reference_deg,
                 delta_scale=0.5, max_offset_deg=3.0,
                 max_shadow_step_deg=5.0, offset_limit_enabled=True):
        self.shadow_reference = np.asarray(
            shadow_reference_rad, dtype=float)
        self.robot_reference = np.asarray(robot_reference_deg, dtype=float)
        self.delta_scale = float(delta_scale)
        self.max_offset = float(max_offset_deg)
        self.max_shadow_step = float(max_shadow_step_deg)
        self.offset_limit_enabled = bool(offset_limit_enabled)
        if (self.shadow_reference.shape != (7,) or
                self.robot_reference.shape != (7,) or
                not np.all(np.isfinite(self.shadow_reference)) or
                not np.all(np.isfinite(self.robot_reference))):
            raise ValueError("shadow and robot references must be finite 7-vectors")
        if (not 0.0 < self.delta_scale <= 1.0 or self.max_offset <= 0.0 or
                self.max_shadow_step <= 0.0):
            raise ValueError("invalid joint-delta mapping limits")
        self.previous_shadow = self.shadow_reference.copy()

    def apply_offset_limit(self, requested_offset):
        requested = np.asarray(requested_offset, dtype=float)
        if requested.shape != (7,) or not np.all(np.isfinite(requested)):
            raise ValueError("requested offset must be a finite 7-vector")
        if not self.offset_limit_enabled:
            return requested.copy()
        return np.clip(requested, -self.max_offset, self.max_offset)

    def process(self, shadow_joints_rad):
        shadow = np.asarray(shadow_joints_rad, dtype=float)
        if shadow.shape != (7,) or not np.all(np.isfinite(shadow)):
            raise ValueError("shadow joints must be a finite 7-vector")
        step_deg = np.degrees(shadow - self.previous_shadow)
        maximum_step = float(np.max(np.abs(step_deg)))
        if maximum_step > self.max_shadow_step:
            # Advance the continuity reference while dropping this command.
            # A stable next frame can recover; alternating discontinuities are
            # counted by the node and become a fault.
            self.previous_shadow = shadow.copy()
            raise ShadowJointJump(step_deg, self.max_shadow_step)
        self.previous_shadow = shadow.copy()
        requested_offset = self.delta_scale * np.degrees(
            shadow - self.shadow_reference)
        applied_offset = self.apply_offset_limit(requested_offset)
        saturated = bool(np.any(
            np.abs(requested_offset - applied_offset) > 1e-9))
        return self.robot_reference + applied_offset, saturated, requested_offset


class JointMotionRecorder:
    """Record the trajectory actually published to the green RobotModel."""

    def __init__(self, reference_deg):
        self.reference = np.asarray(reference_deg, dtype=float)
        if self.reference.shape != (7,) or not np.all(
                np.isfinite(self.reference)):
            raise ValueError("motion reference must be a finite 7-vector")
        self.samples = 0
        self.minimum = np.full(7, np.inf)
        self.maximum = np.full(7, -np.inf)
        self.total_travel = np.zeros(7)
        self.maximum_step = np.zeros(7)
        self.previous = None

    def observe(self, joints_deg):
        joints = np.asarray(joints_deg, dtype=float)
        if joints.shape != (7,) or not np.all(np.isfinite(joints)):
            raise ValueError("recorded command must be a finite 7-vector")
        self.minimum = np.minimum(self.minimum, joints)
        self.maximum = np.maximum(self.maximum, joints)
        if self.previous is not None:
            step = np.abs(joints - self.previous)
            self.total_travel += step
            self.maximum_step = np.maximum(self.maximum_step, step)
        self.previous = joints.copy()
        self.samples += 1

    def summary(self):
        if self.samples == 0:
            return {"samples": 0}
        minimum_offset = self.minimum - self.reference
        maximum_offset = self.maximum - self.reference
        return {
            "samples": self.samples,
            "reference_deg": self.reference.tolist(),
            "absolute_min_deg": self.minimum.tolist(),
            "absolute_max_deg": self.maximum.tolist(),
            "offset_min_deg": minimum_offset.tolist(),
            "offset_max_deg": maximum_offset.tolist(),
            "peak_abs_offset_deg": np.maximum(
                np.abs(minimum_offset), np.abs(maximum_offset)).tolist(),
            "offset_span_deg": (self.maximum - self.minimum).tolist(),
            "total_travel_deg": self.total_travel.tolist(),
            "max_display_step_deg": self.maximum_step.tolist(),
            "final_absolute_deg": self.previous.tolist(),
            "final_offset_deg": (self.previous - self.reference).tolist(),
        }


class TrackingErrorMonitor:
    """Require persistent command/readback mismatch before faulting."""

    def __init__(self, warning_deg=1.0, fault_deg=2.0,
                 consecutive_fault_samples=5):
        self.warning = float(warning_deg)
        self.fault = float(fault_deg)
        self.limit = int(consecutive_fault_samples)
        if not 0.0 < self.warning < self.fault or self.limit < 1:
            raise ValueError("invalid tracking-error monitor thresholds")
        self.consecutive_faults = 0
        self.maximum_error = 0.0

    def process(self, commanded_deg, measured_deg):
        commanded = np.asarray(commanded_deg, dtype=float)
        measured = np.asarray(measured_deg, dtype=float)
        if (commanded.shape != (7,) or measured.shape != (7,) or
                not np.all(np.isfinite(commanded)) or
                not np.all(np.isfinite(measured))):
            raise ValueError("tracking vectors must contain seven finite values")
        error = float(np.max(np.abs(commanded - measured)))
        self.maximum_error = max(self.maximum_error, error)
        self.consecutive_faults = (
            self.consecutive_faults + 1 if error > self.fault else 0)
        return {
            "error_deg": error,
            "warning": error > self.warning,
            "fault": self.consecutive_faults >= self.limit,
            "consecutive_faults": self.consecutive_faults,
        }


class RM75ShadowJointRealBridge(Node):
    """Explicitly armed 125 Hz bridge with independent real-motion guards."""

    def __init__(self, real_motion=False, robot_ip="192.168.1.18",
                 robot_port=8080, auto_start=False):
        super().__init__(
            "rm75_shadow_joint_real_bridge" if real_motion else
            "rm75_shadow_joint_fake_bridge")
        defaults = {
            "input_topic": "/rm75_sim/joint_states",
            "input_latency_topic": "/rm75_sim/latency_trace",
            "input_timeout_sec": 0.25,
            "max_duration_sec": 15.0,
            "duration_limit_enabled": True,
            "observe_shadow_jump_faults": False,
            "joint_delta_scale": 0.5,
            "max_relative_offset_deg": 3.0,
            "relative_offset_limit_enabled": True,
            "max_shadow_step_deg": 5.0,
            "hard_shadow_jump_deg": 15.0,
            "shadow_jump_fault_samples": 3,
            "sender_rate_hz": 125.0,
            "max_speed_deg_s": 6.0,
            "max_acceleration_deg_s2": 20.0,
            "joint_limit_margin_deg": 5.0,
            "safe_home_tolerance_deg": 15.0,
            "tracking_warning_deg": 1.0,
            "tracking_fault_deg": 2.0,
            "tracking_fault_samples": 5,
            "return_timeout_sec": 6.0,
            "return_tolerance_deg": 0.10,
            "shadow_rebase_home_service": "/rm75_sim/rebase_home",
            "upper_arm_rebase_service": "/rm75_sim/rebase_upper_arm",
            "rebase_timeout_sec": 4.0,
            "rebase_stable_samples": 12,
            "rebase_home_tolerance_deg": 0.5,
            "rebase_max_step_deg": 0.15,
            "fake_initial_joints_deg": DEFAULT_SAFE_HOME_DEG.tolist(),
            "fake_joint_min_deg": RM75_MODEL_JOINT_MIN_DEG.tolist(),
            "fake_joint_max_deg": RM75_MODEL_JOINT_MAX_DEG.tolist(),
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        self.real_motion = bool(real_motion)
        self.auto_start = bool(auto_start)
        self.robot_ip = str(robot_ip)
        self.robot_port = int(robot_port)
        self.input_timeout = float(
            self.get_parameter("input_timeout_sec").value)
        self.max_duration = float(
            self.get_parameter("max_duration_sec").value)
        self.duration_limit_enabled = bool(
            self.get_parameter("duration_limit_enabled").value)
        self.observe_shadow_jump_faults = bool(
            self.get_parameter("observe_shadow_jump_faults").value)
        self.delta_scale = float(
            self.get_parameter("joint_delta_scale").value)
        self.max_offset = float(
            self.get_parameter("max_relative_offset_deg").value)
        self.relative_offset_limit_enabled = bool(
            self.get_parameter("relative_offset_limit_enabled").value)
        self.max_shadow_step = float(
            self.get_parameter("max_shadow_step_deg").value)
        self.hard_shadow_jump = float(
            self.get_parameter("hard_shadow_jump_deg").value)
        self.rate_hz = float(self.get_parameter("sender_rate_hz").value)
        self.max_speed = float(
            self.get_parameter("max_speed_deg_s").value)
        self.max_acceleration = float(
            self.get_parameter("max_acceleration_deg_s2").value)
        self.joint_margin = float(
            self.get_parameter("joint_limit_margin_deg").value)
        self.safe_home_tolerance = float(
            self.get_parameter("safe_home_tolerance_deg").value)
        self.return_timeout = float(
            self.get_parameter("return_timeout_sec").value)
        self.return_tolerance = float(
            self.get_parameter("return_tolerance_deg").value)
        self.rebase_timeout = float(
            self.get_parameter("rebase_timeout_sec").value)
        self.rebase_home_tolerance = float(
            self.get_parameter("rebase_home_tolerance_deg").value)
        self.rebase_max_step = float(
            self.get_parameter("rebase_max_step_deg").value)
        self.rebase_stable_samples = int(
            self.get_parameter("rebase_stable_samples").value)
        positive = [
            self.input_timeout, self.max_duration, self.delta_scale,
            self.max_offset, self.max_shadow_step, self.rate_hz,
            self.hard_shadow_jump, self.max_speed, self.max_acceleration,
            self.safe_home_tolerance,
            self.return_timeout, self.return_tolerance,
            self.rebase_timeout, self.rebase_home_tolerance,
            self.rebase_max_step,
        ]
        if (any(value <= 0.0 for value in positive) or
                self.joint_margin < 0.0 or self.delta_scale > 1.0 or
                self.rebase_stable_samples < 1):
            raise ValueError(
                "bridge limits must be positive, joint margin nonnegative, "
                "and scale <= 1")

        self.phase = "WAITING_ENTER"
        self.start_event = threading.Event()
        self.stop_event = threading.Event()
        self.latest_shadow = None
        self.latest_shadow_at = None
        self.latest_measured = None
        self.active_started_at = None
        self.return_started_at = None
        self.return_trigger = ""
        self.return_completed = False
        self.return_duration = 0.0
        self.rebase_started_at = None
        self.rebase_future = None
        self.upper_arm_rebase_future = None
        self.rebase_acknowledged = False
        self.rebase_status = None
        self.rebase_monitor = StableJointReferenceMonitor(
            DEFAULT_SAFE_HOME_DEG,
            home_tolerance_deg=self.rebase_home_tolerance,
            max_step_deg=self.rebase_max_step,
            required_samples=self.rebase_stable_samples)
        self.mapper = None
        self.position_guard_lower = None
        self.position_guard_upper = None
        self.motion_recorder = None
        self.buffer = None
        self.backend = None
        self.sender = None
        self.robot = None
        self.connected = False
        self.closed = False
        self.saturated_samples = 0
        self.saturated_samples_by_joint = np.zeros(7, dtype=int)
        self.latest_requested_offset = np.zeros(7, dtype=float)
        self.max_requested_offset_by_joint = np.zeros(7, dtype=float)
        self.min_requested_offset_by_joint = np.full(7, np.inf)
        self.max_signed_requested_offset_by_joint = np.full(7, -np.inf)
        self.position_guard_clamp_samples = 0
        self.position_guard_clamp_samples_by_joint = np.zeros(7, dtype=int)
        self.target_samples = 0
        self.fault_reason = ""
        self.last_tracking_warning_at = -float("inf")
        self.tracking = TrackingErrorMonitor(
            self.get_parameter("tracking_warning_deg").value,
            self.get_parameter("tracking_fault_deg").value,
            self.get_parameter("tracking_fault_samples").value)
        self.shadow_jumps = ShadowJumpMonitor(
            self.hard_shadow_jump,
            self.get_parameter("shadow_jump_fault_samples").value)
        self.shadow_jump_event_count = 0
        self.shadow_jump_events = deque(maxlen=100)
        self.command_publisher = self.create_publisher(
            JointState, "/rm75_real_trial/command_joint_states", 10)
        self.measured_publisher = self.create_publisher(
            JointState, "/rm75_real_trial/measured_joint_states", 10)
        self.last_diagnostic_at = -float("inf")
        self.last_latency_log_at = -float("inf")
        self.shadow_latency_traces = OrderedDict()
        self.pending_bridge_receives = OrderedDict()
        self.recorded_pose_keys = OrderedDict()
        self.latency_statistics = PipelineLatencyStatistics()

        if self.real_motion:
            self._connect_robot()
        latest_only_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE)
        self.create_subscription(
            JointState, str(self.get_parameter("input_topic").value),
            self._shadow_callback, latest_only_qos)
        self.create_subscription(
            UInt64MultiArray,
            str(self.get_parameter("input_latency_topic").value),
            self._shadow_latency_callback, 10)
        self.rebase_client = self.create_client(
            Trigger,
            str(self.get_parameter("shadow_rebase_home_service").value))
        self.upper_arm_rebase_client = self.create_client(
            Trigger,
            str(self.get_parameter("upper_arm_rebase_service").value))
        self.create_timer(0.02, self._watchdog)
        self.create_timer(1.0 / 60.0, self._publish_command_shadow)
        if self.auto_start:
            self.start_event.set()
            if sys.stdin.isatty():
                threading.Thread(
                    target=self._keyboard_stop_only, daemon=True).start()
        else:
            threading.Thread(target=self._keyboard, daemon=True).start()

        offset_label = (
            "unbounded-to-RM75-joint-guards"
            if not self.relative_offset_limit_enabled else
            f"+/-{self.max_offset:.1f}deg")
        duration_label = (
            "unlimited" if not self.duration_limit_enabled else
            f"{self.max_duration:.1f}s")
        self.get_logger().warning(
            ("REAL RM75 LIMITED TRIAL" if self.real_motion else
             "FAKE BACKEND ONLY") +
            f": shadow_delta_scale={self.delta_scale:.2f}, "
            f"offset={offset_label}, "
            f"speed={self.max_speed:.1f}deg/s, "
            f"accel={self.max_acceleration:.1f}deg/s^2, "
            f"duration={duration_label}")
        if self.observe_shadow_jump_faults:
            self.get_logger().warning(
                "SHADOW JUMP OBSERVE MODE: discontinuous IK frames are held "
                "and recorded instead of terminating the session")
        self.get_logger().warning(
            "AUTO START: restore shadow home, reset PICO reference, wait "
            "for stable frames, then capture references; "
            "press Enter to RETURN TO START+STOP" if self.auto_start else
            "Press Enter to restore shadow home, reset PICO reference, wait "
            "for stable frames, then capture shadow+robot references; "
            "press Enter again to RETURN TO START+STOP")
        self.get_logger().info(
            "shadow JointState QoS=latest-only best-effort depth=1; "
            "PIPELINE_LATENCY reports PICO->PC->IK->shadow->bridge")

    def _connect_robot(self):
        self.robot = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        handle = self.robot.rm_create_robot_arm(
            self.robot_ip, self.robot_port)
        if handle.id < 0:
            raise RuntimeError(
                f"RM75 connection failed: handle={handle.id}")
        self.connected = True
        try:
            code, _state = self.robot.rm_get_current_arm_state()
            if code != 0:
                raise RuntimeError(f"RM75 initial state query failed: {code}")
        except Exception:
            self.robot.rm_delete_robot_arm()
            self.connected = False
            raise
        self.get_logger().warning(
            f"connected to RM75 at {self.robot_ip}:{self.robot_port}; "
            "no motion API is called before operator Enter")

    def _keyboard(self):
        print(
            "\n>>> 保持两枚 Tracker 静止，按 Enter 回影子安全位、重置 PICO 共同零点并开始：",
            end="", flush=True)
        if sys.stdin.readline() == "":
            self.stop_event.set()
            self.get_logger().error(
                "stdin closed before START; refusing to arm")
            return
        self.start_event.set()
        print(
            "\n>>> 正在回影子安全位并等待稳定；启动后再按 Enter 可返回起点：",
            end="", flush=True)
        if sys.stdin.readline() == "":
            self.get_logger().warning(
                "stdin closed while ACTIVE; requesting STOP+HOLD")
        self.stop_event.set()

    def _keyboard_stop_only(self):
        print(
            "\n>>> FAKE 正在自动回影子安全位并捕获共同零点；结束并返回时按 Enter：",
            end="", flush=True)
        if sys.stdin.readline() == "":
            self.get_logger().warning(
                "stdin closed while ACTIVE; requesting STOP+HOLD")
        self.stop_event.set()

    @staticmethod
    def _bounded_put(mapping, key, value, limit=256):
        mapping[key] = value
        mapping.move_to_end(key)
        while len(mapping) > limit:
            mapping.popitem(last=False)

    def _shadow_latency_callback(self, message):
        try:
            trace = parse_shadow_trace(message.data)
        except ValueError as error:
            self.get_logger().warning(
                f"invalid shadow latency trace: {error}",
                throttle_duration_sec=2.0)
            return
        key = trace["shadow_key_ns"]
        bridge_receive_ns = self.pending_bridge_receives.pop(key, None)
        if bridge_receive_ns is None:
            self._bounded_put(self.shadow_latency_traces, key, trace)
        else:
            self._record_pipeline_latency(trace, bridge_receive_ns)

    def _match_pipeline_latency(self, shadow_key_ns, bridge_receive_ns):
        trace = self.shadow_latency_traces.pop(shadow_key_ns, None)
        if trace is None:
            self._bounded_put(
                self.pending_bridge_receives, shadow_key_ns,
                bridge_receive_ns)
            return
        self._record_pipeline_latency(trace, bridge_receive_ns)

    def _record_pipeline_latency(self, trace, bridge_receive_ns):
        pose_key = int(trace.get("pose_key_ns", 0))
        if pose_key <= 0 or pose_key in self.recorded_pose_keys:
            return
        self._bounded_put(self.recorded_pose_keys, pose_key, True, 2048)
        latest = self.latency_statistics.observe(trace, bridge_receive_ns)
        # Keep the interactive start prompt readable. Collection continues,
        # but periodic timing output begins only after the bridge is armed.
        if self.phase == "WAITING_ENTER":
            return
        now = time.monotonic()
        if now - self.last_latency_log_at < 1.0:
            return
        self.last_latency_log_at = now
        ordered_names = (
            "pico_to_pc", "pico_to_pc_relative", "pc_sdk_read",
            "pc_to_ik_receive", "ik_queue",
            "ik_compute", "ik_to_shadow", "shadow_to_bridge",
            "pc_to_bridge", "pico_to_bridge", "pico_to_bridge_relative")
        fields = [
            f"{name}_ms={latest[name]:.3f}"
            for name in ordered_names if name in latest]
        if "pico_to_pc" not in latest:
            status = (
                "OFFSET_CALIBRATED" if "pico_to_pc_relative" in latest else
                "UNAVAILABLE")
            fields.insert(0, f"pico_clock={status}")
        self.get_logger().info("PIPELINE_LATENCY " + " ".join(fields))

    def _shadow_callback(self, message):
        now = time.monotonic()
        self._match_pipeline_latency(
            stamp_to_ns(message.header.stamp), time.time_ns())
        try:
            shadow = joint_state_radians(message)
            self.latest_shadow = shadow
            self.latest_shadow_at = now
            if self.phase == "REBASING" and self.rebase_acknowledged:
                self.rebase_status = self.rebase_monitor.observe(
                    np.degrees(shadow))
                return
            if self.phase != "ACTIVE":
                return
            target, saturated, requested = self.mapper.process(shadow)
            self.shadow_jumps.accepted()
            self.latest_requested_offset = requested.copy()
            self.max_requested_offset_by_joint = np.maximum(
                self.max_requested_offset_by_joint, np.abs(requested))
            self.min_requested_offset_by_joint = np.minimum(
                self.min_requested_offset_by_joint, requested)
            self.max_signed_requested_offset_by_joint = np.maximum(
                self.max_signed_requested_offset_by_joint, requested)
            if saturated:
                self.saturated_samples += 1
                applied = target - self.mapper.robot_reference
                saturated_joints = (
                    np.abs(requested - applied) > 1e-9)
                self.saturated_samples_by_joint += saturated_joints.astype(int)
            guarded_target = np.clip(
                target, self.position_guard_lower, self.position_guard_upper)
            guard_clamped_joints = np.abs(target - guarded_target) > 1e-9
            if np.any(guard_clamped_joints):
                self.position_guard_clamp_samples += 1
                self.position_guard_clamp_samples_by_joint += (
                    guard_clamped_joints.astype(int))
            self.buffer.publish(guarded_target, now)
            self.target_samples += 1
        except ShadowJointJump as error:
            result = self.shadow_jumps.rejected(error.step_deg)
            if self.buffer is not None and self.sender is not None:
                self.buffer.publish(self.sender.current_command(), now)
            if self.observe_shadow_jump_faults:
                self._record_shadow_jump_event(
                    now, shadow, error, result)
                if result["fault"]:
                    # Start a fresh observation cluster after recording the
                    # point which would have terminated a real trial.
                    self.shadow_jumps.accepted()
                return
            if result["fault"]:
                reason = (
                    f"hard shadow {error.joint_name} jump "
                    f"{error.magnitude_deg:.3f} deg"
                    if result["hard"] else
                    f"shadow joint jump persisted for "
                    f"{result['consecutive']} frames; latest "
                    f"{error.joint_name}={error.magnitude_deg:.3f} deg")
                self._fault(reason)
            else:
                self.get_logger().warning(
                    "SHADOW_JUMP_DROP: holding limited command; "
                    f"joint={error.joint_name}, "
                    f"jump={error.magnitude_deg:.3f}deg, "
                    f"step_deg={np.round(error.step_deg, 3).tolist()}, "
                    f"consecutive={result['consecutive']}/"
                    f"{self.shadow_jumps.limit}")
        except ValueError as error:
            self._fault(error)

    def _record_shadow_jump_event(self, now, shadow, error, result):
        requested = self.delta_scale * np.degrees(
            shadow - self.mapper.shadow_reference)
        applied = self.mapper.apply_offset_limit(requested)
        current = (
            self.sender.current_command() - self.mapper.robot_reference
            if self.sender is not None else np.zeros(7, dtype=float))
        event = {
            "elapsed_sec": (
                0.0 if self.active_started_at is None else
                float(now - self.active_started_at)),
            "joint": error.joint_name,
            "jump_deg": error.magnitude_deg,
            "step_deg": error.step_deg.tolist(),
            "shadow_joints_deg": np.degrees(shadow).tolist(),
            "requested_offset_deg": requested.tolist(),
            "clamped_target_offset_deg": applied.tolist(),
            "current_command_offset_deg": current.tolist(),
            "consecutive": int(result["consecutive"]),
            "would_fault": bool(result["fault"]),
            "hard": bool(result["hard"]),
        }
        self.shadow_jump_event_count += 1
        self.shadow_jump_events.append(event)
        message = (
            "SHADOW_JUMP_RECORDED: holding current command and "
            f"continuing; joint={error.joint_name}, "
            f"jump={error.magnitude_deg:.3f}deg, "
            f"consecutive={result['consecutive']}, "
            f"would_fault={result['fault']}, hard={result['hard']}")
        # rclpy Humble binds severity to a logging call site. Keep warning
        # and error on separate source lines instead of dynamically choosing
        # a bound method at one call site.
        if result["fault"]:
            self.get_logger().error(message)
        else:
            self.get_logger().warning(message)

    def _robot_state(self):
        if not self.real_motion:
            return np.asarray(
                self.get_parameter("fake_initial_joints_deg").value,
                dtype=float)
        code, state = self.robot.rm_get_current_arm_state()
        if code != 0:
            raise RuntimeError(f"RM75 state query failed: {code}")
        joints = np.asarray(state.get("joint", []), dtype=float)
        if joints.shape != (7,) or not np.all(np.isfinite(joints)):
            raise RuntimeError("RM75 returned invalid joint state")
        return joints

    def _position_guards(self, initial):
        if self.real_motion:
            joint_min = np.asarray(
                self.robot.rm_algo_get_joint_min_limit(), dtype=float)
            joint_max = np.asarray(
                self.robot.rm_algo_get_joint_max_limit(), dtype=float)
        else:
            joint_min = np.asarray(
                self.get_parameter("fake_joint_min_deg").value, dtype=float)
            joint_max = np.asarray(
                self.get_parameter("fake_joint_max_deg").value, dtype=float)
            if (joint_min.shape != (7,) or joint_max.shape != (7,) or
                    not np.all(np.isfinite(joint_min)) or
                    not np.all(np.isfinite(joint_max))):
                raise RuntimeError("fake RM75 joint limits must be finite 7-vectors")
        hard_lower = joint_min + self.joint_margin
        hard_upper = joint_max - self.joint_margin
        if self.relative_offset_limit_enabled:
            lower = np.maximum(hard_lower, initial - self.max_offset)
            upper = np.minimum(hard_upper, initial + self.max_offset)
        else:
            lower = hard_lower
            upper = hard_upper
        if np.any(lower > initial) or np.any(initial > upper):
            raise RuntimeError(
                "initial RM75 joints violate the configured position guards")
        return lower, upper

    def _arm(self):
        if self.latest_shadow is None or self.latest_shadow_at is None:
            self.get_logger().warning("waiting for the first shadow JointState")
            return
        if time.monotonic() - self.latest_shadow_at > self.input_timeout:
            self.get_logger().warning("latest shadow JointState is stale")
            return
        initial = self._robot_state()
        if self.real_motion:
            safe_error = wrapped_joint_error_deg(
                initial, DEFAULT_SAFE_HOME_DEG)
            maximum_safe_error = float(np.max(np.abs(safe_error)))
            self.get_logger().warning(
                f"RM75 initial_joints_deg={np.round(initial, 3).tolist()}; "
                "safe_home_error_deg="
                f"{np.round(safe_error, 3).tolist()}")
            if maximum_safe_error > self.safe_home_tolerance:
                raise RuntimeError(
                    "RM75 is not near the validated safe home: maximum "
                    f"error {maximum_safe_error:.3f} deg exceeds "
                    f"{self.safe_home_tolerance:.3f} deg; reposition with "
                    "the teach pendant, then restart")
        lower, upper = self._position_guards(initial)
        self.position_guard_lower = lower.copy()
        self.position_guard_upper = upper.copy()
        self.mapper = ShadowJointDeltaMapper(
            self.latest_shadow, initial, self.delta_scale,
            self.max_offset, self.max_shadow_step,
            self.relative_offset_limit_enabled)
        self.buffer = LatestJointTarget()
        self.backend = (
            RealFollowJointBackend(
                self.robot, initial, lower, upper)
            if self.real_motion else FakeJointBackend())
        self.sender = JointSender(
            self.buffer, self.backend, initial,
            rate_hz=self.rate_hz,
            max_speed_deg_s=self.max_speed,
            max_accel_deg_s2=self.max_acceleration,
            max_step_deg=self.max_speed / self.rate_hz,
            min_position_deg=lower,
            max_position_deg=upper)
        self.motion_recorder = JointMotionRecorder(initial)
        self.motion_recorder.observe(initial)
        self.buffer.publish(initial)
        self.sender.start()
        self.phase = "ACTIVE"
        self.active_started_at = time.monotonic()
        self.get_logger().warning(
            "ACTIVE: references captured; real target is scaled shadow delta "
            "around the current RM75 joints" if self.real_motion else
            "ACTIVE FAKE: references captured; no robot motion")

    def _begin_synchronized_rebase(self):
        """Restore shadow home before any real-motion reference is captured."""
        self.phase = "REBASING"
        self.rebase_started_at = time.monotonic()
        self.rebase_acknowledged = False
        self.rebase_status = None
        self.rebase_monitor.reset()
        self.rebase_future = None
        self.upper_arm_rebase_future = None
        self.get_logger().warning(
            "SYNC_ZERO: waiting for the shadow rebase-home service; no "
            "robot-motion reference has been captured")

    def _watch_synchronized_rebase(self, now):
        if self.stop_event.is_set():
            self.phase = "STOP"
            self.get_logger().warning(
                "STOP: operator cancelled while synchronized zero was pending")
            return
        if now - self.rebase_started_at > self.rebase_timeout:
            self._fault(
                "synchronized shadow rebase did not stabilize within "
                f"{self.rebase_timeout:.1f}s; no motion started")
            return
        if not self.rebase_acknowledged:
            if self.rebase_future is None:
                if (not self.rebase_client.service_is_ready() or
                        not self.upper_arm_rebase_client.service_is_ready()):
                    return
                self.rebase_future = self.rebase_client.call_async(
                    Trigger.Request())
                self.upper_arm_rebase_future = (
                    self.upper_arm_rebase_client.call_async(
                        Trigger.Request()))
                self.get_logger().warning(
                    "SYNC_ZERO: requested validated shadow home, wrist "
                    "position+orientation rebase, and upper-arm rebase")
                return
            if (not self.rebase_future.done() or
                    not self.upper_arm_rebase_future.done()):
                return
            try:
                responses = (
                    self.rebase_future.result(),
                    self.upper_arm_rebase_future.result())
            except Exception as error:  # ROS future preserves service error.
                self._fault(f"common rebase service failed: {error}")
                return
            for name, response in zip(
                    ("shadow rebase-home", "upper-arm rebase"), responses):
                if response is None or not response.success:
                    message = (
                        "no response" if response is None else
                        response.message)
                    self._fault(f"{name} was rejected: {message}")
                    return
            self.rebase_acknowledged = True
            self.rebase_monitor.reset()
            self.get_logger().warning(
                "SYNC_ZERO: shadow acknowledged home reset; collecting "
                f"{self.rebase_stable_samples} stable joint frames")
            return
        status = self.rebase_status
        if status is None or not status["ready"]:
            return
        self.get_logger().warning(
            "SYNC_ZERO_READY: shadow home is stable; "
            f"samples={status['consecutive']}, "
            f"home_error={status['max_home_error_deg']:.4f}deg, "
            f"frame_step={status['max_step_deg']:.4f}deg; "
            "capturing shadow and robot joint references now")
        self._arm()

    def _publish_joint_state(self, publisher, joints_deg):
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = EXPECTED_JOINT_NAMES
        message.position = np.radians(joints_deg).tolist()
        publisher.publish(message)

    def _publish_command_shadow(self):
        """Feed the limited command RobotModel independently at 60 Hz."""
        if self.sender is None:
            return
        commanded = self.sender.current_command()
        self.motion_recorder.observe(commanded)
        self._publish_joint_state(self.command_publisher, commanded)
        measured = (
            commanded if self.latest_measured is None else
            self.latest_measured)
        self._publish_joint_state(self.measured_publisher, measured)

    def _diagnose(self, now, measured=None):
        if now - self.last_diagnostic_at < 0.2:
            return
        self.last_diagnostic_at = now
        commanded = self.sender.current_command()
        if measured is None:
            measured = commanded
        self.latest_measured = np.asarray(measured, dtype=float).copy()
        offset = commanded - self.mapper.robot_reference
        error = measured - commanded
        self.get_logger().info(
            f"BRIDGE_DIAG phase={self.phase} requested_offset_deg="
            f"{np.round(self.latest_requested_offset, 3).tolist()} "
            "command_offset_deg="
            f"{np.round(offset, 3).tolist()} "
            f"tracking_error_deg={np.round(error, 3).tolist()}")

    def _begin_return(self, reason):
        """Return through the same rate limiter after a normal trial end."""
        if self.phase != "ACTIVE":
            return
        self.phase = "RETURNING"
        self.return_started_at = time.monotonic()
        self.return_trigger = str(reason)
        self.latest_requested_offset[:] = 0.0
        self.buffer.publish(self.mapper.robot_reference,
                            self.return_started_at)
        self.get_logger().warning(
            "RETURNING_TO_START: rejecting new shadow commands and "
            f"returning to captured startup joints; trigger={reason}; "
            f"timeout={self.return_timeout:.1f}s")

    def _watch_return(self, now):
        if now - self.return_started_at > self.return_timeout:
            self._fault(
                f"return to startup joints exceeded "
                f"{self.return_timeout:.1f}s")
            return
        state, reason, _snapshot = self.buffer.snapshot()
        if state == "FAULT":
            self._fault(reason)
            return

        reference = self.mapper.robot_reference
        # Refresh the unchanged target so sender target-age metrics remain
        # meaningful while the rate limiter brings every joint back to zero.
        self.buffer.publish(reference, now)
        commanded = self.sender.current_command()
        measured = commanded
        if self.real_motion:
            try:
                measured = self._robot_state()
                result = self.tracking.process(commanded, measured)
                if result["fault"]:
                    self._fault(
                        "joint tracking error persisted during return: "
                        f"{result['error_deg']:.3f} deg")
                    return
            except (RuntimeError, ValueError) as error:
                self._fault(error)
                return
        self._diagnose(now, measured)

        command_error = float(np.max(np.abs(commanded - reference)))
        measured_error = float(np.max(np.abs(measured - reference)))
        if (command_error <= self.return_tolerance and
                measured_error <= self.return_tolerance):
            self.buffer.stop("returned to startup joints")
            self.phase = "STOP"
            self.return_completed = True
            self.return_duration = now - self.return_started_at
            self.get_logger().warning(
                "RETURNED_TO_START: captured startup joints reached; "
                f"max_error={measured_error:.3f}deg, "
                f"duration={self.return_duration:.3f}s; STOP+HOLD")

    def _watchdog(self):
        if self.phase == "WAITING_ENTER" and self.stop_event.is_set():
            self.phase = "STOP"
            self.get_logger().warning(
                "STOP: no interactive START confirmation was received")
            return
        if self.phase == "WAITING_ENTER" and self.start_event.is_set():
            try:
                self._begin_synchronized_rebase()
            except (RuntimeError, ValueError) as error:
                self._fault(error)
            return
        if self.phase == "REBASING":
            try:
                self._watch_synchronized_rebase(time.monotonic())
            except (RuntimeError, ValueError) as error:
                self._fault(error)
            return
        if self.phase == "RETURNING":
            self._watch_return(time.monotonic())
            return
        if self.phase != "ACTIVE":
            return
        now = time.monotonic()
        if self.stop_event.is_set():
            self._begin_return("operator Enter")
            return
        if (self.duration_limit_enabled and
                now - self.active_started_at >= self.max_duration):
            self._begin_return(
                f"{self.max_duration:.1f}s maximum duration reached")
            return
        if (self.latest_shadow_at is None or
                now - self.latest_shadow_at > self.input_timeout):
            self._fault(
                f"shadow JointState timeout > {self.input_timeout:.2f}s")
            return
        state, reason, _snapshot = self.buffer.snapshot()
        if state == "FAULT":
            self._fault(reason)
            return
        measured = None
        if self.real_motion:
            try:
                measured = self._robot_state()
                result = self.tracking.process(
                    self.sender.current_command(), measured)
                if result["fault"]:
                    self._fault(
                        "joint tracking error persisted: "
                        f"{result['error_deg']:.3f} deg")
                elif (result["warning"] and
                      now - self.last_tracking_warning_at >= 1.0):
                    self.last_tracking_warning_at = now
                    self.get_logger().warning(
                        "joint tracking warning: "
                        f"{result['error_deg']:.3f} deg")
            except (RuntimeError, ValueError) as error:
                self._fault(error)
                return
        self._diagnose(now, measured)

    def _fault(self, reason):
        if self.phase == "FAULT":
            return
        self.phase = "FAULT"
        self.fault_reason = str(reason)
        if self.buffer is not None:
            self.buffer.fault(self.fault_reason)
        self.get_logger().error(
            f"FAULT: {self.fault_reason}; holding current limited command")

    def _log_command_motion_report(self, motion):
        if motion.get("samples", 0) == 0:
            self.get_logger().warning(
                "COMMAND_MOTION_REPORT unavailable: bridge was not armed")
            return
        self.get_logger().warning(
            "COMMAND_MOTION_REPORT green model trajectory; "
            f"scale={self.delta_scale:.2f}, "
            f"speed={self.max_speed:.1f}deg/s, "
            f"accel={self.max_acceleration:.1f}deg/s^2, "
            "relative_cap=" + (
                "OFF (RM75 joint guards only)"
                if not self.relative_offset_limit_enabled else
                f"+/-{self.max_offset:.1f}deg"))
        for index, name in enumerate(EXPECTED_JOINT_NAMES):
            self.get_logger().warning(
                f"COMMAND_MOTION_JOINT {name} "
                f"absolute=[{motion['absolute_min_deg'][index]:+.3f},"
                f"{motion['absolute_max_deg'][index]:+.3f}]deg "
                f"offset=[{motion['offset_min_deg'][index]:+.3f},"
                f"{motion['offset_max_deg'][index]:+.3f}]deg "
                f"peak_abs={motion['peak_abs_offset_deg'][index]:.3f}deg "
                f"span={motion['offset_span_deg'][index]:.3f}deg "
                f"travel={motion['total_travel_deg'][index]:.3f}deg "
                "hard_guard_hits="
                f"{self.position_guard_clamp_samples_by_joint[index]}")

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.buffer is not None:
            self.buffer.stop("node close")
        if self.sender is not None:
            time.sleep(0.10 if self.real_motion else 0.03)
            self.sender.stop()
        if self.observe_shadow_jump_faults:
            self.get_logger().warning(
                "SHADOW_JUMP_REPORT "
                f"total={self.shadow_jump_event_count}, "
                f"retained={len(self.shadow_jump_events)}")
            for index, event in enumerate(self.shadow_jump_events, start=1):
                self.get_logger().warning(
                    f"SHADOW_JUMP_POINT {index} " +
                    json.dumps(event, sort_keys=True))
        motion = (
            self.motion_recorder.summary()
            if self.motion_recorder is not None else {"samples": 0})
        self._log_command_motion_report(motion)
        requested_min = (
            self.min_requested_offset_by_joint
            if self.target_samples else np.zeros(7))
        requested_max = (
            self.max_signed_requested_offset_by_joint
            if self.target_samples else np.zeros(7))
        summary = {
            "phase": self.phase,
            "real_motion": self.real_motion,
            "targets": self.target_samples,
            "max_relative_offset_deg": self.max_offset,
            "joint_delta_scale": self.delta_scale,
            "max_speed_deg_s": self.max_speed,
            "max_acceleration_deg_s2": self.max_acceleration,
            "relative_offset_limit_enabled": (
                self.relative_offset_limit_enabled),
            "duration_limit_enabled": self.duration_limit_enabled,
            "observe_shadow_jump_faults": self.observe_shadow_jump_faults,
            "shadow_jump_event_count": self.shadow_jump_event_count,
            "shadow_jump_events_retained": list(self.shadow_jump_events),
            "saturated_samples": self.saturated_samples,
            "saturated_samples_by_joint": (
                self.saturated_samples_by_joint.tolist()),
            "max_requested_offset_by_joint_deg": (
                self.max_requested_offset_by_joint.tolist()),
            "requested_offset_min_by_joint_deg": requested_min.tolist(),
            "requested_offset_max_by_joint_deg": requested_max.tolist(),
            "position_guard_lower_deg": (
                None if self.position_guard_lower is None else
                self.position_guard_lower.tolist()),
            "position_guard_upper_deg": (
                None if self.position_guard_upper is None else
                self.position_guard_upper.tolist()),
            "position_guard_clamp_samples": (
                self.position_guard_clamp_samples),
            "position_guard_clamp_samples_by_joint": (
                self.position_guard_clamp_samples_by_joint.tolist()),
            "command_motion": motion,
            "rejected_shadow_jump_samples": (
                self.shadow_jumps.rejected_samples),
            "max_shadow_jump_deg": self.shadow_jumps.maximum_jump,
            "max_abs_shadow_step_by_joint_deg": (
                self.shadow_jumps.maximum_by_joint.tolist()),
            "max_tracking_error_deg": self.tracking.maximum_error,
            "fault_reason": self.fault_reason,
            "return_trigger": self.return_trigger,
            "return_completed": self.return_completed,
            "return_duration_sec": self.return_duration,
            "pipeline_latency": self.latency_statistics.summary(),
            "sender": self.sender.metrics() if self.sender is not None else {},
        }
        if (not self.real_motion and isinstance(self.backend, FakeJointBackend)
                and self.backend.records and self.mapper is not None):
            commands = np.asarray(
                [record[1] for record in self.backend.records], dtype=float)
            summary["max_command_offset_deg"] = float(np.max(np.abs(
                commands - self.mapper.robot_reference)))
        if self.real_motion and self.backend is not None:
            summary["max_command_offset_deg"] = \
                self.backend.max_observed_offset
            summary["terminal_stop_code"] = (
                self.backend.fault_stop() if self.phase == "FAULT"
                else self.backend.hold_stop())
        self.get_logger().warning(
            "SHADOW_REAL_BRIDGE_SUMMARY " + json.dumps(
                summary, sort_keys=True))
        if self.connected:
            self.robot.rm_delete_robot_arm()
            self.connected = False

    def destroy_node(self):
        self.close()
        return super().destroy_node()


def _spin(node):
    try:
        while rclpy.ok() and node.phase not in ("STOP", "FAULT"):
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        if node.buffer is not None:
            node.buffer.stop("KeyboardInterrupt")
        node.phase = "STOP"
    finally:
        node.destroy_node()
        if rclpy.ok():
            with suppress(KeyboardInterrupt):
                rclpy.shutdown()


def fake_main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auto-start", action="store_true")
    args, ros_args = parser.parse_known_args(argv)
    rclpy.init(args=ros_args)
    node = RM75ShadowJointRealBridge(
        real_motion=False, auto_start=args.auto_start)
    _spin(node)
    return 0 if node.phase != "FAULT" else 1


def real_main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-ip", default="192.168.1.18")
    parser.add_argument("--robot-port", type=int, default=8080)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--enable-motion", action="store_true")
    args, ros_args = parser.parse_known_args(argv)
    if not (args.execute and args.enable_motion):
        print(
            "REFUSED: real bridge requires --execute --enable-motion",
            file=sys.stderr)
        return 2
    rclpy.init(args=ros_args)
    node = None
    try:
        node = RM75ShadowJointRealBridge(
            real_motion=True, robot_ip=args.robot_ip,
            robot_port=args.robot_port)
        _spin(node)
        return 0 if node.phase != "FAULT" else 1
    finally:
        if node is None and rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(fake_main())
