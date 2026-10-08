"""Pure offline RM75 follower for an externally supplied TCP pose."""

import math
import time
from collections import OrderedDict
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pinocchio as pin
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point, PoseStamped, Vector3Stamped
from nav_msgs.msg import Path as PathMessage
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, ColorRGBA, Float64, String, UInt64MultiArray
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray

from .offline_trajectory import (
    DEFAULT_HOME_JOINTS_RAD,
    RM75Kinematics,
    elbow_direction_error_rad,
)
from .latency_trace import (
    make_shadow_trace,
    parse_pico_trace,
    stamp_to_ns,
)
from .pose_mapper import (
    InvalidPose,
    matrix_to_quaternion_xyzw,
    normalize_quaternion_xyzw,
    quaternion_angle_rad,
    quaternion_to_matrix_xyzw,
)


NEAR_CONVERGED_IK_ERROR = 1.0e-4


def interpolate_pose(start, end, fraction):
    """Interpolate translation linearly and orientation on the shortest arc."""
    alpha = float(fraction)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("pose interpolation fraction must be in [0, 1]")
    translation = (
        (1.0 - alpha) * start.translation + alpha * end.translation)
    local_rotation_delta = start.rotation.T @ end.rotation
    rotation = start.rotation @ pin.exp3(
        alpha * pin.log3(local_rotation_delta))
    return pin.SE3(rotation, translation)


def _ik_acceptable(accepted, error):
    return bool(accepted) or float(np.linalg.norm(error)) <= NEAR_CONVERGED_IK_ERROR


@dataclass(frozen=True)
class BoundaryIKResult:
    joints: np.ndarray
    target: pin.SE3
    accepted: bool
    status: str
    fraction: float
    iterations: int
    error: np.ndarray


class ElbowAssistController:
    """Ramp a quality-gated upper-arm direction into or out of the IK."""

    MODES = ("off", "observe", "assist_low")

    def __init__(self, mode="off", configured_weight=0.02,
                 timeout_sec=0.35, ramp_up_sec=2.0,
                 ramp_down_sec=0.5):
        self.mode = str(mode).strip().lower()
        self.configured_weight = float(configured_weight)
        self.timeout = float(timeout_sec)
        self.ramp_up = float(ramp_up_sec)
        self.ramp_down = float(ramp_down_sec)
        if self.mode not in self.MODES:
            raise ValueError(
                f"elbow assist mode must be one of {self.MODES}")
        if (self.configured_weight < 0.0 or self.timeout <= 0.0 or
                self.ramp_up <= 0.0 or self.ramp_down <= 0.0):
            raise ValueError("invalid elbow assistance timing or weight")
        self.direction = None
        self.direction_at = None
        self.quality_valid = False
        self.weight = 0.0
        self.last_update_at = None
        self.status = "OFF" if self.mode == "off" else "WAITING"

    def receive_direction(self, direction, now):
        value = np.asarray(direction, dtype=float)
        norm = float(np.linalg.norm(value))
        if value.shape != (3,) or not np.all(np.isfinite(value)) or norm < 1e-6:
            raise ValueError("elbow direction must be a finite nonzero 3-vector")
        self.direction = value / norm
        self.direction_at = float(now)

    def reset(self):
        """Clear the old upper-arm observation during a common rebase."""
        self.direction = None
        self.direction_at = None
        self.quality_valid = False
        self.weight = 0.0
        self.last_update_at = None
        self.status = "OFF" if self.mode == "off" else "WAITING"

    def set_quality(self, valid):
        self.quality_valid = bool(valid)

    def update(self, now):
        now = float(now)
        elapsed = 0.0 if self.last_update_at is None else max(
            0.0, now - self.last_update_at)
        self.last_update_at = now
        fresh = bool(
            self.direction is not None and self.direction_at is not None and
            now - self.direction_at <= self.timeout)
        enabled = bool(
            self.mode == "assist_low" and self.quality_valid and fresh)
        target = self.configured_weight if enabled else 0.0
        duration = self.ramp_up if target > self.weight else self.ramp_down
        maximum_step = self.configured_weight * elapsed / duration
        if self.weight < target:
            self.weight = min(target, self.weight + maximum_step)
        else:
            self.weight = max(target, self.weight - maximum_step)

        if self.mode == "off":
            self.status = "OFF"
        elif self.mode == "observe":
            self.status = "OBSERVE"
        elif not self.quality_valid:
            self.status = "INVALID"
        elif not fresh:
            self.status = "STALE"
        elif self.weight + 1e-9 < self.configured_weight:
            self.status = "RAMPING_IN"
        else:
            self.status = "ACTIVE"
        direction = self.direction if fresh and self.weight > 0.0 else None
        return direction, self.weight, self.status

    def observation(self, now):
        fresh = bool(
            self.direction is not None and self.direction_at is not None and
            float(now) - self.direction_at <= self.timeout)
        return self.direction.copy() if fresh else None


def solve_with_boundary_fallback(kinematics, requested_target, reachable_pose,
                                 seed, nominal, enabled=True,
                                 search_iterations=7,
                                 min_progress_fraction=0.01,
                                 elbow_direction=None,
                                 elbow_weight=0.0,
                                 elbow_deadband_rad=math.radians(2.0),
                                 max_elbow_error_rad=math.radians(10.0),
                                 adaptive_singularity_damping=False):
    """Solve a target or retreat to the furthest IK-reachable pose on its path."""
    elbow_options = {}
    if elbow_direction is not None and float(elbow_weight) > 0.0:
        elbow_options = {
            "elbow_direction": elbow_direction,
            "elbow_weight": elbow_weight,
            "elbow_deadband_rad": elbow_deadband_rad,
            "max_elbow_error_rad": max_elbow_error_rad,
        }
    solve_options = dict(elbow_options)
    if adaptive_singularity_damping:
        solve_options["adaptive_singularity_damping"] = True
    solved, accepted, iterations, error = kinematics.solve(
        requested_target, seed, nominal=nominal, **solve_options)
    error = np.asarray(error, dtype=float)
    if _ik_acceptable(accepted, error):
        return BoundaryIKResult(
            np.asarray(solved, dtype=float), requested_target, True, "direct",
            1.0, int(iterations), error)

    direct_result = BoundaryIKResult(
        np.asarray(seed, dtype=float).copy(), reachable_pose.copy(), False,
        "rejected", 0.0, int(iterations), error)
    if not enabled:
        return direct_result

    count = int(search_iterations)
    minimum = float(min_progress_fraction)
    if count < 1:
        raise ValueError("boundary search iterations must be positive")
    if not 0.0 < minimum <= 1.0:
        raise ValueError("minimum boundary progress must be in (0, 1]")

    lower = 0.0
    upper = 1.0
    best = None
    for _ in range(count):
        fraction = 0.5 * (lower + upper)
        candidate = interpolate_pose(reachable_pose, requested_target, fraction)
        candidate_joints, candidate_accepted, candidate_iterations, candidate_error = (
            kinematics.solve(
                candidate, seed, nominal=nominal, **solve_options))
        candidate_error = np.asarray(candidate_error, dtype=float)
        if _ik_acceptable(candidate_accepted, candidate_error):
            lower = fraction
            best = BoundaryIKResult(
                np.asarray(candidate_joints, dtype=float), candidate, True,
                "boundary", fraction, int(candidate_iterations), candidate_error)
        else:
            upper = fraction

    if best is None or best.fraction < minimum:
        return direct_result
    return best


class OperatorPoseMapper:
    """Map absolute or rebased relative operator poses into RM75 base_link."""

    def __init__(self, home_pose, input_mode="absolute",
                 source_to_base=None, position_scale=1.0):
        mode = str(input_mode).strip().lower()
        if mode not in ("absolute", "relative"):
            raise ValueError("input_mode must be absolute or relative")
        self.input_mode = mode
        self.source_to_base = np.eye(3) if source_to_base is None else np.asarray(
            source_to_base, dtype=float).reshape(3, 3)
        if not np.all(np.isfinite(self.source_to_base)):
            raise ValueError("source_to_base must be finite")
        if not np.allclose(
                self.source_to_base @ self.source_to_base.T,
                np.eye(3), atol=1e-6):
            raise ValueError("source_to_base must be orthonormal")
        if not math.isclose(
                float(np.linalg.det(self.source_to_base)), 1.0, abs_tol=1e-6):
            raise ValueError("source_to_base must have determinant +1")
        self.position_scale = float(position_scale)
        if not np.isfinite(self.position_scale) or self.position_scale <= 0.0:
            raise ValueError("position_scale must be positive")
        self.anchor_pose = home_pose.copy()
        self.source_position = None
        self.source_rotation = None
        self.last_diagnostics = None

    def rebase(self, anchor_pose):
        """Make the next source sample correspond to ``anchor_pose``."""
        self.anchor_pose = anchor_pose.copy()
        self.source_position = None
        self.source_rotation = None

    def process(self, position, quaternion_xyzw):
        position = np.asarray(position, dtype=float)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise InvalidPose("position must contain three finite values")
        quaternion = normalize_quaternion_xyzw(quaternion_xyzw)
        rotation = quaternion_to_matrix_xyzw(quaternion)

        if self.input_mode == "absolute":
            target = pin.SE3(rotation, position)
            self.last_diagnostics = {
                "pico_position": position.copy(),
                "pico_quaternion": quaternion.copy(),
                "relative_position": np.zeros(3),
                "relative_rotation": np.eye(3),
                "mapped_position": position.copy(),
                "mapped_rotation": rotation.copy(),
                "requested_target": target.copy(),
            }
            return target

        if self.source_position is None:
            self.source_position = position.copy()
            self.source_rotation = rotation.copy()
        relative_position = position - self.source_position
        relative_rotation = rotation @ self.source_rotation.T
        mapped_rotation = (
            self.source_to_base @ relative_rotation @ self.source_to_base.T)
        mapped_position = self.source_to_base @ relative_position
        target = pin.SE3(
            mapped_rotation @ self.anchor_pose.rotation,
            self.anchor_pose.translation + self.position_scale * mapped_position,
        )
        self.last_diagnostics = {
            "pico_position": position.copy(),
            "pico_quaternion": quaternion.copy(),
            "relative_position": relative_position.copy(),
            "relative_rotation": relative_rotation.copy(),
            "mapped_position": mapped_position.copy(),
            "mapped_rotation": mapped_rotation.copy(),
            "requested_target": target.copy(),
        }
        return target


class RM75ExternalPoseFollower(Node):
    """Subscribe to PoseStamped, solve RM75 IK, and publish an RViz shadow."""

    def __init__(self, parameter_overrides=None):
        super().__init__(
            "rm75_external_pose_follower",
            parameter_overrides=parameter_overrides)
        defaults = {
            "input_topic": "/rm75_sim/target_pose_cmd",
            "input_latency_topic": "/pico/right_wrist/latency_trace",
            "shadow_latency_topic": "/rm75_sim/latency_trace",
            "input_frame": "base_link",
            "input_mode": "absolute",
            "source_to_base": np.eye(3).reshape(-1).tolist(),
            "position_scale": 1.0,
            "publish_rate_hz": 60.0,
            "input_timeout_sec": 0.5,
            "max_position_jump_m": 0.15,
            "max_rotation_jump_rad": math.radians(60.0),
            "enable_ik_boundary_fallback": True,
            "boundary_search_iterations": 7,
            "min_boundary_progress_fraction": 0.01,
            "enable_singularity_adaptive_damping": False,
            "enable_pose_diagnostics": False,
            "diagnostic_every_n_frames": 1,
            "trace_points": 1800,
            "end_effector_frame": "Link7",
            "home_joints_rad": DEFAULT_HOME_JOINTS_RAD.tolist(),
            "elbow_assist_mode": "off",
            "elbow_direction_topic": "/rm75_sim/elbow_direction_observed",
            "elbow_quality_topic": "/rm75_sim/elbow_observation_valid",
            "elbow_assist_weight": 0.02,
            "elbow_assist_timeout_sec": 0.35,
            "elbow_assist_ramp_up_sec": 2.0,
            "elbow_assist_ramp_down_sec": 0.5,
            "elbow_assist_deadband_deg": 2.0,
            "elbow_assist_max_error_deg": 10.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        self.input_topic = str(self.get_parameter("input_topic").value)
        self.input_frame = str(self.get_parameter("input_frame").value)
        self.publish_rate = float(
            self.get_parameter("publish_rate_hz").value)
        self.timeout = float(self.get_parameter("input_timeout_sec").value)
        self.max_position_jump = float(
            self.get_parameter("max_position_jump_m").value)
        self.max_rotation_jump = float(
            self.get_parameter("max_rotation_jump_rad").value)
        self.trace_points = int(self.get_parameter("trace_points").value)
        self.enable_boundary_fallback = bool(
            self.get_parameter("enable_ik_boundary_fallback").value)
        self.boundary_search_iterations = int(
            self.get_parameter("boundary_search_iterations").value)
        self.min_boundary_progress = float(
            self.get_parameter("min_boundary_progress_fraction").value)
        self.enable_singularity_adaptive_damping = bool(self.get_parameter(
            "enable_singularity_adaptive_damping").value)
        self.enable_pose_diagnostics = bool(
            self.get_parameter("enable_pose_diagnostics").value)
        self.diagnostic_every_n_frames = int(
            self.get_parameter("diagnostic_every_n_frames").value)
        self.elbow_deadband_rad = math.radians(float(
            self.get_parameter("elbow_assist_deadband_deg").value))
        self.max_elbow_error_rad = math.radians(float(
            self.get_parameter("elbow_assist_max_error_deg").value))
        if self.publish_rate <= 0.0 or self.timeout <= 0.0:
            raise ValueError("publish rate and timeout must be positive")
        if self.max_position_jump <= 0.0 or self.max_rotation_jump <= 0.0:
            raise ValueError("input jump limits must be positive")
        if self.trace_points < 2:
            raise ValueError("trace_points must be at least two")
        if self.boundary_search_iterations < 1:
            raise ValueError("boundary_search_iterations must be positive")
        if not 0.0 < self.min_boundary_progress <= 1.0:
            raise ValueError("min_boundary_progress_fraction must be in (0, 1]")
        if self.diagnostic_every_n_frames < 1:
            raise ValueError("diagnostic_every_n_frames must be positive")
        if (self.elbow_deadband_rad < 0.0 or self.max_elbow_error_rad <= 0.0 or
                self.elbow_deadband_rad >= self.max_elbow_error_rad):
            raise ValueError("invalid elbow assist deadband or maximum error")

        urdf = (Path(get_package_share_directory("rm_description")) /
                "urdf" / "rm_75.urdf")
        self.kinematics = RM75Kinematics(
            urdf, str(self.get_parameter("end_effector_frame").value))
        self.home_joints = np.asarray(
            self.get_parameter("home_joints_rad").value, dtype=float)
        if self.home_joints.shape != (7,):
            raise ValueError("home_joints_rad must contain seven values")
        self.joints = self.home_joints.copy()
        self.fk_pose = self.kinematics.forward(self.joints)
        self.target_pose = self.fk_pose.copy()
        self.last_joint_step_deg = np.zeros(7, dtype=float)
        self.cumulative_joint_travel_deg = np.zeros(7, dtype=float)
        self.mapper = OperatorPoseMapper(
            self.fk_pose,
            input_mode=self.get_parameter("input_mode").value,
            source_to_base=self.get_parameter("source_to_base").value,
            position_scale=self.get_parameter("position_scale").value,
        )

        self.previous_input_position = None
        self.previous_input_quaternion = None
        self.last_input_at = None
        self.last_timeout_warning_at = None
        self.accepted_count = 0
        self.rejected_count = 0
        self.input_count = 0
        self.boundary_count = 0
        self.last_elbow_status_log = None
        self.last_elbow_log_at = -float("inf")
        self.pico_latency_traces = OrderedDict()
        self.latest_ik_trace = None
        self.latest_ik_pose_key = 0
        self.target_path = self._empty_path()
        self.fk_path = self._empty_path()
        self.elbow_assist = ElbowAssistController(
            mode=self.get_parameter("elbow_assist_mode").value,
            configured_weight=self.get_parameter(
                "elbow_assist_weight").value,
            timeout_sec=self.get_parameter(
                "elbow_assist_timeout_sec").value,
            ramp_up_sec=self.get_parameter(
                "elbow_assist_ramp_up_sec").value,
            ramp_down_sec=self.get_parameter(
                "elbow_assist_ramp_down_sec").value,
        )

        self.joint_pub = self.create_publisher(
            JointState, "/rm75_sim/joint_states", 10)
        self.shadow_latency_pub = self.create_publisher(
            UInt64MultiArray,
            str(self.get_parameter("shadow_latency_topic").value), 10)
        self.target_pose_pub = self.create_publisher(
            PoseStamped, "/rm75_sim/target_pose", 10)
        self.fk_pose_pub = self.create_publisher(
            PoseStamped, "/rm75_sim/fk_pose", 10)
        self.target_path_pub = self.create_publisher(
            PathMessage, "/rm75_sim/target_path", 10)
        self.fk_path_pub = self.create_publisher(
            PathMessage, "/rm75_sim/fk_path", 10)
        self.elbow_weight_pub = self.create_publisher(
            Float64, "/rm75_sim/elbow_assist_weight", 10)
        self.elbow_status_pub = self.create_publisher(
            String, "/rm75_sim/elbow_assist_status", 10)
        self.elbow_marker_pub = self.create_publisher(
            MarkerArray, "/rm75_sim/elbow_assist_markers", 10)
        latest_only_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE)
        self.create_subscription(
            PoseStamped, self.input_topic, self._pose_callback,
            latest_only_qos)
        self.create_subscription(
            UInt64MultiArray,
            str(self.get_parameter("input_latency_topic").value),
            self._pico_latency_callback, 10)
        self.create_subscription(
            Vector3Stamped,
            str(self.get_parameter("elbow_direction_topic").value),
            self._elbow_direction_callback, 10)
        self.create_subscription(
            Bool,
            str(self.get_parameter("elbow_quality_topic").value),
            self._elbow_quality_callback, 10)
        self.create_service(Trigger, "/rm75_sim/rebase", self._rebase)
        self.create_service(
            Trigger, "/rm75_sim/rebase_home", self._rebase_home)
        self.create_timer(1.0 / self.publish_rate, self._publish)

        self.get_logger().warning(
            "OFFLINE ONLY: external PoseStamped -> Pinocchio IK -> RViz; "
            "no RealMan connection and no robot motion API")
        self.get_logger().info(
            f"input={self.input_topic} frame={self.input_frame!r} "
            f"mode={self.mapper.input_mode}; position=m quaternion=ROS xyzw; "
            "pose QoS=latest-only best-effort depth=1")
        self.get_logger().info(
            "IK boundary fallback="
            f"{self.enable_boundary_fallback} iterations="
            f"{self.boundary_search_iterations}; pose diagnostics="
            f"{self.enable_pose_diagnostics} every_n="
            f"{self.diagnostic_every_n_frames}")
        self.get_logger().warning(
            "IK singularity adaptive damping="
            f"{self.enable_singularity_adaptive_damping}; "
            "legacy damping is preserved for sigma_min >= 0.030")
        self.get_logger().warning(
            "elbow assist mode="
            f"{self.elbow_assist.mode} configured_weight="
            f"{self.elbow_assist.configured_weight:.3f}; "
            "TCP pose remains the primary task and safe-home posture remains active")

    def _elbow_direction_callback(self, message):
        if message.header.frame_id != "base_link":
            self.get_logger().warning(
                f"ignoring elbow direction frame {message.header.frame_id!r}",
                throttle_duration_sec=2.0)
            return
        try:
            self.elbow_assist.receive_direction([
                message.vector.x, message.vector.y, message.vector.z,
            ], time.monotonic())
        except ValueError as error:
            self.elbow_assist.set_quality(False)
            self.get_logger().warning(
                f"invalid elbow direction: {error}",
                throttle_duration_sec=1.0)

    def _elbow_quality_callback(self, message):
        self.elbow_assist.set_quality(message.data)

    def _pico_latency_callback(self, message):
        try:
            trace = parse_pico_trace(message.data)
        except ValueError as error:
            self.get_logger().warning(
                f"invalid PICO latency trace: {error}",
                throttle_duration_sec=2.0)
            return
        key = trace["pose_key_ns"]
        self.pico_latency_traces[key] = trace
        self.pico_latency_traces.move_to_end(key)
        while len(self.pico_latency_traces) > 256:
            self.pico_latency_traces.popitem(last=False)
        # Cross-topic DDS delivery can occasionally deliver the companion
        # trace just after its pose. Fill the accepted record retroactively.
        if key == self.latest_ik_pose_key and self.latest_ik_trace is not None:
            self.latest_ik_trace.update(trace)

    @staticmethod
    def _empty_path():
        message = PathMessage()
        message.header.frame_id = "base_link"
        return message

    @staticmethod
    def _pose_message(stamp, pose):
        message = PoseStamped()
        message.header.stamp = stamp
        message.header.frame_id = "base_link"
        message.pose.position.x = float(pose.translation[0])
        message.pose.position.y = float(pose.translation[1])
        message.pose.position.z = float(pose.translation[2])
        quaternion = pin.Quaternion(pose.rotation)
        quaternion.normalize()
        xyzw = quaternion.coeffs()
        message.pose.orientation.x = float(xyzw[0])
        message.pose.orientation.y = float(xyzw[1])
        message.pose.orientation.z = float(xyzw[2])
        message.pose.orientation.w = float(xyzw[3])
        return message

    def _validate_input_continuity(self, position, quaternion):
        if self.previous_input_position is None:
            return True

        position_jump = float(np.linalg.norm(
            position - self.previous_input_position))
        rotation_jump = quaternion_angle_rad(
            quaternion, self.previous_input_quaternion)

        if (position_jump > self.max_position_jump or
                rotation_jump > self.max_rotation_jump):

            reason = []
            if position_jump > self.max_position_jump:
                reason.append(
                    f"position jump {position_jump:.3f} m > "
                    f"{self.max_position_jump:.3f} m")
            if rotation_jump > self.max_rotation_jump:
                reason.append(
                    f"rotation jump {rotation_jump:.3f} rad > "
                    f"{self.max_rotation_jump:.3f} rad")

            # PICO/Tracker 发生重定位时：
            # 机械臂保持当前位置，不追踪错误的绝对坐标；
            # 重新建立 operator -> RM75 的相对映射。
            self.mapper.rebase(self.fk_pose)

            # 记录当前 raw pose，避免后续一直和旧坐标比较而永久锁死。
            self.previous_input_position = position.copy()
            self.previous_input_quaternion = quaternion.copy()
            self.last_input_at = time.monotonic()

            self.get_logger().warning(
                "PICO tracking discontinuity detected: "
                + ", ".join(reason)
                + "; holding RM75 pose and automatically rebasing",
                throttle_duration_sec=1.0)

            return False

        return True

    def _pose_callback(self, message):
        result = None
        ik_receive_ns = time.time_ns()
        pose_key_ns = stamp_to_ns(message.header.stamp)
        ik_start_ns = 0
        ik_end_ns = 0
        try:
            self.input_count += 1
            if message.header.frame_id != self.input_frame:
                raise InvalidPose(
                    f"frame_id must be {self.input_frame!r}, got "
                    f"{message.header.frame_id!r}")
            position = np.array([
                message.pose.position.x,
                message.pose.position.y,
                message.pose.position.z,
            ], dtype=float)
            quaternion = normalize_quaternion_xyzw(np.array([
                message.pose.orientation.x,
                message.pose.orientation.y,
                message.pose.orientation.z,
                message.pose.orientation.w,
            ], dtype=float))
            if not self._validate_input_continuity(position, quaternion):
                return

            # Continuity belongs to the raw operator stream, not IK success.
            # Update these before solving IK so an IK rejection cannot make
            # later valid PICO frames look like a huge stale-position jump.
            self.previous_input_position = position.copy()
            self.previous_input_quaternion = quaternion.copy()
            self.last_input_at = time.monotonic()

            ik_start_ns = time.time_ns()
            target = self.mapper.process(position, quaternion)
            elbow_direction, elbow_weight, _ = self.elbow_assist.update(
                time.monotonic())
            result = solve_with_boundary_fallback(
                self.kinematics,
                target,
                self.fk_pose,
                self.joints,
                self.home_joints,
                enabled=self.enable_boundary_fallback,
                search_iterations=self.boundary_search_iterations,
                min_progress_fraction=self.min_boundary_progress,
                elbow_direction=elbow_direction,
                elbow_weight=elbow_weight,
                elbow_deadband_rad=self.elbow_deadband_rad,
                max_elbow_error_rad=self.max_elbow_error_rad,
                adaptive_singularity_damping=(
                    self.enable_singularity_adaptive_damping),
            )
            ik_end_ns = time.time_ns()
            ik_error_norm = float(np.linalg.norm(result.error))
            if (self.enable_singularity_adaptive_damping and
                    np.isfinite(
                        self.kinematics.last_minimum_singular_value)):
                self.get_logger().info(
                    "SINGULARITY_DAMPING "
                    f"active={self.kinematics.adaptive_damping_was_active} "
                    "sigma_min="
                    f"{self.kinematics.last_minimum_singular_value:.6f} "
                    f"damping={self.kinematics.last_damping:.8f}",
                    throttle_duration_sec=1.0)

            if not result.accepted:
                raise InvalidPose(
                    f"IK did not converge after {result.iterations} iterations; "
                    f"error={ik_error_norm:.6g}")

            if result.status == "boundary":
                self.boundary_count += 1
                self.get_logger().warning(
                    "requested target unreachable; applying nearest reachable "
                    f"path point at {100.0 * result.fraction:.1f}% "
                    f"(IK error={ik_error_norm:.6g})",
                    throttle_duration_sec=1.0)

            new_joints = np.asarray(result.joints, dtype=float)
            self.last_joint_step_deg = np.degrees(new_joints - self.joints)
            self.cumulative_joint_travel_deg += np.abs(
                self.last_joint_step_deg)
            self.joints = new_joints
            self.target_pose = result.target
            self.fk_pose = self.kinematics.forward(self.joints)
            self.previous_input_position = position.copy()
            self.previous_input_quaternion = quaternion.copy()
            self.last_input_at = time.monotonic()
            self.accepted_count += 1
            upstream = self.pico_latency_traces.pop(pose_key_ns, {})
            self.latest_ik_trace = dict(upstream)
            self.latest_ik_trace.update({
                "pose_key_ns": pose_key_ns,
                "ik_receive_ns": ik_receive_ns,
                "ik_start_ns": ik_start_ns,
                "ik_end_ns": ik_end_ns,
            })
            self.latest_ik_pose_key = pose_key_ns
            self._log_pose_diagnostics(result)
        except (InvalidPose, ValueError, np.linalg.LinAlgError) as error:
            self.rejected_count += 1
            if result is not None:
                self._log_pose_diagnostics(result)
            self.get_logger().warning(
                f"target rejected; holding last pose: {error}",
                throttle_duration_sec=1.0)

    @staticmethod
    def _rotation_summary(rotation):
        rotvec_deg = np.degrees(pin.log3(rotation))
        euler_deg = np.degrees(pin.rpy.matrixToRpy(rotation))
        return rotvec_deg, euler_deg

    def _log_pose_diagnostics(self, result):
        if (not self.enable_pose_diagnostics or
                self.input_count % self.diagnostic_every_n_frames != 0):
            return
        diagnostic = self.mapper.last_diagnostics
        if diagnostic is None:
            return
        relative_rotvec, relative_euler = self._rotation_summary(
            diagnostic["relative_rotation"])
        mapped_rotvec, mapped_euler = self._rotation_summary(
            diagnostic["mapped_rotation"])
        requested = diagnostic["requested_target"]
        final_target = result.target
        requested_q = matrix_to_quaternion_xyzw(requested.rotation)
        final_q = matrix_to_quaternion_xyzw(final_target.rotation)
        final_rpy = np.degrees(pin.rpy.matrixToRpy(final_target.rotation))
        requested_tool_z = requested.rotation[:, 2]
        final_tool_z = final_target.rotation[:, 2]
        joints_deg = np.degrees(self.joints)
        joint_delta_home_deg = np.degrees(
            self.joints - self.home_joints)
        self.get_logger().info(
            "POSE_DIAG "
            f"frame={self.input_count} status={result.status} "
            f"boundary_alpha={result.fraction:.4f} "
            f"pico_p={np.round(diagnostic['pico_position'], 5).tolist()} "
            f"pico_q_xyzw={np.round(diagnostic['pico_quaternion'], 6).tolist()} "
            f"rel_p={np.round(diagnostic['relative_position'], 5).tolist()} "
            f"rel_rotvec_deg={np.round(relative_rotvec, 3).tolist()} "
            f"rel_euler_xyz_deg={np.round(relative_euler, 3).tolist()} "
            f"mapped_p={np.round(diagnostic['mapped_position'], 5).tolist()} "
            f"mapped_rotvec_deg={np.round(mapped_rotvec, 3).tolist()} "
            f"mapped_euler_xyz_deg={np.round(mapped_euler, 3).tolist()} "
            f"requested_xyz={np.round(requested.translation, 5).tolist()} "
            f"requested_q_xyzw={np.round(requested_q, 6).tolist()} "
            f"requested_tool_z={np.round(requested_tool_z, 5).tolist()} "
            f"final_xyz={np.round(final_target.translation, 5).tolist()} "
            f"final_q_xyzw={np.round(final_q, 6).tolist()} "
            f"final_euler_xyz_deg={np.round(final_rpy, 3).tolist()} "
            f"final_tool_z={np.round(final_tool_z, 5).tolist()} "
            f"ik_joints_deg={np.round(joints_deg, 3).tolist()} "
            "ik_joint_delta_home_deg="
            f"{np.round(joint_delta_home_deg, 3).tolist()} "
            f"ik_joint_step_deg={np.round(self.last_joint_step_deg, 3).tolist()} "
            "ik_joint_cumulative_travel_deg="
            f"{np.round(self.cumulative_joint_travel_deg, 3).tolist()} "
            f"ik_error={float(np.linalg.norm(result.error)):.6g}")

    def _rebase(self, _request, response):
        self.mapper.rebase(self.fk_pose)
        self.previous_input_position = None
        self.previous_input_quaternion = None
        response.success = True
        response.message = (
            "next operator sample will become the reference at current RM75 pose")
        self.get_logger().info(response.message)
        return response

    def _rebase_home(self, _request, response):
        """Restore validated shadow home and rebase the next PICO sample."""
        self.joints = self.home_joints.copy()
        self.fk_pose = self.kinematics.forward(self.joints)
        self.target_pose = self.fk_pose.copy()
        self.mapper.rebase(self.fk_pose)
        self.previous_input_position = None
        self.previous_input_quaternion = None
        self.last_input_at = None
        self.last_joint_step_deg = np.zeros(7, dtype=float)
        self.cumulative_joint_travel_deg = np.zeros(7, dtype=float)
        self.elbow_assist.reset()
        self.target_path = self._empty_path()
        self.fk_path = self._empty_path()
        response.success = True
        response.message = (
            "shadow restored to validated home; next operator sample will "
            "become the PICO position+orientation reference")
        self.get_logger().warning(response.message)
        return response

    def _append_trace(self, path, pose_message):
        path.poses.append(pose_message)
        if len(path.poses) > self.trace_points:
            path.poses = path.poses[-self.trace_points:]

    def _publish(self):
        now = time.monotonic()
        stamp = self.get_clock().now().to_msg()
        shadow_publish_ns = time.time_ns()
        shadow_trace = UInt64MultiArray()
        shadow_trace.data = make_shadow_trace(
            stamp_to_ns(stamp), self.latest_ik_trace, shadow_publish_ns)
        self.shadow_latency_pub.publish(shadow_trace)
        joint_message = JointState()
        joint_message.header.stamp = stamp
        joint_message.name = self.kinematics.joint_names
        joint_message.position = self.joints.tolist()
        self.joint_pub.publish(joint_message)

        target_message = self._pose_message(stamp, self.target_pose)
        fk_message = self._pose_message(stamp, self.fk_pose)
        self.target_pose_pub.publish(target_message)
        self.fk_pose_pub.publish(fk_message)
        self._append_trace(self.target_path, target_message)
        self._append_trace(self.fk_path, fk_message)
        self.target_path.header.stamp = stamp
        self.fk_path.header.stamp = stamp
        self.target_path_pub.publish(self.target_path)
        self.fk_path_pub.publish(self.fk_path)

        _, elbow_weight, elbow_status = self.elbow_assist.update(now)
        weight_message = Float64()
        weight_message.data = float(elbow_weight)
        self.elbow_weight_pub.publish(weight_message)
        status_message = String()
        status_message.data = elbow_status
        self.elbow_status_pub.publish(status_message)
        if (elbow_status != self.last_elbow_status_log or
                now - self.last_elbow_log_at >= 1.0):
            self.last_elbow_status_log = elbow_status
            self.last_elbow_log_at = now
            self.get_logger().info(
                f"ELBOW_ASSIST status={elbow_status} "
                f"effective_weight={elbow_weight:.4f} "
                f"configured_weight={self.elbow_assist.configured_weight:.4f}")
        self._publish_elbow_assist_markers(
            stamp, now, elbow_weight, elbow_status)

        if (self.last_input_at is not None and
                now - self.last_input_at > self.timeout and
                (self.last_timeout_warning_at is None or
                 now - self.last_timeout_warning_at > 2.0)):
            self.last_timeout_warning_at = now
            self.get_logger().warning(
                f"operator input timeout > {self.timeout:.2f}s; holding pose")

    @staticmethod
    def _marker_point(values):
        point = Point()
        point.x, point.y, point.z = map(float, values)
        return point

    @staticmethod
    def _marker_color(red, green, blue, alpha=0.9):
        return ColorRGBA(
            r=float(red), g=float(green), b=float(blue), a=float(alpha))

    def _elbow_arrow(self, stamp, marker_id, start, end, color):
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = "base_link"
        marker.ns = "rm75_elbow_assist"
        marker.id = marker_id
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.points = [
            self._marker_point(start), self._marker_point(end)]
        marker.scale.x = 0.01
        marker.scale.y = 0.022
        marker.scale.z = 0.026
        marker.color = color
        marker.lifetime.nanosec = int(0.2 * 1e9)
        return marker

    def _publish_elbow_assist_markers(self, stamp, now, weight, status):
        try:
            _, _, elbow, axis, current, _, projection = (
                self.kinematics.elbow_geometry(self.joints))
        except ValueError:
            return
        markers = MarkerArray()
        markers.markers.append(self._elbow_arrow(
            stamp, 0, projection, projection + 0.15 * current,
            self._marker_color(0.0, 1.0, 1.0)))

        desired = self.elbow_assist.observation(now)
        error_deg = None
        if desired is not None:
            desired = desired - float(desired @ axis) * axis
            desired_norm = float(np.linalg.norm(desired))
            if desired_norm > 1e-6:
                desired /= desired_norm
                error_deg = math.degrees(elbow_direction_error_rad(
                    axis, current, desired))
                markers.markers.append(self._elbow_arrow(
                    stamp, 1, projection, projection + 0.15 * desired,
                    self._marker_color(1.0, 0.1, 0.85)))

        label = Marker()
        label.header.stamp = stamp
        label.header.frame_id = "base_link"
        label.ns = "rm75_elbow_assist"
        label.id = 2
        label.type = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD
        label.pose.position = self._marker_point(
            elbow + np.array([0.0, 0.0, 0.08]))
        label.pose.orientation.w = 1.0
        label.scale.z = 0.035
        label.color = self._marker_color(1.0, 1.0, 1.0)
        error_text = "n/a" if error_deg is None else f"{error_deg:+.1f}deg"
        label.text = (
            f"elbow {status}  w={weight:.3f}  error={error_text}")
        label.lifetime.nanosec = int(0.2 * 1e9)
        markers.markers.append(label)
        self.elbow_marker_pub.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = RM75ExternalPoseFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        with suppress(KeyboardInterrupt):
            node.destroy_node()
        if rclpy.ok():
            with suppress(KeyboardInterrupt):
                rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
