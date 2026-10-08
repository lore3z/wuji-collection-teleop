"""Observe upper-arm Tracker geometry beside the wrist-driven RM75 shadow."""

from contextlib import suppress
from dataclasses import dataclass, replace
import math
import time
from pathlib import Path

import numpy as np
import pinocchio as pin
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point, PoseStamped, Vector3Stamped
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from std_msgs.msg import Bool, ColorRGBA, Float64
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray

from .elbow_arm_angle_mapper import (
    InvalidElbowDirection,
    RelativeArmAngleMapper,
    elbow_swivel_angle_rad,
)
from .offline_trajectory import RM75Kinematics


@dataclass(frozen=True)
class UpperArmObservation:
    raw_delta: np.ndarray
    mapped_arm_point: np.ndarray
    projection_point: np.ndarray
    elbow_direction: np.ndarray
    offset_m: float
    geometric_angle_deg: float
    arm_angle_delta_deg: float
    status: str
    raw_step_m: float = 0.0


class UpperArmGeometryObserver:
    """Map relative upper-arm motion and derive elbow swivel diagnostics."""

    def __init__(self, shoulder, arm_anchor, source_to_base,
                 position_scale=0.5, direction_ema_alpha=0.18,
                 singularity_threshold_m=0.015,
                 caution_threshold_m=0.05,
                 arm_angle_scale=0.25, max_arm_angle_step_deg=5.0,
                 max_raw_step_m=0.08):
        self.shoulder = np.asarray(shoulder, dtype=float)
        self.arm_anchor = np.asarray(arm_anchor, dtype=float)
        self.source_to_base = np.asarray(
            source_to_base, dtype=float).reshape(3, 3)
        self.position_scale = float(position_scale)
        self.ema_alpha = float(direction_ema_alpha)
        self.singularity_threshold = float(singularity_threshold_m)
        self.caution_threshold = float(caution_threshold_m)
        self.max_raw_step = float(max_raw_step_m)
        if self.shoulder.shape != (3,) or self.arm_anchor.shape != (3,):
            raise ValueError("shoulder and arm anchor must be 3-vectors")
        if not np.allclose(
                self.source_to_base @ self.source_to_base.T,
                np.eye(3), atol=1e-6):
            raise ValueError("source_to_base must be orthonormal")
        if not math.isclose(
                float(np.linalg.det(self.source_to_base)), 1.0, abs_tol=1e-6):
            raise ValueError("source_to_base must have determinant +1")
        if self.position_scale <= 0.0 or not 0.0 < self.ema_alpha <= 1.0:
            raise ValueError("scale and EMA alpha must be positive")
        if not 0.0 < self.singularity_threshold < self.caution_threshold:
            raise ValueError("invalid upper-arm quality thresholds")
        if self.max_raw_step <= 0.0:
            raise ValueError("max_raw_step_m must be positive")
        self.raw_reference = None
        self.last_raw = None
        self.last_observation = None
        self.filtered_direction = None
        self.last_arm_angle_delta = 0.0
        self.arm_angle_mapper = RelativeArmAngleMapper(
            0.0, scale=arm_angle_scale,
            max_step_deg=max_arm_angle_step_deg)

    def reset_reference(self):
        """Make the next upper-arm sample a fresh zero-displacement basis."""
        self.raw_reference = None
        self.last_raw = None
        self.last_observation = None
        self.filtered_direction = None
        self.last_arm_angle_delta = 0.0
        self.arm_angle_mapper.previous_direction = None
        self.arm_angle_mapper.previous_target_deg = None

    def process(self, raw_arm_position, wrist_position):
        raw = np.asarray(raw_arm_position, dtype=float)
        wrist = np.asarray(wrist_position, dtype=float)
        if raw.shape != (3,) or wrist.shape != (3,):
            raise ValueError("raw arm and wrist positions must be 3-vectors")
        if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(wrist)):
            raise ValueError("raw arm and wrist positions must be finite")
        if self.raw_reference is None:
            self.raw_reference = raw.copy()
        raw_step = 0.0 if self.last_raw is None else float(
            np.linalg.norm(raw - self.last_raw))
        if (self.last_raw is not None and raw_step > self.max_raw_step and
                self.last_observation is not None):
            # Preserve the preceding mapped point while rebasing across a PICO
            # relocalization. The held sample is explicitly invalid for IK.
            previous_delta = self.last_raw - self.raw_reference
            self.raw_reference = raw - previous_delta
            self.last_raw = raw.copy()
            held = replace(
                self.last_observation,
                raw_delta=previous_delta.copy(),
                status="HELD_RAW_JUMP",
                raw_step_m=raw_step,
            )
            self.last_observation = held
            return held
        self.last_raw = raw.copy()
        raw_delta = raw - self.raw_reference
        mapped_arm = (
            self.arm_anchor +
            self.position_scale * self.source_to_base @ raw_delta)

        shoulder_to_wrist = wrist - self.shoulder
        axis_length = float(np.linalg.norm(shoulder_to_wrist))
        if axis_length < 1e-6:
            raise InvalidElbowDirection("shoulder-wrist axis is singular")
        axis = shoulder_to_wrist / axis_length
        projection = (
            self.shoulder +
            float((mapped_arm - self.shoulder) @ axis) * axis)
        offset = mapped_arm - projection
        offset_m = float(np.linalg.norm(offset))
        if offset_m < self.singularity_threshold:
            if self.filtered_direction is None:
                raise InvalidElbowDirection(
                    "upper-arm point is too close to shoulder-wrist axis")
            direction = self.filtered_direction.copy()
            status = "HELD_NEAR_AXIS"
        else:
            direction = offset / offset_m
            if self.filtered_direction is not None:
                direction = (
                    (1.0 - self.ema_alpha) * self.filtered_direction +
                    self.ema_alpha * direction)
                direction /= np.linalg.norm(direction)
            self.filtered_direction = direction.copy()
            status = "GOOD" if offset_m >= self.caution_threshold else "CAUTION"

        geometric_angle = math.degrees(
            elbow_swivel_angle_rad(shoulder_to_wrist, direction))
        try:
            arm_angle_delta = self.arm_angle_mapper.process(
                shoulder_to_wrist, direction)
            self.last_arm_angle_delta = float(arm_angle_delta)
        except InvalidElbowDirection:
            arm_angle_delta = self.last_arm_angle_delta
            status = "HELD_STEP"
        observation = UpperArmObservation(
            raw_delta=raw_delta,
            mapped_arm_point=mapped_arm,
            projection_point=projection,
            elbow_direction=direction,
            offset_m=offset_m,
            geometric_angle_deg=geometric_angle,
            arm_angle_delta_deg=float(arm_angle_delta),
            status=status,
            raw_step_m=raw_step,
        )
        self.last_observation = observation
        return observation


class UpperArmShadowObserverNode(Node):
    """Publish RM-base upper-arm markers and detailed observation logs."""

    def __init__(self):
        super().__init__("rm75_upper_arm_shadow_observer")
        defaults = {
            "upper_arm_topic": "/pico/right_upper_arm/raw_pose",
            "wrist_target_topic": "/rm75_sim/target_pose",
            "input_frame": "pico_tracking",
            "position_scale": 0.5,
            "source_to_base": [
                0.0, 0.0, -1.0,
                -1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
            ],
            "home_joints_rad": [
                -1.5707963268, 0.4, 0.0, 1.5, 0.0,
                -0.3292036732, 0.1745329252],
            "direction_ema_alpha": 0.18,
            "arm_angle_scale": 0.25,
            "max_arm_angle_step_deg": 5.0,
            "max_raw_step_m": 0.08,
            "input_timeout_sec": 0.5,
            "diagnostic_rate_hz": 5.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.input_frame = str(self.get_parameter("input_frame").value)
        self.timeout = float(self.get_parameter("input_timeout_sec").value)
        diagnostic_rate = float(
            self.get_parameter("diagnostic_rate_hz").value)
        if self.timeout <= 0.0 or diagnostic_rate <= 0.0:
            raise ValueError("timeout and diagnostic rate must be positive")
        self.log_period = 1.0 / diagnostic_rate
        self.last_log_at = -float("inf")
        self.latest_wrist = None
        self.latest_wrist_at = None

        urdf = (Path(get_package_share_directory("rm_description")) /
                "urdf" / "rm_75.urdf")
        kinematics = RM75Kinematics(urdf)
        home = np.asarray(
            self.get_parameter("home_joints_rad").value, dtype=float)
        if home.shape != (7,):
            raise ValueError("home_joints_rad must contain seven values")
        pin.forwardKinematics(kinematics.model, kinematics.data, home)
        shoulder_id = kinematics.model.getJointId("joint2")
        elbow_id = kinematics.model.getJointId("joint4")
        shoulder = kinematics.data.oMi[shoulder_id].translation.copy()
        elbow = kinematics.data.oMi[elbow_id].translation.copy()
        self.geometry = UpperArmGeometryObserver(
            shoulder,
            elbow,
            self.get_parameter("source_to_base").value,
            position_scale=self.get_parameter("position_scale").value,
            direction_ema_alpha=self.get_parameter(
                "direction_ema_alpha").value,
            arm_angle_scale=self.get_parameter("arm_angle_scale").value,
            max_arm_angle_step_deg=self.get_parameter(
                "max_arm_angle_step_deg").value,
            max_raw_step_m=self.get_parameter("max_raw_step_m").value,
        )

        self.marker_pub = self.create_publisher(
            MarkerArray, "/rm75_sim/upper_arm_markers", 10)
        self.direction_pub = self.create_publisher(
            Vector3Stamped, "/rm75_sim/elbow_direction_observed", 10)
        self.arm_angle_pub = self.create_publisher(
            Float64, "/rm75_sim/arm_angle_delta_observed_deg", 10)
        self.quality_pub = self.create_publisher(
            Bool, "/rm75_sim/elbow_observation_valid", 10)
        latest_only_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE)
        self.create_subscription(
            PoseStamped,
            str(self.get_parameter("wrist_target_topic").value),
            self._wrist, latest_only_qos)
        self.create_subscription(
            PoseStamped,
            str(self.get_parameter("upper_arm_topic").value),
            self._upper_arm, latest_only_qos)
        self.create_service(
            Trigger, "/rm75_sim/rebase_upper_arm", self._rebase_upper_arm)
        self.get_logger().warning(
            "upper-arm observer publishes a quality-gated elbow candidate; "
            "the follower decides whether its weight is zero or assist_low")
        self.get_logger().info(
            f"RM anchors: shoulder={np.round(shoulder, 4).tolist()} "
            f"upper_arm={np.round(elbow, 4).tolist()}")

    def _wrist(self, message):
        self.latest_wrist = np.array([
            message.pose.position.x,
            message.pose.position.y,
            message.pose.position.z,
        ], dtype=float)
        self.latest_wrist_at = time.monotonic()

    def _rebase_upper_arm(self, _request, response):
        self.geometry.reset_reference()
        self.latest_wrist = None
        self.latest_wrist_at = None
        self._publish_quality(False)
        response.success = True
        response.message = (
            "upper-arm history cleared; next synchronized sample becomes "
            "the elbow-assist reference")
        self.get_logger().warning(response.message)
        return response

    def _upper_arm(self, message):
        if message.header.frame_id != self.input_frame:
            self.get_logger().warning(
                f"ignoring upper-arm frame {message.header.frame_id!r}",
                throttle_duration_sec=2.0)
            return
        now = time.monotonic()
        if (self.latest_wrist is None or self.latest_wrist_at is None or
                now - self.latest_wrist_at > self.timeout):
            self._publish_quality(False)
            self.get_logger().warning(
                "waiting for fresh RM75 wrist target",
                throttle_duration_sec=2.0)
            return
        raw_arm = np.array([
            message.pose.position.x,
            message.pose.position.y,
            message.pose.position.z,
        ], dtype=float)
        try:
            observation = self.geometry.process(raw_arm, self.latest_wrist)
        except (InvalidElbowDirection, ValueError) as error:
            self._publish_quality(False)
            self.get_logger().warning(
                f"upper-arm geometry unavailable: {error}",
                throttle_duration_sec=1.0)
            return
        self._publish_observation(message.header.stamp, observation)
        if now - self.last_log_at >= self.log_period:
            self.last_log_at = now
            self.get_logger().info(
                "UPPER_ARM_OBS "
                f"status={observation.status} "
                f"raw_delta_m={np.round(observation.raw_delta, 4).tolist()} "
                f"mapped_point={np.round(observation.mapped_arm_point, 4).tolist()} "
                f"wrist={np.round(self.latest_wrist, 4).tolist()} "
                f"offset_m={observation.offset_m:.4f} "
                f"raw_step_m={observation.raw_step_m:.4f} "
                f"elbow_dir={np.round(observation.elbow_direction, 4).tolist()} "
                f"geom_angle_deg={observation.geometric_angle_deg:+.2f} "
                f"arm_angle_delta_deg={observation.arm_angle_delta_deg:+.2f}")

    @staticmethod
    def _point(values):
        point = Point()
        point.x, point.y, point.z = map(float, values)
        return point

    @staticmethod
    def _color(red, green, blue, alpha=0.9):
        return ColorRGBA(
            r=float(red), g=float(green), b=float(blue), a=float(alpha))

    def _arrow(self, stamp, marker_id, name, start, end, color):
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = "base_link"
        marker.ns = "rm75_upper_arm_observer"
        marker.id = marker_id
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.points = [self._point(start), self._point(end)]
        marker.scale.x = 0.012
        marker.scale.y = 0.025
        marker.scale.z = 0.03
        marker.color = color
        marker.lifetime.nanosec = int(0.25 * 1e9)
        return marker

    def _publish_observation(self, stamp, observation):
        self._publish_quality(observation.status == "GOOD")
        direction = Vector3Stamped()
        direction.header.stamp = stamp
        direction.header.frame_id = "base_link"
        direction.vector.x, direction.vector.y, direction.vector.z = map(
            float, observation.elbow_direction)
        self.direction_pub.publish(direction)
        angle = Float64()
        angle.data = observation.arm_angle_delta_deg
        self.arm_angle_pub.publish(angle)

        shoulder = self.geometry.shoulder
        wrist = self.latest_wrist
        arm = observation.mapped_arm_point
        projection = observation.projection_point
        markers = MarkerArray()
        markers.markers.extend([
            self._arrow(stamp, 0, "shoulder_wrist", shoulder, wrist,
                        self._color(1.0, 0.15, 0.1)),
            self._arrow(stamp, 1, "shoulder_arm", shoulder, arm,
                        self._color(0.1, 1.0, 0.15)),
            self._arrow(
                stamp, 2, "elbow_direction", projection,
                projection + 0.15 * observation.elbow_direction,
                self._color(0.15, 0.45, 1.0)),
        ])
        sphere = Marker()
        sphere.header.stamp = stamp
        sphere.header.frame_id = "base_link"
        sphere.ns = "rm75_upper_arm_observer"
        sphere.id = 3
        sphere.type = Marker.SPHERE
        sphere.action = Marker.ADD
        sphere.pose.position = self._point(projection)
        sphere.pose.orientation.w = 1.0
        sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.035
        sphere.color = self._color(1.0, 0.85, 0.1)
        sphere.lifetime.nanosec = int(0.25 * 1e9)
        markers.markers.append(sphere)

        label = Marker()
        label.header.stamp = stamp
        label.header.frame_id = "base_link"
        label.ns = "rm75_upper_arm_observer"
        label.id = 4
        label.type = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD
        label.pose.position = self._point(arm + np.array([0.0, 0.0, 0.06]))
        label.pose.orientation.w = 1.0
        label.scale.z = 0.04
        label.color = self._color(1.0, 1.0, 1.0)
        label.text = (
            f"{observation.status}  arm-angle Δ="
            f"{observation.arm_angle_delta_deg:+.1f} deg")
        label.lifetime.nanosec = int(0.25 * 1e9)
        markers.markers.append(label)
        self.marker_pub.publish(markers)

    def _publish_quality(self, valid):
        message = Bool()
        message.data = bool(valid)
        self.quality_pub.publish(message)


def main(args=None):
    rclpy.init(args=args)
    node = UpperArmShadowObserverNode()
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
