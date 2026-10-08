"""Connect to XRoboToolkit and publish one PICO tracker as PoseStamped."""

from contextlib import suppress
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node

from .xrobotoolkit_client import XRoboToolkitClient


class TrackerSelectionError(ValueError):
    """The requested physical tracker cannot be selected safely."""


def _serial_text(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def select_tracker_pose(poses, serial_numbers, requested_serial=""):
    """Return ``(serial, position, quaternion_xyzw)`` for one tracker.

    Empty ``requested_serial`` is accepted only when exactly one tracker pose is
    available. This avoids silently switching arms when multiple trackers are
    paired and the SDK enumeration order changes.
    """
    pose_list = [] if poses is None else list(poses)
    serials = [_serial_text(value) for value in (serial_numbers or [])]
    requested = str(requested_serial).strip()
    if not pose_list:
        raise TrackerSelectionError("no PICO Motion Tracker pose is available")

    if requested:
        if requested not in serials:
            raise TrackerSelectionError(
                f"tracker {requested!r} not found; available={serials}")
        index = serials.index(requested)
        if index >= len(pose_list):
            raise TrackerSelectionError(
                f"tracker {requested!r} has no matching pose sample")
        selected_serial = requested
    else:
        if len(pose_list) != 1:
            raise TrackerSelectionError(
                "tracker_serial is required when multiple trackers are "
                f"available; available={serials}")
        index = 0
        selected_serial = serials[0] if serials else "tracker_0"

    pose = np.asarray(pose_list[index], dtype=float)
    if pose.shape != (7,) or not np.all(np.isfinite(pose)):
        raise TrackerSelectionError(
            f"tracker {selected_serial!r} returned an invalid 7-value pose")
    quaternion = pose[3:7]
    norm = float(np.linalg.norm(quaternion))
    if norm < 0.95 or norm > 1.05:
        raise TrackerSelectionError(
            f"tracker {selected_serial!r} quaternion norm={norm:.4f}")
    return selected_serial, pose[:3].copy(), quaternion / norm


class PicoTrackerPosePublisher(Node):
    """Minimal live PICO connection for one wrist tracker."""

    def __init__(self, client=None):
        super().__init__("pico_tracker_pose_publisher")
        defaults = {
            "pc_service_host": "127.0.0.1",
            "pc_service_port": 60061,
            "tracker_serial": "",
            "output_topic": "/pico/right_wrist/raw_pose",
            "frame_id": "pico_tracking",
            # Keep the single-tracker entry point on the same 90 Hz polling
            # cadence as the dual collector; duplicate SDK samples are not
            # emitted by the dual source used for FTP-1 collection.
            "publish_rate_hz": 90.0,
            "reconnect_interval_sec": 2.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        self.tracker_serial = str(
            self.get_parameter("tracker_serial").value).strip()
        self.output_topic = str(self.get_parameter("output_topic").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.publish_rate = float(
            self.get_parameter("publish_rate_hz").value)
        self.reconnect_interval = float(
            self.get_parameter("reconnect_interval_sec").value)
        if self.publish_rate <= 0.0 or self.reconnect_interval <= 0.0:
            raise ValueError("publish rate and reconnect interval must be positive")

        self.client = client or XRoboToolkitClient(
            str(self.get_parameter("pc_service_host").value),
            int(self.get_parameter("pc_service_port").value))
        self.connected = False
        self.last_connect_attempt = -float("inf")
        self.selected_serial = None
        self.published_frames = 0
        self.publisher = self.create_publisher(
            PoseStamped, self.output_topic, 10)
        self.create_timer(1.0 / self.publish_rate, self._tick)
        self.get_logger().info(
            f"PICO wrist output={self.output_topic} frame={self.frame_id!r}; "
            f"requested_serial={self.tracker_serial or '<auto: exactly one>'}")

    def _connect(self, now):
        if now - self.last_connect_attempt < self.reconnect_interval:
            return False
        self.last_connect_attempt = now
        self.get_logger().info("connecting to XRoboToolkit PC-Service ...")
        self.connected = bool(self.client.init())
        if self.connected:
            self.get_logger().info("XRoboToolkit SDK initialized")
        else:
            self.get_logger().warning(
                "PICO connection failed; start "
                "/opt/apps/roboticsservice/runService.sh and check the headset")
        return self.connected

    def _tick(self):
        now = time.monotonic()
        if not self.connected and not self._connect(now):
            return
        try:
            serial, position, quaternion = select_tracker_pose(
                self.client.get_motion_tracker_pose(),
                self.client.get_motion_tracker_serial_numbers(),
                self.tracker_serial)
        except TrackerSelectionError as error:
            self.get_logger().warning(
                f"waiting for wrist tracker: {error}",
                throttle_duration_sec=2.0)
            return
        except Exception as error:  # SDK failures must trigger a clean retry.
            self.connected = False
            self.get_logger().warning(
                f"XRoboToolkit read failed; reconnecting: {error}")
            return

        if serial != self.selected_serial:
            self.selected_serial = serial
            self.get_logger().info(f"using PICO tracker serial={serial}")
        message = PoseStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.frame_id
        message.pose.position.x = float(position[0])
        message.pose.position.y = float(position[1])
        message.pose.position.z = float(position[2])
        message.pose.orientation.x = float(quaternion[0])
        message.pose.orientation.y = float(quaternion[1])
        message.pose.orientation.z = float(quaternion[2])
        message.pose.orientation.w = float(quaternion[3])
        self.publisher.publish(message)
        self.published_frames += 1
        if self.published_frames == 1 or self.published_frames % 90 == 0:
            self.get_logger().info(
                f"PICO[{serial}] xyz_m={position.round(4).tolist()} "
                f"quat_xyzw={quaternion.round(4).tolist()} "
                f"frames={self.published_frames}")

    def close(self):
        self.client.close()
        self.connected = False


def main(args=None):
    rclpy.init(args=args)
    node = PicoTrackerPosePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        with suppress(Exception):
            node.close()
        with suppress(KeyboardInterrupt):
            node.destroy_node()
        if rclpy.ok():
            with suppress(KeyboardInterrupt):
                rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
