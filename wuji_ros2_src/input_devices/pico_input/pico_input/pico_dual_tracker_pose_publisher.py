"""Publish synchronized raw poses for one wrist and one upper-arm Tracker."""

from contextlib import suppress
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import UInt64MultiArray

from .pico_tracker_pose_publisher import (
    TrackerSelectionError,
    select_tracker_pose,
)
from .xrobotoolkit_client import XRoboToolkitClient


def select_dual_tracker_poses(poses, serial_numbers, wrist_serial,
                              upper_arm_serial):
    """Select and validate two distinct Tracker samples from one SDK frame."""
    wrist = str(wrist_serial).strip()
    upper = str(upper_arm_serial).strip()
    if not wrist or not upper:
        raise TrackerSelectionError(
            "wrist_serial and upper_arm_serial are both required")
    if wrist == upper:
        raise TrackerSelectionError(
            "wrist_serial and upper_arm_serial must be different")
    wrist_pose = select_tracker_pose(poses, serial_numbers, wrist)
    upper_pose = select_tracker_pose(poses, serial_numbers, upper)
    return wrist_pose, upper_pose


class PicoDualTrackerPosePublisher(Node):
    """Read both physical Trackers in one poll and stamp them identically."""

    def __init__(self, client=None):
        super().__init__("pico_dual_tracker_pose_publisher")
        defaults = {
            "pc_service_host": "127.0.0.1",
            "pc_service_port": 60061,
            "wrist_serial": "PC2310MLKC190056G",
            "upper_arm_serial": "PC2310MLKC190573G",
            "wrist_topic": "/pico/right_wrist/raw_pose",
            "upper_arm_topic": "/pico/right_upper_arm/raw_pose",
            "wrist_latency_topic": "/pico/right_wrist/latency_trace",
            "frame_id": "pico_tracking",
            # Polling cached shared memory at 90 Hz does not imply that the
            # underlying Tracker sample changed. Motion.timeStampNs below is
            # used to suppress cached repeats and preserve acquisition time.
            "publish_rate_hz": 90.0,
            "reconnect_interval_sec": 2.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        self.wrist_serial = str(
            self.get_parameter("wrist_serial").value).strip()
        self.upper_arm_serial = str(
            self.get_parameter("upper_arm_serial").value).strip()
        if not self.wrist_serial or not self.upper_arm_serial:
            raise ValueError("both Tracker serial numbers are required")
        if self.wrist_serial == self.upper_arm_serial:
            raise ValueError("wrist and upper-arm Tracker serials must differ")
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
        self.ready_announced = False
        self.paired_frames = 0
        self.last_pair_at = 0.0
        self.last_sdk_timestamp = None
        self.sdk_to_ros_offset_ns = None
        self.zero_timestamp_warned = False
        self.wrist_pub = self.create_publisher(
            PoseStamped, str(self.get_parameter("wrist_topic").value), 10)
        self.upper_arm_pub = self.create_publisher(
            PoseStamped, str(self.get_parameter("upper_arm_topic").value), 10)
        self.wrist_latency_pub = self.create_publisher(
            UInt64MultiArray,
            str(self.get_parameter("wrist_latency_topic").value), 10)
        self.create_timer(1.0 / self.publish_rate, self._tick)

    def _connect(self, now):
        if now - self.last_connect_attempt < self.reconnect_interval:
            return False
        self.last_connect_attempt = now
        self.connected = bool(self.client.init())
        if not self.connected:
            self.get_logger().warning(
                "PICO connection failed; check PC-Service and headset Send",
                throttle_duration_sec=2.0)
        return self.connected

    def _disconnect(self) -> None:
        """Drop an unavailable SDK session so the next timer tick can reconnect."""
        try:
            self.client.close()
        except Exception as error:
            self.get_logger().warning(f"XRoboToolkit close failed while reconnecting: {error}")
        self.connected = False
        self.ready_announced = False
        self.last_sdk_timestamp = None
        self.sdk_to_ros_offset_ns = None

    def _message(self, stamp, position, quaternion):
        message = PoseStamped()
        message.header.stamp = stamp
        message.header.frame_id = self.frame_id
        message.pose.position.x = float(position[0])
        message.pose.position.y = float(position[1])
        message.pose.position.z = float(position[2])
        message.pose.orientation.x = float(quaternion[0])
        message.pose.orientation.y = float(quaternion[1])
        message.pose.orientation.z = float(quaternion[2])
        message.pose.orientation.w = float(quaternion[3])
        return message

    def _tick(self):
        now = time.monotonic()
        if not self.connected and not self._connect(now):
            return
        pc_poll_start_ns = time.time_ns()
        try:
            poses = self.client.get_motion_tracker_pose()
            serials = self.client.get_motion_tracker_serial_numbers()
            wrist, upper = select_dual_tracker_poses(
                poses, serials, self.wrist_serial, self.upper_arm_serial)
        except TrackerSelectionError as error:
            self.get_logger().warning(
                f"waiting for synchronized Tracker pair: {error}",
                throttle_duration_sec=2.0)
            # Once a stream was healthy, a prolonged loss of the pair is a
            # dead SDK session rather than a transient startup condition.
            if self.ready_announced and now - self.last_pair_at > 0.5:
                self._disconnect()
            return
        except Exception as error:
            self.connected = False
            self.get_logger().warning(
                f"XRoboToolkit read failed; reconnecting: {error}")
            return

        timestamp_getter = getattr(self.client, "get_motion_timestamp_ns", None)
        sdk_timestamp = int(timestamp_getter()) if timestamp_getter else 0
        pc_read_complete_ns = time.time_ns()
        _, wrist_position, wrist_quaternion = wrist
        _, upper_position, upper_quaternion = upper
        if sdk_timestamp > 0:
            if self.last_sdk_timestamp is not None and sdk_timestamp == self.last_sdk_timestamp:
                # PC-Service exposes its latest pose through shared memory.
                # Re-stamping this cached value at every timer tick fabricates
                # a high-rate stream while XYZ may really update near 1 Hz.
                return
            if (self.last_sdk_timestamp is None or
                    sdk_timestamp < self.last_sdk_timestamp):
                if self.last_sdk_timestamp is not None:
                    self.get_logger().warning(
                        "PICO Motion Tracker timestamp restarted; accepting new epoch")
                self.sdk_to_ros_offset_ns = pc_read_complete_ns - sdk_timestamp
            self.last_sdk_timestamp = sdk_timestamp
        else:
            if not self.zero_timestamp_warned:
                self.zero_timestamp_warned = True
                self.get_logger().warning(
                    "PICO Motion.timeStampNs is unavailable; poll-stamping Tracker samples (wrist value-update quality remains independently gated)")

        stamp = self.get_clock().now().to_msg()
        if sdk_timestamp > 0 and self.sdk_to_ros_offset_ns is not None:
            sample_ros_ns = sdk_timestamp + self.sdk_to_ros_offset_ns
            stamp.sec = sample_ros_ns // 1_000_000_000
            stamp.nanosec = sample_ros_ns % 1_000_000_000
        pose_key_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
        latency = UInt64MultiArray()
        # v1: pose key, SDK sample, PC poll start/read complete, publish time.
        latency.data = [
            1, pose_key_ns, max(0, sdk_timestamp), pc_poll_start_ns,
            pc_read_complete_ns, time.time_ns(),
        ]
        # Publish the companion trace first so the single-threaded downstream
        # node normally has it cached before processing the corresponding pose.
        self.wrist_latency_pub.publish(latency)
        self.wrist_pub.publish(self._message(
            stamp, wrist_position, wrist_quaternion))
        self.upper_arm_pub.publish(self._message(
            stamp, upper_position, upper_quaternion))
        self.paired_frames += 1
        self.last_pair_at = now
        if not self.ready_announced:
            self.ready_announced = True
            self.get_logger().info(
                "synchronized dual-Tracker stream ready: "
                f"wrist={self.wrist_serial}, upper_arm={self.upper_arm_serial}")

    def close(self):
        self._disconnect()


def main(args=None):
    rclpy.init(args=args)
    node = PicoDualTrackerPosePublisher()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
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
