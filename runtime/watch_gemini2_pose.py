#!/usr/bin/env python3
"""Print Gemini 2's live IMU-derived ``camera_ego_pose``.

The input is the synchronized ``sensor_msgs/Imu`` topic published by the
Orbbec driver.  Gemini 2 has no absolute position tracker, so xyz is always
zero.  Roll/pitch are gravity-stabilized; yaw is relative to this monitor's
start and will drift over time.
"""

from __future__ import annotations

import argparse
import math
import time
from typing import Any

import numpy as np


class GeminiPoseEstimator:
    """The same complementary filter used by the FTP-1 collector."""

    def __init__(self) -> None:
        self.rpy: np.ndarray | None = None
        self.last_stamp_ns = 0

    def update(self, msg: Any) -> tuple[int, np.ndarray] | None:
        try:
            stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
            accel = np.asarray(
                (msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z), dtype=np.float64
            )
            gyro = np.asarray(
                (msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z), dtype=np.float64
            )
            if not np.all(np.isfinite(accel)) or not np.all(np.isfinite(gyro)) or float(np.linalg.norm(accel)) < 1e-5:
                return None
            accel_roll = math.atan2(float(accel[1]), float(accel[2]))
            accel_pitch = math.atan2(-float(accel[0]), math.hypot(float(accel[1]), float(accel[2])))
            if self.rpy is None:
                rpy = np.asarray((accel_roll, accel_pitch, 0.0), dtype=np.float64)
            else:
                dt = min(max((stamp_ns - self.last_stamp_ns) / 1e9, 0.0), 0.05)
                rpy = self.rpy.astype(np.float64, copy=True) + gyro * dt
                correction = min(0.04, 2.0 * dt)
                rpy[0] = (1.0 - correction) * rpy[0] + correction * accel_roll
                rpy[1] = (1.0 - correction) * rpy[1] + correction * accel_pitch
                rpy[2] = (rpy[2] + math.pi) % (2.0 * math.pi) - math.pi
            self.rpy = rpy.astype(np.float32)
            self.last_stamp_ns = stamp_ns
            return stamp_ns, np.concatenate((np.zeros(3, dtype=np.float32), self.rpy))
        except (AttributeError, TypeError, ValueError, OverflowError):
            return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default="/gemini2/gyro_accel/sample")
    parser.add_argument("--rate", type=float, default=10.0, help="terminal refresh rate in Hz (default: 10)")
    parser.add_argument("--seconds", type=float, default=0.0, help="0=until Ctrl-C (default); otherwise stop after this duration")
    args = parser.parse_args()
    if args.rate <= 0 or args.seconds < 0:
        parser.error("--rate must be positive and --seconds must be non-negative")

    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import Imu
    except ImportError as exc:
        raise SystemExit("找不到 ROS 2 Python；请通过 runtime/watch_gemini2_pose.sh 运行。") from exc

    rclpy.init(args=None)
    node = Node("gemini2_pose_watch")
    estimator = GeminiPoseEstimator()
    latest: tuple[int, np.ndarray] | None = None

    def callback(msg: Imu) -> None:
        nonlocal latest
        estimate = estimator.update(msg)
        if estimate is not None:
            latest = estimate

    node.create_subscription(Imu, args.topic, callback, 240)
    print(f"[LIVE] Gemini 2 IMU: {args.topic}  (Ctrl-C 停止)")
    print("       camera_ego_pose=[0, 0, 0, roll, pitch, yaw]; xyz: m; rpy: rad / deg")
    deadline = time.monotonic() + args.seconds if args.seconds else None
    next_print = 0.0
    last_stamp_ns = -1
    next_waiting_notice = time.monotonic() + 3.0
    try:
        while deadline is None or time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            if latest is None and time.monotonic() >= next_waiting_notice:
                print(
                    "[WAITING] 未收到 IMU 帧：确认 Gemini 采集栈已重启，且 "
                    "WUJI_GEMINI_IMU_ENABLED=1。",
                    flush=True,
                )
                next_waiting_notice = time.monotonic() + 3.0
            if latest is None or latest[0] == last_stamp_ns or time.monotonic() < next_print:
                continue
            stamp_ns, pose = latest
            rpy = pose[3:]
            print(
                f"t={stamp_ns} ns  camera_ego_pose="
                f"[0, 0, 0, {rpy[0]:+.4f}, {rpy[1]:+.4f}, {rpy[2]:+.4f}]  "
                f"rpy_deg=[{np.degrees(rpy[0]):+.1f}, {np.degrees(rpy[1]):+.1f}, {np.degrees(rpy[2]):+.1f}]",
                flush=True,
            )
            last_stamp_ns = stamp_ns
            next_print = time.monotonic() + 1.0 / args.rate
    except KeyboardInterrupt:
        print("\n[LIVE] stopped")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
