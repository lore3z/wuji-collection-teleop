#!/usr/bin/env python3
"""Wait until enabled camera and Tracker topics deliver valid live data."""

from __future__ import annotations

import argparse
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ego", help="first-person RGB topic (omit when disabled)")
    parser.add_argument("--ego-compressed", action="store_true")
    parser.add_argument("--main-compressed", action="store_true")
    parser.add_argument("--main", required=True)
    parser.add_argument("--tracker-camera")
    parser.add_argument("--tracker-wrist")
    parser.add_argument("--timeout", type=float, default=25.0)
    args = parser.parse_args()

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CompressedImage, Image
    from geometry_msgs.msg import PoseStamped

    rclpy.init()
    node = Node("wuji_wait_for_rgb_topics")
    received: dict[str, tuple[int, int, str]] = {}

    def callback(label: str):
        def receive(msg: Image) -> None:
            if msg.height > 0 and msg.width > 0 and len(msg.data) > 0:
                received[label] = (int(msg.width), int(msg.height), str(msg.encoding))
        return receive

    def compressed_callback(label: str):
        def receive(msg: CompressedImage) -> None:
            if not msg.data:
                return
            import cv2
            import numpy as np
            image = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is not None and image.size:
                height, width = image.shape[:2]
                received[label] = (int(width), int(height), str(msg.format or "jpeg"))
        return receive

    main_compressed = args.main_compressed or args.main.endswith("/compressed")
    main_type = CompressedImage if main_compressed else Image
    main_callback = compressed_callback("main") if main_compressed else callback("main")
    subscriptions = [node.create_subscription(main_type, args.main, main_callback, qos_profile_sensor_data)]
    required = {"main"}
    if args.ego:
        ego_compressed = args.ego_compressed or args.ego.endswith("/compressed")
        ego_type = CompressedImage if ego_compressed else Image
        ego_callback = compressed_callback("ego") if ego_compressed else callback("ego")
        subscriptions.append(node.create_subscription(ego_type, args.ego, ego_callback, qos_profile_sensor_data))
        required.add("ego")
    def pose_callback(label: str):
        def receive(msg: PoseStamped) -> None:
            values = (
                msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
                msg.pose.orientation.x, msg.pose.orientation.y,
                msg.pose.orientation.z, msg.pose.orientation.w,
            )
            if all(float(value) == float(value) for value in values):
                received[label] = (1, 1, "PoseStamped")
        return receive
    if args.tracker_camera:
        subscriptions.append(node.create_subscription(
            PoseStamped, args.tracker_camera, pose_callback("tracker_camera"), qos_profile_sensor_data
        ))
        required.add("tracker_camera")
    if args.tracker_wrist:
        subscriptions.append(node.create_subscription(
            PoseStamped, args.tracker_wrist, pose_callback("tracker_wrist"), qos_profile_sensor_data
        ))
        required.add("tracker_wrist")
    deadline = time.monotonic() + args.timeout
    try:
        while rclpy.ok() and time.monotonic() < deadline and not required.issubset(received):
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        # Keep references alive until after the wait.
        del subscriptions
        node.destroy_node()
        rclpy.shutdown()
    if not required.issubset(received):
        missing = sorted(required - set(received))
        raise SystemExit(f"source topic timeout; missing={missing}")
    details = ", ".join(f"{label}={received[label]}" for label in sorted(required))
    print(f"[PASS] live RGB: {details}")


if __name__ == "__main__":
    main()
