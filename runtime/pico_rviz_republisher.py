#!/usr/bin/env python3
"""Decode the PICO CompressedImage stream for RViz without backpressuring capture."""

from __future__ import annotations

import argparse
import array

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image


class PicoRvizRepublisher(Node):
    def __init__(self, compressed_topic: str, raw_topic: str) -> None:
        super().__init__("pico_rviz_republisher")
        self.frames = 0
        # The camera intentionally offers BEST_EFFORT so a visualization can
        # never add latency to collection. A depth of two keeps only fresh UI.
        input_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        # RViz commonly requests RELIABLE for sensor_msgs/Image. This output is
        # isolated from capture, so a one-frame reliable queue is safe.
        output_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.publisher = self.create_publisher(Image, raw_topic, output_qos)
        self.subscription = self.create_subscription(
            CompressedImage, compressed_topic, self._receive, input_qos
        )
        self.get_logger().info(f"RViz image: {compressed_topic} -> {raw_topic}")

    def _receive(self, message: CompressedImage) -> None:
        try:
            encoded = np.frombuffer(message.data, dtype=np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("OpenCV returned an empty image")
            image = np.ascontiguousarray(image)
            output = Image()
            output.header = message.header
            output.height, output.width = image.shape[:2]
            output.encoding = "bgr8"
            output.is_bigendian = 0
            output.step = output.width * 3
            # Assign array('B') so the generated ROS setter takes its fast
            # path. Assigning ``bytes`` makes it validate millions of pixels
            # one by one and reduces this 60 Hz stream to roughly 1 Hz.
            output.data = array.array("B", image.tobytes())
            self.publisher.publish(output)
            self.frames += 1
        except Exception as exc:
            self.get_logger().error(f"PICO JPEG decode failed: {exc}", throttle_duration_sec=2.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="/pico/ego/image_raw/compressed")
    parser.add_argument("--output", default="/pico/ego/image_raw")
    args = parser.parse_args()
    rclpy.init(args=[])
    node = PicoRvizRepublisher(args.input, args.output)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
