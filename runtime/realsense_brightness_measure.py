#!/usr/bin/env python3

import time
import cv2
import numpy as np
import rclpy

from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class Measure(Node):
    def __init__(self, seconds=3.0):
        super().__init__("realsense_brightness_measure")

        self.seconds = seconds
        self.start = None
        self.values = []
        self.sharpness = []
        self.done = False

        self.create_subscription(
            Image,
            "/camera/camera/color/image_raw",
            self.cb,
            qos_profile_sensor_data,
        )

    def cb(self, msg):
        if self.done:
            return

        now = time.monotonic()

        if self.start is None:
            self.start = now

        raw = np.frombuffer(msg.data, dtype=np.uint8)

        try:
            img = raw.reshape(msg.height, msg.step)
            img = img[:, :msg.width * 3]
            img = img.reshape(msg.height, msg.width, 3)
        except Exception as e:
            print(f"[ERROR] image reshape failed: {e}")
            self.done = True
            return

        if msg.encoding.lower() == "rgb8":
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        elif msg.encoding.lower() == "bgr8":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            print(f"[ERROR] unsupported encoding: {msg.encoding}")
            self.done = True
            return

        self.values.append(float(gray.mean()))

        self.sharpness.append(
            float(
                cv2.Laplacian(
                    gray,
                    cv2.CV_64F
                ).var()
            )
        )

        if now - self.start >= self.seconds:
            self.done = True

    def report(self):
        if not self.values:
            print("[FAILED] no frames received")
            return

        arr = np.asarray(self.values)
        sharp = np.asarray(self.sharpness)

        duration = max(time.monotonic() - self.start, 1e-6)

        print(
            f"frames={len(arr)} "
            f"rx_fps={len(arr)/duration:.1f} "
            f"brightness_mean={arr.mean():.2f} "
            f"brightness_median={np.median(arr):.2f} "
            f"lap_median={np.median(sharp):.2f}",
            flush=True
        )


def main():
    rclpy.init()

    node = Measure(seconds=3.0)

    deadline = time.monotonic() + 6.0

    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)

            if time.monotonic() > deadline:
                print("[FAILED] measurement timeout")
                break

        node.report()

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
