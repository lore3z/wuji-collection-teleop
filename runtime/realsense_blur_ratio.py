#!/usr/bin/env python3

import time
import cv2
import numpy as np
import rclpy

from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


TOPIC = "/camera/camera/color/image_raw"
STATIC_SECONDS = 3.0
TOTAL_SECONDS = 15.0


class BlurRatio(Node):
    def __init__(self):
        super().__init__("realsense_blur_ratio")

        self.start = None
        self.done = False

        self.times = []
        self.laps = []
        self.motions = []
        self.brightness = []

        self.prev_small = None

        self.create_subscription(
            Image,
            TOPIC,
            self.cb,
            qos_profile_sensor_data,
        )

        print("=" * 72)
        print("RealSense 动态模糊测试")
        print("=" * 72)
        print("0~3 秒   ：头和手完全静止")
        print("3~15 秒  ：反复手开合/抓取 + 头部左右转动 3~4°")
        print("=" * 72, flush=True)

    def cb(self, msg):
        if self.done:
            return

        now = time.monotonic()

        if self.start is None:
            self.start = now

        elapsed = now - self.start

        raw = np.frombuffer(msg.data, dtype=np.uint8)

        try:
            img = raw.reshape(msg.height, msg.step)
            img = img[:, :msg.width * 3]
            img = img.reshape(msg.height, msg.width, 3)
        except Exception as e:
            print(f"[ERROR] reshape: {e}")
            self.done = True
            return

        enc = msg.encoding.lower()

        if enc == "rgb8":
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        elif enc == "bgr8":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            print(f"[ERROR] unsupported encoding: {msg.encoding}")
            self.done = True
            return

        lap = float(
            cv2.Laplacian(
                gray,
                cv2.CV_64F
            ).var()
        )

        small = cv2.resize(
            gray,
            (240, 135),
            interpolation=cv2.INTER_AREA
        )

        if self.prev_small is None:
            motion = 0.0
        else:
            motion = float(
                cv2.absdiff(
                    small,
                    self.prev_small
                ).mean()
            )

        self.prev_small = small

        self.times.append(elapsed)
        self.laps.append(lap)
        self.motions.append(motion)
        self.brightness.append(float(gray.mean()))

        if elapsed >= TOTAL_SECONDS:
            self.done = True

    def report(self):
        if len(self.laps) < 10:
            print("[FAILED] 收到的图像太少")
            return

        t = np.asarray(self.times)
        lap = np.asarray(self.laps)
        motion = np.asarray(self.motions)
        brightness = np.asarray(self.brightness)

        static_mask = t < STATIC_SECONDS
        moving_period = t >= STATIC_SECONDS

        if static_mask.sum() < 3 or moving_period.sum() < 3:
            print("[FAILED] 静止/动态样本不足")
            return

        static_lap = lap[static_mask]

        threshold = np.percentile(
            motion[moving_period],
            70
        )

        dynamic_mask = (
            moving_period &
            (motion >= threshold)
        )

        dynamic_lap = lap[dynamic_mask]

        if len(dynamic_lap) == 0:
            print("[FAILED] 没检测到动态帧")
            return

        static_med = float(np.median(static_lap))
        static_p10 = float(np.percentile(static_lap, 10))

        dyn_med = float(np.median(dynamic_lap))
        dyn_p10 = float(np.percentile(dynamic_lap, 10))
        dyn_p05 = float(np.percentile(dynamic_lap, 5))

        ratio_med = dyn_med / max(static_med, 1e-9)
        ratio_p10 = dyn_p10 / max(static_med, 1e-9)

        duration = max(t[-1] - t[0], 1e-9)

        print()
        print("=" * 72)
        print("RESULT")
        print("=" * 72)

        print(f"Frames                      : {len(lap)}")
        print(f"Measured receive FPS        : {(len(lap)-1)/duration:.2f} Hz")

        print()
        print("亮度:")
        print(f"  all mean                  : {brightness.mean():.2f}")
        print(f"  static mean               : {brightness[static_mask].mean():.2f}")
        print(f"  moving-period mean        : {brightness[moving_period].mean():.2f}")

        print()
        print("静止 0~3秒:")
        print(f"  Lap median                : {static_med:.2f}")
        print(f"  Lap p10                   : {static_p10:.2f}")

        print()
        print("动态最明显的30%帧:")
        print(f"  dynamic frames            : {len(dynamic_lap)}")
        print(f"  motion threshold          : {threshold:.2f}")
        print(f"  Lap median                : {dyn_med:.2f}")
        print(f"  Lap p10                   : {dyn_p10:.2f}")
        print(f"  Lap p05                   : {dyn_p05:.2f}")

        print()
        print("动态清晰度保持率:")
        print(f"  median/static median      : {ratio_med:.3f}")
        print(f"  p10/static median         : {ratio_p10:.3f}")

        print()
        print("旧现象参考:")
        print("  530 / 1050                : 0.505")

        print()

        if ratio_p10 >= 0.75:
            print("[PASS] 动态清晰度保持较好")
        elif ratio_p10 >= 0.60:
            print("[MARGINAL] 有运动模糊，但明显优于旧基准")
        else:
            print("[FAIL] 动态锐度下降仍较明显")

        print("=" * 72)


def main():
    rclpy.init()
    node = BlurRatio()

    deadline = time.monotonic() + 20.0

    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)

            if time.monotonic() > deadline:
                print("[FAILED] 20秒超时")
                break

        node.report()

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
