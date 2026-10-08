#!/usr/bin/env python3

import argparse
import time
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class LaplacianMonitor(Node):
    def __init__(self, topic, seconds, baseline):
        super().__init__("realsense_dynamic_laplacian")

        self.topic = topic
        self.seconds = seconds
        self.baseline = baseline

        self.t0 = None
        self.prev_small = None

        self.timestamps = []
        self.laps = []
        self.motions = []
        self.brightness = []

        self.last_print = 0.0

        self.sub = self.create_subscription(
            Image,
            topic,
            self.cb,
            qos_profile_sensor_data,
        )

        print("=" * 72)
        print("RealSense Dynamic Laplacian Test")
        print(f"topic    : {topic}")
        print(f"duration : {seconds:.1f} s")
        print(f"baseline : {baseline:.1f}")
        print("=" * 72)
        print()
        print("测试过程中请反复执行：")
        print("  手开合 / 抓取动作 / 头部左右转动 3~4°")
        print()
        print("前 2~3 秒可以保持静止，之后持续运动。")
        print("=" * 72)

    def decode(self, msg):
        h = int(msg.height)
        w = int(msg.width)

        if msg.encoding in ("rgb8", "bgr8"):
            raw = np.frombuffer(msg.data, dtype=np.uint8)
            row = raw.reshape(h, msg.step)
            img = row[:, :w * 3].reshape(h, w, 3)

            if msg.encoding == "rgb8":
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        elif msg.encoding == "mono8":
            raw = np.frombuffer(msg.data, dtype=np.uint8)
            row = raw.reshape(h, msg.step)
            gray = row[:, :w]

        else:
            raise RuntimeError(f"unsupported encoding: {msg.encoding}")

        return gray

    def cb(self, msg):
        try:
            gray = self.decode(msg)
        except Exception as e:
            print("[ERROR]", e)
            return

        now = time.monotonic()

        if self.t0 is None:
            self.t0 = now

        # 与常用清晰度指标一致：
        # grayscale -> Laplacian -> variance
        lap = float(
            cv2.Laplacian(
                gray,
                cv2.CV_64F
            ).var()
        )

        bright = float(gray.mean())

        # 缩小后计算帧间变化，作为运动强度
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

        self.timestamps.append(now)
        self.laps.append(lap)
        self.motions.append(motion)
        self.brightness.append(bright)

        # 每 0.5 秒打印一次，避免疯狂刷屏
        if now - self.last_print >= 0.5:
            self.last_print = now

            recent = np.asarray(self.laps[-30:], dtype=np.float64)

            print(
                f"[LIVE] "
                f"Lap={lap:8.1f}  "
                f"recent_min={recent.min():8.1f}  "
                f"motion={motion:6.2f}  "
                f"brightness={bright:6.1f}"
            )

        if now - self.t0 >= self.seconds:
            self.finish()
            rclpy.shutdown()

    def finish(self):
        lap = np.asarray(self.laps, dtype=np.float64)
        motion = np.asarray(self.motions, dtype=np.float64)
        brightness = np.asarray(self.brightness, dtype=np.float64)
        ts = np.asarray(self.timestamps, dtype=np.float64)

        if len(lap) < 10:
            print("\n[FAILED] 收到的帧太少")
            return

        duration = ts[-1] - ts[0]
        fps = (len(ts) - 1) / duration if duration > 0 else 0

        # 去掉第一帧
        valid_motion = motion[1:]

        # 取运动最明显的 30% 帧。
        # 这样不依赖某个绝对 motion 阈值，
        # 对开合和小角度头动更稳定。
        motion_threshold = float(
            np.percentile(valid_motion, 70)
        )

        dynamic_mask = motion >= motion_threshold
        dynamic_mask[0] = False

        dyn_lap = lap[dynamic_mask]
        dyn_motion = motion[dynamic_mask]

        def p(x, q):
            return float(np.percentile(x, q))

        below = int(np.sum(dyn_lap < self.baseline))
        below_pct = 100.0 * below / len(dyn_lap)

        dyn_p05 = p(dyn_lap, 5)
        dyn_p10 = p(dyn_lap, 10)
        dyn_med = p(dyn_lap, 50)
        dyn_mean = float(np.mean(dyn_lap))
        dyn_min = float(np.min(dyn_lap))

        ratio = dyn_p10 / self.baseline

        print()
        print("=" * 72)
        print("RESULT")
        print("=" * 72)

        print(f"Frames                   : {len(lap)}")
        print(f"Measured receive FPS     : {fps:.2f} Hz")
        print(f"Mean brightness          : {brightness.mean():.1f}")

        print()
        print("全部帧 Laplacian:")
        print(f"  min                    : {lap.min():.1f}")
        print(f"  p05                    : {p(lap, 5):.1f}")
        print(f"  p10                    : {p(lap, 10):.1f}")
        print(f"  median                 : {p(lap, 50):.1f}")
        print(f"  mean                   : {lap.mean():.1f}")

        print()
        print("动态帧（运动最明显的 30%）:")
        print(f"  dynamic frames         : {len(dyn_lap)}")
        print(f"  motion threshold       : {motion_threshold:.2f}")
        print(f"  motion median          : {np.median(dyn_motion):.2f}")
        print(f"  Lap min                : {dyn_min:.1f}")
        print(f"  Lap p05                : {dyn_p05:.1f}")
        print(f"  Lap p10                : {dyn_p10:.1f}")
        print(f"  Lap median             : {dyn_med:.1f}")
        print(f"  Lap mean               : {dyn_mean:.1f}")

        print()
        print(f"原动态基准             : {self.baseline:.1f}")
        print(f"动态帧 < {self.baseline:.0f}       : "
              f"{below}/{len(dyn_lap)} ({below_pct:.1f}%)")
        print(f"dynamic p10 / baseline   : {ratio:.2f}x")

        print()
        print("=" * 72)

        if dyn_p10 >= self.baseline * 1.20:
            print(
                "[PASS] 动态清晰度明显提高："
                f"p10={dyn_p10:.1f}，比原 530 高至少 20%"
            )
        elif dyn_p10 >= self.baseline:
            print(
                "[MARGINAL] 有改善，但提升不算明显："
                f"p10={dyn_p10:.1f}"
            )
        else:
            print(
                "[FAIL] 动态清晰度仍未超过原水平："
                f"p10={dyn_p10:.1f}"
            )

        print("=" * 72)

        out = Path.home() / (
            "realsense_laplacian_"
            + datetime.now().strftime("%Y%m%d_%H%M%S")
            + ".csv"
        )

        data = np.column_stack([
            ts - ts[0],
            lap,
            motion,
            brightness,
            dynamic_mask.astype(np.int32),
        ])

        np.savetxt(
            out,
            data,
            delimiter=",",
            header=(
                "time_s,laplacian_variance,"
                "motion_score,brightness,dynamic"
            ),
            comments=""
        )

        print(f"\nCSV saved: {out}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--topic",
        default="/camera/camera/color/image_raw"
    )

    parser.add_argument(
        "--seconds",
        type=float,
        default=15.0
    )

    parser.add_argument(
        "--baseline",
        type=float,
        default=530.0
    )

    args = parser.parse_args()

    rclpy.init()

    node = LaplacianMonitor(
        args.topic,
        args.seconds,
        args.baseline
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        if len(node.laps) > 10:
            node.finish()
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass


if __name__ == "__main__":
    main()
