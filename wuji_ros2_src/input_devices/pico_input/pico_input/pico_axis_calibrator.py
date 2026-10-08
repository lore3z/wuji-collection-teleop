"""Interactive raw-PICO axis calibration tool. It never connects to a robot."""

from collections import deque
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from .axis_calibrator import SEMANTIC_PAIRS, derive_axis_mapping


PROMPTS = (
    ("center", "保持自然中心姿态"),
    ("forward", "向前移动并保持"),
    ("backward", "向后移动并保持"),
    ("right", "向右移动并保持"),
    ("left", "向左移动并保持"),
    ("up", "向上移动并保持"),
    ("down", "向下移动并保持"),
)


class PicoAxisCalibrator(Node):
    def __init__(self):
        super().__init__("pico_axis_calibrator")
        self.declare_parameter("input_topic", "/pico/right_wrist/raw_pose")
        topic = str(self.get_parameter("input_topic").value)
        self._samples = deque(maxlen=200)
        self._lock = threading.Lock()
        self._last_received = 0.0
        self.create_subscription(PoseStamped, topic, self._callback,
                                 qos_profile_sensor_data)
        self.get_logger().info(
            f"Raw-only calibration subscribed to {topic}; RM75 is not used")

    def _callback(self, msg: PoseStamped):
        if msg.header.frame_id != "pico_tracking":
            return
        p = np.array([msg.pose.position.x, msg.pose.position.y,
                      msg.pose.position.z], dtype=float)
        if not np.all(np.isfinite(p)):
            return
        now = time.monotonic()
        with self._lock:
            self._samples.append((now, p))
            self._last_received = now

    def capture(self, seconds: float = 0.75) -> np.ndarray:
        start = time.monotonic()
        time.sleep(seconds)
        with self._lock:
            points = [p for timestamp, p in self._samples if timestamp >= start]
            age = time.monotonic() - self._last_received
        if not points or age > 0.25:
            raise RuntimeError("没有收到实时 PICO 数据；请保持头显唤醒并确认 Send=开")
        return np.mean(points, axis=0)


def _format_matrix(matrix: np.ndarray) -> str:
    rows = ["[" + ", ".join(f"{value:.1f}" for value in row) + "]"
            for row in matrix]
    return "[" + ",\n ".join(rows) + "]"


def main(args=None):
    rclpy.init(args=args)
    node = PicoAxisCalibrator()
    spinner = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spinner.start()
    captured = {}
    try:
        print("\n只读取 /pico/right_wrist/raw_pose，不连接 RM75。")
        print("每次移动约 5-10 cm，保持静止后按 Enter；程序随后平均采样 0.75 秒。\n")
        for name, instruction in PROMPTS:
            input(f"[{name}] {instruction}，然后按 Enter：")
            captured[name] = node.capture()
            print("  PICO XYZ = " + np.array2string(
                captured[name], precision=4, suppress_small=True))

        result = derive_axis_mapping(captured)
        print("\n=== 标定结果（仅建议，不会写配置） ===")
        for positive, negative, target in SEMANTIC_PAIRS:
            delta = result.pair_displacements_m[f"{positive}_minus_{negative}"]
            print(f"{positive} - {negative} -> {target}: "
                  + np.array2string(delta, precision=4, suppress_small=True))
        print("\naxis_mapping:")
        print(_format_matrix(result.axis_mapping))
        print(f"\n建议首次 dry-run position_scale="
              f"{result.suggested_position_scale:.3f}")
        print("该矩阵仍需在 RM75 Base 坐标显示中逐轴确认后才能用于运动。")
    except (EOFError, KeyboardInterrupt):
        print("\n标定已取消。")
    except (RuntimeError, ValueError) as exc:
        print(f"\n标定失败：{exc}")
        raise SystemExit(1) from exc
    finally:
        node.destroy_node()
        rclpy.shutdown()
        spinner.join(timeout=1.0)


if __name__ == "__main__":
    main()

