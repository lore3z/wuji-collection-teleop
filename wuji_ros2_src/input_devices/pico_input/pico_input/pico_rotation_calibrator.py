"""Interactive raw-PICO rotational-axis calibration; never connects to RM75."""

from collections import deque
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from .rotation_calibrator import (
    ROTATION_PAIRS,
    derive_rotation_axis_mapping,
    normalize_quaternion_xyzw,
)


PROMPTS = (
    ("neutral", "保持自然中立姿态"),
    ("rm_x_positive", "右手拇指指向身体前方，按四指弯曲方向绕前后轴正转 15-25°"),
    ("rm_x_negative", "绕同一前后轴反向转 15-25°"),
    ("rm_y_positive", "右手拇指指向身体左侧，按四指弯曲方向绕左右轴正转 15-25°"),
    ("rm_y_negative", "绕同一左右轴反向转 15-25°"),
    ("rm_z_positive", "右手拇指指向上方，按四指弯曲方向绕竖直轴正转 15-25°"),
    ("rm_z_negative", "绕同一竖直轴反向转 15-25°"),
)


class PicoRotationCalibrator(Node):
    def __init__(self):
        super().__init__("pico_rotation_calibrator")
        self.declare_parameter("input_topic", "/pico/right_wrist/raw_pose")
        topic = str(self.get_parameter("input_topic").value)
        self._samples = deque(maxlen=200)
        self._lock = threading.Lock()
        self._last_received = 0.0
        self.create_subscription(PoseStamped, topic, self._callback,
                                 qos_profile_sensor_data)
        self.get_logger().info(
            f"Raw-only rotation calibration subscribed to {topic}; RM75 is not used")

    def _callback(self, msg: PoseStamped):
        if msg.header.frame_id != "pico_tracking":
            return
        try:
            q = normalize_quaternion_xyzw([
                msg.pose.orientation.x, msg.pose.orientation.y,
                msg.pose.orientation.z, msg.pose.orientation.w])
        except ValueError:
            return
        now = time.monotonic()
        with self._lock:
            self._samples.append((now, q))
            self._last_received = now

    def capture(self, seconds=0.75):
        start = time.monotonic()
        time.sleep(seconds)
        with self._lock:
            quaternions = [q for timestamp, q in self._samples if timestamp >= start]
            age = time.monotonic() - self._last_received
        if not quaternions or age > 0.25:
            raise RuntimeError("没有收到实时 PICO 数据；请保持头显唤醒并确认 Send=开")
        reference = quaternions[0]
        aligned = [(-q if np.dot(q, reference) < 0.0 else q)
                   for q in quaternions]
        return normalize_quaternion_xyzw(np.mean(aligned, axis=0))


def _format_matrix(matrix):
    rows = ["[" + ", ".join(f"{value:.1f}" for value in row) + "]"
            for row in matrix]
    return "[" + ",\n ".join(rows) + "]"


def main(args=None):
    rclpy.init(args=args)
    node = PicoRotationCalibrator()
    spinner = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spinner.start()
    captured = {}
    try:
        print("\n只读取 /pico/right_wrist/raw_pose，不连接 RM75。")
        print("每次按提示旋转并保持，再按 Enter；随后平均采样 0.75 秒。")
        print("正方向严格按右手定则：拇指指轴正向，四指弯曲方向为正转。\n")
        for name, instruction in PROMPTS:
            input(f"[{name}] {instruction}，保持后按 Enter：")
            captured[name] = node.capture()
            print("  quaternion xyzw = " + np.array2string(
                captured[name], precision=5, suppress_small=True))

        result = derive_rotation_axis_mapping(captured)
        print("\n=== 旋转标定结果（仅建议，不会写配置） ===")
        for positive, negative, target in ROTATION_PAIRS:
            vector = result.pair_rotation_vectors_rad[
                f"{positive}_minus_{negative}"]
            print(f"{positive} - {negative} -> {target}: rotvec_deg="
                  + np.array2string(np.degrees(vector), precision=2,
                                    suppress_small=True))
        print("\nrotation_axis_mapping:")
        print(_format_matrix(result.rotation_axis_mapping))
        print(f"\n建议首次 rotation_scale={result.suggested_rotation_scale:.3f}")
        print("下一阶段仍须固定位置、单旋转轴进行 RM75 dry-run。")
    except (EOFError, KeyboardInterrupt):
        print("\n旋转标定已取消。")
    except (RuntimeError, ValueError) as exc:
        print(f"\n旋转标定失败：{exc}")
        raise SystemExit(1) from exc
    finally:
        node.destroy_node()
        rclpy.shutdown()
        spinner.join(timeout=1.0)


if __name__ == "__main__":
    main()

