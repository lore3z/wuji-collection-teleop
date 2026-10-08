"""Interactive RM75 shadow calibration for one PICO wrist Tracker."""

import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node

from .pose_mapper import (
    quaternion_inverse_xyzw,
    quaternion_multiply_xyzw,
    quaternion_to_rotation_vector_xyzw,
)


STEPS = (
    ("RM X", "按新矩阵：手腕上升应为RM +X，下降应为RM -X；做小幅正反动作并回零"),
    ("RM Y", "按新矩阵：手腕后退应为RM +Y，前伸应为RM -Y；做小幅正反动作并回零"),
    ("RM Z", "按新矩阵：手腕向右应为RM +Z，向左应为RM -Z；做小幅正反动作并回零"),
    ("RM ROLL(X)", "位置尽量不动，绕人体竖直轴做正反小角度旋转并回零"),
    ("RM PITCH(Y)", "位置尽量不动，绕人体前后轴做正反小角度旋转并回零"),
    ("RM YAW(Z)", "位置尽量不动，绕人体左右轴做正反小角度旋转并回零"),
)


class SixAxisShadowMonitor(Node):
    def __init__(self):
        super().__init__("rm75_six_axis_monitor")
        self._lock = threading.Lock()
        self.latest = None
        self.reference = None
        self.samples = []
        self.recording = False
        self.create_subscription(
            PoseStamped, "/rm75_shadow/target_pose", self._pose, 10)

    def _pose(self, msg):
        position = np.array([
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        quaternion = np.array([
            msg.pose.orientation.x, msg.pose.orientation.y,
            msg.pose.orientation.z, msg.pose.orientation.w])
        with self._lock:
            self.latest = (position, quaternion)
            if self.recording and self.reference is not None:
                position0, quaternion0 = self.reference
                delta_mm = (position - position0) * 1000.0
                delta_q = quaternion_multiply_xyzw(
                    quaternion, quaternion_inverse_xyzw(quaternion0))
                rotvec_deg = np.degrees(
                    quaternion_to_rotation_vector_xyzw(delta_q))
                self.samples.append((delta_mm, rotvec_deg))

    def ready(self):
        with self._lock:
            return self.latest is not None

    def begin_step(self):
        with self._lock:
            if self.latest is None:
                raise RuntimeError("尚未收到影子目标")
            position, quaternion = self.latest
            self.reference = (position.copy(), quaternion.copy())
            self.samples = []
            self.recording = True

    def end_step(self):
        with self._lock:
            self.recording = False
            samples = list(self.samples)
        if not samples:
            return None
        translations = np.asarray([sample[0] for sample in samples])
        rotations = np.asarray([sample[1] for sample in samples])
        return {
            "translation_min": translations.min(axis=0),
            "translation_max": translations.max(axis=0),
            "rotation_min": rotations.min(axis=0),
            "rotation_max": rotations.max(axis=0),
            "sample_count": len(samples),
        }


def _format_range(minimum, maximum):
    return "[" + ", ".join(
        f"{lo:+.2f}..{hi:+.2f}" for lo, hi in zip(minimum, maximum)
    ) + "]"


def main(args=None):
    rclpy.init(args=args)
    node = SixAxisShadowMonitor()
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    try:
        print("\n=== RM75 单手腕Tracker六轴交互标定 ===", flush=True)
        print("无机器人运动接口；RViz会实时显示125Hz平滑后的影子模型。", flush=True)
        print("正在等待 /rm75_shadow/target_pose ...", flush=True)
        while rclpy.ok() and not node.ready():
            time.sleep(0.1)
        if not rclpy.ok():
            return
        print("已收到影子目标。先把手腕放到舒适、稳定的中立位置。", flush=True)
        input("准备好后按 Enter 开始六轴标定：")
        for index, (name, instruction) in enumerate(STEPS, start=1):
            print(f"\n[{index}/6] {name}", flush=True)
            print(instruction, flush=True)
            input("保持当前起点稳定，按 Enter 锁定本轴基准：")
            node.begin_step()
            input("现在执行动作并观察RViz；完成且回到起点后按 Enter：")
            result = node.end_step()
            if result is None:
                print("本轴没有收到样本，请稍后重新运行该项。", flush=True)
                continue
            print(
                "结果  dXYZ_mm=" + _format_range(
                    result["translation_min"], result["translation_max"]),
                flush=True)
            print(
                "      rotvec_deg=" + _format_range(
                    result["rotation_min"], result["rotation_max"]) +
                f"  samples={result['sample_count']}", flush=True)
        print("\n六轴采集完成。Ctrl+C或按 Enter 关闭影子标定。", flush=True)
        input()
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=1.0)
        node.destroy_node()


if __name__ == "__main__":
    main()
