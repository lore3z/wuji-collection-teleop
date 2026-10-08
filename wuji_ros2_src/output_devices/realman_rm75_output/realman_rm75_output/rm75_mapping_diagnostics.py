"""Single-terminal, read-only/Fake PICO -> RM75 mapping diagnostics."""

import math
import select
import sys
import termios
import threading
import time
import tty

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e

from .cartesian_pose_limiter import CartesianPoseLimiter
from .ik_dry_run import IKDryRunEvaluator
from .pose_mapper import (
    continuous_quaternion_xyzw,
    matrix_to_quaternion_xyzw,
    normalize_quaternion_xyzw,
    quaternion_angle_rad,
    quaternion_inverse_xyzw,
    quaternion_multiply_xyzw,
    quaternion_to_matrix_xyzw,
    quaternion_to_rotation_vector_xyzw,
    quaternion_wxyz_to_xyzw,
)
from .readonly_preflight import collect_read_only_audit
from .realman_rm75_output_node import RealManAlgorithmAdapter


AXIS_MAPPING = np.array([[0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]])
MODE_DURATION_SEC = {1: 10.0, 2: 10.0, 3: 30.0, 4: 15.0,
                     5: 30.0, 6: 15.0, 8: 20.0}
MENU = """
RM75 映射诊断（1～8 只读/Fake，绝不发送运动命令）
  1  原始 PICO 静止稳定性（自动采样 10 秒）
  2  rebase 与相对位姿
  3  平移轴映射
  4  笛卡尔限速器（映射目标 / Fake命令 / RM只读反馈）
  5  相对旋转与轴映射
  6  Base左乘 / TCP右乘对比
  7  RM 坐标系、TCP与安全配置（只读）
  8  IK、关节限位与触发关节诊断
  9  分层真机测试说明（本程序保持锁定，不运动）
  Enter 提前结束当前项；0/q 退出；h 重印菜单
切换 2～8 时，按键后的第一帧自动成为新基准。
"""


def _fmt(values, digits=3):
    return np.asarray(values).round(digits).tolist()


def _rotvec_deg(quaternion):
    return np.degrees(quaternion_to_rotation_vector_xyzw(quaternion))


class RM75MappingDiagnostics(Node):
    def __init__(self, ip="192.168.1.18", port=8080):
        super().__init__("rm75_mapping_diagnostics")
        self.mode = 0
        self.raw = None
        self.raw_time = None
        self.reference = None
        self.previous_q = None
        self.mode_started = time.monotonic()
        self.samples = []
        self.last_print = 0.0
        self.frame_count = 0
        self.keyboard_stop = threading.Event()

        self.robot = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        handle = self.robot.rm_create_robot_arm(ip, port)
        if handle.id < 0:
            raise RuntimeError(f"RM75 read-only connection failed: {handle.id}")
        self.connected = True
        code, state = self.robot.rm_get_current_arm_state()
        if code != 0:
            raise RuntimeError(f"rm_get_current_arm_state failed: {code}")
        pose = np.asarray(state["pose"], dtype=float)
        self.rm_zero_position = pose[:3].copy()
        self.rm_zero_q = quaternion_wxyz_to_xyzw(
            self.robot.rm_algo_euler2quaternion(pose[3:6].tolist()))
        self.actual_joints = np.asarray(state["joint"], dtype=float)
        self.joint_min = np.asarray(self.robot.rm_algo_get_joint_min_limit(), dtype=float)
        self.joint_max = np.asarray(self.robot.rm_algo_get_joint_max_limit(), dtype=float)
        self.ik = self._new_ik()
        self.limiter = CartesianPoseLimiter(
            self.rm_zero_position, self.rm_zero_q, 0.100, 0.020,
            math.radians(20.0))
        self.audit = None

        qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                         history=QoSHistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(PoseStamped, "/pico/right_wrist/raw_pose",
                                 self._pose, qos)
        self.create_timer(0.05, self._tick)
        self.keyboard_thread = threading.Thread(target=self._keyboard)
        self.keyboard_thread.start()
        print(MENU, flush=True)

    def _new_ik(self):
        return IKDryRunEvaluator(
            RealManAlgorithmAdapter(self.robot), self.actual_joints,
            self.joint_min, self.joint_max, 12.0, 5.0, 0.002,
            math.radians(1.0))

    def _keyboard(self):
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while rclpy.ok() and not self.keyboard_stop.is_set():
                readable, _writable, _exceptional = select.select(
                    [sys.stdin], [], [], 0.1)
                if not readable:
                    continue
                key = sys.stdin.read(1).lower()
                if key in "123456789":
                    self._select(int(key))
                elif key in ("0", "q"):
                    rclpy.shutdown()
                    return
                elif key in ("\r", "\n") and self.mode in MODE_DURATION_SEC:
                    self._complete_mode("操作员提前结束")
                elif key == "h":
                    print(MENU, flush=True)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def _select(self, mode):
        self.mode = mode
        self.mode_started = time.monotonic()
        self.samples = []
        self.reference = None
        self.previous_q = None
        self.frame_count = 0
        self.limiter.reset()
        self.ik = self._new_ik()
        titles = {
            1: "保持手环完全静止，开始采样 10 秒",
            2: "检查 rebase：第一帧应为零，随后移动再回原位",
            3: "平移：依次前后、左右、上下，每次回到起点",
            4: "限速：移动手环，比较 mapped / limited / actual",
            5: "旋转：锁定平移，依次绕三个方向小角度正反旋转",
            6: "旋转组合：比较 Base左乘与TCP右乘结果",
            7: "正在读取 RM75 配置（无运动调用）",
            8: "IK诊断：缓慢移动，观察具体限位关节",
            9: "真机阶段仍锁定",
        }
        duration = MODE_DURATION_SEC.get(mode)
        suffix = (f"（{duration:.0f} 秒后自动返回菜单，Enter 可提前结束）"
                  if duration is not None else "")
        print(f"\n[{mode}] {titles[mode]}{suffix}", flush=True)
        if mode == 7:
            self._print_audit()
            self._complete_mode("只读查询完成")
        elif mode == 9:
            print("本诊断程序没有运动后端。顺序应为 X → XYZ → 单轴旋转 → "
                  "三轴旋转 → 6DoF；完成 1～8 并核对记录后，再单独显式解锁。",
                  flush=True)
            self._complete_mode("说明显示完成")

    def _pose(self, msg):
        if msg.header.frame_id != "pico_tracking":
            return
        now = time.monotonic()
        p = np.array([msg.pose.position.x, msg.pose.position.y,
                      msg.pose.position.z], dtype=float)
        q = normalize_quaternion_xyzw([
            msg.pose.orientation.x, msg.pose.orientation.y,
            msg.pose.orientation.z, msg.pose.orientation.w])
        if self.previous_q is not None:
            q = continuous_quaternion_xyzw(q, self.previous_q)
        self.previous_q = q
        self.raw = (p, q, msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
        self.raw_time = now
        self.frame_count += 1
        if self.reference is None and self.mode in range(1, 9):
            self.reference = (p.copy(), q.copy())
            print("基准帧: relative XYZ=[0,0,0] mm, rotation=0 deg", flush=True)
        if self.mode == 1:
            self.samples.append((now, p.copy(), q.copy(), self.raw[2]))

    def _relative(self):
        p, q, _stamp = self.raw
        p0, q0 = self.reference
        dp = p - p0
        dq = quaternion_multiply_xyzw(q, quaternion_inverse_xyzw(q0))
        mapped_dp = AXIS_MAPPING @ dp
        mapped_r = AXIS_MAPPING @ quaternion_to_matrix_xyzw(dq) @ AXIS_MAPPING.T
        mapped_dq = matrix_to_quaternion_xyzw(mapped_r)
        return dp, dq, mapped_dp, mapped_dq

    def _tick(self):
        now = time.monotonic()
        if self.mode == 0 or self.raw is None:
            return
        duration = MODE_DURATION_SEC.get(self.mode)
        if (self.mode != 1 and duration is not None and
                now - self.mode_started >= duration):
            self._complete_mode("到达本项测试时间")
            return
        if self.raw_time is None or now - self.raw_time > 0.5:
            if now - self.last_print > 1.0:
                print("WARN: PICO 超过 0.5 秒没有新帧", flush=True)
                self.last_print = now
            return
        if self.mode == 1:
            if now - self.mode_started >= 10.0:
                self._finish_stability()
            return
        if self.mode in (7, 9) or self.reference is None:
            return
        if now - self.last_print < 0.25:
            return
        self.last_print = now
        dp, dq, mapped_dp, mapped_dq = self._relative()
        if self.mode == 2:
            print(f"relative_mm={_fmt(dp*1000)} rotation_deg="
                  f"{math.degrees(quaternion_angle_rad(dq,[0,0,0,1])):.2f} "
                  f"target_offset_mm={_fmt(mapped_dp*1000)}", flush=True)
        elif self.mode == 3:
            print(f"PICO_dXYZ_mm={_fmt(dp*1000)} A*dXYZ_mm={_fmt(mapped_dp*1000)} "
                  f"RM_target_offset_mm={_fmt(mapped_dp*1000)}", flush=True)
        elif self.mode == 4:
            target = self.rm_zero_position + mapped_dp
            limited, _ = self.limiter.step(target, self.rm_zero_q, 0.05)
            code, state = self.robot.rm_get_current_arm_state()
            actual = (np.asarray(state["pose"][:3])-self.rm_zero_position
                      if code == 0 else np.full(3, np.nan))
            print(f"mapped_mm={_fmt(mapped_dp*1000)} "
                  f"fake_limited_mm={_fmt((limited-self.rm_zero_position)*1000)} "
                  f"actual_readonly_mm={_fmt(actual*1000)}", flush=True)
        elif self.mode == 5:
            print(f"PICO_rotvec_deg={_fmt(_rotvec_deg(dq))} "
                  f"RM_mapped_rotvec_deg={_fmt(_rotvec_deg(mapped_dq))}", flush=True)
        elif self.mode == 6:
            left = quaternion_multiply_xyzw(mapped_dq, self.rm_zero_q)
            right = quaternion_multiply_xyzw(self.rm_zero_q, mapped_dq)
            separation = math.degrees(quaternion_angle_rad(left, right))
            print(f"mapped_rotvec_deg={_fmt(_rotvec_deg(mapped_dq))} "
                  f"Base_left_target_xyzw={_fmt(left,4)} "
                  f"TCP_right_target_xyzw={_fmt(right,4)} diff_deg={separation:.2f}",
                  flush=True)
        elif self.mode == 8:
            target_p = self.rm_zero_position + mapped_dp
            target_q = quaternion_multiply_xyzw(mapped_dq, self.rm_zero_q)
            result = self.ik.evaluate(target_p, target_q)
            if result.target_joints_deg is None:
                print(f"IK FAIL: {result.reason}", flush=True)
                return
            margins = np.minimum(result.target_joints_deg-self.joint_min,
                                 self.joint_max-result.target_joints_deg)
            index = int(np.argmin(margins))
            print(f"IK={'PASS' if result.accepted else 'FAIL'} "
                  f"q_deg={_fmt(result.target_joints_deg,2)} "
                  f"margin_deg={_fmt(margins,2)} nearest=J{index+1}:{margins[index]:.2f} "
                  f"jump={result.max_joint_delta_deg:.2f} reason={result.reason}",
                  flush=True)

    def _finish_stability(self):
        samples = self.samples
        if len(samples) < 2:
            print("[1] FAIL: 样本不足，请检查 PICO 数据", flush=True)
            self._complete_mode("采样结束")
            return
        positions = np.asarray([x[1] for x in samples])
        q0 = samples[0][2]
        angles = np.asarray([quaternion_angle_rad(x[2], q0) for x in samples])
        times = np.asarray([x[0] for x in samples])
        stamps = np.asarray([x[3] for x in samples])
        intervals = np.diff(times)
        repeats = sum(np.array_equal(samples[i][1], samples[i-1][1]) and
                      np.array_equal(samples[i][2], samples[i-1][2])
                      for i in range(1, len(samples)))
        backward_stamps = int(np.sum(np.diff(stamps) <= 0))
        span = (positions.max(axis=0)-positions.min(axis=0))*1000
        print(f"[1] 完成 samples={len(samples)} rate_hz={1/intervals.mean():.1f} "
              f"max_period_ms={intervals.max()*1000:.1f} XYZ_span_mm={_fmt(span)} "
              f"rotation_span_deg={math.degrees(angles.max()-angles.min()):.3f} "
              f"repeated_frames={repeats} nonincreasing_stamps={backward_stamps}",
              flush=True)
        self._complete_mode("采样完成")

    def _complete_mode(self, reason):
        completed = self.mode
        if completed == 0:
            return
        self.mode = 0
        self.reference = None
        self.samples = []
        print(f"[{completed}] 已结束：{reason}", flush=True)
        print("请选择下一项（1～9），按 h 查看完整菜单，按 q 退出：", flush=True)

    def _print_audit(self):
        try:
            if self.audit is None:
                self.audit = collect_read_only_audit(self.robot)
            names = ("rm_get_current_tool_frame", "rm_get_current_work_frame",
                     "rm_get_collision_stage", "rm_get_collision_detection",
                     "rm_get_self_collision_enable",
                     "rm_get_self_endeffector_collision_enable",
                     "rm_get_electronic_fence_enable",
                     "rm_get_electronic_fence_config")
            for name in names:
                print(f"{name}: {self.audit[name]}", flush=True)
            print("CANFD位姿透传的坐标语义属于官方文档/技术支持确认项，"
                  "本地查询不能替代官方确认。", flush=True)
        except Exception as exc:
            print(f"[7] READ-ONLY FAIL: {exc}", flush=True)

    def destroy_node(self):
        self.keyboard_stop.set()
        keyboard_thread = getattr(self, "keyboard_thread", None)
        if (keyboard_thread is not None and keyboard_thread.is_alive() and
                keyboard_thread is not threading.current_thread()):
            keyboard_thread.join(timeout=1.0)
        if getattr(self, "connected", False):
            self.robot.rm_delete_robot_arm()
            self.connected = False
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = RM75MappingDiagnostics()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
