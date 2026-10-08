"""Experimental fixed 1 mm RM Base-X pose-pass-through test."""

import argparse
import math
import sys
import threading
import time

import numpy as np
from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e

from .movep_backend import RealMovePBackend
from .pose_mapper import quaternion_wxyz_to_xyzw
from .readonly_preflight import collect_read_only_audit, run_base_x_preflight


def fixed_offsets(step_count=20, hold_count=10):
    """One second out, half-second hold, one second home at 20 Hz."""
    outward = np.linspace(0.0, 0.001, step_count + 1)[1:]
    hold = np.full(hold_count, 0.001)
    home = np.linspace(0.001, 0.0, step_count + 1)[1:]
    return np.concatenate((outward, hold, home))


def _errors_clear(audit):
    state = audit["rm_get_current_arm_state"][1]
    controller = audit["rm_get_controller_state"]
    joints = audit["rm_get_joint_err_flag"]
    arm_errors = state.get("err", {}).get("err", [])
    return (controller.get("system_error", controller.get("sys_err", 0)) == 0 and
            all(int(value) == 0 for value in joints.get("err_flag", [])) and
            all(str(value) == "0" for value in arm_errors))


def _wait_until_stable(robot, initial_joints, timeout=2.0):
    """Tianji-style feedback poll: require several stationary readbacks."""
    deadline = time.monotonic() + timeout
    previous = np.asarray(initial_joints, dtype=float)
    stable = 0
    while time.monotonic() < deadline:
        ret, state = robot.rm_get_current_arm_state()
        if ret != 0:
            return False, f"state read failed: {ret}"
        current = np.asarray(state.get("joint", []), dtype=float)
        if current.shape != (7,):
            return False, "invalid joint feedback"
        if np.max(np.abs(current - previous)) <= 0.01:
            stable += 1
            if stable >= 3:
                return True, "three stable joint readbacks"
        else:
            stable = 0
        previous = current
        time.sleep(0.05)
    return False, "joint feedback did not become stable before timeout"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-ip", default="192.168.1.18")
    parser.add_argument("--robot-port", type=int, default=8080)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--enable-motion", action="store_true")
    args = parser.parse_args(argv)
    if not (args.execute and args.enable_motion):
        print("REFUSED: real motion requires --execute --enable-motion", file=sys.stderr)
        return 2

    robot = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    connected = False
    backend = None
    stop_requested = threading.Event()
    try:
        handle = robot.rm_create_robot_arm(args.robot_ip, args.robot_port)
        if handle.id < 0:
            raise RuntimeError(f"connection failed: handle.id={handle.id}")
        connected = True
        audit = collect_read_only_audit(robot)
        if not _errors_clear(audit):
            raise RuntimeError("controller, arm, or joint error is nonzero")
        state = audit["rm_get_current_arm_state"][1]
        initial_pose = np.asarray(state["pose"], dtype=float)
        initial_joints = np.asarray(state["joint"], dtype=float)
        preflight = run_base_x_preflight(
            robot, state, max_offset_m=0.001, sample_step_m=0.00025)
        if not preflight["passed"]:
            raise RuntimeError("fixed 1 mm preflight failed")
        quaternion = quaternion_wxyz_to_xyzw(
            robot.rm_algo_euler2quaternion(initial_pose[3:6].tolist()))
        backend = RealMovePBackend(robot, follow=False)

        def wait_for_enter():
            sys.stdin.readline()
            stop_requested.set()

        threading.Thread(target=wait_for_enter, daemon=True).start()
        print("READY: fixed Base +X 1 mm, 1 mm/s, follow=false")
        print("Press Enter at any time for experimental slow stop")
        for remaining in (3, 2, 1):
            print(f"Starting in {remaining}...", flush=True)
            time.sleep(1.0)

        period = 0.05
        next_tick = time.monotonic()
        max_actual_offset = 0.0
        max_command_error = 0.0
        previous_joints = initial_joints.copy()
        for offset in fixed_offsets():
            if stop_requested.is_set():
                code = backend.slow_stop()
                print(f"ENTER slow-stop return_code={code}")
                ok, reason = _wait_until_stable(robot, previous_joints)
                print(f"STOP_CONFIRM ok={ok} reason={reason}")
                return 0 if code == 0 and ok else 3
            target = initial_pose[:3].copy()
            target[0] += offset
            code = backend.send_pose(target, quaternion, time.monotonic())
            if code != 0:
                raise RuntimeError(f"pose pass-through return code {code}")
            ret, feedback = robot.rm_get_current_arm_state()
            if ret != 0:
                raise RuntimeError(f"state feedback return code {ret}")
            actual_pose = np.asarray(feedback.get("pose", []), dtype=float)
            joints = np.asarray(feedback.get("joint", []), dtype=float)
            if actual_pose.shape != (6,) or joints.shape != (7,):
                raise RuntimeError("invalid state feedback shape")
            actual_offset = abs(float(actual_pose[0] - initial_pose[0]))
            command_error = abs(float(actual_pose[0] - target[0]))
            max_actual_offset = max(max_actual_offset, actual_offset)
            max_command_error = max(max_command_error, command_error)
            if actual_offset > 0.003:
                raise RuntimeError(f"actual X escaped 3 mm guard: {actual_offset}")
            if np.max(np.abs(joints - previous_joints)) > 2.0:
                raise RuntimeError("actual joint feedback jumped by more than 2 deg")
            previous_joints = joints
            next_tick += period
            time.sleep(max(0.0, next_tick - time.monotonic()))

        ok, reason = _wait_until_stable(robot, previous_joints)
        print(f"COMPLETE max_actual_offset_m={max_actual_offset:.9f} "
              f"max_command_error_m={max_command_error:.9f} "
              f"stable={ok} reason={reason}")
        return 0 if ok else 3
    except Exception as exc:
        print(f"FAULT: {type(exc).__name__}: {exc}", file=sys.stderr)
        if backend is not None:
            code = backend.emergency_stop()
            print(f"FAULT emergency-stop return_code={code}", file=sys.stderr)
        return 1
    finally:
        if connected:
            robot.rm_delete_robot_arm()
