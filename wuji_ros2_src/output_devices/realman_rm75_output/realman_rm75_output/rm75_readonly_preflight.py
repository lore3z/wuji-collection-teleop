"""CLI for the RM75 read-only audit and Base-X local path preflight."""

import argparse
import json
import sys

from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e

from .readonly_preflight import collect_read_only_audit, run_base_x_preflight


def _json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Read-only RM75 audit and local Base-X IK/FK preflight")
    parser.add_argument("--robot-ip", default="192.168.1.18")
    parser.add_argument("--robot-port", type=int, default=8080)
    parser.add_argument("--max-offset-mm", type=float, default=5.0)
    parser.add_argument("--sample-step-mm", type=float, default=0.5)
    args = parser.parse_args(argv)

    robot = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    connected = False
    try:
        handle = robot.rm_create_robot_arm(args.robot_ip, args.robot_port)
        if handle.id < 0:
            raise RuntimeError(f"RM75 connection failed: handle.id={handle.id}")
        connected = True
        audit = collect_read_only_audit(robot)
        state = audit["rm_get_current_arm_state"][1]
        preflight = run_base_x_preflight(
            robot, state,
            max_offset_m=args.max_offset_mm / 1000.0,
            sample_step_m=args.sample_step_mm / 1000.0,
        )
        report = {
            "safety_notice": (
                "READ-ONLY: no move, stop, enable, or configuration API was called"
            ),
            "audit": audit,
            "base_x_preflight": preflight,
        }
        print(json.dumps(report, ensure_ascii=False, indent=2,
                         default=_json_default, allow_nan=True))
        return 0 if preflight["passed"] else 2
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        if connected:
            robot.rm_delete_robot_arm()


if __name__ == "__main__":
    raise SystemExit(main())
