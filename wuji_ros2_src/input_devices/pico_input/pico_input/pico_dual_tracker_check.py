"""Quiet one-shot verification that two Trackers arrive in the same SDK poll."""

import argparse
from contextlib import suppress
import sys
import time

from .pico_dual_tracker_pose_publisher import select_dual_tracker_poses
from .pico_tracker_pose_publisher import TrackerSelectionError
from .xrobotoolkit_client import XRoboToolkitClient


def collect_synchronized_frames(client, wrist_serial, upper_arm_serial,
                                required_frames=45, timeout_sec=15.0,
                                poll_rate_hz=60.0):
    """Return collected consecutive pair count or raise a concise error."""
    required = int(required_frames)
    timeout = float(timeout_sec)
    rate = float(poll_rate_hz)
    if required < 1 or timeout <= 0.0 or rate <= 0.0:
        raise ValueError("frame count, timeout, and poll rate must be positive")
    if not client.init():
        raise RuntimeError("无法连接 XRoboToolkit PC-Service")

    deadline = time.monotonic() + timeout
    consecutive = 0
    last_error = "尚未收到 Tracker 数据"
    last_sdk_timestamp = None
    while time.monotonic() < deadline:
        try:
            select_dual_tracker_poses(
                client.get_motion_tracker_pose(),
                client.get_motion_tracker_serial_numbers(),
                wrist_serial, upper_arm_serial)
            timestamp_getter = getattr(client, "get_time_stamp_ns", None)
            sdk_timestamp = int(timestamp_getter()) if timestamp_getter else 0
            if sdk_timestamp > 0 and sdk_timestamp == last_sdk_timestamp:
                time.sleep(1.0 / rate)
                continue
            if sdk_timestamp > 0:
                last_sdk_timestamp = sdk_timestamp
            consecutive += 1
            if consecutive >= required:
                return consecutive
        except TrackerSelectionError as error:
            consecutive = 0
            last_error = str(error)
        time.sleep(1.0 / rate)
    raise RuntimeError(
        f"{timeout:.1f} 秒内未形成稳定双 Tracker 同帧数据：{last_error}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--wrist-serial", required=True)
    parser.add_argument("--upper-arm-serial", required=True)
    parser.add_argument("--required-frames", type=int, default=45)
    parser.add_argument("--timeout-sec", type=float, default=15.0)
    args = parser.parse_args(argv)
    client = XRoboToolkitClient()
    try:
        count = collect_synchronized_frames(
            client, args.wrist_serial, args.upper_arm_serial,
            required_frames=args.required_frames,
            timeout_sec=args.timeout_sec)
        print(
            "双 Tracker 数据确认成功："
            f"手腕={args.wrist_serial}，上臂={args.upper_arm_serial}，"
            f"连续同帧={count}")
        return 0
    except (RuntimeError, ValueError) as error:
        print(f"双 Tracker 数据确认失败：{error}", file=sys.stderr)
        return 1
    finally:
        with suppress(Exception):
            client.close()


if __name__ == "__main__":
    raise SystemExit(main())
