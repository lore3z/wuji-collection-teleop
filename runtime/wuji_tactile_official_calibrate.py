#!/usr/bin/env python3
"""Run Wuji's official guided tactile *contact-model* calibration.

This program intentionally delegates calibration to the installed Wuji SDK.
It does *not* infer eight cells from a quiet recording, fill row 10, change
the firmware tactile mask, or alter raw packets. The calibration needs an
operator because it records real physical poses and writes the vendor contact
model used by tactile_residual/tactile_binary.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from wuji_glove_d435_collect import DEFAULT_GLOVE_SN, WUJI_OFFICIAL_ACTIVE_TAXELS


def _prompt(pose: dict[str, Any]) -> str:
    """Show every SDK-directed pose and let the operator control the run."""
    step = pose.get("step_index", "?")
    total = pose.get("step_total", "?")
    name = pose.get("step_name", "unnamed")
    seconds = pose.get("seconds_per_pose", "?")
    payload = pose.get("payload") or {}
    warnings = pose.get("warnings") or []
    print("\n" + "=" * 78)
    print(f"Wuji official tactile calibration — pose {step}/{total}: {name}")
    print(f"Hold time: {seconds} s")
    if payload:
        print("SDK instruction:", json.dumps(payload, ensure_ascii=False, default=str))
    if warnings:
        print("SDK warning:", "; ".join(map(str, warnings)))
    print("Make the requested pose. Do not touch the glove unless the SDK asks for it.")
    while True:
        choice = input("Enter=record  r=redo this pose  q=abort: ").strip().lower()
        if choice in ("", "p", "proceed"):
            return "proceed"
        if choice in ("r", "retry"):
            return "retry"
        if choice in ("q", "quit", "abort"):
            return "abort"
        print("Please enter Enter, r, or q.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glove-sn", default=DEFAULT_GLOVE_SN)
    parser.add_argument("--seconds-per-pose", type=float, default=30.0)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--sensitivity", type=float, default=None)
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    parser.add_argument(
        "--no-install",
        action="store_true",
        help="diagnostic only; do not install the calibrated vendor model (not useful for production)",
    )
    args = parser.parse_args()
    if args.seconds_per_pose <= 0 or args.epochs <= 0 or args.timeout_s <= 0:
        parser.error("seconds-per-pose, epochs and timeout-s must be positive")

    try:
        from wuji_sdk import SdkManager
    except ImportError as exc:
        raise SystemExit("wuji_sdk is unavailable; run ./setup.sh and use the project Python.") from exc

    last_line = ""

    def feedback(status: dict[str, Any]) -> None:
        nonlocal last_line
        operation = status.get("operation", "")
        state = status.get("state", "")
        step = f"{status.get('step_index', '?')}/{status.get('step_total', '?')}"
        if "epoch" in status:
            progress = f"epoch={status.get('epoch')}/{status.get('epoch_total')} best={status.get('best_val')}"
        else:
            progress = f"frames={status.get('frames_collected', '?')} elapsed={status.get('collect_elapsed', 0):.1f}/{status.get('collect_target', 0):.1f}s"
        line = f"[SDK] {operation} {state} step={step} {progress}"
        if line != last_line:
            print(line)
            last_line = line

    print("=" * 78)
    print("Wuji official tactile calibration")
    print(f"glove={args.glove_sn}; target official active taxels={WUJI_OFFICIAL_ACTIVE_TAXELS}")
    print("This changes the SDK contact model, not the raw tactile firmware mask. Keep the glove powered and connected.")
    print("After it completes, run the contract check; collection remains blocked unless raw, zones and point cloud are all 526.")
    print("=" * 78)

    glove = None
    try:
        glove = SdkManager.instance().connect(sn=args.glove_sn, device_name="official_tactile_calibration")
        summary = glove.calibrate_tactile_blocking(
            seconds_per_pose=args.seconds_per_pose,
            epochs=args.epochs,
            install=not args.no_install,
            sensitivity=args.sensitivity,
            timeout_s=args.timeout_s,
            on_feedback=feedback,
            on_pose_prompt=_prompt,
        )
        print("\n[SDK SUMMARY]")
        print(json.dumps(dict(summary), ensure_ascii=False, indent=2, default=str))
        verified = summary.get("verified_alive_taxels")
        installed = bool(summary.get("installed", False))
        if args.no_install:
            print("[NOT INSTALLED] --no-install was requested; raw SDK streams are unchanged.")
            raise SystemExit(3)
        if not installed:
            print("[FAILED] SDK did not report an installed tactile model.")
            raise SystemExit(2)
        if verified is not None and int(verified) != WUJI_OFFICIAL_ACTIVE_TAXELS:
            print(
                f"[FAILED] SDK verified {verified} alive taxels, not official "
                f"{WUJI_OFFICIAL_ACTIVE_TAXELS}. Do not record training data."
            )
            raise SystemExit(2)
        print("[DONE] Official contact-model calibration completed. Reconnect and run the 526 contract check next.")
    finally:
        if glove is not None:
            try:
                glove.disconnect()
            except Exception:
                pass
        # The device needs a moment to publish the installed model after disconnect.
        time.sleep(1.0)


if __name__ == "__main__":
    main()
