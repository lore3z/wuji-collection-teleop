#!/usr/bin/env python3
"""Display or headlessly validate the exact q21 sent by the GOOD teleop loop.

This process consumes only the isolated preview UDP port. It never opens a
glove, ROS publisher, bridge, CAN interface or EMG/PyTorch environment.
"""
from __future__ import annotations

# Resolve project imports independently of the current working directory.
import sys as _project_sys
from pathlib import Path as _ProjectPath
_project_root = _ProjectPath(__file__).resolve().parents[1]
if str(_project_root) not in _project_sys.path:
    _project_sys.path.insert(0, str(_project_root))


import argparse
import contextlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import sys
import time

sys.dont_write_bytecode = True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=17622)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    if not 1024 <= args.port <= 65535 or args.port in (15120, 15121, 17621):
        parser.error("Use an isolated preview port in [1024, 65535]")
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be positive")
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        parser.error("--duration must be finite and positive")
    if args.report is not None and args.report.exists():
        parser.error("Report path exists; choose a new file")

    import mujoco
    import numpy as np
    import yaml
    from emg_teleop_launcher import BASELINE, load_sim, _write_new_json, verify_baseline

    verify_baseline()
    sim = load_sim()
    config_path = BASELINE / "example/config/l20_feature_retarget_wuji_right.yaml"
    adapter = sim.Retargeter.from_yaml(str(config_path), hand_side="right").optimizer.adapter
    config = yaml.safe_load(config_path.read_text())
    model = mujoco.MjModel.from_xml_path(str((config_path.parent / config["optimizer"]["mjcf_path"]).resolve()))
    data = mujoco.MjData(model)
    joint_map = sim.build_joint_map(model, adapter.q_names)
    report = {"headless": args.headless, "port": args.port, "mujoco_frames": 0,
              "rejected_packets": 0, "max_limit_violation_rad": 0.0,
              "max_coupling_residual_rad": 0.0,
              "max_command_saturation_rad": 0.0,
              "torch_imported": "torch" in sys.modules,
              "q_names": list(adapter.q_names), "mapped_joints": len(joint_map)}
    stop = False
    def finish(_signum, _frame):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, finish)
    signal.signal(signal.SIGINT, finish)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", args.port))
    sock.setblocking(False)
    print(f"[EMG MuJoCo] {'headless' if args.headless else 'display'} ready: 127.0.0.1:{args.port}", flush=True)
    if args.headless:
        context = contextlib.nullcontext(None)
    else:
        import mujoco.viewer
        context = mujoco.viewer.launch_passive(model, data)
    start = time.monotonic()
    last_rx = None
    mean, m2 = np.zeros(21), np.zeros(21)
    qmin, qmax = np.full(21, np.inf), np.full(21, -np.inf)
    kp = None
    try:
        with context as viewer:
            while not stop and (viewer is None or viewer.is_running()):
                if args.duration is not None and time.monotonic() - start >= args.duration:
                    break
                for _ in range(512):
                    try:
                        payload, _ = sock.recvfrom(16385)
                    except BlockingIOError:
                        break
                    try:
                        if len(payload) > 16384:
                            raise ValueError("oversize preview packet")
                        msg = json.loads(payload)
                        q = np.asarray(msg["q21"], dtype=np.float64)
                        kp = np.asarray(msg["kp"], dtype=np.float64)
                        if q.shape != (21,) or kp.shape != (21, 3):
                            raise ValueError("wrong preview shapes")
                        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(kp)):
                            raise ValueError("nonfinite preview")
                    except (TypeError, ValueError, KeyError, OverflowError):
                        report["rejected_packets"] += 1
                        continue
                    # Exact pose the hardware path can actually execute:
                    # q21 -> native u16 -> q21.
                    #
                    # Keep command saturation as a diagnostic, but display and
                    # accumulate statistics from q_exec, not the pre-saturation q.
                    u = adapter.compress(q)
                    q_exec = adapter.expand(u)

                    saturation = float(
                        np.max(np.abs(q_exec - q))
                    )

                    u_check = adapter.compress(q_exec)
                    q_roundtrip = adapter.expand(u_check)

                    residual = float(
                        np.max(np.abs(q_roundtrip - q_exec))
                    )

                    violation = float(
                        max(
                            0.0,
                            np.max(adapter.lower - u),
                            np.max(u - adapter.upper),
                        )
                    )

                    for qi, qadr, _name in joint_map:
                        data.qpos[qadr] = q_exec[qi]

                    mujoco.mj_forward(model, data)

                    if not np.all(np.isfinite(data.xpos)):
                        raise RuntimeError(
                            "MuJoCo FK produced nonfinite positions"
                        )

                    report["max_limit_violation_rad"] = max(
                        report["max_limit_violation_rad"],
                        violation,
                    )

                    report["max_coupling_residual_rad"] = max(
                        report["max_coupling_residual_rad"],
                        residual,
                    )

                    report["max_command_saturation_rad"] = max(
                        report["max_command_saturation_rad"],
                        saturation,
                    )

                    report["mujoco_frames"] += 1

                    delta = q_exec - mean
                    mean += delta / report["mujoco_frames"]
                    m2 += delta * (q_exec - mean)

                    qmin = np.minimum(qmin, q_exec)
                    qmax = np.maximum(qmax, q_exec)
                    last_rx = time.monotonic()
                    if args.max_frames is not None and report["mujoco_frames"] >= args.max_frames:
                        stop = True
                        break
                if viewer is not None:
                    if kp is not None:
                        sim.draw_skeleton(viewer.user_scn, kp)
                    viewer.sync()
                time.sleep(1 / 120 if args.headless else 1 / 60)
    finally:
        sock.close()
        report["elapsed_s"] = time.monotonic() - start
        report["last_packet_age_ms"] = (time.monotonic() - last_rx) * 1000 if last_rx else None
        if report["mujoco_frames"]:
            report["q21_mean"] = mean.tolist()
            report["q21_std"] = np.sqrt(m2 / report["mujoco_frames"]).tolist()
            report["q21_min"], report["q21_max"] = qmin.tolist(), qmax.tolist()
        _write_new_json(args.report, report)
        print("[EMG MuJoCo] " + json.dumps(report, allow_nan=False), flush=True)
    exit_code = 0 if report["mujoco_frames"] or not args.headless else 1
    if not args.headless:
        # MuJoCo/GLX has already closed its context above. Avoid a second
        # native teardown during Python finalization, which can hang or emit
        # GLXBadDrawable after the report has been safely written.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
