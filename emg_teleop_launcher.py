"""Input-only adapter around byte-for-byte snapshots of the two GOOD modes.

Only device construction, launch paths, subprocess ownership and UDP transport
are adapted. The calibrated retarget, pinch/grasp/thumb functions and hardware
payload construction execute from the unchanged frozen sources. A private run
directory protects the existing calibration from the baseline's cache writer.
"""

from __future__ import annotations

import argparse
from collections import deque
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import pickle
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import types

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT / "runtime/emg_teleop_baseline_v1"
DEFAULT_VIEWER_PORT = 17622
MODE_LOCK_PATH = Path("/tmp/wuji_emg_teleop_mode.lock")
CONTROLLER_ENTRYPOINTS = {
    "skeleton_teleop_MODE_A_emg.py",
    "skeleton_teleop_MODE_B_emg.py",
    "skeleton_teleop_MODE_A_good_pinch.py",
    "skeleton_teleop_v94_hw.py",
    "skeleton_teleop_MODE_B_thumb5.py",
}
STARTUP_FRESH_FRAMES = 5
STARTUP_FRESH_TIMEOUT_S = 2.0
STARTUP_POLL_INTERVAL_S = 0.005


class HardwareInputSafety:
    """Keep startup draining separate from the irreversible ACTIVE latch."""

    def __init__(self, enabled):
        self.enabled = bool(enabled)
        self.active = False
        self.latched = False
        self.seen_valid = False
        self.session_id = None

    def observe(self, valid, metadata):
        session_id = metadata.get("session_id")
        if valid:
            if self.session_id is None:
                self.session_id = session_id
            elif session_id != self.session_id:
                if self.enabled:
                    self.latched = self.active
                    raise RuntimeError("Skeleton session changed during hardware startup/runtime")
                self.session_id = session_id
        if self.enabled and self.active and self.seen_valid and not valid:
            self.latched = True
        self.seen_valid |= valid
        return valid and not self.latched

    def activate(self):
        if self.enabled and (not self.seen_valid or self.session_id is None):
            raise RuntimeError("Cannot activate hardware without a validated Skeleton session")
        self.active = self.enabled


def _fresh_sample(device, data, expected_session):
    metadata = device.metadata
    session_id = metadata.get("session_id")
    if session_id is not None and session_id != expected_session:
        raise RuntimeError("Skeleton session changed while waiting for hardware startup")
    age_ms = metadata.get("receiver_age_ms")
    quality = metadata.get("quality")
    seq = metadata.get("seq")
    fresh = (
        data.get("right_fingers") is not None
        and metadata.get("fresh") is True
        and isinstance(quality, dict)
        and quality.get("valid") is True
        and isinstance(seq, int)
        and not isinstance(seq, bool)
        and isinstance(age_ms, (int, float))
        and math.isfinite(age_ms)
        and age_ms < device.max_age_s * 1000.0
    )
    return fresh, seq, metadata


def wait_for_post_ready_fresh(device, ready_seq, expected_session, *,
                              required=STARTUP_FRESH_FRAMES,
                              timeout=STARTUP_FRESH_TIMEOUT_S,
                              poll_interval=STARTUP_POLL_INTERVAL_S,
                              monotonic=time.monotonic, sleep=time.sleep):
    """Require distinct, current packets newer than the bridge-ready snapshot."""
    if not isinstance(ready_seq, int) or isinstance(ready_seq, bool):
        raise RuntimeError("Bridge READY snapshot has no valid Skeleton seq")
    if not expected_session:
        raise RuntimeError("Bridge READY snapshot has no Skeleton session")
    deadline = monotonic() + timeout
    fresh_count = 0
    last_counted_seq = ready_seq
    while monotonic() < deadline:
        data = device.get_fingers_data()
        fresh, seq, metadata = _fresh_sample(device, data, expected_session)
        if fresh and seq > ready_seq:
            if seq > last_counted_seq:
                fresh_count += 1
                last_counted_seq = seq
                if fresh_count >= required:
                    return {
                        "required_fresh_frames": required,
                        "fresh_frames": fresh_count,
                        "bridge_ready_seq": ready_seq,
                        "active_seq": seq,
                        "session_id": expected_session,
                        "receiver_age_ms": metadata["receiver_age_ms"],
                    }
            elif seq < last_counted_seq:
                fresh_count = 0
        elif not fresh:
            fresh_count = 0
        sleep(poll_interval)
    raise TimeoutError(
        f"No {required} post-READY fresh Skeleton packets within {timeout:.3f}s"
    )


def wait_bridge_ready_with_skeleton_drain(proc, device, ready_check, *, timeout=10.0,
                                          fresh_timeout=STARTUP_FRESH_TIMEOUT_S,
                                          required=STARTUP_FRESH_FRAMES,
                                          poll_interval=STARTUP_POLL_INTERVAL_S,
                                          monotonic=time.monotonic, sleep=time.sleep):
    """Drain input while the DISARMED bridge starts, then run the fresh gate."""
    deadline = monotonic() + timeout
    drain_calls = 0
    while monotonic() < deadline:
        device.get_fingers_data()
        drain_calls += 1
        if proc.poll() is not None:
            raise SystemExit(f"bridge exited before READY, returncode={proc.returncode}")
        if ready_check():
            # Drain once at the READY boundary so the snapshot is the newest
            # packet already queued at that instant, never an earlier cache.
            device.get_fingers_data()
            drain_calls += 1
            metadata = device.metadata
            expected_session = getattr(device, "startup_session_id", None)
            if expected_session is None:
                expected_session = metadata.get("session_id")
            if metadata.get("session_id") != expected_session:
                raise RuntimeError("Skeleton session changed before bridge READY")
            result = wait_for_post_ready_fresh(
                device, metadata.get("seq"), expected_session,
                required=required, timeout=fresh_timeout,
                poll_interval=poll_interval, monotonic=monotonic, sleep=sleep,
            )
            if proc.poll() is not None:
                raise SystemExit(
                    f"bridge exited during post-READY Skeleton gate, returncode={proc.returncode}"
                )
            result["bridge_wait_drain_calls"] = drain_calls
            return result
        sleep(poll_interval)
    raise SystemExit("bridge startup timed out before UDP listener became READY")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_baseline():
    """Fail on drift rather than silently loading a different stable algorithm."""
    manifest = json.loads((BASELINE / "manifest.json").read_text())
    for relative, entry in manifest["files"].items():
        path = BASELINE / relative
        if sha256(path) != entry["sha256"]:
            raise RuntimeError(f"Frozen baseline checksum mismatch: {path}")
    for filename, expected in manifest["external_dependencies"].items():
        if sha256(ROOT / filename) != expected:
            raise RuntimeError(f"Audited external dependency changed: {filename}")
    return manifest


def verify_original_backups():
    manifest = json.loads((BASELINE / "manifest.json").read_text())
    for filename, expected in manifest["stable_backup_files"].items():
        if sha256(ROOT / filename) != expected:
            raise RuntimeError(f"Original GOOD backup changed: {filename}")
    return len(manifest["stable_backup_files"])


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    if Path(filename).is_relative_to(BASELINE):
        from runtime.project_paths import portable_home
        source = Path(filename).read_text().replace("Path.home()", f"Path({str(portable_home())!r})")
        exec(compile(source, str(filename), "exec"), module.__dict__)
    else:
        spec.loader.exec_module(module)
    return module


def acquire_mode_lock(mode):
    """Keep every GOOD A/B controller mutually exclusive, even on custom ports."""
    handle = MODE_LOCK_PATH.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.seek(0)
        owner = handle.read().strip() or "unknown owner"
        handle.close()
        raise RuntimeError(f"Another GOOD A/B controller owns {MODE_LOCK_PATH}: {owner}")
    conflicts = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            argv = (entry / "cmdline").read_bytes().split(b"\0")
            names = {Path(arg.decode(errors="replace")).name for arg in argv if arg}
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        matched = sorted(names & CONTROLLER_ENTRYPOINTS)
        if matched:
            conflicts.append(f"pid {entry.name} ({matched[0]})")
    if conflicts:
        handle.close()
        raise RuntimeError("Refusing concurrent GOOD A/B controller: " + ", ".join(conflicts))
    handle.seek(0)
    handle.truncate()
    handle.write(json.dumps({"pid": os.getpid(), "mode": mode, "started": time.time()}))
    handle.flush()
    return handle


def load_sim():
    # Substitute the provider *before* the frozen shared module is imported.
    # This also makes EMG preview independent of the Wuji SDK installation.
    from emg_skeleton_device import EMGSkeletonDevice
    provider = types.ModuleType("input_devices.wuji_glove_device")
    provider.WujiGloveDevice = EMGSkeletonDevice
    sys.modules[provider.__name__] = provider
    sys.path.insert(0, str(BASELINE))
    return _load("skeleton_teleop_v9_calibrated_sim",
                 BASELINE / "skeleton_teleop_v9_calibrated_sim.py")


def load_baseline(mode, controller_source=None):
    verify_baseline()
    # The snapshot's local retarget package precedes the evolving workspace.
    sim = load_sim()
    _load("wuji_l20_g20_teleop", BASELINE / "wuji_l20_g20_teleop.py")
    expected = BASELINE / ("mode_" + mode.lower() + ".py")
    source = Path(controller_source).resolve() if controller_source else expected
    if sha256(source) != sha256(expected):
        raise RuntimeError(
            f"Restored Mode {mode} controller does not match the frozen GOOD source: {source}"
        )
    baseline = _load("emg_frozen_mode_" + mode.lower(), source)
    if "torch" in sys.modules:
        raise RuntimeError("Unexpected PyTorch import in the teleop process")
    return baseline, sim


def apply_verified_environment(mode):
    # Reproduce exports from the GOOD launch script without executing the script
    # (which would activate environments and start physical hardware).
    for name in list(os.environ):
        if name.startswith(("V94_", "V10_", "V116_", "V117_", "V118_")):
            del os.environ[name]
    source = (BASELINE / ("mode_" + mode.lower() + ".env.sh")).read_text()
    for name, value in re.findall(r"^export ([A-Z0-9_]+)=([^\n]+)$", source, re.M):
        if not re.fullmatch(r"[-+0-9.]+", value.strip()):
            raise RuntimeError("Unexpected launch export: " + name)
        os.environ[name] = value.strip()
    os.environ["V94_PAD_TARGETS"] = str(BASELINE / "runtime/thumb_all_fingers_manual_pad_IK.npz")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"


def _parser(mode):
    parser = argparse.ArgumentParser(description=f"EMG Skeleton → unchanged GOOD Mode {mode}")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--sim", action="store_true", help="Preview; hardware output stays disabled")
    group.add_argument("--hardware", action="store_true", help="Hardware launcher; additionally requires --arm")
    parser.add_argument("--arm", action="store_true", help="Explicitly authorize this invocation's hardware output")
    parser.add_argument("--port", type=int, default=17621, help="Local EMG skeleton UDP input")
    parser.add_argument("--max-age", type=float, default=0.20, help="Maximum input age in seconds")
    parser.add_argument("--hz", type=float, default=120.0)
    parser.add_argument("--startup-timeout", type=float, default=30.0)
    parser.add_argument("--duration", type=float, help="Stop after this many seconds from input-device creation")
    parser.add_argument("--max-frames", type=int, help="Stop after this many calibrated retarget calls")
    parser.add_argument("--viewer", action="store_true", help="Start isolated MuJoCo viewer")
    parser.add_argument("--headless", action="store_true", help="Start MuJoCo FK validation without a display")
    parser.add_argument("--viewer-port", type=int, default=DEFAULT_VIEWER_PORT)
    parser.add_argument("--viewer-report", type=Path, help="New JSON output path for MuJoCo statistics")
    parser.add_argument("--metrics-json", type=Path, help="New JSON output path for controller statistics")
    parser.add_argument("--calibration", type=Path, help="Trusted local OPEN/FIST/O cache; copied into a new session")
    parser.add_argument("--controller-source", type=Path,
                        help="Restored byte-identical GOOD controller work copy")
    parser.add_argument("--arm-confirmed", action="store_true",
                        help="Launcher already obtained the real-hand Enter confirmation")
    parser.add_argument("--verify-only", action="store_true", help="Verify frozen sources/dependencies and exit")
    return parser


def _write_new_json(path, report):
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as output:
        json.dump(report, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")


def main(mode, argv=None):
    parser = _parser(mode)
    args = parser.parse_args(argv)
    # These checks precede all simulation/SDK imports and subprocess creation.
    if args.arm and not args.hardware:
        parser.error("--arm requires --hardware; simulation entrypoints cannot enable hardware")
    if args.hardware and not args.arm:
        parser.error("Hardware is disabled. Re-run explicitly with --arm to authorize this invocation.")
    if args.arm_confirmed and not (args.hardware and args.arm):
        parser.error("--arm-confirmed requires --hardware --arm")
    if args.hardware and args.headless:
        parser.error("--headless is for offline simulation only")
    for name in ("max_age", "hz", "startup_timeout", "duration"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error("--" + name.replace("_", "-") + " must be finite and positive")
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be positive")
    if not 1024 <= args.port <= 65535 or not 1024 <= args.viewer_port <= 65535:
        parser.error("UDP ports must be in [1024, 65535]")
    if args.port == args.viewer_port or args.port in (15120, 15121) or args.viewer_port in (15120, 15121):
        parser.error("Use separate EMG ports, preserving existing hardware/viewer ports 15120/15121")
    for path in (args.metrics_json, args.viewer_report):
        if path is not None and path.exists():
            parser.error(f"Report path already exists; choose a new file: {path}")
    if args.metrics_json and args.viewer_report and args.metrics_json.resolve() == args.viewer_report.resolve():
        parser.error("Controller and viewer reports need separate paths")

    manifest = verify_baseline()
    backup_count = verify_original_backups()
    if args.controller_source:
        expected = BASELINE / ("mode_" + mode.lower() + ".py")
        if sha256(args.controller_source) != sha256(expected):
            raise RuntimeError(
                f"Restored Mode {mode} controller does not match the frozen GOOD source: {args.controller_source}"
            )
    if args.verify_only:
        print(json.dumps({"verified_original_backup_files": backup_count,
                          "mode_a_origin": manifest["mode_a_origin"],
                          "mode_b_origin": manifest["mode_b_origin"],
                          "frozen_files": len(manifest["files"]),
                          "external_dependencies": len(manifest["external_dependencies"])}))
        return 0

    import numpy as np
    from emg_skeleton_device import EMGSkeletonDevice

    apply_verified_environment(mode)
    baseline, sim = load_baseline(mode, args.controller_source)
    calibration = args.calibration or BASELINE / "runtime/v93_skeleton_calib_right.pkl"
    with calibration.open("rb") as cache:
        saved = pickle.load(cache)
    for pose in ("OPEN", "FIST", "O"):
        if pose not in saved["calib"] or np.asarray(saved["calib_kp"][pose]).shape != (21, 3):
            raise ValueError("Invalid calibration cache: " + str(calibration))

    sessions = ROOT / "runtime/emg_teleop_sessions"
    sessions.mkdir(exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix=f"mode_{mode.lower()}_", dir=sessions))
    (run_dir / "example/config").mkdir(parents=True)
    (run_dir / "runtime").mkdir()
    for name in ("l20_feature_retarget_wuji_right.yaml", "l20_semantic_v4_wuji_right.yaml"):
        import yaml
        original = BASELINE / "example/config" / name
        config = yaml.safe_load(original.read_text())
        for key in ("urdf_path", "mjcf_path"):
            config["optimizer"][key] = str((original.parent / config["optimizer"][key]).resolve())
        (run_dir / "example/config" / name).write_text(yaml.safe_dump(config, sort_keys=False))
    shutil.copy2(calibration, run_dir / "runtime/v93_skeleton_calib_right.pkl")
    baseline.ROOT = run_dir
    # Baseline runtime writers derive their destination from __file__. No source
    # is rewritten; this virtual entry path puts any such output in this session.
    baseline.__file__ = str(run_dir / ("mode_" + mode.lower() + ".py"))

    report = {"mode": mode, "hardware_authorized": args.hardware and args.arm,
              "input_port": args.port, "viewer_port": args.viewer_port,
              "baseline_source_sha256": manifest["files"][f"mode_{mode.lower()}.py"]["sha256"],
              "session_directory": str(run_dir), "calibration_sha256": sha256(calibration),
              "retarget_frames": 0, "hardware_packets_sent": 0,
              "hardware_packets_suppressed": 0, "viewer_packets": 0,
              "stop_reason": "normal", "torch_imported": False}
    devices, durations = [], deque(maxlen=4096)
    timing = {"sum": 0.0, "max": 0.0}
    q_mean, q_m2 = np.zeros(21), np.zeros(21)
    viewer = None
    mode_lock = None
    exit_code = 0

    class GuardedDevice(EMGSkeletonDevice):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw, port=args.port, max_age_s=args.max_age)
            self.started = time.monotonic()
            self.input_safety = HardwareInputSafety(args.hardware)
            devices.append(self)
            print(f"[EMG] Waiting on udp://127.0.0.1:{args.port}; max age {args.max_age:.3f}s", flush=True)

        @property
        def seen_valid(self):
            return self.input_safety.seen_valid

        @property
        def startup_session_id(self):
            return self.input_safety.session_id

        @property
        def hardware_latched(self):
            return self.input_safety.latched

        @hardware_latched.setter
        def hardware_latched(self, value):
            self.input_safety.latched = bool(value)

        def get_fingers_data(self):
            elapsed = time.monotonic() - self.started
            if args.duration is not None and elapsed >= args.duration:
                report["stop_reason"] = "duration reached"
                raise KeyboardInterrupt
            if args.max_frames is not None and report["retarget_frames"] >= args.max_frames:
                report["stop_reason"] = "max frames reached"
                raise KeyboardInterrupt
            if not self.seen_valid and elapsed > args.startup_timeout:
                raise TimeoutError("No valid EMG Skeleton before --startup-timeout")
            if viewer is not None and viewer.poll() is not None:
                if viewer.returncode != 0:
                    raise RuntimeError(f"MuJoCo viewer exited with {viewer.returncode}")
                report["stop_reason"] = "viewer closed"
                raise KeyboardInterrupt
            data = super().get_fingers_data()
            valid = data["right_fingers"] is not None
            was_latched = self.hardware_latched
            allowed = self.input_safety.observe(valid, self.metadata)
            if self.hardware_latched and not was_latched:
                if args.hardware:
                    print("[EMG] Input invalid/stale: output latched off; restart with --arm to resume.", flush=True)
            if not allowed:
                return {"left_fingers": None, "right_fingers": None}
            return data

    sim.WujiGloveDevice = GuardedDevice
    original_retarget = sim.calibrated_retarget

    def observed_retarget(*a, **kw):
        start = time.monotonic()
        result = original_retarget(*a, **kw)
        q = np.asarray(result[0], dtype=np.float64)
        if q.shape != (21,) or not np.all(np.isfinite(q)):
            raise RuntimeError("Baseline retarget returned invalid q21")
        report["retarget_frames"] += 1
        duration = time.monotonic() - start
        durations.append(duration)
        timing["sum"] += duration
        timing["max"] = max(timing["max"], duration)
        delta = q - q_mean
        q_mean[:] += delta / report["retarget_frames"]
        q_m2[:] += delta * (q - q_mean)
        return result

    sim.calibrated_retarget = observed_retarget

    class OutputSocket:
        def __init__(self, *a, **kw):
            self.sock = socket.socket(*a, **kw)

        def sendto(self, payload, addr):
            if addr == ("127.0.0.1", 15120):
                if not args.hardware or not args.arm:
                    raise RuntimeError("Blocked hardware packet in simulation")
                if not devices or devices[-1].hardware_latched or not devices[-1].metadata["fresh"]:
                    report["hardware_packets_suppressed"] += 1
                    if devices:
                        devices[-1].hardware_latched = True
                    return len(payload)
                report["hardware_packets_sent"] += 1
            elif addr == ("127.0.0.1", 15121):
                addr = ("127.0.0.1", args.viewer_port)
                report["viewer_packets"] += 1
            else:
                raise RuntimeError("Unexpected baseline UDP destination: " + repr(addr))
            return self.sock.sendto(payload, addr)

        def close(self):
            self.sock.close()

    baseline.socket = types.SimpleNamespace(socket=OutputSocket, AF_INET=socket.AF_INET,
                                             SOCK_DGRAM=socket.SOCK_DGRAM)
    original_support = baseline.load_old_verified_bridge_support

    def owned_bridge_support():
        if not args.hardware:
            return types.SimpleNamespace(stop_old_bridge=lambda: None, stop_proc=lambda p: None)
        support = original_support()
        def refuse_existing_bridge():
            if support.udp_pids():
                raise RuntimeError("Hardware bridge port already occupied; existing process left running")
        def wait_udp_listener_with_freshness(proc, timeout=10.0):
            if not devices:
                raise RuntimeError("Hardware bridge started before Skeleton input device")
            device = devices[-1]
            startup = wait_bridge_ready_with_skeleton_drain(
                proc, device, lambda: bool(support.udp_pids()), timeout=timeout,
            )
            device.input_safety.activate()
            report["startup_freshness"] = startup
            print(
                "[EMG] Bridge READY + post-READY fresh Skeleton confirmed: "
                f"session={startup['session_id']} seq={startup['active_seq']} "
                f"frames={startup['fresh_frames']} age={startup['receiver_age_ms']:.1f}ms",
                flush=True,
            )
        support.stop_old_bridge = refuse_existing_bridge
        support.wait_udp_listener = wait_udp_listener_with_freshness
        return support

    baseline.load_old_verified_bridge_support = owned_bridge_support

    def interrupted(signum, _frame):
        report["stop_reason"] = f"signal {signum}"
        raise KeyboardInterrupt

    previous_sigterm = signal.signal(signal.SIGTERM, interrupted)
    previous_argv = sys.argv[:]
    try:
        mode_lock = acquire_mode_lock(mode)
        if args.viewer or args.headless:
            command = [sys.executable, "-B", "-u", str(ROOT / "emg_mujoco_viewer.py"),
                       "--port", str(args.viewer_port)]
            if args.headless:
                command.append("--headless")
            if args.viewer_report:
                command.extend(["--report", str(args.viewer_report.resolve())])
            viewer = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ))
        sys.argv = [baseline.__file__, "--hz", str(args.hz)]
        if args.hardware:
            sys.argv.append("--arm")
        if args.arm_confirmed:
            def confirmed_input(prompt=""):
                if "按 Enter 开始实机接管" not in str(prompt):
                    raise RuntimeError("Unexpected frozen controller prompt after launcher confirmation")
                print(str(prompt) + " [confirmed by run_emg_model_hand.sh]", flush=True)
                return ""
            baseline.input = confirmed_input
        baseline.main()
    except KeyboardInterrupt:
        if report["stop_reason"] == "normal":
            report["stop_reason"] = "interrupted"
    except Exception as exc:
        report["stop_reason"] = f"{type(exc).__name__}: {exc}"
        print("[EMG] " + report["stop_reason"], file=sys.stderr, flush=True)
        exit_code = 1
    finally:
        sys.argv = previous_argv
        signal.signal(signal.SIGTERM, previous_sigterm)
        for device in devices:
            report["receiver"] = device.metadata
            device.cleanup()
        if viewer is not None and viewer.poll() is None:
            viewer.terminate()
            try:
                viewer.wait(timeout=3)
            except subprocess.TimeoutExpired:
                viewer.kill()
                viewer.wait(timeout=3)
        report["viewer_returncode"] = viewer.returncode if viewer is not None else None
        if mode_lock is not None:
            mode_lock.close()
        if viewer is not None and viewer.returncode != 0:
            report["viewer_failed"] = True
            exit_code = 1
        if durations:
            report["retarget_ms"] = {"mean": timing["sum"] / report["retarget_frames"] * 1000,
                                     "max": timing["max"] * 1000,
                                     "p95_recent_4096": float(np.percentile(durations, 95) * 1000)}
            report["q21_std"] = np.sqrt(np.maximum(0, q_m2 / report["retarget_frames"])).tolist()
        report["torch_imported"] = "torch" in sys.modules
        report["verified_original_backup_files"] = verify_original_backups()
        _write_new_json(args.metrics_json or run_dir / "controller_metrics.json", report)
        print("[EMG] " + json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)
    return exit_code
