#!/usr/bin/env python3
"""Forward new 8765 online-model skeleton predictions to EMGSkeletonDevice.

The online server remains the sole PyTorch model owner.  This adapter polls its
incremental prediction endpoint at no more than the model's native 25 Hz and
publishes each new 21x3 skeleton on the existing localhost UDP wire protocol.
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import math
from pathlib import Path
import signal
import socket
import time
import urllib.parse
import urllib.request
import uuid

from emg_skeleton_device import validate_skeleton


DEFAULT_URL = "http://127.0.0.1:8765"
DEFAULT_PORT = 17621
MAX_MODEL_HZ = 25.0


class OnlineAPI:
    def __init__(self, url: str):
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != "http" or parsed.username or parsed.password:
            raise ValueError("Online API must be an unauthenticated localhost HTTP URL")
        try:
            address = ipaddress.ip_address(socket.gethostbyname(parsed.hostname or ""))
        except OSError as exc:
            raise ValueError("Cannot resolve online API host") from exc
        if not address.is_loopback:
            raise ValueError("Online API must resolve to localhost")
        self.url = url.rstrip("/")
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def get(self, path: str):
        with self.opener.open(self.url + path, timeout=2.0) as response:
            return json.load(response)

    def status(self):
        return self.get("/api/status")

    def predictions(self, after: int):
        return self.get("/api/predictions?after=" + str(int(after)))


class SkeletonUDP:
    def __init__(self, host: str, port: int, packet_log: Path | None):
        address = ipaddress.ip_address(socket.gethostbyname(host))
        if address.version != 4 or not address.is_loopback:
            raise ValueError("Skeleton UDP destination must be IPv4 localhost")
        if not 1024 <= port <= 65535 or port in (15120, 15121, 17622):
            raise ValueError("Use an isolated Skeleton input port, normally 17621")
        self.destination = (str(address), port)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.session_id = "online-" + uuid.uuid4().hex
        self.seq = 0
        self.log = packet_log.open("x", encoding="utf-8") if packet_log else None

    def publish(self, row, *, valid=True, reason="online_prediction"):
        skeleton = validate_skeleton(row["skeleton"])
        quality = {
            "valid": bool(valid),
            "confidence": None,
            "reason": reason,
        }
        packet = {
            "version": 1,
            "source": "emg2pose",
            "session_id": self.session_id,
            "seq": self.seq,
            "timestamp": time.time(),
            "source_timestamp": float(row["time_seconds"]),
            "skeleton": skeleton.tolist(),
            "quality": quality,
            "coordinate_frame": "wuji_wrist",
            "units": "m",
            "joint_order": "mediapipe",
            "backend": "online_api",
            "backend_session_id": row["session_id"],
            "backend_frame_id": int(row["frame_id"]),
            "model_version": int(row["model_version"]),
            "transition_fraction": float(row["transition_fraction"]),
            "inference_ms": float(row["inference_ms"]),
        }
        data = json.dumps(packet, separators=(",", ":"), allow_nan=False).encode("utf-8")
        if len(data) > 60000:
            raise ValueError("Skeleton UDP datagram is too large")
        self.socket.sendto(data, self.destination)
        if self.log:
            self.log.write(data.decode("utf-8") + "\n")
            self.log.flush()
        self.seq += 1

    def close(self):
        self.socket.close()
        if self.log:
            self.log.close()


def session_checkpoint(status, *, require_personal: bool, require_finger_model: bool = False):
    directory = Path(status["output_directory"]).resolve()
    metadata = json.loads((directory / "session.json").read_text(encoding="utf-8"))
    if metadata.get("session_id") != status.get("session_id"):
        raise RuntimeError("API status and session.json identify different sessions")
    backend = metadata.get("backend", "online_personal")
    if backend == "fingers_wide":
        model_directory = Path(metadata["model_directory"]).resolve()
        files = [Path(path).resolve() for path in metadata.get("weight_files", [])]
        if not model_directory.is_dir() or len(files) != 5 or any(
                path.parent != model_directory or not path.is_file() for path in files):
            raise RuntimeError("Loaded five-finger model package is incomplete")
        if require_personal:
            raise RuntimeError("8765 is using the five-finger backend, not a personal_v*.pt checkpoint")
        return model_directory, metadata.get("model_sha256"), backend
    checkpoint = Path(metadata["checkpoint"]).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError("Loaded checkpoint no longer exists: " + str(checkpoint))
    if require_personal and not checkpoint.name.startswith("personal_v"):
        raise RuntimeError("8765 was not started from a personal_v*.pt checkpoint")
    if require_finger_model:
        raise RuntimeError("8765 is not using the five-finger backend")
    return checkpoint, metadata.get("base_sha256"), backend


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--poll-hz", type=float, default=MAX_MODEL_HZ)
    requirement = ap.add_mutually_exclusive_group()
    requirement.add_argument("--require-personal", action="store_true")
    requirement.add_argument("--require-finger-model", action="store_true")
    ap.add_argument("--duration", type=float)
    ap.add_argument("--max-packets", type=int)
    ap.add_argument("--packet-log", type=Path)
    ap.add_argument("--summary", type=Path)
    return ap


def main(argv=None):
    ap = parser()
    args = ap.parse_args(argv)
    if not math.isfinite(args.poll_hz) or not 0 < args.poll_hz <= MAX_MODEL_HZ:
        ap.error("--poll-hz must be finite and in (0, 25]")
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        ap.error("--duration must be finite and positive")
    if args.max_packets is not None and args.max_packets <= 0:
        ap.error("--max-packets must be positive")
    for path in (args.packet_log, args.summary):
        if path is not None and path.exists():
            ap.error("Output path already exists: " + str(path))

    api = OnlineAPI(args.url)
    status = api.status()
    checkpoint, checkpoint_sha256, backend = session_checkpoint(
        status, require_personal=args.require_personal,
        require_finger_model=args.require_finger_model,
    )
    backend_session = status["session_id"]
    # Forward only predictions created after adapter startup.  Old cached poses
    # must never become a fresh robot command.
    cursor = int(status["frames"]) - 1
    output = SkeletonUDP(args.host, args.port, args.packet_log)
    running = True

    def stop(_signum, _frame):
        nonlocal running
        running = False

    previous_handlers = {
        sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    report = {
        "url": args.url,
        "destination": f"{args.host}:{args.port}",
        "backend_session_id": backend_session,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "backend": backend,
        "poll_hz": args.poll_hz,
        "packets": 0,
        "first_backend_frame": None,
        "last_backend_frame": None,
        "model_versions": [],
        "invalid_predictions": 0,
        "last_invalid_error": None,
    }
    print(json.dumps({"event": "ready", **report}, ensure_ascii=False), flush=True)
    start = time.monotonic()
    last_row = None
    period = 1.0 / args.poll_hz
    next_poll = start
    try:
        while running:
            if args.duration is not None and time.monotonic() - start >= args.duration:
                break
            result = api.predictions(cursor)
            if result.get("session_id") != backend_session:
                raise RuntimeError("Online model session changed; restart this adapter")
            if result.get("history_gap"):
                raise RuntimeError("Online prediction history gap; robot forwarding stopped")
            for row in result.get("predictions", []):
                frame = int(row["frame_id"])
                if frame <= cursor:
                    raise RuntimeError("Online prediction frames are not strictly increasing")
                if row.get("session_id") != backend_session:
                    raise RuntimeError("Prediction belongs to a different online session")
                try:
                    output.publish(row)
                except ValueError as exc:
                    # Fail closed for an isolated invalid model prediction: do
                    # not forward or reuse it, but keep polling later frames.
                    cursor = frame
                    report["invalid_predictions"] += 1
                    report["last_invalid_error"] = str(exc)
                    print(json.dumps({"event": "prediction_rejected",
                                      "backend_frame_id": frame,
                                      "error": str(exc)}, ensure_ascii=False),
                          flush=True)
                    continue
                cursor = frame
                last_row = row
                report["packets"] += 1
                report["first_backend_frame"] = (
                    frame if report["first_backend_frame"] is None
                    else report["first_backend_frame"]
                )
                report["last_backend_frame"] = frame
                version = int(row["model_version"])
                if version not in report["model_versions"]:
                    report["model_versions"].append(version)
                if args.max_packets is not None and report["packets"] >= args.max_packets:
                    running = False
                    break
            next_poll += period
            delay = next_poll - time.monotonic()
            if delay > 0 and running:
                time.sleep(delay)
            elif delay <= -period:
                next_poll = time.monotonic()
    finally:
        if last_row is not None:
            try:
                output.publish(last_row, valid=False, reason="adapter_stopped")
            except OSError:
                pass
        output.close()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        report["elapsed_seconds"] = time.monotonic() - start
        if args.summary:
            args.summary.parent.mkdir(parents=True, exist_ok=True)
            with args.summary.open("x", encoding="utf-8") as handle:
                json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
                handle.write("\n")
        print(json.dumps({"event": "stopped", **report}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
