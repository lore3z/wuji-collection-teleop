#!/usr/bin/env python3
"""Forward only new EMG2Pose predictions to the verified Skeleton UDP input."""

from __future__ import annotations

# Resolve project imports independently of the current working directory.
import sys as _project_sys
from pathlib import Path as _ProjectPath
_project_root = _ProjectPath(__file__).resolve().parents[1]
if str(_project_root) not in _project_sys.path:
    _project_sys.path.insert(0, str(_project_root))


import argparse
import ipaddress
import json
import math
from pathlib import Path
import signal
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

from emg_skeleton_device import validate_skeleton


class LocalAPI:
    def __init__(self, url):
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != "http" or parsed.username or parsed.password or parsed.path not in ("", "/"):
            raise ValueError("Model API must be a bare unauthenticated localhost HTTP URL")
        address = ipaddress.ip_address(socket.gethostbyname(parsed.hostname or ""))
        if not address.is_loopback:
            raise ValueError("Model API must resolve to localhost")
        self.url = url.rstrip("/")
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def get(self, path):
        with self.opener.open(self.url + path, timeout=2.0) as response:
            return json.load(response)


class Publisher:
    def __init__(self, host, port, packet_log):
        address = ipaddress.ip_address(socket.gethostbyname(host))
        if address.version != 4 or not address.is_loopback or port != 17621:
            raise ValueError("Unified EMG Skeleton output is fixed at 127.0.0.1:17621")
        self.destination = (str(address), port)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.session_id = "emg-model-" + uuid.uuid4().hex
        self.seq = 0
        self.log = packet_log.open("x", encoding="utf-8") if packet_log else None

    def send(self, row, *, valid=True, reason="model_prediction"):
        skeleton = validate_skeleton(row["skeleton"])
        packet = {
            "version": 1,
            "source": "emg2pose",
            "session_id": self.session_id,
            "seq": self.seq,
            "timestamp": time.time(),
            "source_timestamp": float(row["time_seconds"]),
            "skeleton": skeleton.tolist(),
            "quality": {"valid": bool(valid), "confidence": None, "reason": reason},
            "coordinate_frame": "wuji_wrist",
            "units": "m",
            "joint_order": "mediapipe",
            "backend": row["backend"],
            "backend_session_id": row["session_id"],
            "backend_frame_id": int(row["frame"]),
            "model_id": row["model_id"],
            "model_version": row.get("model_version"),
        }
        payload = json.dumps(packet, separators=(",", ":"), allow_nan=False).encode("utf-8")
        if len(payload) > 16384:
            raise ValueError("Skeleton UDP packet exceeds receiver limit")
        self.socket.sendto(payload, self.destination)
        if self.log:
            self.log.write(payload.decode("utf-8") + "\n")
            self.log.flush()
        self.seq += 1

    def close(self):
        self.socket.close()
        if self.log:
            self.log.close()


def real(path):
    return str(Path(path).resolve())


def verify_backend(status, expected_checkpoint, model_id, profile):
    if profile == "wavletech_native":
        if status.get("error") or status.get("phase") not in ("starting", "warming", "running"):
            raise RuntimeError("Native Wavletech server is not starting or running healthy inference")
        if status.get("native_hz") != 2000 or status.get("output_hz") != 25:
            raise RuntimeError("Native Wavletech server rate contract changed")
        if status.get("robot_output_enabled") is not False:
            raise RuntimeError("Native model server unexpectedly enables robot output")
        if status.get("selected") != [model_id]:
            raise RuntimeError("Native server selected a different model")
        models = status.get("models")
        selected = [row for row in models if isinstance(row, dict) and row.get("id") == model_id] \
            if isinstance(models, list) else []
        if len(selected) != 1 or real(selected[0].get("checkpoint", "")) != real(expected_checkpoint):
            raise RuntimeError("Native server loaded a different model/checkpoint")
        return None, None, None
    output = Path(status["output_directory"]).resolve()
    metadata_path = output / "session.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("session_id") != status.get("session_id"):
        raise RuntimeError("API status and session.json identify different sessions")
    if not status.get("emg_only"):
        raise RuntimeError("Model server is not in Myo-only inference mode")
    if status.get("stage") != "inference" or status.get("error"):
        raise RuntimeError("Model server is not in a healthy inference stage")
    if profile == "compare_manifest":
        # status.models is authoritative after an onsite calibration promotes
        # new weights without replacing the original session.json.
        models = status.get("models")
        if not isinstance(models, list):
            raise RuntimeError("Compare server status has no model manifest")
        selected = [row for row in models if isinstance(row, dict) and row.get("id") == model_id]
        if len(selected) != 1 or real(selected[0].get("checkpoint", "")) != real(expected_checkpoint):
            raise RuntimeError("Compare server loaded a different model/checkpoint")
        if status.get("model_version") != "personal":
            raise RuntimeError("Compare server is not identifying the loaded checkpoint as personal")
    else:
        if real(metadata.get("checkpoint", "")) != real(expected_checkpoint):
            raise RuntimeError("Model server loaded a different checkpoint")
        if status.get("model") != "personal":
            raise RuntimeError("Model server is not identifying the loaded checkpoint as personal")
    return output, metadata_path.stat().st_ino, status.get("model_revision")


def verify_model_unchanged(status, expected_checkpoint, model_id, profile, model_revision):
    if profile in ("compare_manifest", "wavletech_native"):
        selected = [row for row in status.get("models", [])
                    if isinstance(row, dict) and row.get("id") == model_id]
        if len(selected) != 1 or real(selected[0].get("checkpoint", "")) != real(expected_checkpoint):
            raise RuntimeError("Selected checkpoint changed; forwarding stopped")
        if profile == "compare_manifest" and status.get("model_revision") != model_revision:
            raise RuntimeError("Compare model revision changed; forwarding stopped")
        if profile == "wavletech_native" and status.get("selected") != [model_id]:
            raise RuntimeError("Native selected model changed; forwarding stopped")


def status_row(status, profile, model_id):
    frame = int(status["frames"])
    if profile in ("compare_manifest", "wavletech_native"):
        predictions = status.get("predictions")
        time_seconds = status.get("time_seconds")
        if profile == "wavletech_native":
            time_seconds = float(status.get("samples", 0)) / 2000.0
            if status.get("phase") != "running" or not status.get("prediction_fresh"):
                return None
        if not isinstance(predictions, dict) or model_id not in predictions or time_seconds is None:
            return None
        skeleton = predictions[model_id]
        version = status.get("model_version", "native-2000hz")
    else:
        latest = status.get("latest")
        if not isinstance(latest, dict):
            return None
        skeleton = latest.get("skeleton")
        version = status.get("model")
    return {"frame": frame, "time_seconds": float(time_seconds if profile in ("compare_manifest", "wavletech_native") else latest["time_seconds"]),
            "skeleton": skeleton, "model_id": model_id, "model_version": version,
            "session_id": status["session_id"], "backend": "status.latest"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=17621)
    parser.add_argument("--profile", choices=("compare_manifest", "single_checkpoint", "wavletech_native"), required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--expected-checkpoint", required=True, type=Path)
    parser.add_argument("--poll-hz", type=float, default=50.0)
    parser.add_argument("--packet-log", type=Path)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args(argv)
    if not math.isfinite(args.poll_hz) or not 1 <= args.poll_hz <= 100:
        parser.error("--poll-hz must be finite and in [1,100]")
    for path in (args.packet_log, args.summary):
        if path and path.exists():
            parser.error("Output already exists: " + str(path))

    api = LocalAPI(args.url)
    initial = api.get("/api/status")
    output_dir, metadata_inode, model_revision = verify_backend(
        initial, args.expected_checkpoint, args.model_id, args.profile)
    backend_session = initial["session_id"]
    cursor = int(initial["frames"])
    previous_source_time = None
    source = "status.latest"
    try:
        probe = api.get("/api/predictions?after=" + str(cursor))
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
    else:
        if not isinstance(probe, dict) or not isinstance(probe.get("predictions"), list):
            raise RuntimeError("Incremental prediction endpoint returned an invalid contract")
        source = "http.incremental"
    publisher = Publisher(args.host, args.port, args.packet_log)
    running = True

    def stop(*_):
        nonlocal running
        running = False

    old_handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    report = {"url": args.url, "source": source, "destination": "127.0.0.1:17621",
              "backend_session_id": backend_session, "model_id": args.model_id,
              "checkpoint": real(args.expected_checkpoint), "packets": 0,
              "invalid_predictions": 0, "missing_backend_frames": 0,
              "first_backend_frame": None, "last_backend_frame": None, "last_error": None}
    print(json.dumps({"event": "ready", **report}, ensure_ascii=False), flush=True)
    last_row = None

    def process_row(row):
        nonlocal cursor, previous_source_time, last_row
        frame = int(row["frame"])
        if frame <= cursor:
            raise RuntimeError("Backend frames are not strictly increasing")
        if frame > cursor + 1:
            missing = frame - cursor - 1
            report["missing_backend_frames"] += missing
            print(json.dumps({"event": "missing_backend_frames", "count": missing,
                              "after": cursor, "before": frame}), flush=True)
        source_time = float(row["time_seconds"])
        if previous_source_time is not None and source_time <= previous_source_time:
            raise RuntimeError("Model prediction time_seconds did not increase")
        cursor = frame
        previous_source_time = source_time
        try:
            publisher.send(row)
        except (ValueError, TypeError, KeyError) as exc:
            report["invalid_predictions"] += 1
            report["last_error"] = str(exc)
            print(json.dumps({"event": "prediction_rejected", "backend_frame_id": frame,
                              "error": str(exc)}, ensure_ascii=False), flush=True)
            return
        last_row = row
        report["packets"] += 1
        report["first_backend_frame"] = frame if report["first_backend_frame"] is None else report["first_backend_frame"]
        report["last_backend_frame"] = frame

    period = 1.0 / args.poll_hz
    next_poll = time.monotonic()
    try:
        while running:
            status = api.get("/api/status")
            if status.get("session_id") != backend_session:
                raise RuntimeError("Model server session changed; forwarding stopped")
            if args.profile != "wavletech_native":
                if Path(status.get("output_directory", "")).resolve() != output_dir:
                    raise RuntimeError("Model server output directory changed; forwarding stopped")
                session_file = output_dir / "session.json"
                if not session_file.is_file() or session_file.stat().st_ino != metadata_inode:
                    raise RuntimeError("Model session metadata was replaced; forwarding stopped")
            if status.get("error") or status.get("stage") == "error" or status.get("phase") == "error":
                raise RuntimeError("Model server error: " + str(status.get("error")))
            if args.profile == "wavletech_native" and status.get("phase") not in ("starting", "warming", "running"):
                raise RuntimeError("Native Wavletech inference stopped; forwarding stopped")
            verify_model_unchanged(status, args.expected_checkpoint, args.model_id,
                                   args.profile, model_revision)
            frame = int(status["frames"])
            if frame < cursor:
                raise RuntimeError("Backend frame counter moved backwards")
            if source == "http.incremental":
                result = api.get("/api/predictions?after=" + str(cursor))
                if result.get("session_id") != backend_session or result.get("history_gap"):
                    raise RuntimeError("Incremental prediction session/history changed")
                for raw in result.get("predictions", []):
                    raw_frame = raw.get("frame_id", raw.get("frame"))
                    skeleton = raw.get("skeleton")
                    if skeleton is None and isinstance(raw.get("predictions"), dict):
                        skeleton = raw["predictions"].get(args.model_id)
                    row = {"frame": raw_frame, "time_seconds": raw["time_seconds"],
                           "skeleton": skeleton, "model_id": args.model_id,
                           "model_version": raw.get("model_version"),
                           "session_id": raw.get("session_id", backend_session),
                           "backend": "http.incremental"}
                    if row["session_id"] != backend_session:
                        raise RuntimeError("Incremental prediction belongs to another session")
                    process_row(row)
            elif frame > cursor:
                row = status_row(status, args.profile, args.model_id)
                if row is None or int(row["frame"]) != frame:
                    raise RuntimeError("New backend frame has no identifiable latest prediction")
                process_row(row)
            next_poll += period
            delay = next_poll - time.monotonic()
            if delay > 0 and running:
                time.sleep(delay)
            elif delay <= -period:
                next_poll = time.monotonic()
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise RuntimeError("Model server/status failure; forwarding stopped") from exc
    finally:
        if last_row is not None:
            try:
                publisher.send(last_row, valid=False, reason="adapter_stopped")
            except OSError:
                pass
        publisher.close()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        if args.summary:
            args.summary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"event": "stopped", **report}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
