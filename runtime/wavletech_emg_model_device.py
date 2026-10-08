#!/usr/bin/env python3
"""Feed causal 200 Hz Wavletech EMG to the existing skeleton model server.

Native 2000 Hz samples are reduced by non-overlapping, causal 10-sample means.
The model checkpoint must have been trained with the same preprocessing.
"""

from __future__ import annotations

# Resolve project imports independently of the current working directory.
import sys as _project_sys
from pathlib import Path as _ProjectPath
_project_root = _ProjectPath(__file__).resolve().parents[1]
for _project_path in (_project_root, _project_root / "src"):
    if str(_project_path) not in _project_sys.path:
        _project_sys.path.insert(0, str(_project_path))


import argparse
import fcntl
import json
import os
import signal
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np


class SampleClock:
    """Turn bursty USB arrivals into a bounded nominal 2000 Hz source clock."""

    PERIOD_NS = 500_000
    MAX_CORRECTION_NS = 10_000
    CORRECTION_DIVISOR = 100

    def __init__(self) -> None:
        self.next_ns: int | None = None

    def timestamp(self, arrival_ns: int) -> int:
        if self.next_ns is None:
            # Unlike a Myo BLE notification, one Wavletech AA packet carries
            # one current sample; the first packet is not an older pair member.
            value = int(arrival_ns)
        else:
            error = int(arrival_ns) - self.next_ns
            correction = max(
                -self.MAX_CORRECTION_NS,
                min(self.MAX_CORRECTION_NS, error // self.CORRECTION_DIVISOR),
            )
            value = self.next_ns + correction
        self.next_ns = value + self.PERIOD_NS
        return value


class BlockMeanDecimator:
    """Causal, stateful 2000-to-200 Hz reduction used by training and live input."""

    factor = 10
    preprocessing = "wavletech_block_mean_10_v1"

    def __init__(self) -> None:
        self.values: list[np.ndarray] = []
        self.timestamps_ns: list[int] = []
        self.arrivals_ns: list[int] = []

    def feed(
        self, frames: list[tuple[int, int, int, np.ndarray, int]], clock: SampleClock
    ) -> list[tuple[int, np.ndarray, int]]:
        output: list[tuple[int, np.ndarray, int]] = []
        for frame in frames:
            self.values.append(np.asarray(frame[3], dtype=np.float64))
            self.timestamps_ns.append(clock.timestamp(int(frame[4])))
            self.arrivals_ns.append(int(frame[4]))
            if len(self.values) == self.factor:
                output.append((
                    self.timestamps_ns[-1],
                    np.mean(self.values, axis=0, dtype=np.float64).astype(np.float32),
                    self.arrivals_ns[-1],
                ))
                self.values.clear()
                self.timestamps_ns.clear()
                self.arrivals_ns.clear()
        return output


def post_json(opener, url: str, payload: dict) -> None:
    request = urllib.request.Request(
        url, json.dumps(payload, allow_nan=False).encode(), {"Content-Type": "application/json"}
    )
    with opener.open(request, timeout=10) as response:
        response.read()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collector", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8772")
    parser.add_argument("--tty", required=True)
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--status-file", type=Path)
    parser.add_argument("--min-native-hz", type=float, default=1950.0)
    parser.add_argument("--silence-timeout-s", type=float, default=0.35)
    args = parser.parse_args()
    if min(args.baud, args.min_native_hz, args.silence_timeout_s) <= 0:
        parser.error("baud/rate/timeout must be positive")

    collector = args.collector.expanduser().resolve()
    sys.path.insert(0, str(collector / "src"))
    from wuji_serial_emg_source import SerialEmgSource

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(args.url + "/api/status", timeout=10) as response:
        status = json.load(response)
    if status.get("samples"):
        raise RuntimeError("restart model server before starting a new Wavletech clock")
    session = status["session_id"]

    runtime_dir = Path(os.environ.get("WUJI_RUNTIME_DIR", collector / ".runtime"))
    runtime_dir.mkdir(parents=True, exist_ok=True)
    lock_handle = (runtime_dir / "wavletech_emg.lock").open("a")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("another Wavletech acquisition owns the live-input lock") from exc

    stopping = False

    def stop(*_args) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    source = SerialEmgSource(
        sys.executable, args.tty, baud=args.baud,
        silence_timeout_s=args.silence_timeout_s,
    )
    clock = SampleClock()
    decimator = BlockMeanDecimator()
    origin_ns: int | None = None
    last_seq: int | None = None
    connection_id = 0
    interruption_id = 0
    output_count = 0
    started_mono = time.monotonic()
    last_status_mono = 0.0
    try:
        print(
            f"连接 Wavletech {args.tty} @ {args.baud}; "
            "2000 Hz -> 200 Hz block-mean; 不使用 Wuji/相机/GT。",
            flush=True,
        )
        source.start(wait_for_stability=False)
        while not stopping and not source.stable(
            duration_s=3.0, min_hz=args.min_native_hz,
            max_age_s=args.silence_timeout_s,
        ):
            if time.monotonic() - started_mono > 30.0:
                raise RuntimeError("Wavletech did not become stable: " + source.status())
            time.sleep(0.02)
        connection_id = source.current_connection_id()
        interruption_id = source.interruption_snapshot()[0]
        source.clear()
        print("LIVE：Wavletech 原生流稳定，开始向模型发送 200 Hz 数据。", flush=True)

        while not stopping:
            frames = source.drain()
            if source.current_connection_id() != connection_id:
                raise RuntimeError("Wavletech reconnected; stop live control")
            new_interruption, reason = source.interruption_snapshot()
            if new_interruption != interruption_id:
                raise RuntimeError("Wavletech interrupted: " + reason)
            if frames:
                for frame in frames:
                    seq = int(frame[1])
                    if last_seq is not None and seq != last_seq + 1:
                        raise RuntimeError(f"Wavletech sequence discontinuity: {last_seq}->{seq}")
                    last_seq = seq
                rows = decimator.feed(frames, clock)
                if rows:
                    if origin_ns is None:
                        origin_ns = rows[0][0] - 5_000_000
                    timestamps = np.asarray([row[0] for row in rows], dtype=np.int64)
                    packet = {
                        "session_id": session,
                        "emg": [row[1].tolist() for row in rows],
                        "emg_time_seconds": ((timestamps - origin_ns).astype(np.float64) / 1e9).tolist(),
                        "emg_native_timestamp_ns": timestamps.tolist(),
                        "emg_arrival_timestamp_ns": [row[2] for row in rows],
                        "origin_timestamp_ns": origin_ns,
                        "emg_source": "wavletech_serial_v1",
                        "emg_native_sample_rate_hz": 2000,
                        "emg_model_sample_rate_hz": 200,
                        "emg_preprocessing": decimator.preprocessing,
                        "skeleton": [],
                        "pose_time_seconds": [],
                    }
                    post_json(opener, args.url + "/api/input", packet)
                    output_count += len(rows)
            now = time.monotonic()
            if now - last_status_mono >= 5.0:
                elapsed = max(now - started_mono, 1e-9)
                live_status = {
                    "wall_time": time.time(),
                    "emg": source.status(),
                    "emg_diagnostic": source.diagnostic_tail(),
                    "emg_ready": source.healthy(max_age_s=args.silence_timeout_s),
                    "input_source": "wavletech_serial_v1",
                    "native_hz": 2000,
                    "model_hz": 200,
                    "preprocessing": decimator.preprocessing,
                    "model_samples": output_count,
                    "model_rate_hz": output_count / elapsed,
                }
                print("STREAM", live_status, flush=True)
                if args.status_file:
                    args.status_file.parent.mkdir(parents=True, exist_ok=True)
                    temporary = args.status_file.with_suffix(args.status_file.suffix + ".tmp")
                    temporary.write_text(json.dumps(live_status), encoding="utf-8")
                    temporary.replace(args.status_file)
                last_status_mono = now
            time.sleep(0.005)
    finally:
        source.close()
        lock_handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
