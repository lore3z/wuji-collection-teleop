#!/usr/bin/env python3
"""Verify native Myo rate, continuity, reconnects, and pyomyo duplicate events."""

from __future__ import annotations

import argparse
import json
import queue
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path


def arrival_quality(arrivals_ns: list[int], observed_seconds: float) -> tuple[float, float]:
    """Use measured delivery, including a silent tail, rather than the sample PLL."""
    if len(arrivals_ns) < 2:
        return 0.0, float("inf")
    arrival_span_s = (arrivals_ns[-1] - arrivals_ns[0]) / 1e9
    span_s = max(observed_seconds, arrival_span_s, 1e-9)
    rate_hz = (len(arrivals_ns) - 1) / span_s
    max_gap_ms = max(b - a for a, b in zip(arrivals_ns, arrivals_ns[1:])) / 1e6
    return rate_hz, max_gap_ms


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tty", default="/dev/ttyACM0")
    parser.add_argument("--mac", required=True)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--min-hz", type=float, default=180.0)
    parser.add_argument("--max-gap-ms", type=float, default=1000.0)
    args = parser.parse_args()
    if min(args.seconds, args.min_hz, args.max_gap_ms) <= 0:
        parser.error("--seconds, --min-hz and --max-gap-ms must be positive")

    bridge = Path(__file__).with_name("myo_200hz_stream.py")
    process = subprocess.Popen(
        [
            sys.executable, "-u", str(bridge), "--tty", args.tty,
            "--mac", args.mac,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None and process.stderr is not None
    stderr_tail: list[str] = []

    def read_stderr() -> None:
        for line in process.stderr:
            line = line.rstrip()
            if line:
                stderr_tail.append(line)
                del stderr_tail[:-30]
                print(line, file=sys.stderr, flush=True)

    threading.Thread(target=read_stderr, daemon=True).start()
    lines: queue.Queue[str | None] = queue.Queue()

    def read_stdout() -> None:
        # TextIO can retain multiple lines after the fd becomes non-readable.
        # A dedicated reader drains those lines without waiting for select().
        try:
            for line in process.stdout:
                lines.put(line)
        finally:
            lines.put(None)

    threading.Thread(target=read_stdout, daemon=True).start()
    samples: list[tuple[int, tuple[int, ...]]] = []
    arrivals_ns: list[int] = []
    recovering_count = 0
    observed_seconds = 0.0
    last_sample_mono: float | None = None
    ready_count = 0
    disconnected_count = 0
    error_messages: list[str] = []
    first_sample_mono: float | None = None
    startup_deadline = time.monotonic() + 45.0
    try:
        while process.poll() is None:
            now = time.monotonic()
            if first_sample_mono is None and now >= startup_deadline:
                break
            if first_sample_mono is not None and now - first_sample_mono >= args.seconds:
                break
            try:
                line = lines.get(timeout=0.2)
            except queue.Empty:
                continue
            if line is None:
                break
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                print(f"[VERIFY] non-JSON stdout: {line.rstrip()}", file=sys.stderr)
                continue
            kind = event.get("kind")
            if kind == "ready":
                ready_count += 1
            elif kind == "error":
                message = str(event.get("message", "unknown Myo bridge error"))
                error_messages.append(message)
                # A lock/configuration error is deterministic; waiting for
                # the 45-second startup watchdog only hides the cause.
                if "already in use" in message or "exclusively lock" in message:
                    break
            elif kind == "state" and event.get("state") == "disconnected":
                disconnected_count += 1
            elif kind == "state" and event.get("state") == "recovering":
                recovering_count += 1
            elif kind == "sample":
                last_sample_mono = time.monotonic()
                arrivals_ns.append(int(event.get("arrival_timestamp_ns", event["timestamp_ns"])))
                if first_sample_mono is None:
                    first_sample_mono = time.monotonic()
                samples.append(
                    (int(event["timestamp_ns"]), tuple(int(v) for v in event["emg"]))
                )
    finally:
        stopped_mono = time.monotonic()
        observed_seconds = stopped_mono - first_sample_mono if first_sample_mono is not None else 0.0
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=4.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)

    if len(samples) < 2:
        detail = error_messages[-1] if error_messages else f"tail={stderr_tail[-4:]}"
        print(f"[FAILED] Myo verification received {len(samples)} samples; {detail}")
        return 1
    gaps_ms = [(samples[i][0] - samples[i - 1][0]) / 1e6 for i in range(1, len(samples))]
    # Each real notification carries two samples, so adjacent callbacks can be
    # very close. The pyomyo double-dispatch bug instead emitted A,B,A,B: the
    # same vector at lag 2 within one millisecond.
    duplicate_events = sum(
        1
        for i in range(2, len(samples))
        if samples[i][1] == samples[i - 2][1]
        and samples[i][0] - samples[i - 2][0] < 1_000_000
    )
    native_max_gap_ms = max(gaps_ms)
    arrival_rate_hz, max_gap_ms = arrival_quality(arrivals_ns, observed_seconds)
    if last_sample_mono is not None:
        max_gap_ms = max(max_gap_ms, (stopped_mono - last_sample_mono) * 1000.0)
    ok = (
        arrival_rate_hz >= args.min_hz
        and observed_seconds >= args.seconds
        and recovering_count == 0
        and max_gap_ms <= args.max_gap_ms
        and ready_count == 1
        and disconnected_count == 0
        and duplicate_events == 0
    )
    label = "PASS" if ok else "FAILED"
    print(
        f"[{label}] Myo {observed_seconds:.1f}s: samples={len(samples)}, arrival_rate={arrival_rate_hz:.2f} Hz, "
        f"arrival_max_gap={max_gap_ms:.2f} ms, native_max_gap={native_max_gap_ms:.2f} ms, connections={ready_count}, "
        f"disconnects={disconnected_count}, rearms={recovering_count}, duplicate_BGAPI_events={duplicate_events}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
