#!/usr/bin/env python3
"""Resilient Myo RAW EMG (200 Hz) JSONL bridge for the FTP-1 collector.

The bridge owns BLED112 exclusively, keeps the armband awake, and reconnects
forever after a BLE outage. It never fabricates samples during an outage.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import signal
import sys
import threading
import time
from typing import Any

from myo_transport import (
    PortBusyError,
    acquire_bled112_lock,
    normalize_mac,
    resolve_bled112_tty,
    tty_is_in_use,
)


def reversed_mac(text: str) -> list[int]:
    octets = normalize_mac(text).split(":")
    return list(reversed([int(x, 16) for x in octets]))


def emit(kind: str, **kwargs: Any) -> None:
    print(json.dumps({"kind": kind, **kwargs}, separators=(",", ":")), flush=True)


class EmgSampleClock:
    """Assign acquisition times to RAW samples instead of USB arrival times.

    Myo RAW notifications contain two consecutive samples.  pyomyo invokes
    the callback twice while handling that one packet, so timestamping inside
    the callback makes the pair appear nearly simultaneous.  The armband has
    no timestamp in the BLE payload; sequence plus the documented 200 Hz
    cadence is therefore the least misleading source timeline.
    """

    PERIOD_NS = 5_000_000
    PHASE_CORRECTION_DIVISOR = 100
    # Myo is nominally 200 Hz, but the observed long-term callback cadence can
    # be about 197.7 Hz.  The clock must follow that slow oscillator error or
    # it drifts hundreds of milliseconds away from wall time while the bridge
    # waits for an episode.  Keep every individual step bounded to 4.9--5.1 ms
    # so a delayed BGAPI batch can never create an acquisition-time jump.
    MAX_PHASE_CORRECTION_NS = 100_000

    def __init__(self) -> None:
        self.next_timestamp_ns: int | None = None

    def reset(self) -> None:
        self.next_timestamp_ns = None

    def timestamp(self, arrival_ns: int) -> int:
        phase_error_ns = (
            arrival_ns - self.next_timestamp_ns
            if self.next_timestamp_ns is not None
            else 0
        )
        if self.next_timestamp_ns is None:
            # The first callback is the older member of a two-sample packet.
            timestamp_ns = arrival_ns - self.PERIOD_NS
        else:
            # A very small PLL correction follows oscillator drift without
            # copying ordinary BGAPI/USB packet jitter into the sample axis.
            correction_ns = max(
                -self.MAX_PHASE_CORRECTION_NS,
                min(
                    self.MAX_PHASE_CORRECTION_NS,
                    phase_error_ns // self.PHASE_CORRECTION_DIVISOR,
                ),
            )
            timestamp_ns = self.next_timestamp_ns + correction_ns
        self.next_timestamp_ns = timestamp_ns + self.PERIOD_NS
        return timestamp_ns


def close_myo(myo: Any | None, *, graceful: bool = False) -> None:
    """Release BLED112 without waiting on pyomyo's blocking disconnect.

    ``pyomyo`` opens its serial port with no read timeout.  In particular, a
    failed connection can leave ``Myo.connect()`` blocked in ``recv_packet()``.
    Calling ``Myo.disconnect()`` at that point starts another blocking receive
    before the port is closed, so the reconnect loop never gets another turn.

    A connected stream gets one bounded attempt to release its BLE connection
    before the port is closed.  A connection attempt that is still blocked is
    cancelled by closing the serial object immediately.
    """
    if myo is None:
        return
    serial_port = getattr(getattr(myo, "bt", None), "ser", None)
    if graceful and serial_port is not None and getattr(myo, "conn", None) is not None:
        try:
            # pyomyo normally uses an infinite timeout. Bound this call so a
            # dead dongle cannot turn cleanup into another deadlock. Any final
            # EMG event remains valid JSON on stdout instead of being rerouted.
            serial_port.timeout = 0.25
            myo.disconnect()
        except Exception:
            pass
    try:
        if serial_port is not None:
            serial_port.close()
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tty", default="/dev/ttyACM0")
    parser.add_argument("--mac", required=True)
    parser.add_argument("--connect-timeout-s", type=float, default=12.0)
    parser.add_argument("--silence-timeout-s", type=float, default=0.35)
    parser.add_argument("--rearm-timeout-s", type=float, default=2.0)
    parser.add_argument("--reconnect-initial-s", type=float, default=1.0)
    parser.add_argument("--reconnect-max-s", type=float, default=8.0)
    args = parser.parse_args()
    if min(
        args.connect_timeout_s,
        args.silence_timeout_s,
        args.rearm_timeout_s,
        args.reconnect_initial_s,
        args.reconnect_max_s,
    ) <= 0:
        parser.error("Myo timeout/reconnect values must be positive")

    # Keep the dongle exclusively owned for the whole stream lifetime. A
    # second scanner/battery query can otherwise interleave BGAPI packets and
    # look like a radio dropout.
    transport_lock = acquire_bled112_lock()
    try:
        transport_lock.__enter__()
    except PortBusyError as exc:
        emit("error", message=str(exc))
        return 2
    except OSError as exc:
        emit("error", message=f"cannot create BLED112 lock: {type(exc).__name__}: {exc}")
        return 2

    try:
        from pyomyo import Myo, emg_mode
        from myo_pyomyo_patch import disconnect_detail, install, is_disconnect_event
        install()
    except Exception as exc:
        emit("error", message=f"pyomyo import failed: {exc}")
        transport_lock.__exit__(None, None, None)
        return 1
    try:
        addr = reversed_mac(args.mac)
    except ValueError as exc:
        emit("error", message=f"bad MAC: {exc}")
        transport_lock.__exit__(None, None, None)
        return 1

    seq = 0
    connection_id = 0
    rate_count = 0
    rate_t0 = time.monotonic()
    last_sample_mono = 0.0
    sample_clock = EmgSampleClock()

    def on_emg(emg: Any, movement: Any) -> None:
        nonlocal seq, rate_count, rate_t0, last_sample_mono
        values = [int(value) for value in emg]
        if len(values) != 8:
            print(f"[MYO] bad EMG length={len(values)}", file=sys.stderr, flush=True)
            return
        now_mono = time.monotonic()
        seq += 1
        if rate_count == 0:
            rate_t0 = now_mono
        rate_count += 1
        last_sample_mono = now_mono
        arrival_timestamp_ns = time.time_ns()
        emit(
            "sample",
            timestamp_ns=sample_clock.timestamp(arrival_timestamp_ns),
            arrival_timestamp_ns=arrival_timestamp_ns,
            monotonic_ns=time.perf_counter_ns(),
            seq=seq,
            connection_id=connection_id,
            movement=int(movement) if movement is not None else 0,
            emg=values,
        )
        if now_mono - rate_t0 >= 5.0:
            print(
                f"[MYO] RAW receive rate={rate_count / (now_mono - rate_t0):.2f} Hz samples={seq}",
                file=sys.stderr,
                flush=True,
            )
            rate_count = 0
            rate_t0 = now_mono

    def connect_once() -> Any:
        nonlocal connection_id, last_sample_mono, rate_count, rate_t0
        tty = resolve_bled112_tty(args.tty)
        if tty != args.tty:
            print(
                f"[MYO] configured tty {args.tty} unavailable; using stable path {tty}",
                file=sys.stderr,
                flush=True,
            )
        if tty_is_in_use(tty):
            raise PortBusyError(
                f"BLED112 serial port {tty} is already open by another process"
            )
        myo = Myo(tty, mode=emg_mode.RAW)
        worker: threading.Thread | None = None
        try:
            try:
                myo.bt.ser.exclusive = True
            except (AttributeError, OSError):
                print("[MYO] serial exclusive mode unavailable", file=sys.stderr, flush=True)
            # Bound every BGAPI receive performed by Myo.connect().  The
            # outer worker remains a total-duration watchdog, while this
            # timeout prevents one individual command from hanging it.
            myo.bt.ser.timeout = min(1.0, args.connect_timeout_s / 3.0)
            myo.bt._wuji_bgapi_timeout_s = args.connect_timeout_s
            print(f"[MYO] connecting MAC={args.mac} BGAPI={addr}", file=sys.stderr, flush=True)
            errors: list[BaseException] = []

            def connect_worker() -> None:
                try:
                    # pyomyo prints device discovery details. Connection runs
                    # before our EMG callback is registered, so redirecting in
                    # this one worker cannot reroute live JSON samples.
                    with contextlib.redirect_stdout(sys.stderr):
                        myo.connect(addr)
                except BaseException as exc:
                    errors.append(exc)

            worker = threading.Thread(target=connect_worker, daemon=True)
            worker.start()
            worker.join(args.connect_timeout_s)
            if worker.is_alive():
                stage = getattr(myo.bt, "_wuji_last_operation", "unknown BGAPI stage")
                raise TimeoutError(
                    f"Myo initialization did not finish within {args.connect_timeout_s:.0f}s; {stage}"
                )
            if errors:
                raise errors[0]
            # Myo.connect currently requests this too; make reconnect behavior
            # explicit so the band is never put into automatic sleep.
            myo.sleep_mode(1)
            myo.add_emg_handler(on_emg)
            myo.bt.ser.timeout = min(0.20, args.silence_timeout_s / 3.0)
            myo.bt._wuji_bgapi_timeout_s = args.rearm_timeout_s
        except BaseException:
            close_myo(myo)
            if worker is not None and worker.is_alive():
                worker.join(timeout=1.0)
            raise
        connection_id += 1
        sample_clock.reset()
        last_sample_mono = time.monotonic()
        rate_count = 0
        rate_t0 = last_sample_mono
        emit(
            "ready", tty=tty, mac=args.mac, channels=8, target_hz=200,
            connection_id=connection_id, driver="wuji-pyomyo-bgapi-v3-sample-clock",
        )
        return myo

    myo: Any | None = None
    retry_s = args.reconnect_initial_s
    running = True
    stop_event = threading.Event()

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal running
        running = False
        stop_event.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    def run_packet(active_myo: Any) -> Any:
        packet = active_myo.run()
        if is_disconnect_event(packet):
            raise ConnectionError(disconnect_detail(packet))
        return packet

    def rearm_raw_stream(active_myo: Any) -> bool:
        """Re-enable RAW notifications before tearing down a live BLE link."""
        baseline_seq = seq
        # A genuine notification re-arm starts a new acquisition epoch. This
        # is the correct place to re-anchor; ordinary delayed BGAPI batches
        # with continuous received samples must not create timeline gaps.
        sample_clock.reset()
        emit(
            "state", state="recovering", connection_id=connection_id,
            message=f"no RAW EMG for {args.silence_timeout_s:.1f}s; re-arming",
        )
        print(
            f"[MYO] no RAW EMG for {args.silence_timeout_s:.1f}s; "
            "re-arming notifications on the current BLE connection",
            file=sys.stderr,
            flush=True,
        )
        errors: list[BaseException] = []

        def rearm_worker() -> None:
            try:
                # This worker is the sole serial reader while the main thread
                # waits on join(), so commands cannot race with run_packet().
                active_myo.start_raw_unfiltered()
                active_myo.sleep_mode(1)
            except BaseException as exc:
                errors.append(exc)

        # pyomyo waits synchronously for a BGAPI response after every write.
        # A half-open BLE link can therefore trap start_raw_unfiltered()
        # forever even though the serial read timeout is finite. Run the
        # command sequence behind a watchdog; closing the serial port releases
        # the blocked worker and lets the outer loop make a fresh connection.
        worker = threading.Thread(target=rearm_worker, daemon=True)
        worker.start()
        command_timeout_s = args.rearm_timeout_s
        worker.join(command_timeout_s)
        if worker.is_alive():
            print(
                f"[MYO] RAW re-arm BGAPI command timed out after {command_timeout_s:.1f}s; forcing reconnect",
                file=sys.stderr,
                flush=True,
            )
            close_myo(active_myo)
            worker.join(timeout=1.0)
            return False
        if errors:
            print(
                f"[MYO] RAW re-arm failed: {type(errors[0]).__name__}: {errors[0]}",
                file=sys.stderr,
                flush=True,
            )
            return False
        if seq > baseline_seq:
            print("[MYO] RAW stream recovered during re-arm", file=sys.stderr, flush=True)
            emit("state", state="connected", connection_id=connection_id, message="RAW stream re-armed")
            return True
        deadline = time.monotonic() + args.rearm_timeout_s
        while running and time.monotonic() < deadline:
            run_packet(active_myo)
            if seq > baseline_seq:
                print("[MYO] RAW stream recovered without reconnect", file=sys.stderr, flush=True)
                emit("state", state="connected", connection_id=connection_id, message="RAW stream re-armed")
                return True
        return False

    try:
        while running:
            try:
                emit("state", state="connecting", retry_delay_s=retry_s, connection_id=connection_id + 1)
                myo = connect_once()
                retry_s = args.reconnect_initial_s
                while running:
                    run_packet(myo)
                    if time.monotonic() - last_sample_mono > args.silence_timeout_s:
                        if not rearm_raw_stream(myo):
                            raise ConnectionError(
                                f"no RAW EMG after {args.silence_timeout_s:.1f}s silence "
                                f"and {args.rearm_timeout_s:.1f}s re-arm attempt"
                            )
            except KeyboardInterrupt:
                break
            except PortBusyError as exc:
                # Another process owns the tty. Retrying would only spam the
                # log and can never repair a process-level ownership conflict.
                emit("error", message=str(exc))
                return 2
            except BaseException as exc:
                if not running:
                    break
                emit("state", state="disconnected", connection_id=connection_id, message=f"{type(exc).__name__}: {exc}")
                print(f"[MYO] disconnected: {type(exc).__name__}: {exc}; reconnecting in {retry_s:.1f}s", file=sys.stderr, flush=True)
                # The BLE link is already failed here.  Asking pyomyo to send
                # another synchronous disconnect command can wait forever for
                # a response from a connection which no longer exists and
                # strand the bridge in state=disconnected.  Close the serial
                # transport immediately so the reconnect loop always gets its
                # next turn.
                close_myo(myo, graceful=False)
                myo = None
                stop_event.wait(retry_s)
                retry_s = min(args.reconnect_max_s, retry_s * 2.0)
    finally:
        close_myo(myo, graceful=True)
        transport_lock.__exit__(None, None, None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
