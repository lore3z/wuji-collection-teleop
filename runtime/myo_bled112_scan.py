#!/usr/bin/env python3
"""Verify that the configured Myo is advertising through a BLED112 dongle."""

from __future__ import annotations

import argparse
import sys
import time

from myo_transport import (
    PortBusyError,
    acquire_bled112_lock,
    normalize_mac,
    resolve_bled112_tty,
    tty_is_in_use,
)


MYO_ADV_SUFFIX = bytes.fromhex(
    "064248124A7F2C4847B9DE04A9010006D5"
)


def _normal_mac(raw: list[int]) -> str:
    return ":".join(f"{value:02X}" for value in reversed(raw))


def _is_myo_advertisement(payload: bytes) -> bool:
    return bytes(payload).endswith(MYO_ADV_SUFFIX)


def _auto_select_myo_mac(myo_seen: dict[str, int]) -> str:
    """Return the only advertising Myo, refusing an ambiguous selection."""
    candidates = sorted(myo_seen)
    if not candidates:
        raise ValueError("no advertising Myo was found")
    if len(candidates) > 1:
        raise ValueError(
            "multiple advertising Myos were found: " + ", ".join(candidates)
        )
    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tty", default="/dev/ttyACM0")
    parser.add_argument(
        "--mac",
        default="auto",
        help="target Myo MAC, or 'auto' to select the only advertising Myo",
    )
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument(
        "--select-mac",
        action="store_true",
        help="print only the selected MAC to stdout; send diagnostics to stderr",
    )
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")

    diagnostic_file = sys.stderr if args.select_mac else sys.stdout

    def diagnostic(message: str) -> None:
        print(message, file=diagnostic_file, flush=True)

    try:
        from pyomyo.pyomyo import BT, multiord
        from myo_pyomyo_patch import install
        install()
    except Exception as exc:
        diagnostic(f"[FAILED] Cannot import pyomyo/BLED112 support: {exc}")
        return 2

    auto_select = args.mac.strip().lower() == "auto"
    target = ""
    if not auto_select:
        try:
            target = normalize_mac(args.mac)
        except ValueError as exc:
            parser.error(str(exc))
    tty = resolve_bled112_tty(args.tty)
    seen: dict[str, int] = {}
    myo_seen: dict[str, int] = {}
    bt = None
    try:
        with acquire_bled112_lock():
            try:
                if tty_is_in_use(tty):
                    raise PortBusyError(
                        f"BLED112 serial port {tty} is already open by another process"
                    )
                bt = BT(tty)
                # pyomyo otherwise blocks forever in recv_packet(), which is useful for
                # streaming but inappropriate for a startup health check.
                bt.ser.timeout = 0.25
                bt._wuji_bgapi_timeout_s = 0.75
                bt.end_scan()
                for handle in (0, 1, 2):
                    bt.disconnect(handle)
                bt.discover()
                deadline = time.monotonic() + args.seconds
                while time.monotonic() < deadline:
                    packet = bt.recv_packet()
                    if packet is None or packet.typ != 0x80 or packet.cls != 6 or packet.cmd != 0:
                        continue
                    payload = list(multiord(packet.payload))
                    if len(payload) < 8:
                        continue
                    address = _normal_mac(payload[2:8])
                    seen[address] = seen.get(address, 0) + 1
                    if _is_myo_advertisement(bytes(packet.payload)):
                        myo_seen[address] = myo_seen.get(address, 0) + 1
            finally:
                if bt is not None:
                    try:
                        bt.end_scan()
                    except Exception:
                        pass
                    try:
                        bt.ser.close()
                    except Exception:
                        pass
                    bt = None
    except PortBusyError as exc:
        diagnostic(f"[FAILED] {exc}")
        return 2
    except Exception as exc:
        diagnostic(f"[FAILED] BLED112 scan failed on {tty}: {type(exc).__name__}: {exc}")
        return 2

    visible = ", ".join(sorted(seen)) if seen else "none"
    myo_candidates = ", ".join(sorted(myo_seen)) if myo_seen else "none"
    if auto_select:
        try:
            target = _auto_select_myo_mac(myo_seen)
        except ValueError as exc:
            diagnostic(f"[FAILED] Automatic Myo selection failed on {tty}: {exc}.")
            diagnostic(f"         Visible BLE addresses: {visible}")
            diagnostic(f"         Myo advertisement candidates: {myo_candidates}")
            diagnostic("         Wake/charge and wear exactly one Myo near the BLED112, then retry.")
            return 1
        if args.select_mac:
            print(target, flush=True)
        else:
            diagnostic(
                f"[PASS] Automatically selected Myo through BLED112 {tty}: "
                f"{target} ({myo_seen[target]} Myo advertisements)"
            )
        return 0

    if target in myo_seen:
        if args.select_mac:
            print(target, flush=True)
        else:
            diagnostic(f"[PASS] Myo is advertising through BLED112 {tty}: {target} ({myo_seen[target]} Myo advertisements)")
        return 0

    diagnostic(f"[FAILED] Configured Myo {target} was not visible through BLED112 {tty} in {args.seconds:.0f} s.")
    diagnostic(f"         Visible BLE addresses: {visible}")
    diagnostic(f"         Myo advertisement candidates: {myo_candidates}")
    if target in seen:
        diagnostic("         The configured address is visible but does not match the Myo advertisement signature.")
    diagnostic("         Wake/charge and wear the Myo, keep it near the BLED112, then retry.")
    diagnostic("         Set WUJI_MYO_MAC=auto to select the currently advertising Myo at startup.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
