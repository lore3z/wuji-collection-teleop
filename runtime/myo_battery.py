#!/usr/bin/env python3

import argparse
import contextlib
import sys
import time

from struct import pack

from myo_transport import (
    PortBusyError,
    acquire_bled112_lock,
    normalize_mac,
    resolve_bled112_tty,
    tty_is_in_use,
)


def reversed_mac(text):
    octets = normalize_mac(text).split(":")
    return list(reversed([int(x, 16) for x in octets]))


def read_attribute_value(myo, handle, timeout_s=3.0, attempts=3):
    """Read one exact GATT handle, ignoring unrelated notification events."""
    if attempts < 1:
        raise ValueError("battery read attempts must be positive")

    last_timeout = None
    for attempt in range(1, attempts + 1):
        try:
            return _read_attribute_value_once(myo, handle, timeout_s)
        except TimeoutError as exc:
            last_timeout = exc
            if attempt < attempts:
                print(
                    f"[RETRY] Battery GATT read timed out ({attempt}/{attempts}); retrying...",
                    file=sys.stderr,
                    flush=True,
                )
    raise last_timeout


def _read_attribute_value_once(myo, handle, timeout_s):
    result = [None]

    def capture(packet):
        payload = bytes(getattr(packet, "payload", b""))
        if (
            getattr(packet, "cls", None) == 4
            and getattr(packet, "cmd", None) == 5
            and len(payload) >= 5
            and int.from_bytes(payload[1:3], "little") == handle
        ):
            result[0] = payload[5:]

    myo.bt.add_handler(capture)
    try:
        # BGAPI is byte-packed little-endian. Native ``BH`` inserts one pad
        # byte on this host and turns handle 0x0011 into 0x1100 on the wire.
        myo.bt.send_command(4, 4, pack("<BH", myo.conn, handle))
        deadline = time.monotonic() + timeout_s
        while result[0] is None and time.monotonic() < deadline:
            myo.bt.recv_packet()
        if result[0] is None:
            raise TimeoutError(f"attribute 0x{handle:04x} read timed out")
        return result[0]
    finally:
        myo.bt.remove_handler(capture)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tty", default="/dev/ttyACM0")
    parser.add_argument("--mac", required=True)
    args = parser.parse_args()

    from pyomyo import Myo, emg_mode
    from myo_pyomyo_patch import install
    install()

    addr = reversed_mac(args.mac)
    tty = resolve_bled112_tty(args.tty)

    myo = None

    try:
        with acquire_bled112_lock():
            if tty_is_in_use(tty):
                raise PortBusyError(
                    f"BLED112 serial port {tty} is already open by another process"
                )
            print(f"Myo: {args.mac}")
            print(f"BLED112: {tty}")
            print("Connecting...")

            with contextlib.redirect_stdout(sys.stderr):
                # Battery queries must not enable the 200 Hz EMG stream: an
                # unrelated EMG attribute event can otherwise satisfy
                # pyomyo's generic cls/cmd wait and be misread as >100%.
                myo = Myo(tty, mode=emg_mode.NO_DATA)

                try:
                    myo.bt.ser.exclusive = True
                except Exception:
                    pass
                myo.bt.ser.timeout = 0.25
                myo.bt._wuji_bgapi_timeout_s = 2.0

                myo.connect(addr)
                myo.sleep_mode(1)

            try:
                # Myo Battery Level characteristic handle
                payload = read_attribute_value(myo, 0x11)
                if len(payload) != 1:
                    raise RuntimeError(f"unexpected battery value: {payload!r}")
                value = payload[0]
                battery = value if isinstance(value, int) else ord(value)
                if not 0 <= battery <= 100:
                    raise RuntimeError(f"invalid battery percentage: {battery}")
                print()
                print("==============================")
                print(f"Battery: {battery}%")
                print("==============================")

                if battery <= 10:
                    print("[WARN] Myo 电量很低，建议立即充电")
                elif battery <= 20:
                    print("[WARN] Myo 电量偏低")
                else:
                    print("[PASS] Myo 电量正常")
                return 0
            finally:
                try:
                    myo.disconnect()
                except Exception:
                    pass
                try:
                    myo.bt.ser.close()
                except Exception:
                    pass
                myo = None

    except PortBusyError as e:
        print(f"[FAILED] {e}")
        return 2
    except Exception as e:
        print(f"[FAILED] {type(e).__name__}: {e}")
        return 1

    finally:
        if myo is not None:
            try:
                myo.disconnect()
            except Exception:
                pass

            try:
                myo.bt.ser.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
