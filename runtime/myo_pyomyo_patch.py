"""Project-local correctness fixes for pyomyo 0.0.5's BGAPI transport.

pyomyo's ``recv_packet`` already dispatches event packets, but its
``send_command`` dispatches the same packet a second time while waiting for a
response. For RAW EMG this produces duplicated samples. Its command/event
loops also dereference ``None`` after a finite serial timeout. Keep the small
vendor-facing patch here so a recreated virtualenv receives the same behavior.
"""

from __future__ import annotations

import time
from typing import Any


def install() -> None:
    """Patch pyomyo classes once, preserving its public Myo API."""
    from pyomyo.pyomyo import BT, Myo, multichr, pack

    if getattr(BT, "_wuji_bgapi_fixed", False):
        return

    def recv_packet(self: Any) -> Any:
        # pyomyo samples inWaiting() once and flushes the whole serial input if
        # that stale value exceeds 5096 bytes. Under RAW EMG load this can
        # discard valid BGAPI packets. Consume exactly one packet without an
        # unconditional buffer flush.
        while True:
            value = self.ser.read(1)
            if not value:
                return None
            packet = self.proc_byte(value[0])
            if packet is None:
                continue
            if packet.typ == 0x80:
                self.handle_event(packet)
            return packet

    def wait_event(self: Any, cls: int, cmd: int) -> Any:
        self._wuji_last_operation = f"wait event cls={cls} cmd={cmd}"
        result: list[Any | None] = [None]

        def capture(packet: Any) -> None:
            if packet.cls == cls and packet.cmd == cmd:
                result[0] = packet

        self.add_handler(capture)
        deadline = time.monotonic() + float(getattr(self, "_wuji_bgapi_timeout_s", 10.0))
        try:
            while result[0] is None:
                packet = self.recv_packet()
                if packet is None and time.monotonic() >= deadline:
                    raise TimeoutError(f"BGAPI event timeout cls={cls} cmd={cmd}")
        finally:
            self.remove_handler(capture)
        return result[0]

    def send_command(
        self: Any,
        cls: int,
        cmd: int,
        payload: bytes = b"",
        wait_resp: bool = True,
    ) -> Any:
        self._wuji_last_operation = f"command cls={cls} cmd={cmd} payload={payload.hex()}"
        message = pack("4B", 0, len(payload), cls, cmd) + payload
        self.ser.write(message)
        if not wait_resp:
            return None
        deadline = time.monotonic() + float(getattr(self, "_wuji_bgapi_timeout_s", 10.0))
        while True:
            packet = self.recv_packet()
            if packet is None and time.monotonic() >= deadline:
                raise TimeoutError(f"BGAPI response timeout cls={cls} cmd={cmd}")
            if packet is None:
                continue
            if packet.typ == 0 and packet.cls == cls and packet.cmd == cmd:
                return packet
            # recv_packet() has already called handle_event(packet). Calling it
            # again here is the pyomyo 0.0.5 duplicate-EMG bug.

    def connect(self: Any, addr: list[int]) -> Any:
        # Preserve pyomyo's connection parameters, but route the command
        # through the corrected send_command implementation above.
        return self.send_command(
            6,
            3,
            pack("6sBHHHH", multichr(addr), 0, 6, 6, 64, 0),
        )

    def run(self: Any) -> Any:
        # Returning the packet lets the bridge identify the BLED112
        # connection_disconnected event instead of leaving state=connected.
        return self.bt.recv_packet()

    def start_raw_unfiltered(self: Any) -> None:
        # This bridge consumes only EMG. Vendor RAW mode also enables a 50 Hz
        # IMU stream, wasting BLE/serial bandwidth and callback work.
        for handle in (0x2c, 0x2f, 0x32, 0x35):
            self.write_attr(handle, b"\x01\x00")
        self.write_attr(0x1d, b"\x00\x00")
        self.write_attr(0x19, b"\x01\x03\x03\x00\x00")

    BT.recv_packet = recv_packet
    BT.wait_event = wait_event
    BT.send_command = send_command
    BT.connect = connect
    BT._wuji_bgapi_fixed = True
    Myo.run = run
    Myo.start_raw_unfiltered = start_raw_unfiltered


def is_disconnect_event(packet: Any) -> bool:
    """Return whether packet is a Bluegiga connection_disconnected event."""
    return bool(
        packet is not None
        and getattr(packet, "typ", None) == 0x80
        and getattr(packet, "cls", None) == 3
        and getattr(packet, "cmd", None) == 4
    )


def disconnect_detail(packet: Any) -> str:
    payload = bytes(getattr(packet, "payload", b""))
    connection = payload[0] if payload else -1
    reason = int.from_bytes(payload[1:3], "little") if len(payload) >= 3 else -1
    return f"BGAPI disconnected connection={connection}, reason=0x{reason:04x}"
