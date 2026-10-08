#!/usr/bin/env python3
"""Regression tests for the project-local pyomyo BGAPI fixes."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from myo_pyomyo_patch import disconnect_detail, install, is_disconnect_event


class _Serial:
    def __init__(self) -> None:
        self.writes: list[bytes] = []

    def write(self, value: bytes) -> None:
        self.writes.append(value)


class _ReadSerial(_Serial):
    def __init__(self) -> None:
        super().__init__()
        self.reads = 0

    def read(self, _size: int) -> bytes:
        self.reads += 1
        return b"\x80"

    def flushInput(self) -> None:  # pragma: no cover - must never be called
        raise AssertionError("valid BGAPI input must not be flushed")


class PyomyoPatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        install()

    def test_command_event_is_not_dispatched_twice(self) -> None:
        from pyomyo.pyomyo import BT

        bt = BT.__new__(BT)
        bt.ser = _Serial()
        bt.handlers = []
        seen: list[object] = []
        bt.add_handler(seen.append)
        event = SimpleNamespace(typ=0x80, cls=4, cmd=5, payload=b"")
        response = SimpleNamespace(typ=0, cls=4, cmd=5, payload=b"")
        packets = iter((event, response))

        def receive():
            packet = next(packets)
            if packet.typ == 0x80:
                bt.handle_event(packet)
            return packet

        bt.recv_packet = receive
        self.assertIs(bt.send_command(4, 5, b"x"), response)
        self.assertEqual(seen, [event])

    def test_receive_does_not_flush_buffer(self) -> None:
        from pyomyo.pyomyo import BT

        bt = BT.__new__(BT)
        bt.ser = _ReadSerial()
        bt.handlers = []
        event = SimpleNamespace(typ=0x80, cls=4, cmd=5, payload=b"")
        bt.proc_byte = lambda _value: event
        seen: list[object] = []
        bt.add_handler(seen.append)
        self.assertIs(bt.recv_packet(), event)
        self.assertEqual(seen, [event])

    def test_command_timeout_is_explicit(self) -> None:
        from pyomyo.pyomyo import BT

        bt = BT.__new__(BT)
        bt.ser = _Serial()
        bt._wuji_bgapi_timeout_s = 0.0
        bt.recv_packet = lambda: None
        with self.assertRaisesRegex(TimeoutError, "BGAPI response timeout"):
            bt.send_command(4, 5, b"x")

    def test_disconnect_event(self) -> None:
        packet = SimpleNamespace(typ=0x80, cls=3, cmd=4, payload=b"\x02\x08\x02")
        self.assertTrue(is_disconnect_event(packet))
        self.assertEqual(disconnect_detail(packet), "BGAPI disconnected connection=2, reason=0x0208")


if __name__ == "__main__":
    unittest.main()


def test_raw_mode_disables_unused_imu_without_changing_emg_mode():
    from pyomyo.pyomyo import Myo
    install()
    myo = Myo.__new__(Myo)
    writes = []
    myo.write_attr = lambda handle, value: writes.append((handle, value))
    myo.start_raw_unfiltered()
    assert writes == [(h, b'\x01\x00') for h in (0x2c, 0x2f, 0x32, 0x35)] + [
        (0x1d, b'\x00\x00'), (0x19, b'\x01\x03\x03\x00\x00')]


def test_command_ignores_unrelated_pending_response():
    from pyomyo.pyomyo import BT
    install()
    bt = BT.__new__(BT)
    bt.ser = _Serial()
    stale = SimpleNamespace(typ=0, cls=3, cmd=1, payload=b'')
    expected = SimpleNamespace(typ=0, cls=4, cmd=5, payload=b'')
    packets = iter([stale, expected])
    bt.recv_packet = lambda: next(packets)
    assert bt.send_command(4, 5) is expected
