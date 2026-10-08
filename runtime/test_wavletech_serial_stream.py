import json
import os
import pty
import select
import subprocess
import sys
import time
from pathlib import Path

from wavletech_serial_stream import (
    ACCEL_SCALE_M_S2,
    GYRO_SCALE_RAD_S,
    HEADER,
    PacketParser,
    decode_packet,
    packet_events,
)


def _signed(value: int, width: int) -> bytes:
    return value.to_bytes(width, "big", signed=True)


def _packet(kind: int, seq: int, payload: bytes) -> bytes:
    return HEADER + bytes((kind, seq)) + payload.ljust(24, b"\x00")


def test_decode_emg_signed_24_bit_big_endian() -> None:
    values = (-8_388_608, -2, -1, 0, 1, 2, 1_234_567, 8_388_607)
    packet = _packet(0xAA, 255, b"".join(_signed(value, 3) for value in values))
    decoded = decode_packet(packet)
    assert decoded.kind == "emg"
    assert decoded.hardware_seq == 255
    assert decoded.values == values


def test_parser_resynchronizes_and_handles_split_packets() -> None:
    first = _packet(0xAA, 9, b"".join(_signed(i, 3) for i in range(8)))
    second = _packet(0xBB, 10, b"".join(_signed(i, 2) for i in range(6)))
    parser = PacketParser()
    assert parser.feed(b"noise" + first[:11]) == []
    decoded = parser.feed(first[11:] + b"\xd2\xd2\xd2\x01junk" + second)
    assert [packet.kind for packet in decoded] == ["emg", "imu"]
    assert parser.discarded_bytes >= 10


def test_packet_events_scale_imu_and_report_sequence_gap() -> None:
    emg = decode_packet(_packet(0xAA, 254, b"".join(_signed(1, 3) for _ in range(8))))
    imu_values = (100, -100, 200, 1000, -1000, 2000)
    imu = decode_packet(_packet(0xBB, 0, b"".join(_signed(v, 2) for v in imu_values)))
    state = {"hardware_seq": None, "emg_seq": 0, "imu_seq": 0, "last_timestamp_ns": 0}
    events = list(packet_events((emg, imu), state=state))
    assert [event["kind"] for event in events] == ["sample", "gap", "imu"]
    assert events[1]["missing_packets"] == 1
    assert events[2]["gyro_rad_s"][0] == 100 * GYRO_SCALE_RAD_S
    assert events[2]["accel_m_s2"][1] == -1000 * ACCEL_SCALE_M_S2


def test_jsonl_bridge_reads_a_pseudo_serial_port() -> None:
    master_fd, slave_fd = pty.openpty()
    tty = os.ttyname(slave_fd)
    process = subprocess.Popen(
        [
            sys.executable, "-u", str(Path(__file__).with_name("wavletech_serial_stream.py")),
            "--tty", tty, "--silence-timeout-s", "1.0",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        time.sleep(0.1)
        emg_payload = b"".join(_signed(value, 3) for value in range(8))
        imu_payload = b"".join(_signed(value, 2) for value in range(6))
        os.write(master_fd, _packet(0xAA, 0, emg_payload) + _packet(0xBB, 1, imu_payload))
        kinds = []
        deadline = time.monotonic() + 2.0
        assert process.stdout is not None
        while time.monotonic() < deadline and not {"ready", "sample", "imu"}.issubset(kinds):
            readable, _, _ = select.select([process.stdout], [], [], 0.1)
            if readable:
                kinds.append(json.loads(process.stdout.readline())["kind"])
        assert {"ready", "sample", "imu"}.issubset(kinds)
    finally:
        process.terminate()
        process.wait(timeout=2.0)
        os.close(master_fd)
        os.close(slave_fd)
