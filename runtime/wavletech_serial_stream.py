#!/usr/bin/env python3
"""Receive Wavletech armband EMG/IMU packets from its USB serial receiver.

Wire format (29 bytes): ``D2 D2 D2 TYPE SEQ PAYLOAD[24]``.  ``TYPE=AA``
contains eight signed 24-bit EMG channels and ``TYPE=BB`` contains three
signed 16-bit gyro axes followed by three signed 16-bit accelerometer axes.
The manual states that the right-most byte is least-significant, therefore
both payload formats are decoded as signed big-endian integers.

Stdout is JSONL for :mod:`wuji_serial_emg_source`; diagnostics go to stderr.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from dataclasses import dataclass
from typing import Any, Iterable


HEADER = b"\xd2\xd2\xd2"
PACKET_SIZE = 29
TYPE_EMG = 0xAA
TYPE_IMU = 0xBB
GYRO_SCALE_RAD_S = 0.0012
ACCEL_SCALE_M_S2 = 0.0005978


def emit(kind: str, **values: Any) -> None:
    print(json.dumps({"kind": kind, **values}, separators=(",", ":")), flush=True)


def decode_signed_be(data: bytes) -> int:
    return int.from_bytes(data, byteorder="big", signed=True)


@dataclass(frozen=True)
class DecodedPacket:
    kind: str
    hardware_seq: int
    values: tuple[int, ...]


def decode_packet(packet: bytes) -> DecodedPacket:
    if len(packet) != PACKET_SIZE:
        raise ValueError(f"packet must be {PACKET_SIZE} bytes, got {len(packet)}")
    if packet[:3] != HEADER:
        raise ValueError("invalid D2 D2 D2 packet header")
    packet_type = packet[3]
    hardware_seq = packet[4]
    payload = packet[5:]
    if packet_type == TYPE_EMG:
        values = tuple(decode_signed_be(payload[i:i + 3]) for i in range(0, 24, 3))
        return DecodedPacket("emg", hardware_seq, values)
    if packet_type == TYPE_IMU:
        values = tuple(decode_signed_be(payload[i:i + 2]) for i in range(0, 12, 2))
        return DecodedPacket("imu", hardware_seq, values)
    raise ValueError(f"unsupported packet type 0x{packet_type:02X}")


class PacketParser:
    """Incrementally frame a noisy serial byte stream without fabricating data."""

    def __init__(self) -> None:
        self.buffer = bytearray()
        self.discarded_bytes = 0

    def feed(self, chunk: bytes) -> list[DecodedPacket]:
        self.buffer.extend(chunk)
        result: list[DecodedPacket] = []
        while True:
            offset = self.buffer.find(HEADER)
            if offset < 0:
                keep = min(len(self.buffer), len(HEADER) - 1)
                self.discarded_bytes += len(self.buffer) - keep
                if keep:
                    del self.buffer[:-keep]
                else:
                    self.buffer.clear()
                break
            if offset:
                self.discarded_bytes += offset
                del self.buffer[:offset]
            if len(self.buffer) < PACKET_SIZE:
                break
            if self.buffer[3] not in (TYPE_EMG, TYPE_IMU):
                self.discarded_bytes += 1
                del self.buffer[0]
                continue
            candidate = bytes(self.buffer[:PACKET_SIZE])
            try:
                result.append(decode_packet(candidate))
            except ValueError:
                self.discarded_bytes += 1
                del self.buffer[0]
                continue
            del self.buffer[:PACKET_SIZE]
        return result


def packet_events(
    packets: Iterable[DecodedPacket],
    *,
    state: dict[str, int | None],
) -> Iterable[dict[str, Any]]:
    """Convert decoded packets to bridge events and audit sequence gaps."""
    for packet in packets:
        previous = state.get("hardware_seq")
        missing = 0
        if previous is not None:
            delta = (packet.hardware_seq - int(previous)) & 0xFF
            if delta == 0:
                yield {
                    "kind": "gap", "reason": "duplicate hardware packet sequence",
                    "previous_seq": previous, "hardware_seq": packet.hardware_seq,
                    "missing_packets": 0,
                }
            elif delta != 1:
                missing = delta - 1
                yield {
                    "kind": "gap", "reason": "hardware packet sequence gap",
                    "previous_seq": previous, "hardware_seq": packet.hardware_seq,
                    "missing_packets": missing,
                }
        state["hardware_seq"] = packet.hardware_seq
        arrival_ns = time.time_ns()
        last_timestamp_ns = int(state.get("last_timestamp_ns") or 0)
        timestamp_ns = max(arrival_ns, last_timestamp_ns + 1)
        state["last_timestamp_ns"] = timestamp_ns
        if packet.kind == "emg":
            # Add conservatively detected missing transport packets to the EMG
            # sequence. The collector can then reject loss without assuming a
            # fixed AA:BB packet ratio.
            state["emg_seq"] = int(state.get("emg_seq") or 0) + missing + 1
            yield {
                "kind": "sample", "timestamp_ns": timestamp_ns,
                "arrival_timestamp_ns": arrival_ns, "seq": state["emg_seq"],
                "hardware_seq": packet.hardware_seq, "movement": 0,
                "emg": list(packet.values),
            }
        else:
            state["imu_seq"] = int(state.get("imu_seq") or 0) + missing + 1
            gyro = [value * GYRO_SCALE_RAD_S for value in packet.values[:3]]
            accel = [value * ACCEL_SCALE_M_S2 for value in packet.values[3:6]]
            yield {
                "kind": "imu", "timestamp_ns": timestamp_ns,
                "arrival_timestamp_ns": arrival_ns, "seq": state["imu_seq"],
                "hardware_seq": packet.hardware_seq,
                "gyro_rad_s": gyro, "accel_m_s2": accel,
            }


def main() -> int:
    parser = argparse.ArgumentParser(description="Wavletech serial EMG/IMU JSONL bridge")
    parser.add_argument("--tty", required=True, help="USB serial receiver, preferably /dev/serial/by-id/...")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--silence-timeout-s", type=float, default=0.35)
    parser.add_argument("--reconnect-initial-s", type=float, default=1.0)
    parser.add_argument("--reconnect-max-s", type=float, default=8.0)
    args = parser.parse_args()
    if args.baud <= 0 or min(args.silence_timeout_s, args.reconnect_initial_s, args.reconnect_max_s) <= 0:
        parser.error("baud and timeout values must be positive")

    try:
        import serial
    except ImportError as exc:
        emit("error", message=f"pyserial is required: {exc}")
        return 2

    stopping = False

    def stop(_signum: int, _frame: Any) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    reconnect_delay = args.reconnect_initial_s
    connection_id = 0

    while not stopping:
        port = None
        try:
            emit("state", state="connecting", message=f"opening {args.tty} at {args.baud} baud")
            try:
                port = serial.Serial(
                    args.tty, args.baud, bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE,
                    timeout=0.10, exclusive=True,
                )
            except TypeError:
                port = serial.Serial(
                    args.tty, args.baud, bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE,
                    timeout=0.10,
                )
            parser_state = PacketParser()
            event_state: dict[str, int | None] = {
                "hardware_seq": None, "emg_seq": 0, "imu_seq": 0,
                "last_timestamp_ns": 0,
            }
            last_valid_mono = time.monotonic()
            ready = False
            while not stopping:
                waiting = int(getattr(port, "in_waiting", 0))
                chunk = port.read(min(max(waiting, 1), 4096))
                packets = parser_state.feed(chunk) if chunk else []
                for event in packet_events(packets, state=event_state):
                    if event["kind"] in {"sample", "imu"}:
                        last_valid_mono = time.monotonic()
                    if event["kind"] == "sample" and not ready:
                        connection_id += 1
                        emit(
                            "ready", connection_id=connection_id, tty=args.tty,
                            baud=args.baud, driver="wavletech-serial-v1",
                            discarded_bytes=parser_state.discarded_bytes,
                        )
                        ready = True
                        reconnect_delay = args.reconnect_initial_s
                    emit(**event)
                if time.monotonic() - last_valid_mono > args.silence_timeout_s:
                    raise TimeoutError(
                        f"no valid AA/BB packet for {args.silence_timeout_s:.3f}s"
                    )
        except Exception as exc:
            emit("state", state="recovering", message=f"{type(exc).__name__}: {exc}")
        finally:
            if port is not None:
                try:
                    port.close()
                except Exception:
                    pass
        if stopping:
            break
        deadline = time.monotonic() + reconnect_delay
        while not stopping and time.monotonic() < deadline:
            time.sleep(0.05)
        reconnect_delay = min(args.reconnect_max_s, reconnect_delay * 2.0)

    emit("state", state="stopped", message="requested shutdown")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
