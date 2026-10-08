#!/usr/bin/env python3
"""Serial data interface for the WAVELETECH / 唯理 18-channel EMG wristband.

The wristband sends 41-byte packets when its onboard EMG filter is enabled
(18 signed 16-bit EMG counts) and 59-byte packets when disabled (18 signed
24-bit counts). Both modes interleave AA EMG packets and BB IMU/status packets
on a 2,000,000-baud, 8N1 serial link.

This module does not send device commands. In particular, it will not change
the filter state or start/stop acquisition. ``Weili18EmgDevice`` opens the
serial port exclusively and exposes typed :class:`EmgSample` and
:class:`ImuSample` events through ``read_events`` / ``drain_events``.
"""

from __future__ import annotations

from dataclasses import dataclass
import queue
import threading
import time
from typing import TypeAlias

HEADER = b"\xd2\xd2\xd2"
TYPE_EMG = 0xAA
TYPE_IMU = 0xBB
BAUD_RATE = 2_000_000
EMG_CHANNELS = 18
EMG_RATE_HZ = 2_000
IMU_RATE_HZ = 208
GYRO_RAD_S_PER_COUNT = 0.001225
ACCEL_M_S2_PER_COUNT = 0.0005982


@dataclass(frozen=True, slots=True)
class Protocol:
    name: str
    packet_size: int
    emg_bytes_per_channel: int
    counts_per_microvolt: float
    filter_enabled: bool


FILTERED_PROTOCOL = Protocol("filtered_16bit", 41, 2, 7.5, True)
RAW_PROTOCOL = Protocol("raw_24bit", 59, 3, 60.0, False)


def protocol_for(value: str | int | Protocol) -> Protocol:
    if isinstance(value, Protocol):
        return value
    text = str(value).strip().lower()
    if text in {"filtered", "filtered_16bit", "41", "41-byte"}:
        return FILTERED_PROTOCOL
    if text in {"raw", "raw_24bit", "59", "59-byte"}:
        return RAW_PROTOCOL
    raise ValueError("packet format must be 'filtered' (41 bytes) or 'raw' (59 bytes)")


def _signed_be(data: bytes | bytearray | memoryview) -> int:
    return int.from_bytes(data, byteorder="big", signed=True)


def _decode_emg(payload: bytes, bytes_per_channel: int) -> tuple[int, ...]:
    if bytes_per_channel == 2:
        return tuple(_signed_be(payload[i:i + 2]) for i in range(0, 36, 2))
    values = []
    for i in range(0, 54, 3):
        unsigned = int.from_bytes(payload[i:i + 3], byteorder="big", signed=False)
        values.append(unsigned - (1 << 24) if unsigned & (1 << 23) else unsigned)
    return tuple(values)


@dataclass(frozen=True, slots=True)
class RawPacket:
    kind: str
    packet_sequence: int
    emg_counts: tuple[int, ...] | None = None
    imu_raw: tuple[int, int, int, int, int, int, int, int, int] | None = None
    device_time_ms: int | None = None


def decode_packet(packet: bytes, protocol: str | int | Protocol = FILTERED_PROTOCOL) -> RawPacket:
    spec = protocol_for(protocol)
    if len(packet) != spec.packet_size:
        raise ValueError(f"packet must be {spec.packet_size} bytes, got {len(packet)}")
    if packet[:3] != HEADER:
        raise ValueError("invalid D2 D2 D2 packet header")
    packet_type = packet[3]
    sequence = packet[4]
    payload = packet[5:]
    if packet_type == TYPE_EMG:
        return RawPacket("emg", sequence, emg_counts=_decode_emg(payload, spec.emg_bytes_per_channel))
    if packet_type != TYPE_IMU:
        raise ValueError(f"unsupported packet type 0x{packet_type:02X}")
    if any(payload[25:]):
        raise ValueError("BB packet reserved bytes are not zero")
    raw = (
        _signed_be(payload[0:2]),  # temperature, 0.1 C / count
        _signed_be(payload[2:4]), _signed_be(payload[4:6]), _signed_be(payload[6:8]),
        _signed_be(payload[8:10]), _signed_be(payload[10:12]), _signed_be(payload[12:14]),
        int.from_bytes(payload[14:16], "big", signed=False),  # battery voltage, mV
        payload[16],  # battery percentage
    )
    device_time_ms = int.from_bytes(payload[17:25], "big", signed=False)
    return RawPacket("imu", sequence, imu_raw=raw, device_time_ms=device_time_ms)


class PacketParser:
    """Incrementally frame the fixed-size D2 D2 D2 AA/BB binary stream.

    Since the wire format has no length or checksum field, a candidate header
    is accepted only after a second valid packet appears at the exact packet
    boundary with a forward-moving shared SN. This reduces false framing when
    an EMG payload happens to contain the three-byte header pattern.
    """

    def __init__(self, protocol: str | int | Protocol = FILTERED_PROTOCOL):
        self.protocol = protocol_for(protocol)
        self.buffer = bytearray()
        self.discarded_bytes = 0
        self.invalid_candidates = 0

    def _valid_at(self, offset: int) -> RawPacket | None:
        size = self.protocol.packet_size
        if len(self.buffer) < offset + size:
            return None
        candidate = bytes(self.buffer[offset:offset + size])
        try:
            return decode_packet(candidate, self.protocol)
        except ValueError:
            return None

    def feed(self, chunk: bytes | bytearray | memoryview) -> list[RawPacket]:
        if chunk:
            self.buffer.extend(chunk)
        size = self.protocol.packet_size
        packets: list[RawPacket] = []
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
            if len(self.buffer) < 2 * size:
                break
            first = self._valid_at(0)
            second = self._valid_at(size)
            if first is None or second is None:
                self.invalid_candidates += 1
                self.discarded_bytes += 1
                del self.buffer[0]
                continue
            delta = (second.packet_sequence - first.packet_sequence) & 0xFF
            if not 1 <= delta <= 128:
                self.invalid_candidates += 1
                self.discarded_bytes += 1
                del self.buffer[0]
                continue
            packets.append(first)
            del self.buffer[:size]
        return packets


@dataclass(frozen=True, slots=True)
class EmgSample:
    packet_sequence: int
    sample_index: int
    raw_counts: tuple[int, ...]
    voltage_uv: tuple[float, ...]
    sequence_gap_before: int
    host_monotonic_ns: int
    host_unix_ns: int

    @property
    def kind(self) -> str:
        return "emg"


@dataclass(frozen=True, slots=True)
class ImuSample:
    packet_sequence: int
    raw: tuple[int, int, int, int, int, int, int, int, int]
    device_time_ms: int
    sequence_gap_before: int
    host_monotonic_ns: int
    host_unix_ns: int

    @property
    def kind(self) -> str:
        return "imu"

    @property
    def temperature_c(self) -> float:
        return self.raw[0] / 10.0

    @property
    def gyro_rad_s(self) -> tuple[float, float, float]:
        return tuple(value * GYRO_RAD_S_PER_COUNT for value in self.raw[1:4])

    @property
    def accel_m_s2(self) -> tuple[float, float, float]:
        return tuple(value * ACCEL_M_S2_PER_COUNT for value in self.raw[4:7])

    @property
    def battery_voltage_v(self) -> float:
        return self.raw[7] / 1000.0

    @property
    def battery_percent(self) -> int:
        return self.raw[8]


DataEvent: TypeAlias = EmgSample | ImuSample


class Weili18EmgDevice:
    """Exclusive, bounded-queue serial reader for one 18-channel wristband.

    Typical use::

        device = Weili18EmgDevice("/dev/serial/by-id/...")
        device.start()
        try:
            while True:
                for event in device.read_events(timeout=0.1):
                    if isinstance(event, EmgSample):
                        consume(event.voltage_uv, event.sample_index)
                    else:
                        consume_imu(event.gyro_rad_s, event.accel_m_s2)
                device.raise_if_failed()
        finally:
            device.close()

    Queue overflow and serial silence are reported as fatal reader errors so a
    collector cannot quietly save an incomplete stream as if it were complete.
    """

    def __init__(
        self,
        tty: str,
        *,
        baud: int = BAUD_RATE,
        packet_format: str | int | Protocol = FILTERED_PROTOCOL,
        queue_capacity: int = 50_000,
        silence_timeout_s: float = 0.5,
        read_chunk_bytes: int = 8192,
    ):
        if not tty:
            raise ValueError("tty path is required")
        if baud != BAUD_RATE:
            raise ValueError(f"the hardware specification requires exactly {BAUD_RATE} baud")
        if queue_capacity < 4096:
            raise ValueError("queue_capacity must be at least 4096 packet events")
        if silence_timeout_s <= 0:
            raise ValueError("silence_timeout_s must be positive")
        if read_chunk_bytes < 128:
            raise ValueError("read_chunk_bytes must be at least 128")
        self.tty = str(tty)
        self.baud = int(baud)
        self.protocol = protocol_for(packet_format)
        self.silence_timeout_s = float(silence_timeout_s)
        self.read_chunk_bytes = int(read_chunk_bytes)
        self._events: queue.Queue[DataEvent] = queue.Queue(maxsize=int(queue_capacity))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._serial = None
        self._lock = threading.Lock()
        self._error: str | None = None
        self._state = "created"
        self._stats = {
            "bytes_received": 0,
            "packets": 0,
            "emg_packets": 0,
            "imu_packets": 0,
            "missing_packets": 0,
            "duplicate_packets": 0,
            "out_of_order_packets": 0,
            "queue_overflows": 0,
            "discarded_bytes": 0,
            "invalid_candidates": 0,
        }
        self._last_packet_sequence: int | None = None
        self._emg_sample_index = 0

    def _set_error(self, message: str) -> None:
        with self._lock:
            if self._error is None:
                self._error = str(message)
                self._state = "error"

    def start(self) -> "Weili18EmgDevice":
        if self._thread is not None:
            raise RuntimeError("device has already been started")
        try:
            import serial
        except ImportError as exc:
            raise RuntimeError("pyserial is required; install requirements-collector.txt") from exc
        try:
            self._serial = serial.Serial(
                self.tty,
                self.baud,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.05,
                exclusive=True,
            )
        except TypeError:
            self._serial = serial.Serial(
                self.tty,
                self.baud,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.05,
            )
        with self._lock:
            self._state = "starting"
        self._thread = threading.Thread(target=self._reader_loop, name="weili18-serial-reader", daemon=True)
        self._thread.start()
        return self

    def _reader_loop(self) -> None:
        assert self._serial is not None
        parser = PacketParser(self.protocol)
        last_valid = time.monotonic()
        try:
            while not self._stop.is_set():
                waiting = int(getattr(self._serial, "in_waiting", 0))
                chunk = self._serial.read(min(max(waiting, 1), self.read_chunk_bytes))
                if not chunk:
                    if time.monotonic() - last_valid > self.silence_timeout_s:
                        raise TimeoutError(
                            f"no valid AA/BB packet for {self.silence_timeout_s:.3f}s on {self.tty}"
                        )
                    continue
                mono_ns = time.monotonic_ns()
                unix_ns = time.time_ns()
                with self._lock:
                    self._stats["bytes_received"] += len(chunk)
                packets = parser.feed(chunk)
                accepted_events: list[DataEvent] = []
                for packet in packets:
                    gap = 0
                    previous = self._last_packet_sequence
                    if previous is not None:
                        delta = (packet.packet_sequence - previous) & 0xFF
                        if delta == 0:
                            with self._lock:
                                self._stats["duplicate_packets"] += 1
                            continue
                        if delta > 128:
                            with self._lock:
                                self._stats["out_of_order_packets"] += 1
                            continue
                        if delta > 1:
                            gap = delta - 1
                            with self._lock:
                                self._stats["missing_packets"] += gap
                    self._last_packet_sequence = packet.packet_sequence
                    if packet.kind == "emg":
                        assert packet.emg_counts is not None
                        volts = tuple(value / self.protocol.counts_per_microvolt for value in packet.emg_counts)
                        accepted_events.append(EmgSample(
                            packet.packet_sequence,
                            self._emg_sample_index,
                            packet.emg_counts,
                            volts,
                            gap,
                            mono_ns,
                            unix_ns,
                        ))
                        self._emg_sample_index += 1
                    else:
                        assert packet.imu_raw is not None and packet.device_time_ms is not None
                        accepted_events.append(ImuSample(
                            packet.packet_sequence,
                            packet.imu_raw,
                            packet.device_time_ms,
                            gap,
                            mono_ns,
                            unix_ns,
                        ))
                with self._lock:
                    self._stats["discarded_bytes"] = parser.discarded_bytes
                    self._stats["invalid_candidates"] = parser.invalid_candidates
                for event in accepted_events:
                    try:
                        self._events.put_nowait(event)
                    except queue.Full as exc:
                        with self._lock:
                            self._stats["queue_overflows"] += 1
                        raise BufferError(
                            "consumer fell behind; bounded event queue overflowed, collection stopped"
                        ) from exc
                    with self._lock:
                        self._stats["packets"] += 1
                        self._stats["emg_packets" if event.kind == "emg" else "imu_packets"] += 1
                    last_valid = time.monotonic()
                    with self._lock:
                        if self._state == "starting":
                            self._state = "running"
        except Exception as exc:
            if not self._stop.is_set():
                self._set_error(f"{type(exc).__name__}: {exc}")
        finally:
            try:
                self._serial.close()
            except Exception:
                pass
            with self._lock:
                if self._state != "error":
                    self._state = "stopped" if self._stop.is_set() else "closed"

    def read_events(self, *, max_items: int = 4096, timeout: float = 0.1) -> list[DataEvent]:
        if max_items < 1 or timeout < 0:
            raise ValueError("max_items must be positive and timeout non-negative")
        result: list[DataEvent] = []
        try:
            result.append(self._events.get(timeout=timeout))
        except queue.Empty:
            return result
        while len(result) < max_items:
            try:
                result.append(self._events.get_nowait())
            except queue.Empty:
                break
        return result

    def drain_events(self, *, max_items: int = 4096) -> list[DataEvent]:
        return self.read_events(max_items=max_items, timeout=0.0)

    def raise_if_failed(self) -> None:
        with self._lock:
            error = self._error
        if error is not None:
            raise RuntimeError(error)

    @property
    def stats(self) -> dict[str, int | str | None]:
        with self._lock:
            result: dict[str, int | str | None] = dict(self._stats)
            result.update(state=self._state, error=self._error, queue_depth=self._events.qsize())
        return result

    def close(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
            if thread.is_alive():
                self._set_error("reader thread did not stop within 2 seconds")

    def __enter__(self) -> "Weili18EmgDevice":
        return self.start()

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()
