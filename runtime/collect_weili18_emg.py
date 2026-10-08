#!/usr/bin/env python3
"""Collect native-rate 唯理 / WAVELETECH-18 EMG and IMU into a Zarr episode."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import zarr
from numcodecs import Blosc

from weili18_emg import (
    BAUD_RATE,
    EMG_CHANNELS,
    EMG_RATE_HZ,
    GYRO_RAD_S_PER_COUNT,
    IMU_RATE_HZ,
    EmgSample,
    ImuSample,
    Weili18EmgDevice,
    protocol_for,
)


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_TTY = os.environ.get("WEILI18_TTY", "/dev/ttyUSB0")
DEFAULT_OUTPUT_DIR = PROJECT / "data" / "weili18_emg"
FORMAT = "weili18_emg_zarr_v1"
EMG_CHUNK_ROWS = 4096
IMU_CHUNK_ROWS = 512


def _create_array(group, name: str, tail_shape: tuple[int, ...], dtype, chunk_rows: int):
    return group.create_dataset(
        name,
        shape=(0, *tail_shape),
        chunks=(chunk_rows, *tail_shape),
        dtype=dtype,
        compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE),
    )


def _append(array, values: np.ndarray) -> None:
    if len(values) == 0:
        return
    start = array.shape[0]
    array.resize((start + len(values), *array.shape[1:]))
    array[start:start + len(values)] = values


class ZarrEpisodeWriter:
    def __init__(self, path: Path, *, protocol, tty: str, started_mono_ns: int, started_unix_ns: int):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f"Output already exists: {path}")
        self.final_path = path
        self.partial_path = path.with_name(path.name + ".partial")
        if self.partial_path.exists():
            raise FileExistsError(f"Incomplete output already exists: {self.partial_path}")
        self.protocol = protocol
        self.root = zarr.open_group(str(self.partial_path), mode="w-")
        self.streams = self.root.require_group("streams")
        self.arrays = {
            "emg_raw_counts": _create_array(self.streams, "emg_raw_counts", (EMG_CHANNELS,), "<i4", EMG_CHUNK_ROWS),
            "emg_voltage_uv": _create_array(self.streams, "emg_voltage_uv", (EMG_CHANNELS,), "<f4", EMG_CHUNK_ROWS),
            "emg_sample_index": _create_array(self.streams, "emg_sample_index", (), "<i8", EMG_CHUNK_ROWS),
            "emg_packet_sequence": _create_array(self.streams, "emg_packet_sequence", (), "u1", EMG_CHUNK_ROWS),
            "emg_sequence_gap_before": _create_array(self.streams, "emg_sequence_gap_before", (), "u1", EMG_CHUNK_ROWS),
            "emg_host_monotonic_ns": _create_array(self.streams, "emg_host_monotonic_ns", (), "<i8", EMG_CHUNK_ROWS),
            "emg_host_unix_ns": _create_array(self.streams, "emg_host_unix_ns", (), "<i8", EMG_CHUNK_ROWS),
            "imu_raw": _create_array(self.streams, "imu_raw", (9,), "<i4", IMU_CHUNK_ROWS),
            "imu_si": _create_array(self.streams, "imu_si", (9,), "<f4", IMU_CHUNK_ROWS),
            "imu_device_time_ms": _create_array(self.streams, "imu_device_time_ms", (), "<u8", IMU_CHUNK_ROWS),
            "imu_packet_sequence": _create_array(self.streams, "imu_packet_sequence", (), "u1", IMU_CHUNK_ROWS),
            "imu_sequence_gap_before": _create_array(self.streams, "imu_sequence_gap_before", (), "u1", IMU_CHUNK_ROWS),
            "imu_host_monotonic_ns": _create_array(self.streams, "imu_host_monotonic_ns", (), "<i8", IMU_CHUNK_ROWS),
            "imu_host_unix_ns": _create_array(self.streams, "imu_host_unix_ns", (), "<i8", IMU_CHUNK_ROWS),
        }
        self.pending_emg: list[EmgSample] = []
        self.pending_imu: list[ImuSample] = []
        self.emg_count = 0
        self.imu_count = 0
        self.root.attrs.update({
            "format": FORMAT,
            "device_model": "唯理科技 WAVELETECH-18 EMG wristband",
            "serial_port": tty,
            "serial_baud": BAUD_RATE,
            "serial_format": "8N1",
            "packet_format": protocol.name,
            "packet_size_bytes": protocol.packet_size,
            "emg_channels": [f"CH{i}" for i in range(1, EMG_CHANNELS + 1)],
            "emg_channel_count": EMG_CHANNELS,
            "emg_nominal_rate_hz": EMG_RATE_HZ,
            "imu_nominal_rate_hz": IMU_RATE_HZ,
            "emg_bits_per_channel": protocol.emg_bytes_per_channel * 8,
            "onboard_emg_filter_enabled": protocol.filter_enabled,
            "emg_counts_per_microvolt": protocol.counts_per_microvolt,
            "imu_gyro_rad_s_per_count": GYRO_RAD_S_PER_COUNT,
            "imu_accel_m_s2_per_count": 0.0005982,
            "emg_axis_order": "CH1..CH18 in the device packet order; no channel reordering",
            "imu_axis_order": "device X,Y,Z axes shown in the hardware manual",
            "imu_si_columns": ["temperature_C", "gyro_x_rad_s", "gyro_y_rad_s", "gyro_z_rad_s",
                               "accel_x_m_s2", "accel_y_m_s2", "accel_z_m_s2",
                               "battery_voltage_V", "battery_percent"],
            "imu_raw_columns": ["temperature_0p1C", "gyro_x_count", "gyro_y_count", "gyro_z_count",
                                "accel_x_count", "accel_y_count", "accel_z_count",
                                "battery_voltage_mV", "battery_percent"],
            "emg_voltage_units": "microvolt",
            "timestamp_semantics": (
                "host_monotonic_ns and host_unix_ns are serial-reader receive times and can repeat for packets "
                "decoded from one read batch; emg_sample_index is the observed AA-packet order at nominal 2000 Hz. "
                "The shared packet sequence and gap arrays audit transport loss."
            ),
            "capture_started_monotonic_ns": int(started_mono_ns),
            "capture_started_unix_ns": int(started_unix_ns),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "capture_status": "recording",
            "capture_complete": False,
        })

    def add(self, event) -> None:
        if isinstance(event, EmgSample):
            self.pending_emg.append(event)
        elif isinstance(event, ImuSample):
            self.pending_imu.append(event)
        else:
            raise TypeError(f"unsupported event type: {type(event).__name__}")
        if len(self.pending_emg) >= EMG_CHUNK_ROWS or len(self.pending_imu) >= IMU_CHUNK_ROWS:
            self.flush()

    def flush(self) -> None:
        if self.pending_emg:
            rows = self.pending_emg
            _append(self.arrays["emg_raw_counts"], np.asarray([x.raw_counts for x in rows], dtype=np.int32))
            _append(self.arrays["emg_voltage_uv"], np.asarray([x.voltage_uv for x in rows], dtype=np.float32))
            _append(self.arrays["emg_sample_index"], np.asarray([x.sample_index for x in rows], dtype=np.int64))
            _append(self.arrays["emg_packet_sequence"], np.asarray([x.packet_sequence for x in rows], dtype=np.uint8))
            _append(self.arrays["emg_sequence_gap_before"], np.asarray([x.sequence_gap_before for x in rows], dtype=np.uint8))
            _append(self.arrays["emg_host_monotonic_ns"], np.asarray([x.host_monotonic_ns for x in rows], dtype=np.int64))
            _append(self.arrays["emg_host_unix_ns"], np.asarray([x.host_unix_ns for x in rows], dtype=np.int64))
            self.emg_count += len(rows)
            self.pending_emg = []
        if self.pending_imu:
            rows = self.pending_imu
            raw = np.asarray([x.raw for x in rows], dtype=np.int32)
            si = np.empty((len(rows), 9), dtype=np.float32)
            si[:, 0] = raw[:, 0] / 10.0
            si[:, 1:4] = raw[:, 1:4] * 0.001225
            si[:, 4:7] = raw[:, 4:7] * 0.0005982
            si[:, 7] = raw[:, 7] / 1000.0
            si[:, 8] = raw[:, 8]
            _append(self.arrays["imu_raw"], raw)
            _append(self.arrays["imu_si"], si)
            _append(self.arrays["imu_device_time_ms"], np.asarray([x.device_time_ms for x in rows], dtype=np.uint64))
            _append(self.arrays["imu_packet_sequence"], np.asarray([x.packet_sequence for x in rows], dtype=np.uint8))
            _append(self.arrays["imu_sequence_gap_before"], np.asarray([x.sequence_gap_before for x in rows], dtype=np.uint8))
            _append(self.arrays["imu_host_monotonic_ns"], np.asarray([x.host_monotonic_ns for x in rows], dtype=np.int64))
            _append(self.arrays["imu_host_unix_ns"], np.asarray([x.host_unix_ns for x in rows], dtype=np.int64))
            self.imu_count += len(rows)
            self.pending_imu = []

    def finalize(self, *, status: str, stop_reason: str, device_stats: dict) -> Path:
        self.flush()
        self.root.attrs.update({
            "capture_status": status,
            "capture_complete": status == "complete",
            "stop_reason": stop_reason,
            "emg_sample_count": int(self.emg_count),
            "imu_sample_count": int(self.imu_count),
            "device_stats": json.dumps(device_stats, ensure_ascii=False, sort_keys=True),
            "capture_finished_utc": datetime.now(timezone.utc).isoformat(),
        })
        store = getattr(self.root, "store", None)
        if store is not None and hasattr(store, "close"):
            store.close()
        if self.emg_count == 0:
            raise RuntimeError(f"No EMG rows were captured; incomplete data remains at {self.partial_path}")
        self.partial_path.rename(self.final_path)
        return self.final_path


def _output_path(args) -> Path:
    if args.output:
        return Path(args.output).expanduser().resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (Path(args.output_dir).expanduser().resolve() / f"episode_{stamp}.zarr")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tty", default=DEFAULT_TTY, help="USB communication port; use /dev/serial/by-id/... when available")
    parser.add_argument("--packet-format", choices=("filtered", "raw"), default="filtered",
                        help="filtered=41-byte packets (default, 16-bit); raw=59-byte packets (24-bit)")
    parser.add_argument("--duration-seconds", type=float, default=10.0,
                        help="capture duration; 0 captures until Ctrl-C (default: 10 seconds)")
    parser.add_argument("--output", help="exact output .zarr path; defaults to a timestamped episode under --output-dir")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--silence-timeout-s", type=float, default=0.5)
    args = parser.parse_args(argv)
    if args.duration_seconds < 0 or args.silence_timeout_s <= 0:
        parser.error("duration must be non-negative and silence timeout positive")
    protocol = protocol_for(args.packet_format)
    started_mono_ns = time.monotonic_ns()
    started_unix_ns = time.time_ns()
    output_path = _output_path(args)
    try:
        writer = ZarrEpisodeWriter(
            output_path, protocol=protocol, tty=args.tty,
            started_mono_ns=started_mono_ns, started_unix_ns=started_unix_ns,
        )
    except Exception as exc:
        print(f"[FAILED] Cannot create output: {exc}", file=sys.stderr)
        return 2

    device = Weili18EmgDevice(
        args.tty, baud=BAUD_RATE, packet_format=protocol,
        silence_timeout_s=args.silence_timeout_s, read_chunk_bytes=512,
    )
    status = "failed"
    reason = "unexpected exit"
    error = None
    try:
        device.start()
        print(
            f"[LIVE] 唯理 WAVELETECH-18: {args.tty} @ {BAUD_RATE} baud, "
            f"{protocol.name}, 18x{EMG_RATE_HZ} Hz EMG + {IMU_RATE_HZ} Hz IMU\n"
            f"[SAVE] {output_path}\n"
            "串口已独占；不发送设备配置指令。按 Ctrl-C 保存并结束。",
            flush=True,
        )
        next_report = time.monotonic() + 1.0
        end_at = time.monotonic() + args.duration_seconds if args.duration_seconds else None
        while end_at is None or time.monotonic() < end_at:
            timeout = 0.1
            if end_at is not None:
                timeout = max(0.0, min(timeout, end_at - time.monotonic()))
            for event in device.read_events(max_items=4096, timeout=timeout):
                writer.add(event)
            device.raise_if_failed()
            now = time.monotonic()
            if now >= next_report:
                stats = device.stats
                print(
                    f"[RECORD] emg={writer.emg_count + len(writer.pending_emg)} "
                    f"imu={writer.imu_count + len(writer.pending_imu)} "
                    f"seq_missing={stats['missing_packets']} queue={stats['queue_depth']} "
                    f"discarded_bytes={stats['discarded_bytes']}",
                    flush=True,
                )
                next_report = now + 1.0
        # Drain events that were decoded just before the duration boundary.
        for event in device.drain_events(max_items=50_000):
            writer.add(event)
        device.raise_if_failed()
        status = "complete"
        reason = "duration_elapsed" if args.duration_seconds else "stopped"
    except KeyboardInterrupt:
        status = "interrupted"
        reason = "keyboard_interrupt"
        print("\n[STOP] 收到 Ctrl-C，正在落盘已收到的数据。", flush=True)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        reason = "reader_or_storage_error"
        print(f"[FAILED] {error}", file=sys.stderr)
    finally:
        device.close()
        # Collect data already queued before the stop request without waiting
        # for new samples from the serial reader.
        try:
            for event in device.drain_events(max_items=50_000):
                writer.add(event)
            device.raise_if_failed()
        except Exception as exc:
            if error is None:
                error = f"{type(exc).__name__}: {exc}"
                status = "failed"
                reason = "reader_or_storage_error"
        stats = device.stats
        if error:
            stats["error"] = error
        try:
            saved = writer.finalize(status=status, stop_reason=reason, device_stats=stats)
            print(f"[SAVE] {saved}: EMG={writer.emg_count}, IMU={writer.imu_count}, status={status}", flush=True)
        except Exception as exc:
            print(f"[FAILED] Finalizing capture: {exc}", file=sys.stderr)
            return_code = 1
        else:
            return_code = 1 if status == "failed" else 0
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
