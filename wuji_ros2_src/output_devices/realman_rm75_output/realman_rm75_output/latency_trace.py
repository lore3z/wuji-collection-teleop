"""Compact timestamp trace shared by the RM75 shadow and safety bridge."""

from collections import deque

import numpy as np


PICO_TRACE_VERSION = 1
PICO_TRACE_LENGTH = 6
SHADOW_TRACE_VERSION = 1
SHADOW_TRACE_LENGTH = 11


def stamp_to_ns(stamp):
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def parse_pico_trace(data):
    values = [int(value) for value in data]
    if len(values) != PICO_TRACE_LENGTH or values[0] != PICO_TRACE_VERSION:
        raise ValueError("invalid PICO latency trace")
    return {
        "pose_key_ns": values[1],
        "sdk_sample_ns": values[2],
        "pc_poll_start_ns": values[3],
        "pc_read_complete_ns": values[4],
        "pico_publish_ns": values[5],
    }


def make_shadow_trace(shadow_key_ns, source, shadow_publish_ns):
    source = source or {}
    return [
        SHADOW_TRACE_VERSION,
        int(shadow_key_ns),
        int(source.get("pose_key_ns", 0)),
        int(source.get("sdk_sample_ns", 0)),
        int(source.get("pc_poll_start_ns", 0)),
        int(source.get("pc_read_complete_ns", 0)),
        int(source.get("pico_publish_ns", 0)),
        int(source.get("ik_receive_ns", 0)),
        int(source.get("ik_start_ns", 0)),
        int(source.get("ik_end_ns", 0)),
        int(shadow_publish_ns),
    ]


def parse_shadow_trace(data):
    values = [int(value) for value in data]
    if len(values) != SHADOW_TRACE_LENGTH or values[0] != SHADOW_TRACE_VERSION:
        raise ValueError("invalid RM75 shadow latency trace")
    names = (
        "shadow_key_ns", "pose_key_ns", "sdk_sample_ns",
        "pc_poll_start_ns", "pc_read_complete_ns", "pico_publish_ns",
        "ik_receive_ns", "ik_start_ns", "ik_end_ns",
        "shadow_publish_ns",
    )
    return dict(zip(names, values[1:]))


class PipelineLatencyStatistics:
    """Bounded rolling statistics for one accepted source pose per sample."""

    PAIRS = {
        "pico_to_pc": ("sdk_sample_ns", "pc_read_complete_ns"),
        "pc_sdk_read": ("pc_poll_start_ns", "pc_read_complete_ns"),
        "pc_to_ik_receive": ("pc_read_complete_ns", "ik_receive_ns"),
        "ik_queue": ("ik_receive_ns", "ik_start_ns"),
        "ik_compute": ("ik_start_ns", "ik_end_ns"),
        "ik_to_shadow": ("ik_end_ns", "shadow_publish_ns"),
        "shadow_to_bridge": ("shadow_publish_ns", "bridge_receive_ns"),
        "pc_to_bridge": ("pc_poll_start_ns", "bridge_receive_ns"),
        "pico_to_bridge": ("sdk_sample_ns", "bridge_receive_ns"),
    }
    CALIBRATED_NAMES = (
        "pico_to_pc_relative", "pico_to_bridge_relative")

    def __init__(self, capacity=1800, maximum_valid_ms=10_000.0):
        self.values = {
            name: deque(maxlen=int(capacity)) for name in self.PAIRS}
        self.values.update({
            name: deque(maxlen=int(capacity))
            for name in self.CALIBRATED_NAMES})
        self.latest = {}
        self.samples = 0
        self.sdk_clock_valid_samples = 0
        self.sdk_clock_invalid_samples = 0
        self.sdk_clock_offset_min_ns = None
        self.maximum_valid_ns = int(float(maximum_valid_ms) * 1e6)

    def observe(self, trace, bridge_receive_ns):
        values = dict(trace)
        values["bridge_receive_ns"] = int(bridge_receive_ns)
        latest = {}
        sdk_valid = True
        for name, (start_name, end_name) in self.PAIRS.items():
            start = int(values.get(start_name, 0))
            end = int(values.get(end_name, 0))
            delta = end - start
            valid = start > 0 and end > 0 and 0 <= delta <= self.maximum_valid_ns
            if name in ("pico_to_pc", "pico_to_bridge") and not valid:
                sdk_valid = False
            if valid:
                milliseconds = delta / 1e6
                self.values[name].append(milliseconds)
                latest[name] = milliseconds
        sdk_sample = int(values.get("sdk_sample_ns", 0))
        pc_complete = int(values.get("pc_read_complete_ns", 0))
        bridge_receive = int(values.get("bridge_receive_ns", 0))
        raw_clock_delta = pc_complete - sdk_sample
        # When PICO and PC clocks are offset, the minimum observed arrival
        # delta estimates that offset plus the best-path transit time. The
        # corrected values therefore show delay above the best observed path;
        # they are useful for queue/jitter diagnosis but not absolute one-way
        # network latency.
        if (sdk_sample > 0 and pc_complete > 0 and
                abs(raw_clock_delta) <= 3_600_000_000_000):
            if (self.sdk_clock_offset_min_ns is None or
                    raw_clock_delta < self.sdk_clock_offset_min_ns):
                self.sdk_clock_offset_min_ns = raw_clock_delta
            relative_pc_ns = max(
                0, raw_clock_delta - self.sdk_clock_offset_min_ns)
            relative_bridge_ns = relative_pc_ns + max(
                0, bridge_receive - pc_complete)
            for name, delta_ns in (
                    ("pico_to_pc_relative", relative_pc_ns),
                    ("pico_to_bridge_relative", relative_bridge_ns)):
                milliseconds = delta_ns / 1e6
                self.values[name].append(milliseconds)
                latest[name] = milliseconds
        self.samples += 1
        if sdk_valid:
            self.sdk_clock_valid_samples += 1
        else:
            self.sdk_clock_invalid_samples += 1
        self.latest = latest
        return latest

    def summary(self):
        metrics = {}
        for name, samples in self.values.items():
            if not samples:
                continue
            array = np.asarray(samples, dtype=float)
            metrics[name] = {
                "latest": float(self.latest.get(name, array[-1])),
                "p50": float(np.percentile(array, 50)),
                "p95": float(np.percentile(array, 95)),
                "max": float(np.max(array)),
            }
        return {
            "samples": self.samples,
            "sdk_clock_valid_samples": self.sdk_clock_valid_samples,
            "sdk_clock_invalid_samples": self.sdk_clock_invalid_samples,
            "sdk_clock_offset_estimate_ms": (
                None if self.sdk_clock_offset_min_ns is None else
                self.sdk_clock_offset_min_ns / 1e6),
            "metric_ms": metrics,
        }
