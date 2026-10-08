"""Synthetic startup-order tests; loopback UDP only, never hardware port 15120."""
from __future__ import annotations

import json
from pathlib import Path
import socket
import sys
import threading
import time
import unittest

import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from emg_model_skeleton_adapter import Publisher
from emg_skeleton_device import EMGSkeletonDevice
from emg_teleop_launcher import (
    HardwareInputSafety,
    wait_bridge_ready_with_skeleton_drain,
    wait_for_post_ready_fresh,
)


def hand():
    result = np.zeros((21, 3))
    for i, x in enumerate((-.05, -.025, 0, .025, .05)):
        base = 1 + 4 * i
        for j in range(4):
            result[base + j] = (x + (j * -.005 if i == 0 else 0),
                                .035 + .025 * j, .002 * j)
    return result


class PeriodicPublisher(threading.Thread):
    def __init__(self, destination, hz=25.0, session_id="startup-test"):
        super().__init__(daemon=True)
        self.destination = destination
        self.period = 1.0 / hz
        self.session_id = session_id
        self.seq = 0
        self.done = threading.Event()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def run(self):
        next_send = time.monotonic()
        while not self.done.is_set():
            packet = {
                "source": "emg2pose",
                "session_id": self.session_id,
                "seq": self.seq,
                "timestamp": time.time(),
                "source_timestamp": self.seq * self.period,
                "quality": {"valid": True, "source_age_ms": 1.0},
                "skeleton": hand().tolist(),
            }
            self.sock.sendto(json.dumps(packet).encode(), self.destination)
            self.seq += 1
            next_send += self.period
            self.done.wait(max(0.0, next_send - time.monotonic()))

    def stop(self):
        self.done.set()
        self.join(timeout=1.0)
        self.sock.close()


class RunningProcess:
    returncode = None

    @staticmethod
    def poll():
        return None


def receive_initial(device, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if device.get_fingers_data()["right_fingers"] is not None:
            return device.metadata
        time.sleep(.005)
    raise AssertionError("publisher did not produce initial fresh Skeleton")


class StartupFreshnessTests(unittest.TestCase):
    def test_blocked_startup_without_drain_reproduces_stale(self):
        device = EMGSkeletonDevice(port=0, max_age_s=.20)
        # A deliberately small receive buffer makes a blocked latest-frame
        # consumer retain the oldest datagrams while newer 25 Hz packets drop.
        device._socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2048)
        publisher = PeriodicPublisher(device.address)
        try:
            publisher.start()
            receive_initial(device)
            time.sleep(.70)
            publisher.stop()
            self.assertIsNone(device.get_fingers_data()["right_fingers"])
            self.assertGreater(device.stats["rejected"], 0)
            self.assertIn("publication timestamp is stale", device.metadata["last_rejection"])
        finally:
            if publisher.is_alive():
                publisher.stop()
            device.cleanup()

    def test_bridge_wait_drains_and_requires_post_ready_fresh(self):
        device = EMGSkeletonDevice(port=0, max_age_s=.20)
        publisher = PeriodicPublisher(device.address)
        try:
            publisher.start()
            receive_initial(device)
            ready_at = time.monotonic() + .70
            result = wait_bridge_ready_with_skeleton_drain(
                RunningProcess(), device, lambda: time.monotonic() >= ready_at,
                timeout=1.2, fresh_timeout=.6, required=5,
            )
            self.assertEqual(result["fresh_frames"], 5)
            self.assertGreater(result["active_seq"], result["bridge_ready_seq"])
            self.assertEqual(result["session_id"], "startup-test")
            self.assertLess(result["receiver_age_ms"], 200.0)
            self.assertGreater(result["bridge_wait_drain_calls"], 50)
        finally:
            publisher.stop()
            device.cleanup()

    def test_publisher_stops_after_ready_fails_closed(self):
        device = EMGSkeletonDevice(port=0, max_age_s=.20)
        publisher = PeriodicPublisher(device.address)
        try:
            publisher.start()
            ready = receive_initial(device)
            publisher.stop()
            time.sleep(.21)
            with self.assertRaises(TimeoutError):
                wait_for_post_ready_fresh(
                    device, ready["seq"], ready["session_id"],
                    required=3, timeout=.10,
                )
            self.assertFalse(device.metadata["fresh"])
        finally:
            if publisher.is_alive():
                publisher.stop()
            device.cleanup()

    def test_active_stale_latches_and_does_not_auto_recover(self):
        safety = HardwareInputSafety(True)
        self.assertTrue(safety.observe(True, {"session_id": "one"}))
        safety.activate()
        self.assertFalse(safety.observe(False, {"session_id": "one"}))
        self.assertTrue(safety.latched)
        self.assertFalse(safety.observe(True, {"session_id": "one"}))

    def test_cached_ready_frame_cannot_pass_startup_gate(self):
        device = EMGSkeletonDevice(port=0, max_age_s=.20)
        publisher = PeriodicPublisher(device.address)
        try:
            publisher.start()
            ready = receive_initial(device)
            publisher.stop()
            with self.assertRaises(TimeoutError):
                wait_for_post_ready_fresh(
                    device, ready["seq"], ready["session_id"],
                    required=1, timeout=.08,
                )
        finally:
            if publisher.is_alive():
                publisher.stop()
            device.cleanup()

    def test_adapter_final_invalidation_remains_effective(self):
        device = EMGSkeletonDevice(port=0, max_age_s=.20)
        publisher = Publisher("127.0.0.1", 17621, None)
        publisher.destination = device.address
        row = {
            "skeleton": hand(), "time_seconds": 123.0, "backend": "test",
            "session_id": "backend", "frame": 1, "model_id": "n_motion",
        }
        try:
            publisher.send(row)
            self.assertIsNotNone(device.get_fingers_data()["right_fingers"])
            self.assertEqual(device.metadata["source_timestamp"], 123.0)
            self.assertLess(abs(device.metadata["timestamp"] - time.time()), .20)
            publisher.send(row, valid=False, reason="adapter_stopped")
            self.assertIsNone(device.get_fingers_data()["right_fingers"])
            self.assertEqual(device.metadata["quality"]["reason"], "adapter_stopped")
            self.assertEqual(device.stats["invalid_quality"], 1)
        finally:
            publisher.close()
            device.cleanup()


if __name__ == "__main__":
    unittest.main(verbosity=2)
