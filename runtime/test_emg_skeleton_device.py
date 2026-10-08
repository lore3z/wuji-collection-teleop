"""Real loopback UDP and fail-closed launcher tests; no device access."""
from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from emg_skeleton_device import EMGSkeletonDevice, validate_skeleton
from emg_teleop_launcher import verify_baseline, verify_original_backups


def hand():
    result = np.zeros((21, 3))
    for i, x in enumerate((-.05, -.025, 0, .025, .05)):
        base = 1 + 4 * i
        for j in range(4):
            result[base + j] = (x + (j * -.005 if i == 0 else 0), .035 + .025 * j, .002 * j)
    return result


class Clock:
    wall = 1000.0
    mono = 100.0

    def advance(self, seconds):
        self.wall += seconds
        self.mono += seconds


class ReceiverTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.device = EMGSkeletonDevice(port=0, wall_time=lambda: self.clock.wall,
                                       monotonic=lambda: self.clock.mono)
        self.sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def tearDown(self):
        self.sender.close()
        self.device.cleanup()

    def packet(self, **changes):
        msg = dict(source="emg2pose", session_id="one", seq=0,
                   timestamp=self.clock.wall, source_timestamp=10.0,
                   quality=dict(valid=True, source_age_ms=5.0, confidence=None),
                   skeleton=hand().tolist())
        msg.update(changes)
        return msg

    def send(self, msg=None, raw=None):
        self.sender.sendto(raw if raw is not None else json.dumps(msg).encode(), self.device.address)
        return self.device.get_fingers_data()["right_fingers"]

    def test_no_data_and_cleanup(self):
        self.assertEqual(self.device.get_fingers_data(), dict(left_fingers=None, right_fingers=None))
        self.device.cleanup()
        self.device.cleanup()
        self.assertIsNone(self.device.get_fingers_data()["right_fingers"])

    def test_roundtrip_and_copy_isolation(self):
        value = self.send(self.packet())
        np.testing.assert_array_equal(value, hand().astype(np.float32))
        self.assertEqual(value.dtype, np.float32)
        self.assertIsNone(self.device.get_fingers_data()["left_fingers"])
        value[:] = 99
        np.testing.assert_array_equal(self.device.get_fingers_data()["right_fingers"], hand().astype(np.float32))
        self.assertEqual(self.device.metadata["source_timestamp"], 10.0)
        self.assertIsNone(self.device.metadata["quality"]["confidence"])

    def test_monotonic_expiry_survives_wall_clock_jump(self):
        self.send(self.packet())
        self.clock.mono += .201
        self.clock.wall -= 999
        self.assertIsNone(self.device.get_fingers_data()["right_fingers"])
        self.assertFalse(self.device.metadata["fresh"])

    def test_duplicate_does_not_refresh_dead_stream(self):
        self.send(self.packet())
        self.clock.advance(.15)
        self.send(self.packet())
        self.clock.advance(.06)
        self.assertIsNone(self.device.get_fingers_data()["right_fingers"])
        self.assertEqual(self.device.stats["out_of_order"], 1)

    def test_sequence_gap_and_queue_drain(self):
        self.send(self.packet(seq=1))
        self.sender.sendto(json.dumps(self.packet(seq=3)).encode(), self.device.address)
        self.sender.sendto(json.dumps(self.packet(seq=4)).encode(), self.device.address)
        self.assertIsNotNone(self.device.get_fingers_data()["right_fingers"])
        self.assertEqual(self.device.metadata["seq"], 4)
        self.assertEqual(self.device.stats["missing_sequences"], 1)

    def test_new_session_cannot_take_over_until_expired(self):
        self.send(self.packet(seq=5))
        self.send(self.packet(session_id="two"))
        self.assertEqual(self.device.metadata["session_id"], "one")
        self.assertEqual(self.device.stats["other_publisher"], 1)
        self.clock.advance(.201)
        self.assertIsNotNone(self.send(self.packet(session_id="two")))
        self.assertEqual(self.device.metadata["session_id"], "two")

    def test_invalid_quality_immediately_invalidates_and_can_recover(self):
        self.send(self.packet())
        self.assertIsNone(self.send(self.packet(seq=1, skeleton=None, quality=dict(valid=False, reason="stopped"))))
        self.assertEqual(self.device.stats["invalid_quality"], 1)
        self.assertIsNotNone(self.send(self.packet(seq=2)))

    def test_source_age_independent_from_publication_age(self):
        self.assertIsNone(self.send(self.packet(quality=dict(valid=True, source_age_ms=250))))
        self.assertEqual(self.device.stats["invalid_quality"], 1)

    def test_publication_age_and_future_rejected(self):
        for delta in (-.21, .051):
            with self.subTest(delta=delta):
                self.assertIsNone(self.send(self.packet(timestamp=self.clock.wall + delta)))
        self.assertEqual(self.device.stats["rejected"], 2)

    def test_bad_protocol_inputs_rejected(self):
        tests = [dict(source="other"), dict(seq=True), dict(seq=-1), dict(seq=.5),
                 dict(seq=2**63), dict(session_id=""), dict(timestamp=True),
                 dict(timestamp=float("nan")), dict(source_timestamp=float("inf")),
                 dict(quality=dict(valid=1)), dict(quality=dict(valid=True, source_age_ms=float("nan"))),
                 dict(skeleton=np.zeros((21, 3)).tolist()), dict(skeleton=hand()[:20].tolist()),
                 dict(skeleton=(hand() * 1000).tolist()), dict(skeleton=hand().astype(str).tolist())]
        for changes in tests:
            with self.subTest(changes=list(changes)):
                self.assertIsNone(self.send(self.packet(**changes)))
        for value in (b"{", b"\xff", b"[]", b"x" * 17000):
            self.assertIsNone(self.send(raw=value))
        self.assertEqual(self.device.stats["rejected"], len(tests) + 4)

    def test_geometrically_singular_palm_rejected(self):
        value = hand()
        value[5] = value[9] * .8
        with self.assertRaises(ValueError):
            validate_skeleton(value)

    def test_exclusive_socket_and_local_only(self):
        with self.assertRaises(OSError):
            EMGSkeletonDevice(port=self.device.address[1])
        with self.assertRaises(ValueError):
            EMGSkeletonDevice(host="0.0.0.0", port=0)
        with self.assertRaises(ValueError):
            EMGSkeletonDevice(hand_side="left", port=0)
        self.assertIsNotNone(self.send(self.packet()))


class LauncherSafetyTests(unittest.TestCase):
    def run_cli(self, *args, executable=None):
        return subprocess.run([executable or sys.executable, *args], cwd=ROOT,
                              env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, timeout=10)

    def test_hw_scripts_require_explicit_arm_without_python_or_ros(self):
        for mode in ("a", "b"):
            result = self.run_cli(str(ROOT / "scripts" / f"run_mode_{mode}_emg_hw.sh"), executable="bash")
            self.assertEqual(result.returncode, 2)
            self.assertIn("explicitly add --arm", result.stderr)

    def test_sim_cannot_be_armed(self):
        for mode in ("A", "B"):
            for options in (("--sim", "--arm"), ("--hardware",), ("--arm",)):
                with self.subTest(mode=mode, options=options):
                    result = self.run_cli("-B", f"src/skeleton_teleop_MODE_{mode}_emg.py", *options)
                    self.assertEqual(result.returncode, 2)
                    self.assertNotIn("[V9.4]", result.stdout)

    def test_unsafe_ports_are_rejected_before_startup(self):
        result = self.run_cli("-B", "src/skeleton_teleop_MODE_A_emg.py", "--sim", "--port", "15120")
        self.assertEqual(result.returncode, 2)

    def test_originals_and_frozen_sources_match(self):
        manifest = verify_baseline()
        self.assertEqual(verify_original_backups(), 15)
        self.assertIn("V94_GOOD_ROOT_ONLY_PINCH", manifest["mode_a_origin"])
        self.assertIn("NATURAL_THUMB5", manifest["mode_b_origin"])
        for mode, original in (("a", "skeleton_teleop_v94_hw.py"),
                               ("b", "skeleton_teleop_MODE_B_thumb5.py")):
            entry = manifest["files"][f"mode_{mode}.py"]
            self.assertEqual(Path(entry["origin"]).name, original)


if __name__ == "__main__":
    unittest.main(verbosity=2)
