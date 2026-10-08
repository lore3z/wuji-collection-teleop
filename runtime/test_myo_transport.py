#!/usr/bin/env python3

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from myo_transport import PortBusyError, acquire_bled112_lock, normalize_mac, resolve_bled112_tty
from myo_bled112_scan import MYO_ADV_SUFFIX, _auto_select_myo_mac, _is_myo_advertisement


class MyoTransportTest(unittest.TestCase):
    def test_myo_advertisement_signature(self) -> None:
        self.assertTrue(_is_myo_advertisement(b"prefix" + MYO_ADV_SUFFIX))
        self.assertFalse(_is_myo_advertisement(b"prefix" + b"not-myo"))

    def test_auto_selects_the_only_advertising_myo(self) -> None:
        self.assertEqual(
            _auto_select_myo_mac({"E3:9D:20:B1:F8:5C": 12}),
            "E3:9D:20:B1:F8:5C",
        )

    def test_auto_selection_refuses_zero_or_multiple_myos(self) -> None:
        with self.assertRaises(ValueError):
            _auto_select_myo_mac({})
        with self.assertRaises(ValueError):
            _auto_select_myo_mac({
                "E3:9D:20:B1:F8:5C": 12,
                "ED:94:F0:1E:61:D3": 8,
            })

    def test_normalize_mac_accepts_colons_and_dashes(self) -> None:
        self.assertEqual(normalize_mac("ed-94-f0-1e-61-d3"), "ED:94:F0:1E:61:D3")

    def test_normalize_mac_rejects_placeholder(self) -> None:
        with self.assertRaises(ValueError):
            normalize_mac("<confirmed Myo MAC>")

    def test_normalize_mac_rejects_malformed_value(self) -> None:
        with self.assertRaises(ValueError):
            normalize_mac("ED:94:F0:1E:61")

    def test_existing_configured_path_wins(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            configured = Path(root) / "ttyACM7"
            configured.touch()
            by_id = Path(root) / "by-id"
            by_id.mkdir()
            self.assertEqual(
                resolve_bled112_tty(str(configured), by_id_dir=str(by_id)),
                str(configured),
            )

    def test_unique_bluegiga_path_is_used_when_configured_path_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            by_id = Path(root) / "by-id"
            by_id.mkdir()
            candidate = by_id / "usb-Bluegiga_Low_Energy_Dongle_1-if00"
            candidate.touch()
            self.assertEqual(
                resolve_bled112_tty(str(Path(root) / "missing"), by_id_dir=str(by_id)),
                str(candidate),
            )

    def test_unique_bluegiga_path_is_preferred_over_numeric_tty(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            by_id = Path(root) / "by-id"
            by_id.mkdir()
            candidate = by_id / "usb-Bluegiga_Low_Energy_Dongle_1-if00"
            candidate.touch()
            configured = Path(root) / "ttyACM0"
            configured.touch()
            self.assertEqual(
                resolve_bled112_tty(str(configured), by_id_dir=str(by_id)),
                str(candidate),
            )

    def test_multiple_bluegiga_paths_do_not_guess(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            by_id = Path(root) / "by-id"
            by_id.mkdir()
            (by_id / "usb-Bluegiga_Low_Energy_Dongle_1-if00").touch()
            (by_id / "usb-Bluegiga_Low_Energy_Dongle_2-if00").touch()
            configured = str(Path(root) / "missing")
            self.assertEqual(resolve_bled112_tty(configured, by_id_dir=str(by_id)), configured)

    def test_lock_rejects_second_owner(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            old_dir = os.environ.get("WUJI_MYO_LOCK_DIR")
            os.environ["WUJI_MYO_LOCK_DIR"] = root
            try:
                with acquire_bled112_lock():
                    with self.assertRaises(PortBusyError):
                        with acquire_bled112_lock():
                            pass
            finally:
                if old_dir is None:
                    os.environ.pop("WUJI_MYO_LOCK_DIR", None)
                else:
                    os.environ["WUJI_MYO_LOCK_DIR"] = old_dir


if __name__ == "__main__":
    unittest.main()
