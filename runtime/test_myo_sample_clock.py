#!/usr/bin/env python3

from __future__ import annotations

import unittest

from myo_200hz_stream import EmgSampleClock, close_myo


class EmgSampleClockTest(unittest.TestCase):
    def test_failed_link_cleanup_never_sends_disconnect_command(self) -> None:
        class FakeSerial:
            def __init__(self) -> None:
                self.closed = False

            def close(self) -> None:
                self.closed = True

        class FakeBt:
            def __init__(self) -> None:
                self.ser = FakeSerial()

        class FakeMyo:
            def __init__(self) -> None:
                self.bt = FakeBt()
                self.conn = 1
                self.disconnect_called = False

            def disconnect(self) -> None:
                self.disconnect_called = True
                raise AssertionError("failed-link cleanup must not wait for BLE response")

        myo = FakeMyo()
        close_myo(myo, graceful=False)
        self.assertTrue(myo.bt.ser.closed)
        self.assertFalse(myo.disconnect_called)

    def test_batched_arrivals_become_uniform_sample_times(self) -> None:
        clock = EmgSampleClock()
        arrivals = [100_000_000, 100_200_000, 110_000_000, 110_300_000]
        timestamps = [clock.timestamp(value) for value in arrivals]
        self.assertEqual(timestamps[0], 95_000_000)
        self.assertTrue(all(4_900_000 <= dt <= 5_100_000 for dt in _diff(timestamps)))

    def test_reset_starts_a_new_connection_timeline(self) -> None:
        clock = EmgSampleClock()
        clock.timestamp(100_000_000)
        clock.reset()
        self.assertEqual(clock.timestamp(500_000_000), 495_000_000)

    def test_delayed_batch_does_not_create_a_fake_sample_gap(self) -> None:
        clock = EmgSampleClock()
        timestamps = [clock.timestamp(value) for value in (100_000_000, 100_200_000, 318_000_000)]
        assert 4_900_000 <= timestamps[2] - timestamps[1] <= 5_100_000

    def test_small_long_term_rate_error_is_followed_without_time_jumps(self) -> None:
        clock = EmgSampleClock()
        # pyomyo emits two callbacks per BLE packet. Model a 197.6 Hz stream
        # for about ten minutes, including the near-simultaneous callback pair.
        arrivals = [
            100_000_000 + packet * 10_120_000 + within_packet
            for packet in range(60_000)
            for within_packet in (0, 200_000)
        ]
        timestamps = [clock.timestamp(value) for value in arrivals]
        self.assertTrue(all(4_900_000 <= dt <= 5_100_000 for dt in _diff(timestamps)))
        self.assertLess(abs(timestamps[-1] - arrivals[-1]), 10_000_000)


def _diff(values: list[int]) -> list[int]:
    return [right - left for left, right in zip(values, values[1:])]


if __name__ == "__main__":
    unittest.main()
