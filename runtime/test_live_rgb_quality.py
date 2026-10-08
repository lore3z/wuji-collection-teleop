import threading

from wuji_glove_d435_collect import D435Node


def _monitor(max_gap_ns=50_000_000, max_missing_ratio=0.02):
    monitor = D435Node.__new__(D435Node)
    monitor.lock = threading.Lock()
    monitor.episode_rgb_timestamps_ns = []
    monitor.episode_rgb_fault = ""
    monitor.episode_max_rgb_gap_ns = 0
    monitor.episode_missing_ratio = 0.0
    monitor.live_max_rgb_gap_ns = max_gap_ns
    monitor.live_max_missing_ratio = max_missing_ratio
    return monitor


def test_live_rgb_quality_refuses_large_gap_immediately():
    monitor = _monitor()
    monitor._update_live_rgb_quality(1_000_000_000)
    monitor._update_live_rgb_quality(1_071_400_000)
    assert "71.40 ms" in monitor.episode_rgb_fault


def test_live_rgb_quality_does_not_refuse_from_callback_ratio():
    monitor = _monitor()
    stamp = 1_000_000_000
    period = 16_666_667
    for index in range(120):
        if index in {30, 60, 90}:
            stamp += period
        monitor._update_live_rgb_quality(stamp)
        stamp += period
    assert monitor.episode_rgb_fault == ""


def test_live_rgb_quality_accepts_continuous_frames():
    monitor = _monitor()
    period = 16_666_667
    for index in range(180):
        monitor._update_live_rgb_quality(1_000_000_000 + index * period)
    assert monitor.episode_rgb_fault == ""


def test_live_rgb_quality_tolerates_one_frame_delivery_jitter():
    monitor = _monitor(max_gap_ns=100_000_000, max_missing_ratio=0.02)
    period = 16_666_667
    stamp = 1_000_000_000
    for index in range(180):
        monitor._update_live_rgb_quality(stamp)
        stamp += period + (period // 2 if index % 30 == 0 else 0)
    assert monitor.episode_rgb_fault == ""
