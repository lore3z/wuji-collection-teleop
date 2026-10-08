import time
from unittest.mock import Mock

from wuji_glove_d435_collect import (
    WUJI_OFFICIAL_ACTIVE_TAXELS,
    WujiGloveSource,
)


def _source() -> WujiGloveSource:
    return WujiGloveSource(
        "test-glove",
        120.0,
        capture_skeleton=True,
        capture_zones=True,
        capture_right_wrist_pose=True,
    )


def test_250ms_whole_stream_pause_remains_healthy_with_episode_budget():
    source = _source()
    now = time.monotonic()
    source.last_angle_mono = now - 0.250
    source.last_tactile_mono = now - 0.251
    source.last_skeleton_mono = now - 0.250
    source.last_zone_mono = now - 0.251
    source.last_right_wrist_pose_mono = now - 0.255
    source.last_tactile_active_taxels = WUJI_OFFICIAL_ACTIVE_TAXELS
    source.error = "one transient recv error"

    assert source.healthy(max_age_s=0.5)


def test_stalled_requested_stream_triggers_reconnect_after_budget():
    source = _source()
    source.reconnect_after_s = 0.5
    source.connection_started_mono = 100.0
    source.last_angle_mono = 100.8
    source.last_tactile_mono = 100.8
    source.last_skeleton_mono = 100.8
    source.last_zone_mono = 100.8
    source.last_right_wrist_pose_mono = 100.5

    assert not source._transport_stale(100.99)
    assert source._transport_stale(101.01)


def test_successful_reconnect_rebuilds_transport_and_resets_backoff():
    source = _source()
    source._close_transport = Mock()
    source._open_transport = Mock()
    source.reconnect_retry_s = 4.0

    source._reconnect_transport(time.monotonic())

    source._close_transport.assert_called_once_with()
    source._open_transport.assert_called_once_with()
    assert source.reconnects == 1
    assert source.reconnect_retry_s == 1.0
    assert source.next_reconnect_mono == 0.0


def test_failed_reconnect_keeps_retryable_disconnected_state():
    source = _source()
    source._close_transport = Mock()
    source._open_transport = Mock(side_effect=RuntimeError("device unavailable"))

    source._reconnect_transport(time.monotonic())

    assert "device unavailable" in source.error
    assert source.next_reconnect_mono > time.monotonic()
    assert source.reconnect_retry_s == 2.0
