"""Outages must discard captured buffers without waiting for operator save."""
import threading
from types import SimpleNamespace
from unittest.mock import Mock

from wuji_glove_d435_ftp1_collect import FTP1Collector


def _collector():
    c = FTP1Collector.__new__(FTP1Collector)
    c.state_lock = threading.Lock()
    c.stop_event = threading.Event()
    c.recording = True
    c.baseline_capturing = c.joint_preflight_capturing = False
    for name in ('angle', 'tactile', 'skeleton', 'zone', 'right_wrist_pose', 'emg'):
        setattr(c, name + '_frames', [object()])
    c.glove = SimpleNamespace(
        drain=lambda: ([], []), drain_skeleton=lambda: [], drain_zones=lambda: [],
        drain_right_wrist_pose=lambda: [], clear=Mock())
    c.emg_enabled = True
    c.emg_invalid_reason = c.emg_warning_reason = ''
    c.emg_episode_connection_id = 1
    c.emg_episode_interruption_id = 0
    c.emg_last_episode_ts_ns = 0
    c.max_emg_gap_ns = 350_000_000
    c.emg = SimpleNamespace(
        drain=lambda: [], current_connection_id=lambda: 1,
        interruption_snapshot=lambda: (1, 're-arming'), healthy=lambda **kw: True,
        silence_timeout_s=0.35, age_ms=lambda: 5, clear=Mock())
    c.camera = SimpleNamespace(capture_ego=True, trackers_enabled=False,
                               discard_episode=Mock(side_effect=c.stop_event.set))
    c._save_thread = None
    return c


def test_recovered_outage_discards_episode_in_drain_loop(capsys):
    c = _collector()
    c._drain_loop()
    assert not c.recording
    assert c.emg_frames == []
    assert c.angle_frames == []
    c.camera.discard_episode.assert_called_once()
    assert 'AUTO REFUSE' in capsys.readouterr().out
    c.save()
    assert 'SAVE REFUSED' not in capsys.readouterr().out


def test_save_catches_outage_before_drain_monitor_runs():
    c = _collector()
    c.save()
    assert not c.recording
    c.camera.discard_episode.assert_called_once()
    assert c.emg_frames == []


def test_silence_is_rejected_before_rearm_event():
    c = _collector()
    c.emg.interruption_snapshot = lambda: (0, '')
    c.emg.healthy = lambda **kw: False
    assert c._emg_live_fault()


def test_healthy_stream_and_no_myo_are_accepted():
    c = _collector()
    c.emg.interruption_snapshot = lambda: (0, '')
    assert c._emg_live_fault() == ''
    c.emg_enabled = False
    c.emg = None
    assert c._emg_live_fault() == ''


def test_native_gap_is_latched_and_discards_even_if_source_is_healthy():
    c = _collector()
    c.emg.interruption_snapshot = lambda: (0, '')
    c.emg_last_episode_ts_ns = 1_000_000_000
    c.emg.drain = lambda: [(1_400_000_000, 2, 0, None, 1_400_000_000)]
    c._drain_loop()
    assert not c.recording
    assert '400 ms' in c.last_refusal_reason
    assert c.emg_frames == []


def test_reconnect_discards_even_without_a_long_arrival_gap():
    c = _collector()
    c.emg.interruption_snapshot = lambda: (0, '')
    c.emg.current_connection_id = lambda: 2
    c._drain_loop()
    assert not c.recording
    assert 'reconnected' in c.last_refusal_reason
