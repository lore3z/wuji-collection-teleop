from types import SimpleNamespace
from unittest.mock import patch

from wuji_myo_emg_source import MyoEmgSource


class _AliveProcess:
    @staticmethod
    def poll():
        return None


def _source_with_recent_rate(rate_hz: float, now: float = 100.0) -> MyoEmgSource:
    source = MyoEmgSource("/bin/true", "/dev/null", "ED:94:F0:1E:61:D3")
    source.process = _AliveProcess()
    source.connection_state = "connected"
    source.connection_id = 2
    source.connection_started_mono = 1.0
    source.connection_received_start = 100_000  # Deliberately irrelevant old epoch.
    count = round(rate_hz) + 1
    source.sample_mono_window.extend(now - 1.0 + i / rate_hz for i in range(count))
    source.last_mono = source.sample_mono_window[-1]
    return source


def test_stable_uses_current_recent_window_not_connection_lifetime_average():
    source = _source_with_recent_rate(185.0)
    with patch("wuji_myo_emg_source.time.monotonic", return_value=100.0):
        assert source.stable(duration_s=1.0, min_hz=180.0, max_age_s=0.5)


def test_stable_rejects_a_genuinely_slow_recent_window():
    source = _source_with_recent_rate(170.0)
    with patch("wuji_myo_emg_source.time.monotonic", return_value=100.0):
        assert not source.stable(duration_s=1.0, min_hz=180.0, max_age_s=0.5)


def test_stable_rejects_stale_recent_delivery():
    source = _source_with_recent_rate(190.0)
    source.last_mono = 99.0
    with patch("wuji_myo_emg_source.time.monotonic", return_value=100.0):
        assert not source.stable(duration_s=1.0, min_hz=180.0, max_age_s=0.5)


def test_default_silence_timeout_is_shorter_than_save_gap_budget():
    source = MyoEmgSource("/bin/true", "/dev/null", "ED:94:F0:1E:61:D3")
    assert 0 < source.silence_timeout_s < 0.5


def test_auto_mac_uses_scanner_result():
    source = MyoEmgSource("/bin/true", "/dev/ttyACM0", "auto")
    completed = SimpleNamespace(
        returncode=0,
        stdout="E3:9D:20:B1:F8:5C\n",
        stderr="",
    )
    with patch("wuji_myo_emg_source.subprocess.run", return_value=completed) as run:
        assert source._resolve_mac() == "E3:9D:20:B1:F8:5C"
    command = run.call_args.args[0]
    assert command[-1] == "--select-mac"
    assert command[command.index("--mac") + 1] == "auto"


def test_auto_mac_surfaces_scanner_failure():
    source = MyoEmgSource("/bin/true", "/dev/ttyACM0", "auto")
    completed = SimpleNamespace(
        returncode=1,
        stdout="",
        stderr="multiple advertising Myos were found",
    )
    with patch("wuji_myo_emg_source.subprocess.run", return_value=completed):
        try:
            source._resolve_mac()
        except RuntimeError as exc:
            assert "multiple advertising Myos" in str(exc)
        else:
            raise AssertionError("scanner failure was not raised")


def test_stable_rejects_pause_followed_by_burst_with_good_average():
    source = _source_with_recent_rate(200.0)
    source.sample_mono_window.clear()
    source.sample_mono_window.extend([99.0 + i * .005 for i in range(60)])
    source.sample_mono_window.extend([99.8 + i * .001 for i in range(201)])
    source.last_mono = 100.0
    with patch('wuji_myo_emg_source.time.monotonic', return_value=100.0):
        assert not source.stable(duration_s=1.0, min_hz=180.0)


def test_rearm_clears_stability_and_latches_fault_after_recovery():
    import io
    import json
    source = _source_with_recent_rate(200.0)
    source.process.stdout = io.StringIO('\n'.join(json.dumps(event) for event in [
        {'kind': 'state', 'state': 'recovering', 'message': 'no RAW EMG'},
        {'kind': 'state', 'state': 'connected', 'message': 'recovered'},
    ]))
    source._read_stdout()
    assert source.connection_state == 'connected'
    assert source.interruption_snapshot() == (1, 'no RAW EMG')
    assert not source.sample_mono_window


def test_status_does_not_shorten_stability_history():
    source = _source_with_recent_rate(200.0)
    source.sample_mono_window.appendleft(90.0)
    with patch('wuji_myo_emg_source.time.monotonic', return_value=100.0):
        source.status()
    assert source.sample_mono_window[0] == 90.0
