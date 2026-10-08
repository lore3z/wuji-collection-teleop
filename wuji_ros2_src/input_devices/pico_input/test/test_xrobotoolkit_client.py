from types import SimpleNamespace

import pico_input.xrobotoolkit_client as client_module


def test_timestamp_falls_back_when_motion_specific_sdk_value_is_zero(
        monkeypatch):
    fake_sdk = SimpleNamespace(
        get_motion_timestamp_ns=lambda: 0,
        get_time_stamp_ns=lambda: 1_787_567_589_197_859_072,
    )
    monkeypatch.setattr(client_module, "_USE_PYBIND", True)
    monkeypatch.setattr(client_module, "xrt", fake_sdk)
    client = client_module.XRoboToolkitClient()
    client._connected = True
    assert client.get_time_stamp_ns() == 1_787_567_589_197_859_072


def test_motion_specific_timestamp_remains_preferred_when_available(
        monkeypatch):
    fake_sdk = SimpleNamespace(
        get_motion_timestamp_ns=lambda: 123,
        get_time_stamp_ns=lambda: 456,
    )
    monkeypatch.setattr(client_module, "_USE_PYBIND", True)
    monkeypatch.setattr(client_module, "xrt", fake_sdk)
    client = client_module.XRoboToolkitClient()
    client._connected = True
    assert client.get_time_stamp_ns() == 123
