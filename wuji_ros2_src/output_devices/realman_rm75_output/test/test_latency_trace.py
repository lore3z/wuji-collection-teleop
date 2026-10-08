import pytest

from realman_rm75_output.latency_trace import (
    PipelineLatencyStatistics,
    make_shadow_trace,
    parse_pico_trace,
    parse_shadow_trace,
)


def test_trace_schema_preserves_every_pipeline_timestamp_exactly():
    pico = parse_pico_trace([1, 10, 20, 30, 40, 50])
    pico.update({"ik_receive_ns": 60, "ik_start_ns": 61, "ik_end_ns": 70})
    encoded = make_shadow_trace(80, pico, 75)
    assert parse_shadow_trace(encoded) == {
        "shadow_key_ns": 80,
        "pose_key_ns": 10,
        "sdk_sample_ns": 20,
        "pc_poll_start_ns": 30,
        "pc_read_complete_ns": 40,
        "pico_publish_ns": 50,
        "ik_receive_ns": 60,
        "ik_start_ns": 61,
        "ik_end_ns": 70,
        "shadow_publish_ns": 75,
    }


@pytest.mark.parametrize("values", [[], [2, 1, 2, 3, 4, 5], [1, 2]])
def test_invalid_pico_trace_is_rejected(values):
    with pytest.raises(ValueError):
        parse_pico_trace(values)


def test_pipeline_statistics_report_each_stage_in_milliseconds():
    trace = parse_shadow_trace(make_shadow_trace(90, {
        "pose_key_ns": 1,
        "sdk_sample_ns": 1_000_000,
        "pc_poll_start_ns": 2_000_000,
        "pc_read_complete_ns": 3_000_000,
        "pico_publish_ns": 4_000_000,
        "ik_receive_ns": 5_000_000,
        "ik_start_ns": 6_000_000,
        "ik_end_ns": 8_000_000,
    }, 10_000_000))
    latest = PipelineLatencyStatistics().observe(trace, 13_000_000)
    assert latest == pytest.approx({
        "pico_to_pc": 2.0,
        "pico_to_pc_relative": 0.0,
        "pc_sdk_read": 1.0,
        "pc_to_ik_receive": 2.0,
        "ik_queue": 1.0,
        "ik_compute": 2.0,
        "ik_to_shadow": 2.0,
        "shadow_to_bridge": 3.0,
        "pc_to_bridge": 11.0,
        "pico_to_bridge": 12.0,
        "pico_to_bridge_relative": 10.0,
    })


def test_unsynchronised_sdk_clock_does_not_corrupt_pc_stage_statistics():
    trace = parse_shadow_trace(make_shadow_trace(90, {
        "pose_key_ns": 1,
        "sdk_sample_ns": 99_000_000_000,
        "pc_poll_start_ns": 2_000_000,
        "pc_read_complete_ns": 3_000_000,
        "pico_publish_ns": 4_000_000,
        "ik_receive_ns": 5_000_000,
        "ik_start_ns": 6_000_000,
        "ik_end_ns": 8_000_000,
    }, 10_000_000))
    stats = PipelineLatencyStatistics()
    latest = stats.observe(trace, 13_000_000)
    assert "pico_to_pc" not in latest
    assert latest["pico_to_pc_relative"] == pytest.approx(0.0)
    assert latest["pico_to_bridge_relative"] == pytest.approx(10.0)
    assert latest["pc_to_bridge"] == pytest.approx(11.0)
    assert stats.summary()["sdk_clock_invalid_samples"] == 1
