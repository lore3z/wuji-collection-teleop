import numpy as np
import pytest
from sensor_msgs.msg import JointState

from realman_rm75_output.shadow_joint_real_bridge import (
    EXPECTED_JOINT_NAMES,
    JointMotionRecorder,
    ShadowJointJump,
    ShadowJointDeltaMapper,
    ShadowJumpMonitor,
    StableJointReferenceMonitor,
    TrackingErrorMonitor,
    joint_state_radians,
    real_main,
    wrapped_joint_error_deg,
)


def _joint_state(names=None, positions=None):
    message = JointState()
    message.name = list(names or EXPECTED_JOINT_NAMES)
    message.position = list(positions or np.arange(7, dtype=float))
    return message


def test_joint_state_is_validated_and_reordered_by_name():
    message = _joint_state(
        list(reversed(EXPECTED_JOINT_NAMES)),
        list(reversed(np.arange(7, dtype=float))))
    np.testing.assert_allclose(joint_state_radians(message), np.arange(7))


@pytest.mark.parametrize("message", [
    _joint_state(names=EXPECTED_JOINT_NAMES[:-1], positions=[0.0] * 6),
    _joint_state(names=["bad"] * 7, positions=[0.0] * 7),
    _joint_state(positions=[0.0] * 6 + [float("nan")]),
])
def test_joint_state_rejects_invalid_contract(message):
    with pytest.raises(ValueError):
        joint_state_radians(message)


def test_mapper_applies_half_shadow_delta_around_real_reference():
    shadow_reference = np.zeros(7)
    robot_reference = np.arange(7, dtype=float) * 10.0
    mapper = ShadowJointDeltaMapper(
        shadow_reference, robot_reference, delta_scale=0.5,
        max_offset_deg=3.0, max_shadow_step_deg=5.0)
    target, saturated, requested = mapper.process(
        np.radians([2.0, -2.0, 0.0, 1.0, 0.0, 0.0, 0.0]))
    np.testing.assert_allclose(
        target, robot_reference + [1.0, -1.0, 0.0, 0.5, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(
        requested, [1.0, -1.0, 0.0, 0.5, 0.0, 0.0, 0.0])
    assert not saturated


def test_safe_home_error_uses_shortest_equivalent_joint_angle():
    np.testing.assert_allclose(
        wrapped_joint_error_deg(
            [359.0, 1.0, 180.0, -180.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, -180.0, 180.0, 0.0, 0.0, 0.0]),
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def test_stable_reference_monitor_requires_quiet_consecutive_home_frames():
    monitor = StableJointReferenceMonitor(
        np.zeros(7), home_tolerance_deg=0.5,
        max_step_deg=0.15, required_samples=3)

    assert not monitor.observe([2.0] + [0.0] * 6)["ready"]
    first = monitor.observe([0.1] + [0.0] * 6)
    assert not first["ready"] and first["consecutive"] == 0
    assert not monitor.observe([0.1] + [0.0] * 6)["ready"]
    assert not monitor.observe([0.12] + [0.0] * 6)["ready"]
    ready = monitor.observe([0.11] + [0.0] * 6)
    assert ready["ready"] and ready["consecutive"] == 3

    reset = monitor.observe([0.4] + [0.0] * 6)
    assert not reset["ready"] and reset["consecutive"] == 0


def test_mapper_clamps_total_offset_and_rejects_one_frame_jump():
    mapper = ShadowJointDeltaMapper(
        np.zeros(7), np.zeros(7), delta_scale=0.5,
        max_offset_deg=3.0, max_shadow_step_deg=5.0)
    target, saturated, requested = mapper.process(
        np.radians([4.0] * 7))
    assert not saturated
    np.testing.assert_allclose(target, [2.0] * 7)
    target, saturated, requested = mapper.process(
        np.radians([8.0] * 7))
    assert saturated
    np.testing.assert_allclose(target, [3.0] * 7)
    np.testing.assert_allclose(requested, [4.0] * 7)
    with pytest.raises(ShadowJointJump, match="shadow joint1 jump") as caught:
        mapper.process(np.radians([14.0] * 7))
    assert caught.value.joint_name == "joint1"
    np.testing.assert_allclose(caught.value.step_deg, [6.0] * 7)
    # The rejected sample becomes the new continuity reference, allowing a
    # stable new branch to recover while total output remains clamped.
    target, saturated, _requested = mapper.process(np.radians([14.5] * 7))
    assert saturated
    np.testing.assert_allclose(target, [3.0] * 7)


def test_shadow_jump_monitor_drops_one_but_faults_persistent_or_hard_jump():
    monitor = ShadowJumpMonitor(15.0, 3)
    first = monitor.rejected([0.0, 5.9, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert not first["fault"] and first["consecutive"] == 1
    assert first["joint_name"] == "joint2"
    monitor.accepted()
    assert monitor.consecutive == 0
    assert not monitor.rejected([6.0] + [0.0] * 6)["fault"]
    assert not monitor.rejected([0.0, 0.0, -6.1] + [0.0] * 4)["fault"]
    persistent = monitor.rejected([0.0] * 6 + [6.2])
    assert persistent["fault"] and not persistent["hard"]
    monitor.accepted()
    hard = monitor.rejected([0.0, 0.0, 0.0, 15.0, 0.0, 0.0, 0.0])
    assert hard["fault"] and hard["hard"]
    np.testing.assert_allclose(
        monitor.maximum_by_joint, [6.0, 5.9, 6.1, 15.0, 0.0, 0.0, 6.2])


def test_mapper_supports_six_and_twelve_degree_range_profiles():
    for max_offset, expected_at_twenty, expected_at_twenty_eight in [
            (6.0, 6.0, 6.0),
            (12.0, 10.0, 12.0),
    ]:
        mapper = ShadowJointDeltaMapper(
            np.zeros(7), np.zeros(7), delta_scale=0.5,
            max_offset_deg=max_offset, max_shadow_step_deg=5.0)
        for degrees in (4.0, 8.0, 12.0, 16.0, 20.0):
            target, _, _ = mapper.process(
                np.radians([degrees, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        assert target[0] == pytest.approx(expected_at_twenty)
        for degrees in (24.0, 28.0):
            target, saturated, requested = mapper.process(
                np.radians([degrees, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        assert target[0] == pytest.approx(expected_at_twenty_eight)
        assert requested[0] == pytest.approx(14.0)
        assert saturated


def test_mapper_can_disable_relative_cap_for_fake_diagnostics():
    mapper = ShadowJointDeltaMapper(
        np.zeros(7), np.zeros(7), delta_scale=1.0,
        max_offset_deg=12.0, max_shadow_step_deg=20.0,
        offset_limit_enabled=False)

    target, saturated, requested = mapper.process(
        np.radians([15.0, -15.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

    np.testing.assert_allclose(target[:2], [15.0, -15.0])
    np.testing.assert_allclose(requested[:2], [15.0, -15.0])
    assert not saturated


def test_motion_recorder_reports_green_model_extrema_and_travel():
    recorder = JointMotionRecorder([10.0] * 7)
    recorder.observe([10.0] * 7)
    recorder.observe([12.0] * 7)
    recorder.observe([9.0] * 7)
    report = recorder.summary()

    assert report["samples"] == 3
    np.testing.assert_allclose(report["absolute_min_deg"], [9.0] * 7)
    np.testing.assert_allclose(report["absolute_max_deg"], [12.0] * 7)
    np.testing.assert_allclose(report["offset_min_deg"], [-1.0] * 7)
    np.testing.assert_allclose(report["offset_max_deg"], [2.0] * 7)
    np.testing.assert_allclose(report["peak_abs_offset_deg"], [2.0] * 7)
    np.testing.assert_allclose(report["offset_span_deg"], [3.0] * 7)
    np.testing.assert_allclose(report["total_travel_deg"], [5.0] * 7)


def test_tracking_error_requires_persistence_and_resets():
    monitor = TrackingErrorMonitor(1.0, 2.0, 3)
    assert monitor.process([0.0] * 7, [1.5] * 7)["warning"]
    assert not monitor.process([0.0] * 7, [2.1] * 7)["fault"]
    assert not monitor.process([0.0] * 7, [2.1] * 7)["fault"]
    assert monitor.process([0.0] * 7, [2.1] * 7)["fault"]
    assert monitor.process([0.0] * 7, [0.0] * 7)["consecutive_faults"] == 0


def test_real_cli_refuses_without_both_motion_flags(capsys):
    assert real_main([]) == 2
    assert "REFUSED" in capsys.readouterr().err
