import pytest

from realman_rm75_output.rm75_6dof_teleop import RM756DoFTeleop


def _guard(phase_started_at=0.0):
    node = object.__new__(RM756DoFTeleop)
    node.phase_started_at = phase_started_at
    node.position_error_started_at = None
    node.position_error_soft_limit_m = 0.040
    node.position_error_hard_limit_m = 0.080
    node.position_error_grace_sec = 0.50
    node.position_error_hold_sec = 0.25
    return node


def test_tracking_error_grace_accepts_normal_low_follow_lag():
    node = _guard()
    node._check_position_tracking_error(0.060, 0.49)
    assert node.position_error_started_at is None


def test_tracking_error_hard_limit_is_immediate_even_during_grace():
    node = _guard()
    with pytest.raises(RuntimeError, match="hard limit"):
        node._check_position_tracking_error(0.081, 0.10)


def test_tracking_error_soft_limit_must_be_sustained():
    node = _guard()
    node._check_position_tracking_error(0.050, 0.60)
    node._check_position_tracking_error(0.050, 0.84)
    with pytest.raises(RuntimeError, match="stayed above"):
        node._check_position_tracking_error(0.050, 0.86)


def test_tracking_error_recovery_clears_hold_timer():
    node = _guard()
    node._check_position_tracking_error(0.050, 0.60)
    node._check_position_tracking_error(0.020, 0.70)
    assert node.position_error_started_at is None
