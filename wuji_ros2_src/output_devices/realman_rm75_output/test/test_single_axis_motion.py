import numpy as np
import pytest

from realman_rm75_output.motion_safety_gate import MotionSafetyGate, MotionState
from realman_rm75_output.movep_backend import FakeMovePBackend
from realman_rm75_output.single_axis_target_limiter import SingleAxisTargetLimiter


def limiter():
    return SingleAxisTargetLimiter([0.2, -0.1, 0.3], [0, 0, 0, 1], 0.005, 0.005)


def test_limiter_locks_y_z_orientation_and_steps_025_mm_at_20hz():
    position, quaternion = limiter().step([1.0, 9.0, 9.0], 0.05)
    np.testing.assert_allclose(position, [0.20025, -0.1, 0.3])
    np.testing.assert_allclose(quaternion, [0, 0, 0, 1])


def test_limiter_nominal_step_cap_is_not_expanded_by_timer_jitter():
    item = SingleAxisTargetLimiter(
        [0.2, -0.1, 0.3], [0, 0, 0, 1], 0.005, 0.005,
        max_step_m=0.00025)
    position, _ = item.step([1.0, 0, 0], 0.051)
    assert position[0] - 0.2 == pytest.approx(0.00025)


def test_limiter_never_exceeds_five_mm():
    item = limiter()
    for _ in range(100):
        position, _ = item.step([1.0, 0, 0], 0.05)
    assert position[0] == pytest.approx(0.205)


def test_gate_requires_deadman_and_fault_is_latched():
    gate = MotionSafetyGate()
    assert not gate.activate_if_ready(True)
    gate.set_deadman(True)
    assert gate.activate_if_ready(True)
    gate.trip("timeout")
    gate.set_deadman(False)
    assert gate.state is MotionState.FAULT
    gate.reset()
    assert gate.state is MotionState.DISARMED


def test_fault_reset_requires_deadman_release():
    gate = MotionSafetyGate()
    gate.set_deadman(True)
    gate.trip("bad pose")
    with pytest.raises(RuntimeError, match="release deadman"):
        gate.reset()


def test_fake_backend_only_records():
    backend = FakeMovePBackend()
    assert backend.send_pose([0.2, 0, 0.3], [0, 0, 0, 1], 1.0) == 0
    assert len(backend.sent) == 1
