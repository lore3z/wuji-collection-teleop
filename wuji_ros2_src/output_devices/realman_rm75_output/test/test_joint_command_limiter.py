import numpy as np
import pytest

from realman_rm75_output.joint_command_limiter import JointCommandLimiter
from realman_rm75_output.movej_backend import FakeMoveJBackend


def test_joint_command_is_bounded_to_one_degree_and_point_zero_two_per_tick():
    limiter = JointCommandLimiter([0] * 7, 1.0, 1.0, 0.02)
    first = limiter.step([100] * 7)
    np.testing.assert_allclose(first, [0.02] * 7)
    for _ in range(100):
        last = limiter.step([100] * 7)
    np.testing.assert_allclose(last, [1.0] * 7)


def test_return_home_uses_same_limited_trajectory():
    limiter = JointCommandLimiter([10] * 7, 1.0, 1.0, 0.02)
    limiter.step([11] * 7)
    returned = limiter.step_home()
    np.testing.assert_allclose(returned, [10] * 7)
    assert limiter.at_home()


def test_invalid_joint_command_is_rejected():
    limiter = JointCommandLimiter([0] * 7)
    with pytest.raises(ValueError):
        limiter.step([0] * 6)


def test_fake_movej_backend_records_without_robot():
    backend = FakeMoveJBackend()
    assert backend.send([0] * 7) == 0
    assert backend.sent == [[0] * 7]
