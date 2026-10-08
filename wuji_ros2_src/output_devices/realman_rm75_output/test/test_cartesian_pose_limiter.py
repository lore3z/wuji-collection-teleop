import math
import numpy as np
from realman_rm75_output.cartesian_pose_limiter import CartesianPoseLimiter


def test_6dof_limiter_caps_translation_range_and_rate():
    limiter = CartesianPoseLimiter([0, 0, 0], [0, 0, 0, 1], 0.1, 0.02,
                                   math.radians(20))
    position, _ = limiter.step([1, 1, 1], [0, 0, 0, 1], 0.05)
    assert np.linalg.norm(position) <= 0.001 + 1e-12
    for _ in range(1000):
        position, _ = limiter.step([1, 1, 1], [0, 0, 0, 1], 0.05)
    assert np.all(position <= 0.1 + 1e-12)


def test_acceleration_limited_translation_at_80_hz():
    dt = 1.0 / 80.0
    limiter = CartesianPoseLimiter(
        [0, 0, 0], [0, 0, 0, 1], 1.0, 0.25,
        math.radians(20), 3.6)
    previous_position = np.zeros(3)
    previous_velocity = np.zeros(3)
    for _ in range(20):
        position, _ = limiter.step([1, 0, 0], [0, 0, 0, 1], dt)
        velocity = (position - previous_position) / dt
        acceleration = np.linalg.norm(velocity - previous_velocity) / dt
        assert np.linalg.norm(velocity) <= 0.25 + 1e-12
        assert acceleration <= 3.6 + 1e-9
        previous_position = position
        previous_velocity = velocity


def test_acceleration_limiter_reset_clears_velocity():
    limiter = CartesianPoseLimiter(
        [0, 0, 0], [0, 0, 0, 1], 1.0, 0.25,
        math.radians(20), 3.6)
    limiter.step([1, 0, 0], [0, 0, 0, 1], 0.0125)
    limiter.reset()
    np.testing.assert_array_equal(limiter.linear_velocity, np.zeros(3))
