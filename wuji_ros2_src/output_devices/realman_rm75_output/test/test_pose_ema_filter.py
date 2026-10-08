import math

import numpy as np

from realman_rm75_output.pose_ema_filter import PoseEMAFilter
from realman_rm75_output.pose_mapper import quaternion_angle_rad


def test_position_ema_uses_time_constant():
    ema = PoseEMAFilter(0.05, 0.05)
    ema.reset([0, 0, 0], [0, 0, 0, 1])
    position, _ = ema.step([1, 0, 0], [0, 0, 0, 1], 0.01)
    expected = 1.0 - math.exp(-0.01 / 0.05)
    np.testing.assert_allclose(position, [expected, 0, 0])


def test_position_ema_is_rate_independent_for_equal_elapsed_time():
    slow = PoseEMAFilter(0.05, 0.05)
    fast = PoseEMAFilter(0.05, 0.05)
    slow.reset([0, 0, 0], [0, 0, 0, 1])
    fast.reset([0, 0, 0], [0, 0, 0, 1])
    for _ in range(5):
        slow_position, _ = slow.step([1, 0, 0], [0, 0, 0, 1], 0.02)
    for _ in range(10):
        fast_position, _ = fast.step([1, 0, 0], [0, 0, 0, 1], 0.01)
    np.testing.assert_allclose(slow_position, fast_position, atol=1e-12)


def test_quaternion_ema_takes_shortest_path():
    ema = PoseEMAFilter(0.05, 0.05)
    start = np.array([0.0, 0.0, 0.0, 1.0])
    target = np.array([0.0, 0.0, math.sin(math.pi/4), math.cos(math.pi/4)])
    ema.reset([0, 0, 0], start)
    _, positive = ema.step([0, 0, 0], target, 0.01)
    ema.reset([0, 0, 0], start)
    _, negative = ema.step([0, 0, 0], -target, 0.01)
    assert quaternion_angle_rad(positive, negative) < 1e-12


def test_reset_discards_previous_filter_state():
    ema = PoseEMAFilter(0.05, 0.05)
    ema.reset([0, 0, 0], [0, 0, 0, 1])
    ema.step([1, 0, 0], [0, 0, 1, 0], 0.01)
    ema.reset([0.2, 0.3, 0.4], [0, 0, 0, 1])
    np.testing.assert_allclose(ema.position, [0.2, 0.3, 0.4])
    np.testing.assert_allclose(ema.quaternion, [0, 0, 0, 1])
