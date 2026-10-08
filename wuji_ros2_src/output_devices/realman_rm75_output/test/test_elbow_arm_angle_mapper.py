import math

import numpy as np
import pytest

from realman_rm75_output.elbow_arm_angle_mapper import (
    InvalidElbowDirection,
    RelativeArmAngleMapper,
    elbow_swivel_angle_rad,
)


def test_signed_swivel_angle_uses_shoulder_wrist_axis():
    wrist = [1.0, 0.0, 0.0]
    assert math.degrees(elbow_swivel_angle_rad(wrist, [0, -1, 0])) == pytest.approx(0)
    assert math.degrees(elbow_swivel_angle_rad(wrist, [0, 0, -1])) == pytest.approx(90)


def test_relative_mapping_rebases_to_rm_angle_and_scales_delta():
    mapper = RelativeArmAngleMapper(140.0, scale=0.25, max_step_deg=5.0)
    assert mapper.process([1, 0, 0], [0, -1, 0]) == pytest.approx(140.0)
    angle = math.radians(20)
    direction = [0, -math.cos(angle), -math.sin(angle)]
    assert mapper.process([1, 0, 0], direction) == pytest.approx(145.0)


def test_angle_unwrap_is_continuous_across_pi():
    mapper = RelativeArmAngleMapper(10.0, scale=0.25, max_step_deg=5.0)
    a = math.radians(179)
    mapper.process([1, 0, 0], [0, -math.cos(a), -math.sin(a)])
    b = math.radians(-179)
    assert mapper.process([1, 0, 0], [0, -math.cos(b), -math.sin(b)]) == pytest.approx(10.5)


def test_large_arm_angle_step_is_rejected():
    mapper = RelativeArmAngleMapper(0.0, scale=1.0, max_step_deg=5.0)
    mapper.process([1, 0, 0], [0, -1, 0])
    with pytest.raises(InvalidElbowDirection, match="target step"):
        mapper.process([1, 0, 0], [0, 0, -1])
    # The discontinuous input becomes the new basis while the RM target holds,
    # so the following stable frame recovers instead of rejection-latching.
    assert mapper.process([1, 0, 0], [0, 0, -1]) == pytest.approx(0.0)


def test_moving_wrist_axis_does_not_use_a_flipping_absolute_basis():
    mapper = RelativeArmAngleMapper(140.0, scale=0.25, max_step_deg=5.0)
    assert mapper.process([1, 0, 0], [0, 0, -1]) == pytest.approx(140.0)
    # Move the shoulder-wrist axis almost onto the old fixed -Y reference;
    # a transported direction remains continuous.
    target = mapper.process([0.01, -1.0, 0], [0, 0, -1])
    assert target == pytest.approx(140.0)


@pytest.mark.parametrize("wrist,direction", [
    ([0, 0, 0], [0, 1, 0]),
    ([1, 0, 0], [1, 0, 0]),
    ([np.nan, 0, 0], [0, 1, 0]),
])
def test_singular_or_invalid_geometry_is_rejected(wrist, direction):
    with pytest.raises(InvalidElbowDirection):
        elbow_swivel_angle_rad(wrist, direction)
