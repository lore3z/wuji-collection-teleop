import math

import numpy as np
import pytest

from realman_rm75_output.elbow_arm_angle_mapper import InvalidElbowDirection
from realman_rm75_output.upper_arm_shadow_observer import (
    UpperArmGeometryObserver,
)


def make_observer(**kwargs):
    return UpperArmGeometryObserver(
        shoulder=[0.0, 0.0, 0.0],
        arm_anchor=[0.5, -0.2, 0.0],
        source_to_base=np.eye(3),
        position_scale=1.0,
        direction_ema_alpha=1.0,
        arm_angle_scale=0.25,
        max_arm_angle_step_deg=5.0,
        **kwargs,
    )


def test_initial_sample_rebases_to_rm_upper_arm_anchor():
    observation = make_observer().process([7.0, 8.0, 9.0], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(observation.raw_delta, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(observation.mapped_arm_point, [0.5, -0.2, 0.0])
    np.testing.assert_allclose(observation.projection_point, [0.5, 0.0, 0.0])
    np.testing.assert_allclose(observation.elbow_direction, [0.0, -1.0, 0.0])
    assert observation.offset_m == pytest.approx(0.2)
    assert observation.arm_angle_delta_deg == pytest.approx(0.0)
    assert observation.status == "GOOD"


def test_reset_reference_makes_next_sample_the_new_upper_arm_zero():
    observer = make_observer()
    observer.process([1.0, 2.0, 3.0], [1.0, 0.0, 0.0])
    observer.process([1.0, 2.02, 3.0], [1.0, 0.0, 0.0])

    observer.reset_reference()
    reset = observer.process([8.0, 9.0, 10.0], [1.0, 0.0, 0.0])

    np.testing.assert_allclose(reset.raw_delta, [0.0, 0.0, 0.0])
    assert reset.arm_angle_delta_deg == pytest.approx(0.0)


def test_upper_arm_rotation_produces_scaled_relative_arm_angle():
    observer = make_observer()
    observer.process([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    angle = math.radians(10.0)
    observation = observer.process(
        [0.0, 0.2 * (1.0 - math.cos(angle)), -0.2 * math.sin(angle)],
        [1.0, 0.0, 0.0],
    )
    assert observation.geometric_angle_deg == pytest.approx(10.0)
    assert observation.arm_angle_delta_deg == pytest.approx(2.5)
    assert observation.status == "GOOD"


def test_source_to_base_matrix_maps_raw_position_delta():
    source_to_base = np.array([
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    observer = UpperArmGeometryObserver(
        shoulder=[0.0, 0.0, 0.0],
        arm_anchor=[0.5, -0.2, 0.0],
        source_to_base=source_to_base,
        position_scale=0.5,
        direction_ema_alpha=1.0,
    )
    observer.process([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    observation = observer.process([0.02, 0.04, -0.06], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(
        observation.mapped_arm_point,
        [0.53, -0.21, 0.02],
    )


def test_singular_upper_arm_point_is_rejected_before_direction_exists():
    observer = UpperArmGeometryObserver(
        shoulder=[0.0, 0.0, 0.0],
        arm_anchor=[0.5, 0.0, 0.0],
        source_to_base=np.eye(3),
    )
    with pytest.raises(InvalidElbowDirection, match="too close"):
        observer.process([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])


def test_raw_relocalization_jump_is_held_invalid_and_rebased_continuously():
    observer = make_observer(max_raw_step_m=0.08)
    observer.process([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    before = observer.process([0.0, 0.01, 0.0], [1.0, 0.0, 0.0])
    jumped = observer.process([0.0, 0.60, 0.0], [1.0, 0.0, 0.0])
    assert jumped.status == "HELD_RAW_JUMP"
    assert jumped.raw_step_m == pytest.approx(0.59)
    np.testing.assert_allclose(jumped.mapped_arm_point, before.mapped_arm_point)

    recovered = observer.process([0.0, 0.61, 0.0], [1.0, 0.0, 0.0])
    assert recovered.status == "GOOD"
    np.testing.assert_allclose(
        recovered.mapped_arm_point - before.mapped_arm_point,
        [0.0, 0.01, 0.0], atol=1e-12)
