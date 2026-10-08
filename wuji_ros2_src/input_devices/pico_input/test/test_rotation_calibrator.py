import math

import numpy as np
import pytest

from pico_input.rotation_calibrator import derive_rotation_axis_mapping


def axis_quaternion(axis, degrees):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    half = math.radians(degrees) / 2.0
    return np.r_[axis * math.sin(half), math.cos(half)]


def test_derives_expected_rotation_mapping():
    samples = {
        "neutral": [0, 0, 0, 1],
        "rm_x_positive": axis_quaternion([0, 0, -1], 20),
        "rm_x_negative": axis_quaternion([0, 0, -1], -20),
        "rm_y_positive": axis_quaternion([-1, 0, 0], 20),
        "rm_y_negative": axis_quaternion([-1, 0, 0], -20),
        "rm_z_positive": axis_quaternion([0, 1, 0], 20),
        "rm_z_negative": axis_quaternion([0, 1, 0], -20),
    }
    result = derive_rotation_axis_mapping(samples)
    np.testing.assert_array_equal(result.rotation_axis_mapping, [
        [0, 0, -1], [-1, 0, 0], [0, 1, 0]])
    assert result.suggested_rotation_scale == pytest.approx(0.25)


def test_rejects_mixed_axis_rotation():
    mixed = axis_quaternion([1, 1, 0], 20)
    samples = {
        "neutral": [0, 0, 0, 1],
        "rm_x_positive": mixed,
        "rm_x_negative": axis_quaternion([1, 1, 0], -20),
        "rm_y_positive": axis_quaternion([0, 1, 0], 20),
        "rm_y_negative": axis_quaternion([0, 1, 0], -20),
        "rm_z_positive": axis_quaternion([0, 0, 1], 20),
        "rm_z_negative": axis_quaternion([0, 0, 1], -20),
    }
    with pytest.raises(ValueError, match="clean single-axis"):
        derive_rotation_axis_mapping(samples)

