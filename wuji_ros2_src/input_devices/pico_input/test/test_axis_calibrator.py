import numpy as np
import pytest

from pico_input.axis_calibrator import derive_axis_mapping


def test_derives_expected_signed_permutation():
    samples = {
        "forward": [0.0, 0.0, 0.05],
        "backward": [0.0, 0.0, -0.05],
        "right": [0.05, 0.0, 0.0],
        "left": [-0.05, 0.0, 0.0],
        "up": [0.0, 0.05, 0.0],
        "down": [0.0, -0.05, 0.0],
    }
    result = derive_axis_mapping(samples)
    np.testing.assert_array_equal(result.axis_mapping, [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    assert result.suggested_position_scale == pytest.approx(0.4)


def test_rejects_reused_source_axis():
    samples = {
        "forward": [0.05, 0.0, 0.0], "backward": [-0.05, 0.0, 0.0],
        "left": [0.06, 0.0, 0.0], "right": [-0.06, 0.0, 0.0],
        "up": [0.0, 0.05, 0.0], "down": [0.0, -0.05, 0.0],
    }
    with pytest.raises(ValueError, match="three distinct"):
        derive_axis_mapping(samples)

