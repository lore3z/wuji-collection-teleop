import numpy as np
import pytest

from pico_input.pico_tracker_pose_publisher import (
    TrackerSelectionError,
    select_tracker_pose,
)


def test_selects_requested_serial_and_normalizes_quaternion():
    poses = [
        [1, 2, 3, 0, 0, 0, 1],
        [4, 5, 6, 0, 0, 0, 1.01],
    ]
    serial, position, quaternion = select_tracker_pose(
        poses, [b"LEFT", b"RIGHT"], "RIGHT")
    assert serial == "RIGHT"
    np.testing.assert_allclose(position, [4, 5, 6])
    np.testing.assert_allclose(quaternion, [0, 0, 0, 1])


def test_auto_selects_when_exactly_one_tracker_is_available():
    serial, position, quaternion = select_tracker_pose(
        [[0.1, 0.2, 0.3, 0, 0, 0, 1]], ["WRIST"])
    assert serial == "WRIST"
    np.testing.assert_allclose(position, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(quaternion, [0, 0, 0, 1])


def test_requires_serial_when_multiple_trackers_are_available():
    with pytest.raises(TrackerSelectionError, match="tracker_serial is required"):
        select_tracker_pose(
            [[0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1]],
            ["A", "B"])


def test_rejects_missing_serial_and_invalid_pose():
    with pytest.raises(TrackerSelectionError, match="not found"):
        select_tracker_pose([[0, 0, 0, 0, 0, 0, 1]], ["A"], "B")
    with pytest.raises(TrackerSelectionError, match="quaternion norm"):
        select_tracker_pose([[0, 0, 0, 0, 0, 0, 0]], ["A"], "A")
