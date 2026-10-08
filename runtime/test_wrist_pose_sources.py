import numpy as np

from wuji_glove_d435_ftp1_collect import _wrist_pose_with_tracker_translation


def test_wrist_pose_uses_tracker_xyz_and_wuji_orientation():
    wuji = np.asarray([
        [0.10, -0.15, 0.05, 0.1, 0.2, 0.3],
        [0.10, -0.15, 0.05, 0.4, 0.5, 0.6],
    ], dtype=np.float32)
    tracker = np.asarray([
        [-1.0, -0.3, -0.7, 1.1, 1.2, 1.3],
        [-0.6, -0.2, -0.5, 1.4, 1.5, 1.6],
    ], dtype=np.float32)
    result = _wrist_pose_with_tracker_translation(wuji, tracker)
    np.testing.assert_array_equal(result[:, :3], tracker[:, :3])
    np.testing.assert_array_equal(result[:, 3:], wuji[:, 3:])


def test_wrist_pose_source_shapes_must_match():
    with np.testing.assert_raises(ValueError):
        _wrist_pose_with_tracker_translation(
            np.zeros((2, 6), dtype=np.float32),
            np.zeros((3, 6), dtype=np.float32),
        )
