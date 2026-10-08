import numpy as np

from visualize_glove_trajectory import trajectory_quality


def test_trajectory_quality_separates_message_rate_from_xyz_update_rate():
    timestamps_ns = np.arange(601, dtype=np.int64) * 10_000_000
    xyz = np.zeros((601, 3), dtype=np.float64)
    xyz[:, 0] = (np.arange(601) // 100) * 0.01

    quality = trajectory_quality(xyz, timestamps_ns)

    assert quality["source_hz"] == 100.0
    assert quality["xyz_update_count"] == 6
    assert quality["xyz_update_hz"] == 1.0
    assert quality["adjacent_same_ratio"] == 0.99
    assert np.isclose(quality["path_length_m"], 0.06)
