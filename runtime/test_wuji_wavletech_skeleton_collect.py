from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wuji_wavletech_skeleton_collect import build_episode, nearest_indices, write_episode


def test_nearest_indices_uses_earlier_sample_on_tie() -> None:
    indices, ages = nearest_indices(
        np.asarray([100, 200, 300], dtype=np.int64),
        np.asarray([150, 280], dtype=np.int64),
    )
    np.testing.assert_array_equal(indices, [0, 2])
    np.testing.assert_array_equal(ages, [-50, 20])


def test_build_episode_preserves_native_streams_and_aligns_to_skeleton() -> None:
    base_ns = 1_800_000_000_000_000_000
    skeleton = [
        (base_ns // 1_000 + i * 10_000, i, np.full((21, 3), i, dtype=np.float32))
        for i in range(4)
    ]
    emg = [
        (base_ns + i * 1_000_000, i + 1, 0, np.full(8, i, dtype=np.int32), base_ns + i * 1_000_000)
        for i in range(32)
    ]
    imu = [(base_ns, 1, np.arange(6, dtype=np.float32), base_ns)]
    episode = build_episode(
        skeleton, emg, imu, instruction="pinch", max_alignment_age_ms=2.0,
    )
    assert episode.data["wuji_hand_skeleton_mediapipe"].shape == (4, 21, 3)
    assert episode.data["right_forearm_emg_wavletech"].shape == (4, 8)
    assert episode.streams["right_forearm_emg_wavletech_raw"].shape == (32, 8)
    assert episode.streams["right_forearm_imu_wavletech_raw"].shape == (1, 6)
    np.testing.assert_array_equal(
        episode.data["right_forearm_emg_wavletech"][:, 0], [0, 10, 20, 30]
    )


def test_write_episode_round_trip_and_training_sidecar(tmp_path: Path) -> None:
    import zarr

    base_ns = 1_800_000_000_000_000_000
    skeleton = [
        (
            base_ns // 1_000 + i * 10_000,
            i,
            np.full((21, 3), i / 100.0, dtype=np.float32),
            base_ns + i * 10_000_000 + 1_000_000,
        )
        for i in range(4)
    ]
    emg = [
        (base_ns + i * 1_000_000, i + 1, 0, np.full(8, i, dtype=np.int32), base_ns + i * 1_000_000)
        for i in range(32)
    ]
    episode = build_episode(skeleton, emg, [], instruction="open", max_alignment_age_ms=2.0)
    path = tmp_path / "episode_000000.zarr"
    write_episode(path, episode)
    root = zarr.open_group(str(path), mode="r")
    assert root.attrs["format"] == "wuji_wavletech_emg_skeleton_zarr_v1"
    assert root["data/right_forearm_emg_wavletech"].shape == (4, 8)
    assert root["audit/wuji_hand_skeleton_mediapipe"].shape == (4, 21, 3)
    with np.load(path / "calibration.npz", allow_pickle=False) as sidecar:
        assert sidecar["emg"].shape == (31, 8)
        assert sidecar["raw_glove_skeleton"].shape == (4, 21, 3)
        assert sidecar["emg_time_seconds"][0] == 0.0


def test_build_episode_rejects_non_overlapping_timelines() -> None:
    skeleton = [(1_000_000 + i * 1_000, i, np.zeros((21, 3), np.float32)) for i in range(2)]
    emg = [(5_000_000_000 + i, i, 0, np.zeros(8, np.int32), 5_000_000_000 + i) for i in range(2)]
    try:
        build_episode(skeleton, emg, [], instruction="", max_alignment_age_ms=10.0)
    except ValueError as exc:
        assert "没有重叠时间段" in str(exc)
    else:
        raise AssertionError("non-overlapping episode was accepted")
