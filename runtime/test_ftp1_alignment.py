import numpy as np

from wuji_glove_d435_ftp1_collect import (
    ADDUCTION_CHANNELS,
    _canonical_with_stable_adduction,
    _emg_corrected_timestamps_ns,
    _emg_episode_clock_arrays,
    _emg_alignment_drop_run_limit,
    _is_full_identity_selection,
    _can_drop_sparse_invalid_rows,
    _only_tracker_rows_invalid,
    _only_wuji_rows_invalid,
    _max_false_run,
    _nearest_indices,
    _timeline_quality,
    _tracker_xyz_quality,
    _joint_preflight_quality,
    _live_canonical_latest_ns,
)
from wuji_ftp1_hand_geometry import FTP1_HAND_NAMES


def test_live_canonical_watermark_allows_empty_ego_at_save_boundary():
    assert _live_canonical_latest_ns(
        {"ego": []}, capture_ego=True, skeleton_frames=[]
    ) == 0


def test_live_canonical_watermark_uses_latest_ego_timestamp():
    assert _live_canonical_latest_ns(
        {"ego": [100, 200]}, capture_ego=True, skeleton_frames=[]
    ) == 200


def test_emg_episode_phase_correction_removes_accumulated_clock_offset():
    base_ns = 1_700_000_000_000_000_000
    frames = []
    for index in range(400):
        raw_ns = base_ns + 350_000_000 + index * 5_000_000
        packet_latency_ns = 5_000_000 if index % 2 == 0 else 200_000
        if index % 47 == 0:
            packet_latency_ns += 180_000_000
        arrival_ns = base_ns + index * 5_000_000 + packet_latency_ns
        frames.append((raw_ns, index, 0, np.zeros(8, dtype=np.int8), arrival_ns))

    corrected_ns, offset_ns = _emg_corrected_timestamps_ns(frames)

    assert -351_000_000 < offset_ns < -349_000_000
    np.testing.assert_array_equal(np.diff(corrected_ns), np.full(399, 5_000_000))
    assert abs(corrected_ns[200] - (base_ns + 200 * 5_000_000)) < 1_000_000


def test_emg_clock_correction_removes_linear_oscillator_drift():
    base_ns = 1_700_000_000_000_000_000
    frames = []
    # Reconstructed bridge clock is 200 Hz, while host-observed acquisition is
    # 196 Hz. Alternating packet latency and sparse stalls must not become EMG
    # sample gaps in the corrected acquisition timeline.
    for index in range(4000):
        raw_ns = base_ns + index * 5_000_000
        arrival_ns = base_ns + round(index * 1_000_000_000 / 196.0)
        arrival_ns += 300_000 if index % 2 else 4_000_000
        if index % 401 == 0:
            arrival_ns += 150_000_000
        frames.append((raw_ns, index, 0, np.zeros(8, dtype=np.int8), arrival_ns))

    corrected_ns, _offset_ns = _emg_corrected_timestamps_ns(frames)

    assert 195.5 < (len(corrected_ns) - 1) / ((corrected_ns[-1] - corrected_ns[0]) / 1e9) < 196.5
    assert abs(corrected_ns[-1] - (base_ns + round(3999 * 1_000_000_000 / 196.0))) < 3_000_000


def test_emg_episode_filter_preserves_affine_clock_slope():
    base_ns = 1_700_000_000_000_000_000
    frames = []
    for index in range(2000):
        raw_ns = base_ns + index * 5_000_000
        arrival_ns = base_ns + round(index * 1_000_000_000 / 196.0) + 500_000
        frames.append((raw_ns, index, 0, np.zeros(8, dtype=np.int8), arrival_ns))
    keep = np.ones(len(frames), dtype=bool)
    keep[:10] = False
    keep[-10:] = False

    selected, corrected, raw, arrival, _ = _emg_episode_clock_arrays(frames, keep)

    assert len(selected) == len(corrected) == len(raw) == len(arrival) == 1980
    corrected_hz = (len(corrected) - 1) / ((corrected[-1] - corrected[0]) / 1e9)
    raw_hz = (len(raw) - 1) / ((raw[-1] - raw[0]) / 1e9)
    assert 195.5 < corrected_hz < 196.5
    assert 199.5 < raw_hz < 200.5


def test_canonical_replaces_only_unstable_adduction_channels():
    published = np.arange(63, dtype=np.float32).reshape(3, 21)
    stable = published + 1000.0

    canonical = _canonical_with_stable_adduction(published, stable)

    np.testing.assert_array_equal(
        canonical[:, ADDUCTION_CHANNELS], stable[:, ADDUCTION_CHANNELS]
    )
    other = np.setdiff1d(np.arange(21), ADDUCTION_CHANNELS)
    np.testing.assert_array_equal(canonical[:, other], published[:, other])


def test_joint_preflight_reports_the_three_locked_source_channels(monkeypatch):
    joints = np.zeros((5, 21), dtype=np.float32)
    joints[:, FTP1_HAND_NAMES.index("index_pip")] = 0.0087
    joints[:, FTP1_HAND_NAMES.index("index_dip")] = 0.0087
    joints[:, FTP1_HAND_NAMES.index("middle_pip")] = 0.0087
    frames = [(index, index, np.zeros((21, 3), dtype=np.float32)) for index in range(5)]
    monkeypatch.setattr(
        "wuji_glove_d435_ftp1_collect.ftp1_right_hand_joints_from_mediapipe",
        lambda _landmarks: joints,
    )

    frozen, ranges = _joint_preflight_quality(frames, np.deg2rad(3.0))

    assert frozen == ["index_pip", "index_dip", "middle_pip"]
    assert all(value == 0.0 for value in ranges.values())


def test_joint_preflight_accepts_real_flexion(monkeypatch):
    joints = np.zeros((5, 21), dtype=np.float32)
    for name in ("index_pip", "index_dip", "middle_pip"):
        joints[:, FTP1_HAND_NAMES.index(name)] = np.linspace(0.01, 0.5, 5)
    frames = [(index, index, np.zeros((21, 3), dtype=np.float32)) for index in range(5)]
    monkeypatch.setattr(
        "wuji_glove_d435_ftp1_collect.ftp1_right_hand_joints_from_mediapipe",
        lambda _landmarks: joints,
    )

    frozen, ranges = _joint_preflight_quality(frames, np.deg2rad(3.0))

    assert frozen == []
    assert all(value > np.deg2rad(20.0) for value in ranges.values())


def test_dropping_final_ego_row_is_not_mistaken_for_full_identity():
    assert _is_full_identity_selection(np.arange(388), 388)
    assert not _is_full_identity_selection(np.arange(387), 388)


def test_lower_rate_numeric_tracker_aligns_to_nearest_real_samples():
    """A 57.3 Hz pose source must remain usable on a 59.7 Hz RGB axis."""
    target_us = np.rint(np.arange(240) * 1_000_000 / 59.7).astype(np.int64)
    source_us = np.rint(np.arange(230) * 1_000_000 / 57.3).astype(np.int64)

    indices, ages_us = _nearest_indices(source_us, target_us)

    assert len(indices) == len(target_us)
    assert np.all((0 <= indices) & (indices < len(source_us)))
    assert len(np.unique(indices)) == len(source_us)
    assert len(indices) - len(np.unique(indices)) == 10
    assert np.max(np.abs(ages_us)) < 20_000
    np.testing.assert_array_equal(ages_us, source_us[indices] - target_us)


def test_wrist_xyz_quality_detects_fresh_stamps_on_cached_positions():
    frames = []
    for index in range(720):
        update = index // 60
        pose = np.asarray([update * 0.01, 0, 0, 0, 0, 0], dtype=np.float32)
        frames.append((index * 1_000_000_000 // 60, index, pose))

    quality = _tracker_xyz_quality(frames)

    assert quality["adjacent_same_count"] == 708
    assert quality["xyz_update_count"] == 11
    assert quality["xyz_update_hz"] < 1.0
    assert quality["xyz_span_m"] > 0.10


def test_static_camera_xyz_is_described_without_being_misclassified_as_motion():
    pose = np.asarray([1, 2, 3, 0, 0, 0], dtype=np.float32)
    frames = [(index * 10_000_000, index, pose) for index in range(100)]

    quality = _tracker_xyz_quality(frames)

    assert quality["adjacent_same_ratio"] == 1.0
    assert quality["xyz_update_count"] == 0
    assert quality["xyz_span_m"] == 0.0


def test_sparse_alignment_drop_run_is_measured_without_hiding_outage():
    sparse = np.ones(210, dtype=bool)
    sparse[103] = False
    assert _max_false_run(sparse) == 1
    assert _can_drop_sparse_invalid_rows(sparse, max_drop_ratio=0.03)

    outage = np.ones(210, dtype=bool)
    outage[100:104] = False
    assert _max_false_run(outage) == 4
    assert not _can_drop_sparse_invalid_rows(outage, max_drop_ratio=0.03)

    too_many = np.ones(210, dtype=bool)
    too_many[::20] = False
    assert _max_false_run(too_many) == 1
    assert not _can_drop_sparse_invalid_rows(too_many, max_drop_ratio=0.03)


def test_sparse_tracker_only_invalid_rows_use_auxiliary_source_policy():
    good = np.ones(2468, dtype=bool)
    tracker_camera = good.copy()
    tracker_wrist = good.copy()
    tracker_camera[1200:1206] = False
    tracker_wrist[1200:1206] = False

    assert _only_tracker_rows_invalid(
        tracker_camera,
        tracker_wrist,
        (good, good, good, good, good, good, good, None),
    )
    combined = tracker_camera & tracker_wrist
    assert _can_drop_sparse_invalid_rows(
        combined,
        max_drop_ratio=0.05,
        max_drop_run=12,
    )


def test_tracker_policy_does_not_hide_an_invalid_required_source():
    good = np.ones(100, dtype=bool)
    tracker_camera = good.copy()
    tracker_camera[50] = False
    bad_emg = good.copy()
    bad_emg[70] = False

    assert not _only_tracker_rows_invalid(
        tracker_camera,
        good,
        (good, good, good, good, good, good, bad_emg, None),
    )


def test_short_whole_wuji_pause_uses_recovery_policy():
    good = np.ones(2468, dtype=bool)
    wuji_masks = []
    for _ in range(5):
        mask = good.copy()
        mask[1200:1216] = False
        wuji_masks.append(mask)

    assert _only_wuji_rows_invalid(
        tuple(wuji_masks),
        (good, good, None, good, good),
    )
    combined = np.logical_and.reduce(wuji_masks)
    assert _can_drop_sparse_invalid_rows(
        combined,
        max_drop_ratio=0.05,
        max_drop_run=30,
    )

    too_long = good.copy()
    too_long[1200:1231] = False
    assert not _can_drop_sparse_invalid_rows(
        too_long,
        max_drop_ratio=0.05,
        max_drop_run=30,
    )


def test_wuji_recovery_policy_does_not_hide_camera_failure():
    good = np.ones(200, dtype=bool)
    stale_wuji = good.copy()
    stale_wuji[50:60] = False
    bad_camera = good.copy()
    bad_camera[100] = False

    assert not _only_wuji_rows_invalid(
        (stale_wuji, stale_wuji, stale_wuji, stale_wuji, stale_wuji),
        (bad_camera, good, None, good, good),
    )


def test_two_percent_total_drops_cannot_hide_an_eleven_frame_outage():
    valid = np.ones(3323, dtype=bool)
    valid[1200:1211] = False
    valid[::600] = False

    assert np.count_nonzero(~valid) / len(valid) < 0.03
    assert _max_false_run(valid) == 11
    assert not _can_drop_sparse_invalid_rows(valid, max_drop_ratio=0.03, max_drop_run=3)


def test_high_rate_emg_short_hole_uses_configured_gap_budget():
    valid = np.ones(1798, dtype=bool)
    valid[900:909] = False
    period_ns = round(1_000_000_000 / 60)
    run_limit = _emg_alignment_drop_run_limit(
        period_ns, 500_000_000, native_quality_ok=True
    )

    assert run_limit == 29
    assert _can_drop_sparse_invalid_rows(valid, max_drop_ratio=0.03, max_drop_run=run_limit)
    assert not _can_drop_sparse_invalid_rows(valid, max_drop_ratio=0.03, max_drop_run=3)


def test_native_valid_emg_hole_is_not_limited_by_generic_drop_ratio():
    valid = np.ones(186, dtype=bool)
    valid[80:94] = False
    period_ns = round(1_000_000_000 / 60)
    run_limit = _emg_alignment_drop_run_limit(
        period_ns, 500_000_000, native_quality_ok=True
    )

    assert np.count_nonzero(~valid) / len(valid) > 0.05
    assert _can_drop_sparse_invalid_rows(valid, max_drop_ratio=1.0, max_drop_run=run_limit)


def test_bad_native_emg_quality_keeps_strict_three_row_limit():
    assert _emg_alignment_drop_run_limit(
        round(1_000_000_000 / 60), 500_000_000, native_quality_ok=False
    ) == 3


def test_one_sparse_drop_keeps_final_rgb_timeline_within_quality_gate():
    timestamps_ns = np.rint(np.arange(210) * 1_000_000_000 / 60).astype(np.int64)
    keep = np.ones(210, dtype=bool)
    keep[103] = False

    quality = _timeline_quality(timestamps_ns[keep])

    assert quality["max_gap_ns"] < 50_000_000
    assert quality["missing_slot_ratio"] < 0.03


def test_isolated_45ms_rgb_gaps_pass_combined_50ms_and_3pct_gate():
    timestamps_ns = np.rint(np.arange(195) * 1_000_000_000 / 60).astype(np.int64)
    timestamps_ns[70:] += 28_500_000
    timestamps_ns[140:] += 28_500_000

    quality = _timeline_quality(timestamps_ns)

    assert 45_000_000 < quality["max_gap_ns"] < 50_000_000
    assert quality["missing_slot_ratio"] < 0.03


def test_two_approved_alignment_drops_allow_three_period_final_gap():
    timestamps_ns = np.rint(np.arange(540) * 1_000_000_000 / 60).astype(np.int64)
    raw_quality = _timeline_quality(timestamps_ns)
    keep = np.ones(len(timestamps_ns), dtype=bool)
    keep[200:202] = False
    final_quality = _timeline_quality(timestamps_ns[keep])
    period_ns = int(np.median(np.diff(timestamps_ns)))
    allowed_ns = max(50_000_000, raw_quality["max_gap_ns"] + 2 * period_ns)

    assert 50_000_000 <= final_quality["max_gap_ns"] <= allowed_ns
    assert final_quality["missing_slot_ratio"] < 0.03
