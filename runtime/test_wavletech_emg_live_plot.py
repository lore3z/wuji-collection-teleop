#!/usr/bin/env python3

from collections import deque

import numpy as np

from wavletech_emg_live_plot import center_channels, display_arrays, robust_limit


def test_display_arrays_breaks_at_real_gap() -> None:
    timestamps = deque([1_000_000_000, 1_005_000_000, 1_200_000_000])
    samples = deque([np.full(8, index, dtype=np.int32) for index in range(3)])
    times, values = display_arrays(timestamps, samples, gap_seconds=0.1)
    assert np.allclose(times, [-0.2, -0.195, 0.0])
    assert np.isnan(values[2]).all()
    assert np.isfinite(values[:2]).all()


def test_robust_limit_has_floor_and_expands() -> None:
    assert robust_limit(np.asarray([0.0, 1.0]), 100.0) == 100.0
    assert robust_limit(np.asarray([-1000.0, 1000.0]), 100.0) > 1000.0


def test_center_channels_removes_independent_dc_offsets_and_keeps_gaps() -> None:
    values = np.asarray([
        [10.0, -100.0],
        [12.0, np.nan],
        [14.0, -96.0],
    ])
    centered = center_channels(values)
    assert np.allclose(np.nanmedian(centered, axis=0), [0.0, 0.0])
    assert np.isnan(centered[1, 1])
    assert np.allclose(centered[:, 0], [-2.0, 0.0, 2.0])


def test_each_channel_gets_an_independent_symmetric_limit() -> None:
    centered = center_channels(np.asarray([
        [-1000.0, 20.0],
        [0.0, 40.0],
        [1000.0, 60.0],
    ]))
    limits = [robust_limit(centered[:, channel], 100.0) for channel in range(2)]
    assert limits[0] > 1000.0
    assert limits[1] == 100.0
    assert limits[0] != limits[1]
    assert all((-limit, limit) == (-abs(limit), abs(limit)) for limit in limits)
