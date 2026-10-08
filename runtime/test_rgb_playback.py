import numpy as np
from read_wuji_glove_ftp1_zarr import nearest_frame_indices, playback_index
from read_wuji_glove_ftp1_zarr import comparison_timestamps
import pytest


def test_ego_delay_selects_later_ego_pixels_without_rewriting_timestamps():
    stamps = np.arange(30, dtype=np.int64) * 16_666_667 + 1_000_000_000
    original = stamps.copy()
    ego, main, targets = comparison_timestamps(stamps, stamps, 166.66667)
    assert nearest_frame_indices(ego, targets)[0] == 10
    assert nearest_frame_indices(main, targets)[0] == 0
    assert np.array_equal(stamps, original)
    assert len(targets) == 20


def test_delay_comparison_rejects_nonoverlap_and_nonfinite_offset():
    with pytest.raises(ValueError, match="overlap"):
        comparison_timestamps([100, 200], [100, 200], 1)
    with pytest.raises(ValueError, match="finite"):
        comparison_timestamps([100, 200], [100, 200], float('nan'))


def test_shared_clock_accounts_for_different_camera_start_and_missing_frames():
    ego = np.array([100, 110, 120, 130, 140])
    main = np.array([98, 108, 128, 138])
    assert nearest_frame_indices(main, ego).tolist() == [0, 1, 2, 2, 3]


def test_nearest_ties_choose_earlier_frame_and_clamp_edges():
    assert nearest_frame_indices(np.array([10, 20]), np.array([0, 15, 30])).tolist() == [0, 0, 1]


def test_playback_catches_up_after_decode_delay_without_cumulative_drift():
    stamps = np.array([1_000_000_000, 1_016_000_000, 1_032_000_000, 1_064_000_000])
    assert playback_index(stamps, 40_000_000) == 2
    assert playback_index(stamps, 64_000_000) == 3


def test_two_camera_player_shares_pause_and_step_controls(monkeypatch, tmp_path):
    import cv2
    import read_wuji_glove_ftp1_zarr as reader
    ego = np.zeros((3, 20, 40, 3), dtype=np.uint8)
    main = np.zeros((2, 20, 20, 3), dtype=np.uint8)
    stamps = np.array([1_000_000_000, 1_016_000_000, 1_032_000_000])
    root = {'streams': {'camera_main_rgb_raw': main, 'camera_main_rgb_timestamp_ns': stamps[[0, 2]]}}
    group = {'camera_ego_rgb': ego, 'timestamps': stamps}
    monkeypatch.setattr(reader, '_groups', lambda _: (root, group, None, {}))
    monkeypatch.setattr(reader.time, 'monotonic', lambda: 0.0)
    for method in ('namedWindow', 'resizeWindow', 'destroyAllWindows'):
        monkeypatch.setattr(cv2, method, lambda *a: None)
    captions, displays = [], []
    monkeypatch.setattr(cv2, 'putText', lambda image, text, *a: captions.append(text))
    monkeypatch.setattr(cv2, 'imshow', lambda _, image: displays.append(image.shape))
    keys = iter([32, ord('d'), ord('a'), ord('q')])
    monkeypatch.setattr(cv2, 'waitKey', lambda _: next(keys))
    reader.play_rgb(tmp_path, 0, 'both')
    assert displays == [(520, 1280, 3)] * 4
    assert captions[4].startswith('ego 2/3')
    assert captions[5].startswith('main 1/2')  # equal-distance tie goes earlier
    assert captions[6].startswith('ego 1/3')
