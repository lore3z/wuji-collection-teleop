from types import SimpleNamespace

import cv2
import numpy as np

from wuji_glove_d435_collect import (
    _decode_compressed_messages,
    _decode_messages_to_memmap,
)
from wuji_glove_d435_ftp1_collect import (
    _create_array,
    _sharpness_stats,
)
import zarr


def _jpeg_message(rgb: np.ndarray) -> SimpleNamespace:
    ok, encoded = cv2.imencode(".jpg", rgb[..., ::-1])
    assert ok
    return SimpleNamespace(data=encoded.tobytes())


def test_parallel_jpeg_decode_preserves_tuple_order_and_rgb_channels():
    messages = []
    for index, color in enumerate(((240, 10, 10), (10, 240, 10), (10, 10, 240))):
        rgb = np.full((32, 48, 3), color, dtype=np.uint8)
        messages.append((1000 + index, index, _jpeg_message(rgb)))

    decoded = _decode_compressed_messages(messages, max_workers=3)

    assert [(item[0], item[1]) for item in decoded] == [
        (1000, 0),
        (1001, 1),
        (1002, 2),
    ]
    for item, expected_color in zip(decoded, ((240, 10, 10), (10, 240, 10), (10, 10, 240))):
        np.testing.assert_allclose(item[2].mean(axis=(0, 1)), expected_color, atol=3)


def test_bounded_jpeg_decode_uses_mmap_and_cleans_up():
    messages = []
    colors = ((240, 10, 10), (10, 240, 10), (10, 10, 240), (80, 80, 80))
    for index, color in enumerate(colors):
        rgb = np.full((24, 32, 3), color, dtype=np.uint8)
        messages.append((1000 + index, index, _jpeg_message(rgb)))

    store = _decode_messages_to_memmap(messages, message_index=2, compressed=True, batch_size=2)
    assert store is not None
    path = store.path
    assert isinstance(store.array, np.memmap)
    assert store.array.shape == (4, 24, 32, 3)
    np.testing.assert_allclose(store.array[0].mean(axis=(0, 1)), (240, 10, 10), atol=3)
    store.close()
    assert not path.exists()


def test_parallel_sharpness_matches_single_worker():
    rng = np.random.default_rng(42)
    frames = rng.integers(0, 256, size=(8, 48, 64, 3), dtype=np.uint8)

    single = _sharpness_stats(frames, max_workers=1)
    parallel = _sharpness_stats(frames, max_workers=4)

    assert parallel == single


def test_rgb_zarr_uses_transparent_jpeg_chunks(tmp_path):
    yy, xx = np.mgrid[:96, :128]
    frame = np.stack((xx % 256, yy % 256, (xx + yy) % 256), axis=-1).astype(np.uint8)
    frames = np.stack((frame, np.roll(frame, 7, axis=1)))
    group = zarr.open_group(str(tmp_path / "rgb.zarr"), mode="w")

    _create_array(group, "camera_ego_rgb", frames)

    restored = group["camera_ego_rgb"][:]
    assert restored.shape == frames.shape
    assert restored.dtype == np.uint8
    assert group["camera_ego_rgb"].compressor.codec_id == "imagecodecs_jpeg"
    assert np.mean(np.abs(restored.astype(np.int16) - frames.astype(np.int16))) < 2.0


def test_fast_save_rgb_uses_compact_jpeg_chunks(tmp_path, monkeypatch):
    yy, xx = np.mgrid[:32, :48]
    frame = np.stack((xx % 256, yy % 256, (xx + yy) % 256), axis=-1).astype(np.uint8)
    frames = np.stack((frame, np.roll(frame, 3, axis=1)))
    monkeypatch.setenv("WUJI_FAST_SAVE", "1")
    group = zarr.open_group(str(tmp_path / "fast-rgb.zarr"), mode="w")

    _create_array(group, "camera_ego_rgb", frames)

    restored = group["camera_ego_rgb"][:]
    assert restored.shape == frames.shape
    assert restored.dtype == np.uint8
    assert group["camera_ego_rgb"].chunks[0] == 1
    assert group["camera_ego_rgb"].compressor.codec_id == "imagecodecs_jpeg"
