#!/usr/bin/env python3
"""Atomically replace raw-RGB Zarr arrays with transparent JPEG chunks."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import zarr
from imagecodecs.numcodecs import Jpeg, register_codecs


RGB_PATHS = ("data/camera_ego_rgb", "streams/camera_main_rgb_raw")


def recompress(root: zarr.Group, path: str, level: int) -> None:
    if path not in root:
        return
    parent_name, name = path.rsplit("/", 1)
    parent = root[parent_name]
    source = parent[name]
    if getattr(source.compressor, "codec_id", "") == "imagecodecs_jpeg":
        print(f"[SKIP] {path} is already JPEG-compressed")
        return
    if source.ndim != 4 or source.shape[-1] != 3 or source.dtype != np.uint8:
        raise ValueError(f"{path} is not uint8 (T,H,W,3): {source.shape} {source.dtype}")
    temporary = f".{name}.jpeg-migration"
    if temporary in parent:
        del parent[temporary]
    target = parent.create_dataset(
        temporary,
        shape=source.shape,
        chunks=(1, *source.shape[1:]),
        dtype=np.uint8,
        compressor=Jpeg(
            level=level,
            colorspace_data="RGB",
            colorspace_jpeg="YCbCr",
            subsampling="420",
        ),
    )
    for index in range(len(source)):
        target[index] = source[index]
        if (index + 1) % 100 == 0 or index + 1 == len(source):
            print(f"[JPEG] {path}: {index + 1}/{len(source)}", flush=True)
    # Verify the entire replacement is decodable before changing keys.
    for index in range(len(target)):
        frame = target[index]
        if frame.shape != source.shape[1:] or frame.dtype != np.uint8:
            raise RuntimeError(f"verification failed for {path} frame {index}")
    del parent[name]
    parent.move(temporary, name)
    print(f"[DONE] {path}: compressor={parent[name].compressor}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("episode", type=Path)
    parser.add_argument("--quality", type=int, default=90)
    args = parser.parse_args()
    if not 1 <= args.quality <= 100:
        parser.error("--quality must be in [1,100]")
    register_codecs()
    root = zarr.open_group(str(args.episode.expanduser()), mode="a")
    for path in RGB_PATHS:
        recompress(root, path, args.quality)
    root.attrs["camera_rgb_zarr_compression"] = (
        f"imagecodecs JPEG quality={args.quality} subsampling=4:2:0; "
        "transparent uint8 ndarray decode"
    )


if __name__ == "__main__":
    main()
