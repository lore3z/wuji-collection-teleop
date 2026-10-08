#!/usr/bin/env python3

# Resolve project imports independently of the current working directory.
import sys as _project_sys
from pathlib import Path as _ProjectPath
_project_root = _ProjectPath(__file__).resolve().parents[1]
if str(_project_root) not in _project_sys.path:
    _project_sys.path.insert(0, str(_project_root))


import argparse
from pathlib import Path

import numpy as np
import zarr


# ============================================================
# 默认数据路径
# ============================================================

DEFAULT_PATH = Path(
    "~/wuji_datasets/l20_teleop_pressure_sidecar.zarr/episodes"
).expanduser()


# ============================================================
# Finger names
# ============================================================

FINGER_NAMES = [
    "thumb",
    "index",
    "middle",
    "ring",
    "pinky",
]


def open_dataset(path):
    path = Path(path).expanduser()

    if not path.exists():
        raise FileNotFoundError(
            f"数据目录不存在:\n{path}"
        )

    root = zarr.open_group(
        str(path),
        mode="r",
    )

    return root


def get_episodes(root):
    return sorted(
        list(root.group_keys())
    )


def select_episode(root, name):
    episodes = get_episodes(root)

    if not episodes:
        raise RuntimeError(
            "episodes 目录下没有 episode"
        )

    if name == "latest":
        name = episodes[-1]

    if name not in episodes:
        raise RuntimeError(
            f"episode 不存在: {name}\n"
            f"现有 episodes: {episodes}"
        )

    return name, root[name]


def print_episode_list(root):
    episodes = get_episodes(root)

    print()
    print("=" * 80)
    print("EPISODES")
    print("=" * 80)

    for name in episodes:
        ep = root[name]

        frame_count = None

        # 尽可能找到帧数
        for key in ep.array_keys():
            arr = ep[key]

            if len(arr.shape) >= 1:
                frame_count = arr.shape[0]
                break

        print(
            f"{name:25s}",
            f"frames={frame_count}"
        )

    print()


def print_summary(name, ep):
    print()
    print("=" * 80)
    print("EPISODE SUMMARY")
    print("=" * 80)

    print("episode:", name)

    print()
    print("attrs:")
    print(dict(ep.attrs))

    print()
    print("arrays:")

    for key in sorted(ep.array_keys()):
        arr = ep[key]

        print(
            f"  {key:32s}"
            f" shape={str(arr.shape):18s}"
            f" dtype={arr.dtype}"
        )

    print()


def _percentile_text(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return "无有效值"
    return (
        f"median={np.median(values):.2f}, "
        f"p95={np.percentile(values, 95):.2f}, "
        f"max={np.max(values):.2f}"
    )


def print_quality_report(name, ep):
    """Audit tactile validity, pressure consistency, and source freshness."""
    print()
    print("=" * 80)
    print("QUALITY REPORT")
    print("=" * 80)
    print("episode:", name)

    if "glove_tactile" in ep:
        raw = np.asarray(ep["glove_tactile"][:], dtype=np.float32)
        if raw.ndim == 2 and raw.shape[1] == 24 * 31:
            raw = raw.reshape((-1, 24, 31))
            derived = np.isfinite(raw) & (raw >= 0.0)
            if "glove_tactile_valid_mask" in ep:
                stored = np.asarray(ep["glove_tactile_valid_mask"][:]).astype(bool)
                mask = derived & stored
                mismatch = int(np.count_nonzero(stored != derived))
            else:
                mask = derived
                mismatch = 0
            valid_per_frame = mask.sum(axis=(1, 2))
            active_per_frame = (np.where(mask, raw, 0.0) > 0.0).sum(axis=(1, 2))
            print(
                "glove tactile: valid taxels/frame "
                f"min={valid_per_frame.min()}, max={valid_per_frame.max()}, "
                f"active median={np.median(active_per_frame):.0f}"
            )
            if "glove_tactile_valid_mask" in ep:
                print("glove mask: stored-vs-derived mismatch =", mismatch)
            if "glove_tactile_source_valid" in ep:
                good = int(np.count_nonzero(ep["glove_tactile_source_valid"][:]))
                print(f"glove source valid: {good}/{raw.shape[0]} frames")
        else:
            print("glove tactile: shape is not (T, 744); skipped")

    if "robot_pressure_matrix" in ep and "robot_pressure_mass_g" in ep:
        matrix = np.asarray(ep["robot_pressure_matrix"][:], dtype=np.float32)
        mass = np.asarray(ep["robot_pressure_mass_g"][:], dtype=np.float32)
        matrix_sum = np.where(np.isfinite(matrix), matrix, 0.0).sum(axis=(2, 3))
        residual = mass - matrix_sum
        finite = residual[np.isfinite(residual)]
        max_abs = float(np.max(np.abs(finite))) if finite.size else float("nan")
        print(f"G20 mass - matrix.sum: max_abs={max_abs:.4f}")
        if "robot_pressure_source_valid" in ep:
            good = int(np.count_nonzero(ep["robot_pressure_source_valid"][:]))
            print(f"G20 pressure source valid: {good}/{matrix.shape[0]} frames")

    for key, label in (
        ("glove_tactile_age_ms", "glove tactile age (ms)"),
        ("robot_pressure_age_ms", "G20 tactile age (ms)"),
        ("sidecar_packet_age_ms", "sidecar receive age (ms)"),
    ):
        if key in ep:
            print(f"{label}: {_percentile_text(ep[key][:])}")


def print_glove_angles(data):
    data = np.asarray(data)

    print()
    print("glove_angles")
    print("shape =", data.shape)

    print()
    print("rad:")

    for i in range(
        min(
            len(FINGER_NAMES),
            data.shape[0],
        )
    ):
        print(
            f"{FINGER_NAMES[i]:7s}",
            np.round(
                data[i],
                4,
            ).tolist(),
        )

    print()
    print("deg:")

    deg = np.degrees(data)

    for i in range(
        min(
            len(FINGER_NAMES),
            deg.shape[0],
        )
    ):
        print(
            f"{FINGER_NAMES[i]:7s}",
            np.round(
                deg[i],
                2,
            ).tolist(),
        )


def print_robot_pressure(data):
    data = np.asarray(data)

    print()
    print(
        "robot_pressure_matrix shape =",
        data.shape,
    )

    if data.ndim != 3:
        print(data)
        return

    for i in range(data.shape[0]):
        name = (
            FINGER_NAMES[i]
            if i < len(FINGER_NAMES)
            else str(i)
        )

        print()
        print(
            f"[{name}]"
        )

        print(
            np.round(
                data[i],
                2,
            )
        )

        print(
            "sum =",
            round(
                float(
                    np.sum(
                        data[i]
                    )
                ),
                3,
            ),
            "max =",
            round(
                float(
                    np.max(
                        data[i]
                    )
                ),
                3,
            ),
        )


def print_glove_tactile(data, valid_mask=None):
    """Print Wuji tactile data without treating -1 geometry sentinels as force."""
    data = np.asarray(data)

    print()
    print(
        "glove_tactile original shape =",
        data.shape,
    )

    flat = data.reshape(-1)

    print(
        "elements =",
        flat.size,
    )

    if flat.size == 24 * 31:

        matrix = flat.reshape(
            24,
            31,
        )

        print()
        print(
            "reshape -> 24 x 31"
        )

        print(
            np.round(
                matrix,
                2,
            )
        )

        # Wuji SDK uses -1 for grid positions that do not physically exist.
        # Old v2 episodes have no explicit mask, so derive the same mask from
        # the raw matrix.  A v3 stored mask additionally documents this rule.
        derived_mask = np.isfinite(matrix) & (matrix >= 0.0)
        if valid_mask is not None:
            supplied_mask = np.asarray(valid_mask).reshape(matrix.shape).astype(bool)
            valid_mask = derived_mask & supplied_mask
        else:
            valid_mask = derived_mask
        valid = matrix[valid_mask]

        pressure_sum = float(valid.sum()) if valid.size else 0.0
        pressure_mean = float(valid.mean()) if valid.size else 0.0
        pressure_max = float(valid.max()) if valid.size else 0.0
        active_count = int(np.count_nonzero(valid > 0.0))

        print()
        print("valid cells =", int(valid.size))
        print("invalid / missing cells =", int(matrix.size - valid.size))
        print("valid pressure sum =", round(pressure_sum, 4))
        print("valid pressure mean =", round(pressure_mean, 4))
        print("valid pressure max =", round(pressure_max, 4))
        print("active cells (>0) =", active_count)

    else:
        print(
            np.round(
                data,
                3,
            )
        )


def print_raw20(name, data):
    data = np.asarray(
        data
    ).reshape(-1)

    print()
    print(name)

    print(
        np.round(
            data,
            2,
        ).tolist()
    )

    if data.size == 20:
        print()

        print(
            "thumb raw [0,5,10,15] =",
            np.round(
                data[
                    [0, 5, 10, 15]
                ],
                2,
            ).tolist(),
        )

        print(
            "index raw [1,6,16]    =",
            np.round(
                data[
                    [1, 6, 16]
                ],
                2,
            ).tolist(),
        )

        print(
            "middle raw [2,7,17]   =",
            np.round(
                data[
                    [2, 7, 17]
                ],
                2,
            ).tolist(),
        )

        print(
            "ring raw [3,8,18]     =",
            np.round(
                data[
                    [3, 8, 18]
                ],
                2,
            ).tolist(),
        )

        print(
            "pinky raw [4,9,19]    =",
            np.round(
                data[
                    [4, 9, 19]
                ],
                2,
            ).tolist(),
        )


def print_field(name, data, valid_mask=None):
    data = np.asarray(data)

    if name == "glove_angles":
        print_glove_angles(data)
        return

    if name == "glove_tactile":
        print_glove_tactile(data, valid_mask=valid_mask)
        return

    if name == "robot_pressure_matrix":
        print_robot_pressure(data)
        return

    if name in (
        "robot_command_raw20",
        "robot_actual_raw20",
    ):
        print_raw20(
            name,
            data,
        )
        return

    print()
    print(name)
    print("shape =", data.shape)
    print("dtype =", data.dtype)

    print(
        np.round(
            data,
            4,
        )
        if np.issubdtype(
            data.dtype,
            np.number,
        )
        else data
    )


def show_frame(ep, frame):
    arrays = list(
        ep.array_keys()
    )

    if not arrays:
        raise RuntimeError(
            "episode 中没有数组"
        )

    # 找最大合法 frame
    frame_counts = []

    for key in arrays:
        arr = ep[key]

        if len(arr.shape) > 0:
            frame_counts.append(
                arr.shape[0]
            )

    if not frame_counts:
        raise RuntimeError(
            "无法判断帧数"
        )

    total = min(
        frame_counts
    )

    if frame < 0:
        frame = total + frame

    if not (
        0 <= frame < total
    ):
        raise IndexError(
            f"frame={frame} 越界，"
            f"有效范围 0~{total-1}"
        )

    print()
    print("=" * 80)
    print(
        f"FRAME {frame} / {total-1}"
    )
    print("=" * 80)

    preferred = [
        "timestamp_ns",
        "glove_angles",
        "glove_tactile",
        "glove_tactile_summary",
        "glove_tactile_source_valid",
        "glove_tactile_age_ms",
        "finger_curls",
        "finger_proximal_curls",
        "finger_distal_curls",
        "u16",
        "robot_command_raw20",
        "robot_actual_raw20",
        "robot_pressure_mass_g",
        "robot_pressure_matrix",
        "robot_pressure_matrix_sum",
        "robot_pressure_mass_minus_matrix_sum",
        "robot_pressure_source_valid",
        "robot_pressure_age_ms",
        "sidecar_packet_age_ms",
    ]

    shown = set()

    for key in preferred:

        if key not in ep:
            continue

        shown.add(key)

        print()
        print("-" * 80)

        print_field(
            key,
            ep[key][frame],
            valid_mask=(
                ep["glove_tactile_valid_mask"][frame]
                if key == "glove_tactile" and "glove_tactile_valid_mask" in ep
                else None
            ),
        )

    # 显示剩余字段
    for key in arrays:

        if key in shown:
            continue

        # These are large v3 helper tensors.  Their information is represented
        # by glove_tactile_summary / --quality; print them only on --field.
        if key in {"glove_tactile_valid_mask", "glove_tactile_pressure"}:
            continue

        print()
        print("-" * 80)

        print_field(
            key,
            ep[key][frame],
        )


def show_one_field(
    ep,
    field,
    frame,
):
    if field not in ep:
        raise RuntimeError(
            f"字段不存在: {field}\n"
            f"可用字段:\n"
            + "\n".join(
                sorted(
                    ep.array_keys()
                )
            )
        )

    arr = ep[field]

    if frame is None:

        print()
        print(field)
        print(
            "shape =",
            arr.shape,
        )
        print(
            "dtype =",
            arr.dtype,
        )

        return

    total = arr.shape[0]

    if frame < 0:
        frame = total + frame

    if not (
        0 <= frame < total
    ):
        raise IndexError(
            f"frame={frame} 越界"
        )

    print_field(
        field,
        arr[frame],
        valid_mask=(
            ep["glove_tactile_valid_mask"][frame]
            if field == "glove_tactile" and "glove_tactile_valid_mask" in ep
            else None
        ),
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Read Wuji + LinkerHand "
            "pressure sidecar Zarr dataset"
        )
    )

    parser.add_argument(
        "--path",
        default=str(
            DEFAULT_PATH
        ),
        help=(
            "episodes directory"
        ),
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="列出全部 episodes",
    )

    parser.add_argument(
        "--episode",
        default="latest",
        help=(
            "episode 名称，"
            "默认 latest"
        ),
    )

    parser.add_argument(
        "--summary",
        action="store_true",
        help=(
            "显示 episode 字段/shape"
        ),
    )

    parser.add_argument(
        "--quality",
        action="store_true",
        help="检查手套有效掩码、G20 pressure sum 和采样时延",
    )

    parser.add_argument(
        "--frame",
        type=int,
        default=None,
        help=(
            "读取指定帧，"
            "-1 表示最后一帧"
        ),
    )

    parser.add_argument(
        "--field",
        default=None,
        help=(
            "只读取某个字段"
        ),
    )

    args = parser.parse_args()

    root = open_dataset(
        args.path
    )

    if args.list:
        print_episode_list(
            root
        )

        if (
            not args.summary
            and args.frame is None
            and args.field is None
        ):
            return

    name, ep = select_episode(
        root,
        args.episode,
    )

    if args.summary:
        print_summary(
            name,
            ep,
        )

    if args.quality:
        print_quality_report(name, ep)

    if args.field is not None:

        show_one_field(
            ep,
            args.field,
            args.frame,
        )

        return

    if args.frame is not None:

        show_frame(
            ep,
            args.frame,
        )

        return

    if not args.summary and not args.quality:

        print_summary(
            name,
            ep,
        )


if __name__ == "__main__":
    main()
