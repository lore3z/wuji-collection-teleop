#!/usr/bin/env python3
"""Inspect, validate, and play FTP-1 Zarrs produced by wuji_glove_d435_ftp1_collect.py.

Usage (run this script from its output directory):
  python3 read_wuji_glove_ftp1_zarr.py --list
  python3 read_wuji_glove_ftp1_zarr.py --summary
  python3 read_wuji_glove_ftp1_zarr.py --episode episode_000000.zarr --frame 100
  python3 read_wuji_glove_ftp1_zarr.py --latest --pose-frame 0
  python3 read_wuji_glove_ftp1_zarr.py --episode episode_000000.zarr --play
  python3 read_wuji_glove_ftp1_zarr.py --latest --play --camera main
"""

from __future__ import annotations

# Resolve project imports independently of the current working directory.
import sys as _project_sys
from pathlib import Path as _ProjectPath
_project_root = _ProjectPath(__file__).resolve().parents[1]
if str(_project_root) not in _project_sys.path:
    _project_sys.path.insert(0, str(_project_root))


import argparse
import json
import os
import sys
import time
from pathlib import Path


def _reexec_project_python() -> None:
    """Make the documented ``python3 reader.py`` command codec-complete."""
    if os.environ.get("WUJI_READER_VENV_REEXEC") == "1":
        return
    script = Path(__file__).resolve()
    for parent in (script.parent, *script.parents):
        venv_dir = parent / ".venv"
        python = venv_dir / "bin" / "python"
        if python.is_file():
            if Path(sys.prefix).resolve() == venv_dir.resolve():
                return
            env = os.environ.copy()
            env["WUJI_READER_VENV_REEXEC"] = "1"
            os.execve(str(python), [str(python), str(script), *sys.argv[1:]], env)
    return


_reexec_project_python()

import numpy as np
import zarr
try:
    from imagecodecs.numcodecs import register_codecs as _register_image_codecs

    _register_image_codecs()
except ImportError:
    # Legacy episodes use only built-in numcodecs and remain readable. JPEG-
    # chunked RGB episodes will report their precise missing codec on access.
    pass


REQUIRED = (
    "timestamps",
    "camera_ego_rgb",
    "right_hand_joints",
    "right_hand_joints_idx",
    "right_tactile_data_wuji",
    "right_tactile_area_wuji",
    "right_tactile_sensor_wuji",
    "right_tactile_type_wuji",
    "sub_task_instruction",
)
VALID_TACTILE_TYPES = {"state", "binary", "image", "matrix"}


def _groups(path: Path) -> tuple[zarr.Group, zarr.Group, zarr.Group | None, zarr.Group]:
    """Open FTP-1 v5 ``data/meta/audit`` and legacy root-only episodes."""
    root = zarr.open_group(str(path), mode="r")
    if "data" in root and "meta" in root:
        return root, root["data"], (root["audit"] if "audit" in root else None), root["meta"]
    # Read old root-only captures for inspection, but identify them clearly.
    return root, root, None, root


def _episode_paths(root: Path) -> list[Path]:
    """Return only directories that are actual Zarr groups.

    A previous collector invocation could leave an ``episode_*.zarr`` wrapper
    directory containing the actual Zarr episode one level below it.  A
    directory name alone is therefore not sufficient: Zarr v2 requires a
    ``.zgroup`` file (and Zarr v3 uses ``zarr.json``).  Searching for group
    metadata makes ``--latest`` work for both the normal flat layout and those
    already-recorded nested episodes.
    """
    root = root.expanduser().resolve()

    def is_zarr_group(path: Path) -> bool:
        return path.is_dir() and ((path / ".zgroup").is_file() or (path / "zarr.json").is_file())

    if root.name.endswith(".zarr"):
        if not is_zarr_group(root):
            raise SystemExit(
                f"不是有效 Zarr group：{root}\n"
                "目录中缺少 .zgroup/zarr.json；请将 --path 指向实际 episode 目录。"
            )
        return [root]

    return sorted(
        path.resolve()
        for path in root.rglob("episode_*.zarr")
        if is_zarr_group(path)
    )


def _hz(ts: np.ndarray) -> float:
    if len(ts) < 2:
        return 0.0
    return float((len(ts) - 1) / max((int(ts[-1]) - int(ts[0])) / 1e9, 1e-9))


def _attr_list(value: object) -> list[str] | None:
    """Normalize Zarr JSON-string and native-list attributes."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return None


def _timeline_quality(ts: np.ndarray) -> dict[str, float | int]:
    """Report real RGB timing quality; never infer fake replacement frames."""
    ts = np.asarray(ts, dtype=np.int64)
    if len(ts) < 2:
        return {"nominal_period_ns": 0, "max_gap_ns": 0, "missing_slot_count": 0, "missing_slot_ratio": 0.0}
    dt = np.diff(ts)
    if np.any(dt <= 0):
        return {"nominal_period_ns": 0, "max_gap_ns": int(dt.max(initial=0)), "missing_slot_count": -1, "missing_slot_ratio": 1.0}
    nominal = int(np.median(dt))
    steps = np.maximum(np.rint(dt / max(nominal, 1)).astype(np.int64), 1)
    missing = int(np.maximum(steps - 1, 0).sum())
    return {
        "nominal_period_ns": nominal,
        "max_gap_ns": int(dt.max()),
        "missing_slot_count": missing,
        "missing_slot_ratio": float(missing / max(len(ts) + missing, 1)),
    }


def validate(path: Path, frame: int | None = None, *, strict_quality: bool = False) -> bool:
    root, group, audit, meta = _groups(path)
    keys = set(group.array_keys())
    missing = [key for key in REQUIRED if key not in keys]
    if root.attrs.get("camera_ego_rgb_role") == "disabled":
        missing = [key for key in missing if key != "camera_ego_rgb"]
    if missing:
        print(f"[FAIL] {path.name}: missing {missing}")
        return False
    t = len(group["timestamps"])
    standard_contract = "episode_ends" in meta and group is not root
    if standard_contract:
        ends = meta["episode_ends"][:]
        if ends.shape != (1,) or int(ends[-1]) != t:
            print(f"[FAIL] {path.name}: meta/episode_ends={ends.tolist()} does not equal T={t}")
            return False
    else:
        print(f"  [WARN] legacy root-only layout: not direct FTP-1 ReplayBuffer input (missing data/meta/episode_ends)")
    wrong = [(key, group[key].shape) for key in keys if len(group[key].shape) == 0 or group[key].shape[0] != t]
    ts = group["timestamps"][:]
    monotonic = bool(np.all(np.diff(ts) > 0))
    timeline = _timeline_quality(ts)
    tactile = group["right_tactile_data_wuji"]
    tactile_type = str(group["right_tactile_type_wuji"][0])
    tactile_sensor = str(group["right_tactile_sensor_wuji"][0])
    print(f"{path.name}: T={t}, RGB={_hz(ts):.2f} Hz, timestamp_monotonic={monotonic}")
    print(
        "  canonical timing: nominal=%.2f ms max_gap=%.2f ms missing=%s (%.2f%%)"
        % (
            timeline["nominal_period_ns"] / 1e6,
            timeline["max_gap_ns"] / 1e6,
            timeline["missing_slot_count"],
            100.0 * timeline["missing_slot_ratio"],
        )
    )
    max_gap_ms = float(root.attrs.get("quality_gate_max_rgb_gap_ms", 25.0))
    max_missing_ratio = float(root.attrs.get("quality_gate_max_missing_ratio", 0.02))
    timing_good = timeline["max_gap_ns"] <= round(max_gap_ms * 1e6) and timeline["missing_slot_ratio"] <= max_missing_ratio
    if not timing_good:
        print(f"  [WARN] timing exceeds quality gate: gap<={max_gap_ms:.2f} ms, missing<={max_missing_ratio:.2%}")
    alignment_dropped = int(root.attrs.get("alignment_rows_dropped", 0))
    if alignment_dropped:
        print(
            "  alignment rows dropped:",
            alignment_dropped,
            "| ratio=%.2f%%" % (100.0 * float(root.attrs.get("alignment_rows_drop_ratio", 0.0))),
            "| max consecutive=%s" % root.attrs.get("alignment_max_consecutive_rows_dropped", "?"),
        )
    if "camera_ego_rgb" in keys:
        ego_source = str(root.attrs.get("camera_ego_rgb_source", "RealSense D435"))
        print(f"  ego RGB ({ego_source})", group["camera_ego_rgb"].shape, group["camera_ego_rgb"].dtype)
    else:
        print("  ego RGB: disabled")
    ego_mp4 = str(root.attrs.get("camera_ego_rgb_mp4", ""))
    if ego_mp4:
        print("  ego MP4:", ego_mp4, "|", root.attrs.get("camera_ego_rgb_mp4_status", ""))
    if "camera_main_rgb" in keys:
        print("  main RGB (Gemini)", group["camera_main_rgb"].shape, group["camera_main_rgb"].dtype)
        main_mp4 = str(root.attrs.get("camera_main_rgb_mp4", ""))
        if main_mp4:
            print("  main MP4:", main_mp4, "|", root.attrs.get("camera_main_rgb_mp4_status", ""))
        if audit is not None and "camera_main_rgb_age_us" in audit:
            camera_age = np.abs(audit["camera_main_rgb_age_us"][:])
            print(f"  Gemini alignment age: median={np.median(camera_age)/1000:.2f} ms max={np.max(camera_age)/1000:.2f} ms")
    elif "streams" in root and "camera_main_rgb_raw" in root["streams"]:
        raw_main = root["streams"]["camera_main_rgb_raw"]
        print(
            "  main RGB (Gemini, independent timeline)", raw_main.shape, raw_main.dtype,
            "| native=%.2f Hz" % float(root.attrs.get("camera_main_rgb_raw_hz", 0.0)),
        )
    if "streams" in root and "camera_ego_pose_raw" in root["streams"]:
        print(
            "  Gemini IMU pose RAW", root["streams"]["camera_ego_pose_raw"].shape,
            "| native=%.2f Hz" % float(root.attrs.get("camera_ego_pose_raw_hz", 0.0)),
        )
    if "streams" in root and "camera_tracker_pose_raw" in root["streams"]:
        print(
            "  PICO camera tracker pose RAW", root["streams"]["camera_tracker_pose_raw"].shape,
            "| native=%.2f Hz" % float(root.attrs.get("camera_tracker_pose_raw_hz", 0.0)),
        )
    if "streams" in root and "right_wrist_tracker_pose_raw" in root["streams"]:
        print(
            "  PICO wrist tracker pose RAW", root["streams"]["right_wrist_tracker_pose_raw"].shape,
            "| native=%.2f Hz" % float(root.attrs.get("right_wrist_tracker_pose_raw_hz", 0.0)),
        )
    print("  joints", group["right_hand_joints"].shape, "idx", group["right_hand_joints_idx"].shape)
    if bool(root.attrs.get("myo_enabled", True)) and "right_forearm_emg" in keys:
        print(
            "  canonical Myo EMG",
            group["right_forearm_emg"].shape,
            group["right_forearm_emg"].dtype,
            "| native=%.2f Hz samples=%s missing=%s (%.2f%%)"
            % (
                float(root.attrs.get("right_forearm_emg_native_hz", 0.0)),
                root.attrs.get("right_forearm_emg_native_sample_count", "?"),
                root.attrs.get("right_forearm_emg_missing_count", "?"),
                100.0 * float(root.attrs.get("right_forearm_emg_missing_ratio", 0.0)),
            ),
        )
        if "streams" in root and "right_forearm_emg_raw" in root["streams"]:
            print("  native Myo EMG", root["streams"]["right_forearm_emg_raw"].shape, "at streams/right_forearm_emg_raw")
    elif root.attrs.get("myo_enabled") is False:
        print("  Myo EMG: disabled for this episode")
    if bool(root.attrs.get("wavletech_emg_enabled", root.attrs.get("emg_enabled", False))) and "right_forearm_emg_wavletech" in keys:
        print(
            "  canonical Wavletech EMG",
            group["right_forearm_emg_wavletech"].shape,
            group["right_forearm_emg_wavletech"].dtype,
            "| native=%.2f Hz samples=%s missing=%s (%.2f%%)"
            % (
                float(root.attrs.get("right_forearm_emg_wavletech_native_hz", 0.0)),
                root.attrs.get("right_forearm_emg_wavletech_native_sample_count", "?"),
                root.attrs.get("right_forearm_emg_wavletech_missing_count", "?"),
                100.0 * float(root.attrs.get("right_forearm_emg_wavletech_missing_ratio", 0.0)),
            ),
        )
        if "streams" in root and "right_forearm_emg_wavletech_raw" in root["streams"]:
            print("  native Wavletech EMG", root["streams"]["right_forearm_emg_wavletech_raw"].shape, "at streams/right_forearm_emg_wavletech_raw")
        if "streams" in root and "right_forearm_imu_wavletech_raw" in root["streams"]:
            print("  Wavletech IMU", root["streams"]["right_forearm_imu_wavletech_raw"].shape, "at streams/right_forearm_imu_wavletech_raw")
    joint_names = _attr_list(root.attrs.get("right_hand_joint_names"))
    if joint_names:
        print("  joint layout:", root.attrs.get("right_hand_joints_layout", "unknown"))
        print("  joint names :", joint_names)
    joints = group["right_hand_joints"][:]
    joint_good = True
    if len(joints) >= 2:
        joint_delta = np.abs(np.diff(joints, axis=0))
        flat_idx = int(np.argmax(joint_delta))
        step_frame, step_joint = np.unravel_index(flat_idx, joint_delta.shape)
        max_step = float(joint_delta[step_frame, step_joint])
        step_name = joint_names[step_joint] if isinstance(joint_names, list) and step_joint < len(joint_names) else f"joint_{step_joint}"
        max_joint_step = float(root.attrs.get("quality_gate_max_joint_step_rad", 1.0))
        joint_good = max_step <= max_joint_step
        print(f"  joint smoothness: max_step={max_step:.4f} rad ({step_name}, frame {step_frame}->{step_frame + 1})")
        if not joint_good:
            print(f"  [WARN] joint step exceeds quality gate {max_joint_step:.3f} rad")
    if audit is not None and "wuji_local_joint_idx" in audit:
        print("  local joint idx:", audit["wuji_local_joint_idx"][0].tolist())
    idx_semantics = root.attrs.get("right_hand_joints_idx_semantics")
    if idx_semantics:
        print("  joints_idx semantics:", idx_semantics, "| FAAS verified:", root.attrs.get("right_hand_joints_faas_verified", False))
    if audit is not None and "wuji_hand_joints_raw_wuji_5x5" in audit:
        print("  raw Wuji SDK angles", audit["wuji_hand_joints_raw_wuji_5x5"].shape, "(audit only)")
    if audit is not None and "wuji_hand_skeleton_mediapipe" in audit:
        print("  Wuji MediaPipe skeleton", audit["wuji_hand_skeleton_mediapipe"].shape, "(source for canonical joints)")
    print("  tactile", tactile.shape, f"sensor={tactile_sensor!r} type={tactile_type!r}")
    if tactile_type not in VALID_TACTILE_TYPES:
        print("  [WARN] tactile type is not supported: expected state/binary/image/matrix")
    print("  instruction:", str(group["sub_task_instruction"][0]))
    if "right_tactile_data_wuji_zones" in keys:
        zone_area = group["right_tactile_area_wuji_zones"][0].tolist()
        print(
            "  tactile zones",
            group["right_tactile_data_wuji_zones"].shape,
            f"areas={zone_area} type={str(group['right_tactile_type_wuji_zones'][0])!r}",
        )
    if "right_wrist_pose" not in keys:
        print("  [INFO] right_wrist_pose missing (legacy episode; record a new episode after enabling Wuji TF capture)")
    if "camera_ego_pose" not in keys:
        print("  [INFO] camera_ego_pose missing (legacy episode or Gemini IMU capture disabled)")
    sharpness = root.attrs.get("camera_ego_rgb_sharpness_laplacian")
    if sharpness:
        print("  ego sharpness:", sharpness)
    if root.attrs.get("privacy_face_blur_enabled", False):
        print(
            "  privacy face blur detections:",
            f"ego={root.attrs.get('privacy_face_blur_detections_ego', 0)}",
            f"main={root.attrs.get('privacy_face_blur_detections_main', 0)}",
        )
    print("  all arrays time-major:", "YES" if not wrong else f"NO {wrong}")
    if audit is not None and "right_tactile_valid_mask_wuji" in audit:
        valid = audit["right_tactile_valid_mask_wuji"][0, 0].astype(bool)
        raw = audit["right_tactile_raw_wuji"][0, 0] if "right_tactile_raw_wuji" in audit else tactile[0, 0]
        expected = int(root.attrs.get("wuji_tactile_official_active_taxels", 526))
        invalid_count = int(np.count_nonzero(~valid))
        print(f"  tactile geometry: physical={int(valid.sum())}/744 (official={expected}), raw_invalid_-1={int(np.count_nonzero(raw[~valid] == -1.0))}/{invalid_count}, training_invalid_zero={int(np.count_nonzero(tactile[0, 0][~valid] == 0.0))}/{invalid_count}")
    if audit is not None and "right_hand_joints_age_us" in audit and "right_tactile_age_us" in audit:
        ages = np.concatenate((np.abs(audit["right_hand_joints_age_us"][:]), np.abs(audit["right_tactile_age_us"][:])))
        print(f"  alignment age: median={np.median(ages)/1000:.2f} ms max={np.max(ages)/1000:.2f} ms")
    if audit is not None and "wuji_hand_skeleton_age_us" in audit:
        skeleton_ages = np.abs(audit["wuji_hand_skeleton_age_us"][:])
        print(f"  skeleton age: median={np.median(skeleton_ages)/1000:.2f} ms max={np.max(skeleton_ages)/1000:.2f} ms")
    if frame is not None:
        if not 0 <= frame < t:
            raise SystemExit(f"--frame must be within [0, {t - 1}]")
        print(f"  frame={frame} timestamp_ns={int(ts[frame])}")
        print("  joints:", np.array2string(group["right_hand_joints"][frame], precision=4))
        print("  instruction:", str(group["sub_task_instruction"][frame]))
        if audit is not None and "right_hand_joints_age_us" in audit:
            print("  source ages ms: joints=%.3f tactile=%.3f" % (
                audit["right_hand_joints_age_us"][frame] / 1000.0,
                audit["right_tactile_age_us"][frame] / 1000.0,
            ))
        if "right_forearm_emg" in keys:
            print("  Myo EMG (8ch):", group["right_forearm_emg"][frame].astype(int).tolist())
            if audit is not None and "right_forearm_emg_age_us" in audit:
                print("  Myo source age ms: %.3f" % (audit["right_forearm_emg_age_us"][frame] / 1000.0))
        if "right_forearm_emg_wavletech" in keys:
            print("  Wavletech EMG (8ch µV):", group["right_forearm_emg_wavletech"][frame].astype(int).tolist())
            if audit is not None and "right_forearm_emg_wavletech_age_us" in audit:
                print("  Wavletech source age ms: %.3f" % (audit["right_forearm_emg_wavletech_age_us"][frame] / 1000.0))
    return not wrong and monotonic and (not strict_quality or (standard_contract and timing_good and joint_good))


def show_wuji_frame(path: Path, frame: int, show_tactile_matrix: bool) -> None:
    """Print all Wuji source/audit data retained for one canonical RGB frame."""
    root, group, audit, _meta = _groups(path)
    t = len(group["timestamps"])
    if not 0 <= frame < t:
        raise SystemExit(f"--wuji-frame must be within [0, {t - 1}]")
    keys = set(group.array_keys())
    print(f"\n[WUJI] {path.name} frame={frame} canonical_timestamp_ns={int(group['timestamps'][frame])}")

    names = _attr_list(root.attrs.get("right_hand_joint_names")) or [
        f"joint_{i}" for i in range(group["right_hand_joints"].shape[1])
    ]
    canonical = group["right_hand_joints"][frame]
    print("  FTP-1 canonical joints (rad):")
    print("   ", ", ".join(f"{name}={value:+.4f}" for name, value in zip(names, canonical)))
    if "right_hand_joints_idx" in keys:
        print("  FAAS idx:", group["right_hand_joints_idx"][frame].tolist())
    for key in ("right_hand_joints_source_timestamp_us", "right_hand_joints_source_seq", "right_hand_joints_age_us"):
        if audit is not None and key in audit:
            value = int(audit[key][frame])
            suffix = " ms" if key.endswith("age_us") else ""
            value_text = f"{value / 1000.0:.3f}{suffix}" if suffix else str(value)
            print(f"  {key}: {value_text}")

    if audit is not None and "wuji_hand_joints_anatomical" in audit:
        anatomical_names = _attr_list(root.attrs.get("wuji_hand_joints_anatomical_names")) or []
        anatomical = audit["wuji_hand_joints_anatomical"][frame]
        print("  Wuji anatomical 21 DoF (rad):")
        print("   ", ", ".join(
            f"{anatomical_names[i] if i < len(anatomical_names) else i}={value:+.4f}"
            for i, value in enumerate(anatomical)
        ))
    if audit is not None and "wuji_hand_joints_raw_wuji_5x5" in audit:
        print("  Wuji raw SDK 5x5 (rad):\n", audit["wuji_hand_joints_raw_wuji_5x5"][frame])
    if audit is not None and "wuji_hand_skeleton_mediapipe" in audit:
        skeleton = audit["wuji_hand_skeleton_mediapipe"][frame]
        skeleton_names = [
            "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
            "index_mcp", "index_pip", "index_dip", "index_tip",
            "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
            "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
            "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
        ]
        print("  HandSkeleton positions (21 points):")
        for index, (name, xyz) in enumerate(zip(skeleton_names, skeleton)):
            print(
                f"    {index:02d} {name:12s} "
                f"x={xyz[0]:+.6f} y={xyz[1]:+.6f} z={xyz[2]:+.6f}"
            )
        for key in (
            "wuji_hand_skeleton_source_seq",
            "wuji_hand_skeleton_source_timestamp_us",
            "wuji_hand_skeleton_age_us",
        ):
            if key in audit:
                value = int(audit[key][frame])
                print(f"  {key}: {value / 1000.0:.3f} ms" if key.endswith("age_us") else f"  {key}: {value}")

    if "right_tactile_data_wuji" in keys:
        tactile = group["right_tactile_data_wuji"][frame, 0]
        valid = audit["right_tactile_valid_mask_wuji"][frame, 0].astype(bool) if audit is not None and "right_tactile_valid_mask_wuji" in audit else np.ones_like(tactile, dtype=bool)
        values = tactile[valid]
        pressure_min = float(values.min()) if values.size else 0.0
        pressure_mean = float(values.mean()) if values.size else 0.0
        pressure_max = float(values.max()) if values.size else 0.0
        print(
            "  tactile 24x31 (baseline-subtracted): physical=%d/744, min=%.5f, mean=%.5f, max=%.5f, >0.10=%d"
            % (valid.sum(), pressure_min, pressure_mean, pressure_max, np.count_nonzero(values > 0.10))
        )
        if show_tactile_matrix:
            print("  training tactile matrix (invalid cells are 0):\n", np.array2string(tactile, precision=4, max_line_width=160))
    for key in ("right_tactile_source_timestamp_us", "right_tactile_source_seq", "right_tactile_age_us"):
        if audit is not None and key in audit:
            value = int(audit[key][frame])
            suffix = " ms" if key.endswith("age_us") else ""
            value_text = f"{value / 1000.0:.3f}{suffix}" if suffix else str(value)
            print(f"  {key}: {value_text}")


def _print_pose(name: str, pose: np.ndarray, semantics: str) -> None:
    """Print a stored [x, y, z, roll, pitch, yaw] pose with useful units."""
    xyz = pose[:3]
    rpy = pose[3:]
    print(f"  {name} [x,y,z,roll,pitch,yaw]: "
          f"[{xyz[0]:+.5f}, {xyz[1]:+.5f}, {xyz[2]:+.5f}, "
          f"{rpy[0]:+.5f}, {rpy[1]:+.5f}, {rpy[2]:+.5f}]")
    print(f"    xyz = {np.array2string(xyz, precision=5)} m; "
          f"rpy = {np.array2string(rpy, precision=5)} rad "
          f"/ {np.array2string(np.degrees(rpy), precision=2)} deg")
    print(f"    {semantics}")


def show_pose_frame(path: Path, frame: int) -> None:
    """Print the two pose streams aligned to one canonical RGB frame."""
    root, group, audit, _meta = _groups(path)
    t = len(group["timestamps"])
    if not 0 <= frame < t:
        raise SystemExit(f"--pose-frame must be within [0, {t - 1}]")
    keys = set(group.array_keys())
    print(f"\n[POSE] {path.name} frame={frame} canonical_timestamp_ns={int(group['timestamps'][frame])}")
    if "right_wrist_pose" not in keys:
        raise SystemExit("该 episode 没有 right_wrist_pose；请重启采集程序后录制一个新 episode。")

    _print_pose(
        "right_wrist_pose",
        group["right_wrist_pose"][frame],
        str(root.attrs.get("right_wrist_pose_semantics", "waist -> r_wrist")),
    )
    for key in ("right_wrist_pose_source_timestamp_us", "right_wrist_pose_source_seq", "right_wrist_pose_age_us"):
        if audit is not None and key in audit:
            value = int(audit[key][frame])
            print(f"    {key}: {value / 1000.0:.3f} ms" if key.endswith("age_us") else f"    {key}: {value}")

    if "camera_ego_pose" in keys:
        _print_pose(
            "camera_ego_pose",
            group["camera_ego_pose"][frame],
            str(root.attrs.get("camera_ego_pose_semantics", "Gemini IMU pose")),
        )
        for key in ("camera_ego_pose_source_timestamp_ns", "camera_ego_pose_source_seq", "camera_ego_pose_age_us"):
            if audit is not None and key in audit:
                value = int(audit[key][frame])
                print(f"    {key}: {value / 1000.0:.3f} ms" if key.endswith("age_us") else f"    {key}: {value}")
    else:
        print("  camera_ego_pose: unavailable (Gemini IMU capture disabled or legacy episode)")
    for key, fallback in (("camera_tracker_pose", "PICO camera tracker pose"), ("right_wrist_tracker_pose", "PICO wrist tracker pose")):
        if key in keys:
            _print_pose(key, group[key][frame], str(root.attrs.get(f"{key}_semantics", fallback)))
            for audit_key in (f"{key}_source_timestamp_ns", f"{key}_source_seq", f"{key}_age_us"):
                if audit is not None and audit_key in audit:
                    value = int(audit[audit_key][frame])
                    print(f"    {audit_key}: {value / 1000.0:.3f} ms" if audit_key.endswith("age_us") else f"    {audit_key}: {value}")


def nearest_frame_indices(source_ns: np.ndarray, target_ns: np.ndarray) -> np.ndarray:
    """Match actual timestamps; equal distances choose the earlier frame."""
    source_ns = np.asarray(source_ns, dtype=np.int64)
    target_ns = np.asarray(target_ns, dtype=np.int64)
    if not len(source_ns) or np.any(np.diff(source_ns) <= 0):
        raise ValueError("RGB timestamps must be nonempty and strictly increasing")
    right = np.clip(np.searchsorted(source_ns, target_ns), 0, len(source_ns) - 1)
    left = np.maximum(right - 1, 0)
    return np.where(abs(source_ns[left] - target_ns) <= abs(source_ns[right] - target_ns), left, right)


def playback_index(timestamps: np.ndarray, elapsed_ns: int) -> int:
    """Use one absolute playback clock, including time spent decoding frames."""
    target = int(timestamps[0]) + elapsed_ns
    return max(0, min(len(timestamps) - 1, int(np.searchsorted(timestamps, target, side="right")) - 1))


def comparison_timestamps(ego_ns, main_ns, ego_delay_ms: float):
    """Subtract observed ego latency for comparison only; never rewrite data."""
    if not np.isfinite(ego_delay_ms):
        raise ValueError("ego delay must be finite")
    ego = np.asarray(ego_ns, dtype=np.int64) - round(ego_delay_ms * 1e6)
    main = np.asarray(main_ns, dtype=np.int64)
    for stamps in (ego, main):
        if not len(stamps) or np.any(np.diff(stamps) <= 0):
            raise ValueError("RGB timestamps must be nonempty and strictly increasing")
    targets = ego[(ego >= main[0]) & (ego <= main[-1])]
    if not len(targets):
        raise ValueError("camera timelines do not overlap after delay compensation")
    return ego, main, targets


def play_rgb(path: Path, requested_fps: float, camera: str, ego_delay_ms: float = 0.0) -> None:
    """Play one or both cameras on a shared recorded-time clock."""
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("视频播放需要 OpenCV：pip install opencv-python") from exc
    root, group, _audit, _meta = _groups(path)
    views = []
    if camera in {"ego", "both"}:
        if "camera_ego_rgb" not in group:
            raise SystemExit(f"{path.name} lacks camera_ego_rgb")
        views.append(("ego", group["camera_ego_rgb"], group["timestamps"][:]))
    if camera in {"main", "both"}:
        if "camera_main_rgb" in group:
            views.append(("main", group["camera_main_rgb"], group["timestamps"][:]))
        elif "streams" in root and "camera_main_rgb_raw" in root["streams"]:
            streams = root["streams"]
            views.append(("main", streams["camera_main_rgb_raw"], streams["camera_main_rgb_timestamp_ns"][:]))
        else:
            raise SystemExit(f"{path.name} lacks Gemini RGB data")
    timestamps = views[0][2]
    if camera == "both":
        ego_stamps, main_stamps, timestamps = comparison_timestamps(
            views[0][2], views[1][2], ego_delay_ms
        )
        views = [(views[0][0], views[0][1], ego_stamps),
                 (views[1][0], views[1][1], main_stamps)]
    if not len(timestamps):
        raise SystemExit("RGB 序列为空")
    maps = [nearest_frame_indices(stamps, timestamps) for _, _, stamps in views]
    fps = _hz(timestamps)
    speed = requested_fps / fps if requested_fps > 0 and fps > 0 else 1.0
    title = f"FTP-1 {camera} RGB: {path.name}"
    frame, paused = 0, False
    origin = time.monotonic()
    period_ns = int(np.median(np.diff(timestamps))) if len(timestamps) > 1 else 33_333_333
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 1280 if camera == "both" else 960, 520)
    print("播放控制：Space=暂停/继续，←/→ 或 a/d=逐帧，q/Esc=退出；显示相对时间和各路真实帧号。")
    if camera == "both":
        print(f"双路按时间戳配对；ego 延迟补偿={ego_delay_ms:.3f} ms（仅回放，未修改数据）。")
    try:
        while True:
            if not paused:
                elapsed_ns = int((time.monotonic() - origin) * speed * 1e9)
                if elapsed_ns > int(timestamps[-1] - timestamps[0]) + period_ns:
                    break
                frame = playback_index(timestamps, elapsed_ns)
            panels = []
            target = int(timestamps[frame])
            for (label, rgb, stamps), mapping in zip(views, maps):
                index = int(mapping[frame])
                im = np.ascontiguousarray(rgb[index][..., ::-1])
                if camera == "both":
                    scale = min(640 / im.shape[1], 480 / im.shape[0])
                    scaled = cv2.resize(im, (max(1, round(im.shape[1]*scale)), max(1, round(im.shape[0]*scale))))
                    im = np.zeros((520, 640, 3), dtype=np.uint8)
                    h, w = scaled.shape[:2]
                    im[40:40+h, (640-w)//2:(640-w)//2+w] = scaled
                delta_ms = (int(stamps[index]) - target) / 1e6
                caption = f"{label} {index+1}/{len(rgb)}  t={(target-int(timestamps[0]))/1e9:.3f}s  dt={delta_ms:+.1f}ms"
                cv2.putText(im, caption, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, .55, (0,255,0), 1, cv2.LINE_AA)
                panels.append(im)
            cv2.imshow(title, np.hstack(panels) if len(panels) > 1 else panels[0])
            key = cv2.waitKey(0 if paused else 1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                paused = not paused
                origin = time.monotonic() - (int(timestamps[frame])-int(timestamps[0])) / 1e9 / speed
            elif key in (81, ord("a"), 83, ord("d")):
                frame = max(0, min(len(timestamps)-1, frame + (-1 if key in (81, ord("a")) else 1)))
                paused = True
    finally:
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="FTP-1 data directory or one episode .zarr (defaults to the reader's directory)",
    )
    parser.add_argument("--episode", help="episode filename, e.g. episode_000000.zarr")
    parser.add_argument("--latest", action="store_true", help="automatically select the most recently written episode")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--strict-quality", action="store_true", help="with --summary/frame, return failure when timing or joint-smoothness quality gates are exceeded")
    parser.add_argument("--frame", type=int)
    parser.add_argument("--wuji-frame", type=int, metavar="N", help="print detailed saved Wuji data for frame N")
    parser.add_argument("--pose-frame", type=int, metavar="N", help="print saved right_wrist_pose and camera_ego_pose for frame N")
    parser.add_argument("--tactile-matrix", action="store_true", help="with --wuji-frame, also print the full 24x31 tactile matrix")
    parser.add_argument("--play", action="store_true", help="play the selected episode RGB like a video")
    parser.add_argument("--play-fps", type=float, default=0.0, help="0=recorded FPS (default); otherwise playback FPS")
    parser.add_argument("--camera", choices=("ego", "main", "both"), default="ego", help="ego=first-person configured camera; main=third-person Gemini; both=synchronized timestamp comparison")
    parser.add_argument("--ego-delay-ms", type=float, default=0.0,
                        help="with --play --camera both: subtract observed ego latency in ms for comparison only")
    parser.add_argument("--render-sync", type=Path, metavar="OUTPUT.mp4",
                        help="render video, hand skeleton, wrist trajectory and EMG on one timestamp axis")
    parser.add_argument("--render-fps", type=float, default=30.0)
    args = parser.parse_args()
    if not np.isfinite(args.ego_delay_ms):
        parser.error("--ego-delay-ms must be finite")
    if args.ego_delay_ms and (not args.play or args.camera != "both"):
        parser.error("--ego-delay-ms requires --play --camera both")
    root = args.path.expanduser().resolve()
    paths = _episode_paths(root)
    if args.episode:
        paths = [path for path in paths if path.name == args.episode]
        if len(paths) > 1:
            matches = "\n".join(f"  {path.relative_to(root)}" for path in paths)
            raise SystemExit(
                f"{args.episode} 在多个采集目录中重名：\n{matches}\n"
                "请用 --path 指向具体的 被采集者/tN/有手套或无手套 目录。"
            )
    if not paths:
        raise SystemExit(f"No episode_*.zarr found under {root}")
    if args.latest:
        paths = [max(paths, key=lambda path: path.stat().st_mtime_ns)]
    if args.list:
        for path in paths:
            try:
                print(path.relative_to(root))
            except ValueError:
                print(path)
        return
    if args.render_sync:
        if len(paths) != 1:
            raise SystemExit("--render-sync 请配合 --episode、--latest，或把 --path 指向单个 .zarr")
        renderer = Path(__file__).with_name("render_episode_sync.py")
        if not renderer.is_file():
            renderer = Path(__file__).resolve().parents[1] / "runtime/render_episode_sync.py"
        if not renderer.is_file():
            raise SystemExit(f"缺少同步渲染器：{renderer}")
        output = args.render_sync.expanduser()
        if not output.is_absolute():
            # Relative output belongs beside the selected episode, so the
            # command is portable when run from any shell working directory.
            output = paths[0].parent / output
        import subprocess
        subprocess.run([sys.executable, str(renderer), str(paths[0]), "--output",
                        str(output), "--fps", str(args.render_fps)], check=True)
        return
    if args.play:
        if len(paths) != 1:
            raise SystemExit("--play 请配合 --episode 指定一条 episode，或把 --path 指向单个 .zarr")
        play_rgb(paths[0], args.play_fps, args.camera, args.ego_delay_ms)
        return
    if args.wuji_frame is not None:
        if len(paths) != 1:
            raise SystemExit("--wuji-frame 请配合 --episode、--latest，或把 --path 指向单个 .zarr")
        show_wuji_frame(paths[0], args.wuji_frame, args.tactile_matrix)
        return
    if args.pose_frame is not None:
        if len(paths) != 1:
            raise SystemExit("--pose-frame 请配合 --episode、--latest，或把 --path 指向单个 .zarr")
        show_pose_frame(paths[0], args.pose_frame)
        return
    selected = paths if args.summary else [paths[-1]]
    success = all(validate(path, args.frame, strict_quality=args.strict_quality) for path in selected)
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
