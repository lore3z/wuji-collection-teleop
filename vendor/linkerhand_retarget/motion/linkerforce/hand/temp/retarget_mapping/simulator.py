from __future__ import annotations

from pathlib import Path
from typing import Any

from haocun_glove.calibration_store import load_raw_calibration
from haocun_glove.coordinate_projection1 import project_hand_positions
from haocun_glove.models import GloveFrame, HandFrame, Vec3
from haocun_glove.o20_mapping import (
    O20_INDEX_ABD_CHANNEL,
    O20_INDEX_BASE_CHANNEL,
    O20_INDEX_MIDDLE_CHANNEL,
    O20_MIDDLE_ABD_CHANNEL,
    O20_MIDDLE_BASE_CHANNEL,
    O20_MIDDLE_MIDDLE_CHANNEL,
    O20_OPEN_POSE,
    O20_PINKY_ABD_CHANNEL,
    O20_PINKY_BASE_CHANNEL,
    O20_PINKY_MIDDLE_CHANNEL,
    O20_RING_ABD_CHANNEL,
    O20_RING_BASE_CHANNEL,
    O20_RING_MIDDLE_CHANNEL,
    O20_THUMB_ABD_CHANNEL,
    O20_THUMB_BASE_CHANNEL,
    O20_THUMB_MIDDLE_CHANNEL,
    O20_THUMB_ROTATE_CHANNEL,
    o20_pose_to_motor_angles,
)
from haocun_glove.splay_mapping import splay_features_from_hand
from haocun_glove.urdf_tools import apply_mimic, parse_urdf_metadata

from .profiles import MappingProfile, get_profile


def simulate_saved_pose(
    *,
    project_root: str | Path,
    profile: str | MappingProfile,
    pose_name: str,
) -> dict[str, Any]:
    root = Path(project_root)
    mapping_profile = _profile(profile)
    calibration = load_raw_calibration(root)
    pose = calibration.get("nodes", {}).get(mapping_profile.hand_side, {}).get("poses", {}).get(pose_name)
    if not pose:
        raise ValueError(f"unknown raw calibration pose: {pose_name}")
    glove_raw = pose.get("glove_raw") or {}
    values = _float_dict(glove_raw.get("values", {}))
    if not values:
        raise ValueError(f"raw calibration pose has no glove values: {pose_name}")
    return map_glove_values(
        project_root=root,
        profile=mapping_profile,
        glove_values=values,
        pose_name=pose_name,
        received_at=glove_raw.get("received_at"),
    )


def map_glove_values(
    *,
    project_root: str | Path,
    profile: str | MappingProfile,
    glove_values: dict[str, Any],
    pose_name: str | None = None,
    received_at: Any = None,
) -> dict[str, Any]:
    root = Path(project_root)
    mapping_profile = _profile(profile)
    values = _float_dict(glove_values)
    frame = glove_frame_from_flat_values(values, received_at=_received_at_float(received_at))
    mapped = _map_frame(root, mapping_profile, frame)
    return {
        "profile": mapping_profile.key,
        "pose_name": pose_name,
        "hand_model": mapping_profile.hand_model,
        "hand_side": mapping_profile.hand_side,
        "source": "glove_calibration" if pose_name is not None else "glove_values",
        "input_model": mapping_profile.input_model,
        "glove_values": values,
        "source_features": mapped["source_features"],
        "correspondence": mapped["correspondence"],
        "urdf_joints": mapped["urdf_joints"],
        "device_pose": mapped["device_pose"],
        "motor_values": mapped["motor_values"],
        "received_at": received_at,
    }


def glove_frame_from_flat_values(values: dict[str, float], *, received_at: float = 0.0) -> GloveFrame:
    return GloveFrame(
        left_hand=_hand_from_flat_values(values, side="left", suffix="L"),
        right_hand=_hand_from_flat_values(values, side="right", suffix="R"),
        received_at=received_at,
        raw={},
    )


def extract_source_features(frame: GloveFrame, profile: str | MappingProfile) -> dict[str, float]:
    mapping_profile = _profile(profile)
    side = mapping_profile.hand_side
    hand = frame.right_hand if side == "right" else frame.left_hand
    suffix = "R" if side == "right" else "L"
    qpos = project_hand_positions(hand, hand_suffix=suffix)
    splay_features = splay_features_from_hand(hand, hand_suffix=suffix)
    features: dict[str, float] = {
        "thumb.roll_xy": qpos[0],
        "thumb.yaw_yz": qpos[1],
        "thumb.root_pitch": qpos[2],
        "thumb.end_pitch": qpos[3],
        "thumb.yaw_xz": qpos[25],
        "thumb.roll_yx": qpos[26],
        "thumb.root_angle": qpos[27],
        "middle.fixed": 0.0,
    }
    for finger_index, finger_name in enumerate(("index", "middle", "ring", "pinky")):
        base = 5 + 5 * finger_index
        features[f"{finger_name}.side_yz"] = qpos[base]
        features[f"{finger_name}.root_flexion_yz"] = qpos[base + 2]
        features[f"{finger_name}.middle_flexion_yz"] = qpos[base + 3]
        features[f"{finger_name}.end_flexion_yz"] = qpos[base + 4]
        features[f"{finger_name}.splay_xz"] = splay_features[f"{finger_name}.splay_x"]
    return features


def _map_frame(root: Path, profile: MappingProfile, frame: GloveFrame) -> dict[str, Any]:
    if profile.device_backend != "o20_canfd":
        raise ValueError(f"unsupported mapping profile backend: {profile.device_backend}")
    metadata = parse_urdf_metadata(root / profile.urdf_path)
    source_features = extract_source_features(frame, profile)
    correspondence = _build_correspondence(root, profile)
    urdf_joints = _map_source_features_to_urdf_joints(source_features, correspondence, metadata)
    pose = _o20_pose_from_urdf_joints(urdf_joints, metadata=metadata)
    return {
        "source_features": source_features,
        "correspondence": correspondence,
        "urdf_joints": urdf_joints,
        "device_pose": pose,
        "motor_values": o20_pose_to_motor_angles(pose),
    }


def _build_correspondence(root: Path, profile: MappingProfile) -> dict[str, Any]:
    calibration = load_raw_calibration(root)
    target_calibration = _load_target_calibration(root, profile)
    poses = calibration.get("nodes", {}).get(profile.hand_side, {}).get("poses", {})
    output: dict[str, Any] = {}
    for channel_name, channel in profile.channels.items():
        anchors = []
        for anchor_pose in channel.anchor_poses:
            pose = poses.get(anchor_pose) or {}
            values = _float_dict((pose.get("glove_raw") or {}).get("values", {}))
            if not values:
                continue
            frame = glove_frame_from_flat_values(values)
            source_features = extract_source_features(frame, profile)
            target_joints = target_calibration.get("poses", {}).get(anchor_pose, {}).get("urdf_joints", {})
            anchors.append(
                {
                    "pose": anchor_pose,
                    "source": source_features.get(channel.source_feature),
                    "target_urdf": target_joints.get(channel.target_joint),
                }
            )
        output[channel_name] = {
            "source_feature": channel.source_feature,
            "target_joint": channel.target_joint,
            "mode": channel.mode,
            "anchors": anchors,
        }
    return output


def _map_source_features_to_urdf_joints(
    source_features: dict[str, float],
    correspondence: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, float]:
    joints: dict[str, float] = {}
    for channel in correspondence.values():
        target_joint = str(channel["target_joint"])
        source_feature = str(channel["source_feature"])
        source_value = source_features.get(source_feature)
        if source_value is None:
            continue
        target_value = _interpolate_urdf_value(
            float(source_value),
            anchors=list(channel.get("anchors", [])),
            mode=str(channel.get("mode", "piecewise_linear")),
        )
        if target_value is not None:
            joints[target_joint] = target_value
    return apply_mimic(metadata, joints)


def _interpolate_urdf_value(value: float, *, anchors: list[dict[str, Any]], mode: str) -> float | None:
    points = [
        (float(anchor["source"]), float(anchor["target_urdf"]))
        for anchor in anchors
        if anchor.get("source") is not None and anchor.get("target_urdf") is not None
    ]
    if not points:
        return None
    if mode == "fixed" or len(points) == 1:
        return points[0][1]
    if mode == "linear_fit":
        return _linear_fit_value(value, points)
    return _piecewise_linear_value(value, points)


def _linear_fit_value(value: float, points: list[tuple[float, float]]) -> float:
    sources = [source for source, _target in points]
    targets = [target for _source, target in points]
    clamped = min(max(value, min(sources)), max(sources))
    count = len(points)
    sum_x = sum(sources)
    sum_y = sum(targets)
    sum_xx = sum(source * source for source in sources)
    sum_xy = sum(source * target for source, target in points)
    denominator = count * sum_xx - sum_x * sum_x
    if abs(denominator) < 1e-12:
        return sum_y / count
    slope = (count * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / count
    return slope * clamped + intercept


def _piecewise_linear_value(value: float, points: list[tuple[float, float]]) -> float:
    sorted_points = sorted(points, key=lambda item: item[0])
    if value <= sorted_points[0][0]:
        return sorted_points[0][1]
    if value >= sorted_points[-1][0]:
        return sorted_points[-1][1]
    for (start_source, start_target), (end_source, end_target) in zip(sorted_points, sorted_points[1:]):
        if start_source <= value <= end_source:
            delta = end_source - start_source
            if abs(delta) < 1e-12:
                return start_target
            ratio = (value - start_source) / delta
            return start_target + ratio * (end_target - start_target)
    return sorted_points[-1][1]


def _load_target_calibration(root: Path, profile: MappingProfile) -> dict[str, Any]:
    path = root / profile.target_calibration_path
    if not path.exists():
        return {"poses": {}}
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _o20_pose_to_urdf_joints(pose: list[int], metadata: dict[str, Any]) -> dict[str, float]:
    pose_to_joint = {
        O20_THUMB_BASE_CHANNEL: ("thumb_cmc_pitch", False),
        O20_THUMB_MIDDLE_CHANNEL: ("thumb_mcp", False),
        O20_THUMB_ABD_CHANNEL: ("thumb_cmc_yaw", True),
        O20_THUMB_ROTATE_CHANNEL: ("thumb_cmc_roll", False),
        O20_INDEX_ABD_CHANNEL: ("index_mcp_roll", False),
        O20_INDEX_BASE_CHANNEL: ("index_mcp_pitch", False),
        O20_INDEX_MIDDLE_CHANNEL: ("index_dip", False),
        O20_MIDDLE_ABD_CHANNEL: ("middle_mcp_roll", False),
        O20_MIDDLE_BASE_CHANNEL: ("middle_mcp_pitch", False),
        O20_MIDDLE_MIDDLE_CHANNEL: ("middle_dip", False),
        O20_RING_ABD_CHANNEL: ("ring_mcp_roll", False),
        O20_RING_BASE_CHANNEL: ("ring_mcp_pitch", False),
        O20_RING_MIDDLE_CHANNEL: ("ring_dip", False),
        O20_PINKY_ABD_CHANNEL: ("pinky_mcp_roll", False),
        O20_PINKY_BASE_CHANNEL: ("pinky_mcp_pitch", False),
        O20_PINKY_MIDDLE_CHANNEL: ("pinky_dip", False),
    }
    values: dict[str, float] = {}
    joints = metadata.get("joints", {})
    for pose_index, (joint_name, reverse) in pose_to_joint.items():
        if joint_name == "middle_mcp_roll":
            values[joint_name] = 0.0
            continue
        joint = joints.get(joint_name)
        if not joint or not joint.get("limit"):
            continue
        limit = joint["limit"]
        ratio = _uint8_ratio(pose[pose_index])
        if reverse:
            ratio = 1.0 - ratio
        value = float(limit["lower"]) + ratio * (float(limit["upper"]) - float(limit["lower"]))
        if joint_name in {"index_mcp_roll", "ring_mcp_roll", "pinky_mcp_roll"}:
            value = min(value, 0.0)
        values[joint_name] = value
    return apply_mimic(metadata, values)


def _o20_pose_from_urdf_joints(urdf_joints: dict[str, float], *, metadata: dict[str, Any]) -> list[int]:
    joint_to_pose = {
        "thumb_cmc_pitch": (O20_THUMB_BASE_CHANNEL, False),
        "thumb_mcp": (O20_THUMB_MIDDLE_CHANNEL, False),
        "thumb_cmc_yaw": (O20_THUMB_ABD_CHANNEL, True),
        "thumb_cmc_roll": (O20_THUMB_ROTATE_CHANNEL, False),
        "index_mcp_roll": (O20_INDEX_ABD_CHANNEL, False),
        "index_mcp_pitch": (O20_INDEX_BASE_CHANNEL, False),
        "index_dip": (O20_INDEX_MIDDLE_CHANNEL, False),
        "middle_mcp_pitch": (O20_MIDDLE_BASE_CHANNEL, False),
        "middle_dip": (O20_MIDDLE_MIDDLE_CHANNEL, False),
        "ring_mcp_roll": (O20_RING_ABD_CHANNEL, False),
        "ring_mcp_pitch": (O20_RING_BASE_CHANNEL, False),
        "ring_dip": (O20_RING_MIDDLE_CHANNEL, False),
        "pinky_mcp_roll": (O20_PINKY_ABD_CHANNEL, False),
        "pinky_mcp_pitch": (O20_PINKY_BASE_CHANNEL, False),
        "pinky_dip": (O20_PINKY_MIDDLE_CHANNEL, False),
    }
    pose = list(O20_OPEN_POSE)
    pose[O20_MIDDLE_ABD_CHANNEL] = 128
    for joint_name, (pose_index, reverse) in joint_to_pose.items():
        joint = metadata.get("joints", {}).get(joint_name)
        if not joint or not joint.get("limit"):
            continue
        limit = joint["limit"]
        pose[pose_index] = _urdf_joint_to_uint8(
            urdf_joints.get(joint_name, 0.0),
            float(limit["lower"]),
            float(limit["upper"]),
            reverse=reverse,
        )
    return pose


def _urdf_joint_to_uint8(value: float, lower: float, upper: float, *, reverse: bool = False) -> int:
    if upper == lower:
        return 255
    clamped = min(upper, max(lower, float(value)))
    ratio = (clamped - lower) / (upper - lower)
    if reverse:
        ratio = 1.0 - ratio
    return round(ratio * 255.0)


def _hand_from_flat_values(values: dict[str, float], *, side: str, suffix: str) -> HandFrame:
    prefix = f"{side}."
    joints: dict[str, Vec3] = {}
    for finger in ("Thumb", "Index", "Middle", "Ring", "Pinky"):
        for point_index in range(1, 5):
            name = f"hc_{finger}{point_index}_{suffix}"
            joints[name] = Vec3(
                values[f"{prefix}{name}.x"],
                values[f"{prefix}{name}.y"],
                values[f"{prefix}{name}.z"],
            )
    return HandFrame(
        palm_origin=Vec3(
            values[f"{prefix}PalmOriginLocal.x"],
            values[f"{prefix}PalmOriginLocal.y"],
            values[f"{prefix}PalmOriginLocal.z"],
        ),
        joints=joints,
        raw={},
    )


def _profile(profile: str | MappingProfile) -> MappingProfile:
    if isinstance(profile, MappingProfile):
        return profile
    return get_profile(profile)


def _float_dict(values: dict[str, Any]) -> dict[str, float]:
    return {str(key): float(value) for key, value in values.items()}


def _received_at_float(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _uint8_ratio(value: Any) -> float:
    return min(1.0, max(0.0, float(value) / 255.0))


__all__ = ["extract_source_features", "glove_frame_from_flat_values", "map_glove_values", "simulate_saved_pose"]
