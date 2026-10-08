from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class MappingChannel:
    source_feature: str
    target_joint: str
    anchor_poses: tuple[str, ...]
    mode: str = "piecewise_linear"


@dataclass(frozen=True, slots=True)
class MappingProfile:
    key: str
    hand_model: str
    hand_side: str
    input_model: str
    urdf_path: Path
    glove_calibration_path: Path
    target_calibration_path: Path
    channels: dict[str, MappingChannel]
    device_backend: str


def get_profile(profile: str) -> MappingProfile:
    key = profile.strip().lower()
    try:
        return PROFILES[key]
    except KeyError as exc:
        raise ValueError(f"unknown retarget mapping profile: {profile}") from exc


O20_RIGHT_CHANNELS = {
    "thumb_cmc_pitch": MappingChannel(
        source_feature="thumb.root_pitch",
        target_joint="thumb_cmc_pitch",
        anchor_poses=(
            "open_four_fingers_together",
            "pinch_index_thumb",
            "pinch_middle_thumb",
            "pinch_ring_thumb",
            "fist",
        ),
    ),
    "thumb_mcp": MappingChannel(
        source_feature="thumb.end_pitch",
        target_joint="thumb_mcp",
        anchor_poses=("open_four_fingers_together", "pinch_index_thumb", "pinch_middle_thumb", "pinch_ring_thumb"),
    ),
    "thumb_cmc_yaw": MappingChannel(
        source_feature="thumb.yaw_xz",
        target_joint="thumb_cmc_yaw",
        anchor_poses=("open_four_fingers_together", "pinch_index_thumb", "pinch_middle_thumb", "pinch_ring_thumb"),
        mode="linear_fit",
    ),
    "thumb_cmc_roll": MappingChannel(
        source_feature="thumb.roll_xy",
        target_joint="thumb_cmc_roll",
        anchor_poses=("open_four_fingers_together", "pinch_index_thumb", "pinch_middle_thumb", "pinch_ring_thumb"),
    ),
    "index_mcp_roll": MappingChannel(
        source_feature="index.splay_xz",
        target_joint="index_mcp_roll",
        anchor_poses=("open_four_fingers_together", "open_all_spread"),
        mode="linear",
    ),
    "index_mcp_pitch": MappingChannel(
        source_feature="index.root_flexion_yz",
        target_joint="index_mcp_pitch",
        anchor_poses=("open_four_fingers_together", "pinch_index_thumb", "fist"),
    ),
    "index_dip": MappingChannel(
        source_feature="index.middle_flexion_yz",
        target_joint="index_dip",
        anchor_poses=("open_four_fingers_together", "pinch_index_thumb", "fist"),
    ),
    "middle_mcp_roll": MappingChannel(
        source_feature="middle.fixed",
        target_joint="middle_mcp_roll",
        anchor_poses=("open_four_fingers_together",),
        mode="fixed",
    ),
    "middle_mcp_pitch": MappingChannel(
        source_feature="middle.root_flexion_yz",
        target_joint="middle_mcp_pitch",
        anchor_poses=("open_four_fingers_together", "pinch_middle_thumb", "fist"),
    ),
    "middle_dip": MappingChannel(
        source_feature="middle.middle_flexion_yz",
        target_joint="middle_dip",
        anchor_poses=("open_four_fingers_together", "pinch_middle_thumb", "fist"),
    ),
    "ring_mcp_roll": MappingChannel(
        source_feature="ring.splay_xz",
        target_joint="ring_mcp_roll",
        anchor_poses=("open_four_fingers_together", "open_all_spread"),
        mode="linear",
    ),
    "ring_mcp_pitch": MappingChannel(
        source_feature="ring.root_flexion_yz",
        target_joint="ring_mcp_pitch",
        anchor_poses=("open_four_fingers_together", "pinch_ring_thumb", "fist"),
    ),
    "ring_dip": MappingChannel(
        source_feature="ring.middle_flexion_yz",
        target_joint="ring_dip",
        anchor_poses=("open_four_fingers_together", "pinch_ring_thumb", "fist"),
    ),
    "pinky_mcp_roll": MappingChannel(
        source_feature="pinky.splay_xz",
        target_joint="pinky_mcp_roll",
        anchor_poses=("open_four_fingers_together", "open_all_spread"),
        mode="linear",
    ),
    "pinky_mcp_pitch": MappingChannel(
        source_feature="pinky.root_flexion_yz",
        target_joint="pinky_mcp_pitch",
        anchor_poses=("open_four_fingers_together", "pinch_ring_thumb", "fist"),
    ),
    "pinky_dip": MappingChannel(
        source_feature="pinky.middle_flexion_yz",
        target_joint="pinky_dip",
        anchor_poses=("open_four_fingers_together", "pinch_ring_thumb", "fist"),
    ),
}


PROFILES = {
    "o20-right": MappingProfile(
        key="o20-right",
        hand_model="O20",
        hand_side="right",
        input_model="coordinate_projection1",
        urdf_path=Path("urdf/o20/right/linkerhand_o20_right.urdf"),
        glove_calibration_path=Path("calibration/glove_calibration.json"),
        target_calibration_path=Path("calibration/o20-right/raw_calibration.json"),
        channels=O20_RIGHT_CHANNELS,
        device_backend="o20_canfd",
    ),
}


__all__ = ["MappingChannel", "MappingProfile", "get_profile"]
