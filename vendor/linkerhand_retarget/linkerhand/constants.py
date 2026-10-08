import enum
from pathlib import Path
from typing import Optional

import numpy as np

OPERATOR2MANO_RIGHT = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ]
)

OPERATOR2MANO_LEFT = np.array(
    [
        [0, 0, -1],
        [1, 0, 0],
        [0, -1, 0],
    ]
)


class RobotName(enum.Enum):
    o7 = enum.auto() 
    o7v1 = enum.auto()
    o7v3 = enum.auto()
    o6 = enum.auto() 
    l6 = enum.auto() 
    l7 = enum.auto()
    l10 = enum.auto()
    l10v7 = enum.auto()
    l20 = enum.auto()
    l20lite = enum.auto()
    l21 = enum.auto()
    o20 = enum.auto()
    o30 = enum.auto()
    o30i = enum.auto()
    l25 = enum.auto()
    g20 = enum.auto()


class RetargetingType(enum.Enum):
    vector = enum.auto()  # For teleoperation, no finger closing prior
    position = enum.auto()  # For offline data processing, especially hand-object interaction data
    dexpilot = enum.auto()  # For teleoperation, with finger closing prior
    projection = enum.auto()


class HandType(enum.Enum):
    right = enum.auto()
    left = enum.auto()


class DataSource(enum.Enum):
    motion = enum.auto()
    video = enum.auto()
    vr = enum.auto()


class MotionSource(enum.Enum):
    vtrdyn = enum.auto()
    udexreal = enum.auto()
    udexrealv2t = enum.auto()
    linkerforce = enum.auto()
    sensenova = enum.auto()
    linkermcg = enum.auto()
    linkereg1 = enum.auto()
    linkereg2 = enum.auto()


ROBOT_NAME_MAP = {
    RobotName.o7: "linker_hand_o7",
    RobotName.l7: "linker_hand_l7",
    RobotName.o6: "linker_hand_o6",
    RobotName.l6: "linker_hand_l6",
    RobotName.o7v1: "linker_hand_o7v1",
    RobotName.o7v3: "linker_hand_o7v3",    
    RobotName.l10: "linker_hand_l10",
    RobotName.l10v7: "linker_hand_l10v7",   
    RobotName.l20: "linker_hand_l20",
    RobotName.l20lite: "linker_hand_l20lite",
    RobotName.l21: "linker_hand_l21",
    RobotName.o20: "linker_hand_o20",
    RobotName.o30: "linker_hand_o30",
    RobotName.o30i: "linker_hand_o30i",
    RobotName.l25: "linker_hand_l25",
    RobotName.g20: "linker_hand_g20",
}

ROBOT_NAMES = list(ROBOT_NAME_MAP.keys())


ROBOT_LEN_MAP ={
    RobotName.o7: 7,
    RobotName.l7: 7,
    RobotName.o6: 6,
    RobotName.l6: 6,
    RobotName.o7v1: 7,
    RobotName.o7v3: 7,
    RobotName.l10: 10,
    RobotName.l10v7: 10,
    RobotName.l20: 20,
    RobotName.l20lite: 10,
    RobotName.l21: 25,
    RobotName.o20: 20,
    RobotName.o30: 20,
    RobotName.o30i: 20,
    RobotName.l25: 20,
    RobotName.g20: 20,
}

ROBOT_LEN = list(ROBOT_LEN_MAP.keys())


def get_default_config_path(
        robot_name: RobotName, retargeting_type: RetargetingType, hand_type: HandType
) -> Optional[Path]:
    config_path = Path(__file__).parent.parent / "config"
    if retargeting_type is RetargetingType.position:
        config_path = config_path / "offline"
    else:
        config_path = config_path / "teleop"

    robot_name_str = ROBOT_NAME_MAP[robot_name]
    hand_type_str = hand_type.name
    if "gripper" in robot_name_str:  # For gripper robots, only use gripper config file.
        if retargeting_type == RetargetingType.dexpilot:
            config_name = f"{robot_name_str}_dexpilot.yml"
        else:
            config_name = f"{robot_name_str}.yml"
    else:
        if retargeting_type == RetargetingType.dexpilot:
            config_name = f"{robot_name_str}_{hand_type_str}_dexpilot.yml"
        else:
            config_name = f"{robot_name_str}_{hand_type_str}.yml"
    return config_path / config_name

OPERATOR2MANO = {
    HandType.right: OPERATOR2MANO_RIGHT,
    HandType.left: OPERATOR2MANO_LEFT,
}
