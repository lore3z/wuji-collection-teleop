#!/usr/bin/env python3

import json
import xml.etree.ElementTree as ET
import queue
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np


# ============================================================
# Compatibility for LinkerHand legacy imports
#
# Some official motion modules still use:
#
#     from linkerhand.handcore import HandCore
#
# instead of:
#
#     from linkerhand_retarget.linkerhand.handcore import HandCore
#
# Therefore expose linkerhand_retarget's package directory as
# an additional Python import root.
# ============================================================

_LINKER_WS = (
    Path.home()
    / "linkerhand-telop-ros2"
)

_LEGACY_IMPORT_CANDIDATES = [
    _LINKER_WS
    / "build"
    / "linkerhand_retarget"
    / "linkerhand_retarget",

    _LINKER_WS
    / "src"
    / "linkerhand_retarget"
    / "linkerhand_retarget",
]

for _path in _LEGACY_IMPORT_CANDIDATES:

    if (
        _path
        / "linkerhand"
        / "__init__.py"
    ).exists():

        sys.path.insert(
            0,
            str(_path),
        )

        break

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from ament_index_python.packages import (
    get_package_share_directory,
)

from linkerhand_retarget.linkerhand.config import (
    HandConfig,
)

from linkerhand_retarget.linkerhand.handcore import (
    HandCore,
)

from linkerhand_retarget.motion.linkerforce.hand.linkerforce_l20 import (
    RightHand as L20RightHand,
)


UDP_HOST = "127.0.0.1"
UDP_PORT = 15120


# ============================================================
# REAL PHYSICAL L20 PAD-TO-PAD ANCHORS
# Measured on this physical hand.
# ============================================================

REAL_PINCH_RAW20 = {
    "index": [
        76, 7, 4, 0, 5,
        162, 158, 144, 127, 104,
        67,
        0, 0, 0, 0,
        211, 173, 8, 6, 10,
    ],

    "middle": [
        74, 7, 3, 0, 5,
        124, 156, 143, 127, 106,
        66,
        0, 0, 0, 0,
        211, 172, 175, 6, 10,
    ],

    "ring": [
        72, 7, 3, 0, 5,
        83, 156, 143, 127, 106,
        66,
        0, 0, 0, 0,
        211, 172, 175, 182, 10,
    ],

    "pinky": [
        72, 7, 3, 0, 5,
        47, 156, 143, 127, 106,
        66,
        0, 0, 0, 0,
        211, 172, 175, 182, 183,
    ],
}

REAL_PINCH_CHANNELS = {
    "index":  [0, 5, 10, 15, 1, 6, 16],
    "middle": [0, 5, 10, 15, 2, 7, 17],
    "ring":   [0, 5, 10, 15, 3, 8, 18],
    "pinky":  [0, 5, 10, 15, 4, 9, 19],
}


# Exact MuJoCo V9.4 L20 pad targets.
SIM_PINCH_U16 = {
    "index": [
         0.1722629, 1.3298130, 0.2459128,
        -0.05445618, 0.2273901, 1.7593180,
        -0.1423602, 0.5355923, 1.7259440,
         0.1337923, 0.7652571, 1.1014890,
         0.07718863, 1.0966840, 0.4270508,
         0.00004125487,
    ],

    "middle": [
        0.0, 0.0, 0.0,
        0.1791707, 1.3300000, 0.2192964,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.2452562, 1.0829600, 0.4725981,
        0.000005440248,
    ],

    "ring": [
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.1799855, 1.3299990, 0.2192697,
        0.0, 0.0, 0.0,
        0.2143044, 1.5699990, 0.5102261,
        0.00001633439,
    ],

    "pinky": [
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.1567072, 1.3298590, 0.2555852,
        0.4142706, 1.5697690, 0.4772227,
        0.0001521863,
    ],
}


TOPIC = "/cb_right_hand_control_cmd"

U16_NAMES = [
    "index_mcp_roll",
    "index_mcp_pitch",
    "index_pip",

    "middle_mcp_roll",
    "middle_mcp_pitch",
    "middle_pip",

    "ring_mcp_roll",
    "ring_mcp_pitch",
    "ring_pip",

    "pinky_mcp_roll",
    "pinky_mcp_pitch",
    "pinky_pip",

    "thumb_cmc_roll",
    "thumb_cmc_yaw",
    "thumb_cmc_pitch",
    "thumb_mcp",
]


# ============================================================
# Locate official configs / assets
# ============================================================

WS = (
    Path.home()
    / "linkerhand-telop-ros2"
)


def find_config_root():

    # ========================================================
    # HandConfig expects:
    #
    #     config_root/
    #         config/
    #             hand_config.yml
    #             base_config.yml
    #             retarget_config.yml
    #             model_config.yml
    #
    # So config_root is the PARENT of the "config" directory.
    # ========================================================

    candidates = []

    # --------------------------------------------------------
    # 1. Imported Python package location
    # --------------------------------------------------------

    try:

        import linkerhand_retarget

        pkg_dir = Path(
            linkerhand_retarget.__file__
        ).resolve().parent

        candidates.append(
            pkg_dir
        )

    except Exception:
        pass

    # --------------------------------------------------------
    # 2. Source/build/install locations
    # --------------------------------------------------------

    candidates += [
        WS
        / "src"
        / "linkerhand_retarget"
        / "linkerhand_retarget",

        WS
        / "build"
        / "linkerhand_retarget"
        / "linkerhand_retarget",

        WS
        / "src"
        / "linkerhand_retarget",

        WS
        / "install"
        / "linkerhand_retarget"
        / "share"
        / "linkerhand_retarget",
    ]

    # --------------------------------------------------------
    # Check explicit candidates first
    # --------------------------------------------------------

    required = [
        "hand_config.yml",
        "base_config.yml",
        "retarget_config.yml",
        "model_config.yml",
    ]

    for root in candidates:

        config_dir = (
            root
            / "config"
        )

        if all(
            (
                config_dir
                / name
            ).exists()
            for name in required
        ):

            print(
                "[CONFIG] found:",
                config_dir,
            )

            return root

    # --------------------------------------------------------
    # 3. Robust fallback:
    # search the workspace for hand_config.yml.
    # --------------------------------------------------------

    for hand_file in WS.rglob(
        "hand_config.yml"
    ):

        if (
            hand_file.parent.name
            != "config"
        ):
            continue

        config_dir = (
            hand_file.parent
        )

        if not all(
            (
                config_dir
                / name
            ).exists()
            for name in required
        ):
            continue

        root = (
            config_dir.parent
        )

        print(
            "[CONFIG] auto-detected:",
            config_dir,
        )

        return root

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    print()
    print(
        "Checked config roots:"
    )

    for root in candidates:
        print(
            "  ",
            root,
        )

    print()

    raise RuntimeError(
        "找不到完整的 LinkerHand config 目录 "
        "(hand_config.yml/base_config.yml/"
        "retarget_config.yml/model_config.yml)"
    )


def find_robot_dir():

    candidates = [
        WS
        / "src"
        / "linkerhand_retarget"
        / "linkerhand_retarget"
        / "assets"
        / "robots"
        / "hands",

        WS
        / "install"
        / "linkerhand_retarget"
        / "lib"
        / "python3.10"
        / "site-packages"
        / "linkerhand_retarget"
        / "assets"
        / "robots"
        / "hands",
    ]

    for c in candidates:

        urdf = (
            c
            / "linker_hand"
            / "l20_right"
            / "linkerhand_l20_right.urdf"
        )

        if urdf.exists():
            return c

    matches = list(
        WS.glob(
            "**/assets/robots/hands/"
            "linker_hand/l20_right/"
            "linkerhand_l20_right.urdf"
        )
    )

    if matches:
        # .../hands/linker_hand/l20_right/file
        return matches[0].parents[2]

    raise RuntimeError(
        "找不到 L20 right URDF"
    )


# ============================================================
# Bridge
# ============================================================

class L20Bridge(Node):

    def __init__(self):

        super().__init__(
            "wuji_l20_hw_bridge"
        )

        self.config_root = (
            find_config_root()
        )

        self.robot_dir = (
            find_robot_dir()
        )

        print()
        print(
            "============================================"
        )
        print(
            "L20 WUJI HARDWARE BRIDGE"
        )
        print(
            "============================================"
        )

        print(
            "config_root =",
            self.config_root,
        )

        print(
            "robot_dir   =",
            self.robot_dir,
        )

        # ------------------------------------------------------
        # Official HandCore
        # ------------------------------------------------------

        hc_cfg = HandConfig(
            str(self.robot_dir),
            str(self.config_root),
        )

        self.handcore = HandCore(
            hc_cfg
        )

        print(
            "official right robot =",
            self.handcore.robot_name_str_r,
        )

        # ------------------------------------------------------
        # IMPORTANT:
        #
        # This physical L20 is exposed by the official driver
        # configuration as "g20".  That is intentional for this
        # hardware/firmware stack.
        #
        # Do NOT force base_config.yml from g20 -> l20 here.
        # HandCore must keep using the official hardware mapping
        # selected by the installed configuration.
        # ------------------------------------------------------

        if self.handcore.robot_name_str_r not in (
            "l20",
            "g20",
        ):

            raise RuntimeError(
                "Unsupported right-hand hardware mapping: "
                f"{self.handcore.robot_name_str_r!r}. "
                "Expected l20 or g20."
            )

        if self.handcore.robot_name_str_r == "g20":

            print(
                "[HW MAP] g20 selected by official config "
                "(expected for this L20 hardware)"
            )

        else:

            print(
                "[HW MAP] l20 selected by official config"
            )

        # Official L20 helper:
        # only reuse motor constraints + smoothing.
        try:

            self.hw_helper = L20RightHand(
                self.handcore,
                length=20,
                is_debug=False,
            )

        except TypeError:

            self.hw_helper = L20RightHand(
                self.handcore,
                length=20,
            )

        # More conservative than repo default.
        self.hw_helper.smooth_enabled = True

        # V9.3 hardware:
        # keep ONE smoothing / slew layer only.
        #
        # At 120 Hz:
        # alpha=0.50 gives low latency;
        # max_step=8 still prevents large instantaneous motor jumps.
        self.hw_helper.smooth_alpha = 0.50
        self.hw_helper.max_step = 8

        # ------------------------------------------------------
        # Official URDF non-fixed joint order
        # ------------------------------------------------------

        self.urdf_joint_names = []

        for name, joint in (
            self.handcore
            .RightHandId
            .joint_map
            .items()
        ):

            if joint.type == "fixed":
                continue

            self.urdf_joint_names.append(
                name
            )

        required = {
            "index_mcp_roll",
            "index_mcp_pitch",
            "index_pip",
            "index_dip",

            "middle_mcp_roll",
            "middle_mcp_pitch",
            "middle_pip",
            "middle_dip",

            "ring_mcp_roll",
            "ring_mcp_pitch",
            "ring_pip",
            "ring_dip",

            "pinky_mcp_roll",
            "pinky_mcp_pitch",
            "pinky_pip",
            "pinky_dip",

            "thumb_cmc_roll",
            "thumb_cmc_yaw",
            "thumb_cmc_pitch",
            "thumb_mcp",
            "thumb_ip",
        }

        missing = (
            required
            - set(
                self.urdf_joint_names
            )
        )

        if missing:

            raise RuntimeError(
                "官方 L20 URDF 缺少关节: "
                + str(
                    sorted(missing)
                )
            )


        # ------------------------------------------------------
        # TRUE L20 joint limits
        #
        # HandCore intentionally uses the official G20 hardware
        # configuration for this physical hand.  However, the
        # incoming u16 values are L20 model coordinates.
        #
        # Therefore we must NOT interpret L20 radians directly
        # as G20 radians.  Preserve normalized joint travel:
        #
        #   s = (q_l20 - l20_lower) /
        #       (l20_upper - l20_lower)
        #
        #   q_g20 = g20_lower +
        #           s * (g20_upper - g20_lower)
        # ------------------------------------------------------

        self.l20_urdf_path = (
            Path(self.robot_dir)
            / "linker_hand"
            / "l20_right"
            / "linkerhand_l20_right.urdf"
        )

        if not self.l20_urdf_path.exists():
            raise RuntimeError(
                "L20 URDF not found: "
                + str(self.l20_urdf_path)
            )

        self.l20_joint_limits = {}

        _l20_root = ET.parse(
            self.l20_urdf_path
        ).getroot()

        for _joint in _l20_root.findall(
            "joint"
        ):

            if (
                _joint.attrib.get("type")
                == "fixed"
            ):
                continue

            _limit = _joint.find(
                "limit"
            )

            if _limit is None:
                continue

            _name = _joint.attrib[
                "name"
            ]

            self.l20_joint_limits[
                _name
            ] = (
                float(
                    _limit.attrib[
                        "lower"
                    ]
                ),
                float(
                    _limit.attrib[
                        "upper"
                    ]
                ),
            )

        _missing_l20 = (
            required
            - set(
                self.l20_joint_limits
            )
        )

        if _missing_l20:
            raise RuntimeError(
                "L20 URDF limits missing: "
                + str(
                    sorted(
                        _missing_l20
                    )
                )
            )

        print(
            "[HW MAP] normalized "
            "L20 joint travel -> G20 hardware travel ENABLED"
        )

        print(
            "[HW MAP] L20 URDF =",
            self.l20_urdf_path,
        )

        # ------------------------------------------------------
        # UDP
        # ------------------------------------------------------

        self.sock = socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM,
        )

        self.sock.bind(
            (
                UDP_HOST,
                UDP_PORT,
            )
        )

        self.sock.setblocking(
            False
        )

        # ------------------------------------------------------
        # ROS
        # ------------------------------------------------------

        self.publisher = (
            self.create_publisher(
                JointState,
                TOPIC,
                10,
            )
        )

        # ------------------------------------------------------
        # REAL hardware feedback.
        #
        # V9.4 starts with no previous command publisher, so the
        # old "capture previous command" takeover cannot work.
        #
        # Capture the physical hand's CURRENT raw20 position from
        # /cb_right_hand_state and use that as the seamless
        # takeover start point.
        # ------------------------------------------------------
        self.state_subscription = (
            self.create_subscription(
                JointState,
                "/cb_right_hand_state",
                self.state_monitor_cb,
                10,
            )
        )

        # ------------------------------------------------------
        # State
        # ------------------------------------------------------

        self.armed = False
        # Only initial startup may arm automatically. Any HOLD
        # requires an explicit arm command or a bridge restart.
        self.auto_arm_pending = True

        self.last_udp = None
        self.last_udp_rx = 0.0

        self.captured_raw = None
        self.captured_time = 0.0

        self.last_out = None

        # Extra bridge-side slew limit:
        # raw counts per 1/60 sec.
        self.raw_step = 2.0

        self.cmd_queue = queue.Queue()

        self.print_counter = 0

        threading.Thread(
            target=self.stdin_worker,
            daemon=True,
        ).start()

        self.timer = self.create_timer(
            1.0 / 120.0,
            self.loop,
        )

        print()
        print(
            "UDP listening:",
            f"{UDP_HOST}:{UDP_PORT}",
        )

        print(
            "ROS topic:",
            TOPIC,
        )

        print()
        print(
            "START STATE = DISARMED"
        )

        print()
        print(
            "Commands:"
        )
        print(
            "  status"
        )
        print(
            "  arm"
        )
        print(
            "  hold"
        )
        print(
            "  quit"
        )

        print()
        print(
            "ARM requires:"
        )
        print(
            "  1) Wuji calibration complete"
        )
        print(
            "  2) fresh UDP u16"
        )
        print(
            "  3) real raw20 feedback no older than 0.25 s"
        )
        print(
            "  4) no other command publisher"
        )
        print(
            "============================================"
        )
        print()

    # ========================================================
    # Capture REAL current motor state for seamless takeover
    # ========================================================

    def state_monitor_cb(
        self,
        msg,
    ):

        # Fast G20 feedback should provide the full 20 channels.
        if len(msg.position) != 20:
            return

        arr = np.asarray(
            msg.position,
            dtype=np.float64,
        )

        if not np.all(
            np.isfinite(arr)
        ):
            return

        # Reject the driver's unavailable-state sentinel (-1).
        if (
            np.min(arr) < 0.0
            or
            np.max(arr) > 255.0
        ):
            return

        # Channels 11~14 are reserved.  Feedback commonly reports
        # zero there, while the official command convention keeps
        # them at 255.  Preserve the official command convention.
        arr = arr.copy()
        arr[11:15] = 255.0

        first = (
            self.captured_raw
            is None
        )

        # Keep feedback fresh while ARMED as well. This does not
        # overwrite last_out or the official smoothing history.
        self.captured_raw = (
            arr.copy()
        )

        self.captured_time = (
            time.monotonic()
        )

        if first:
            print(
                "[HW STATE] captured initial raw20:",
                np.round(
                    self.captured_raw
                ).astype(int).tolist(),
                flush=True,
            )

    # ========================================================
    # Human semantic u16 -> L20 mechanical joint angles
    # ========================================================

    def u16_to_angles(
        self,
        u,
    ):

        u = np.asarray(
            u,
            dtype=np.float64,
        )

        if u.shape != (16,):
            raise ValueError(
                f"u16 shape={u.shape}"
            )

        d = dict(
            zip(
                U16_NAMES,
                u,
            )
        )

        angles = {
            # index
            "index_mcp_roll":
                d["index_mcp_roll"],

            "index_mcp_pitch":
                d["index_mcp_pitch"],

            "index_pip":
                d["index_pip"],

            "index_dip":
                0.80790960
                * d["index_pip"],

            # middle
            "middle_mcp_roll":
                d["middle_mcp_roll"],

            "middle_mcp_pitch":
                d["middle_mcp_pitch"],

            "middle_pip":
                d["middle_pip"],

            "middle_dip":
                0.80790960
                * d["middle_pip"],

            # ring
            "ring_mcp_roll":
                d["ring_mcp_roll"],

            "ring_mcp_pitch":
                d["ring_mcp_pitch"],

            "ring_pip":
                d["ring_pip"],

            "ring_dip":
                0.80790960
                * d["ring_pip"],

            # pinky
            "pinky_mcp_roll":
                d["pinky_mcp_roll"],

            "pinky_mcp_pitch":
                d["pinky_mcp_pitch"],

            "pinky_pip":
                d["pinky_pip"],

            "pinky_dip":
                0.80790960
                * d["pinky_pip"],

            # thumb
            "thumb_cmc_roll":
                d["thumb_cmc_roll"],

            "thumb_cmc_yaw":
                d["thumb_cmc_yaw"],

            "thumb_cmc_pitch":
                d["thumb_cmc_pitch"],

            "thumb_mcp":
                d["thumb_mcp"],

            "thumb_ip":
                0.8079
                * d["thumb_mcp"],
        }

        # ====================================================
        # L20 model radians -> G20 hardware radians
        #
        # Preserve normalized mechanical travel instead of
        # treating numerically equal radians as equivalent.
        # ====================================================

        for name in list(
            angles.keys()
        ):

            if (
                name
                not in self.l20_joint_limits
            ):
                continue

            g20_joint = (
                self.handcore
                .RightHandId
                .joint_map[name]
            )

            if g20_joint.limit is None:
                continue

            l20_lo, l20_hi = (
                self.l20_joint_limits[
                    name
                ]
            )

            g20_lo = float(
                g20_joint.limit.lower
            )

            g20_hi = float(
                g20_joint.limit.upper
            )

            q_l20 = float(
                np.clip(
                    angles[name],
                    l20_lo,
                    l20_hi,
                )
            )

            if (
                abs(
                    l20_hi
                    - l20_lo
                )
                < 1e-12
            ):
                ratio = 0.0
            else:
                ratio = (
                    (q_l20 - l20_lo)
                    /
                    (l20_hi - l20_lo)
                )

            ratio = float(
                np.clip(
                    ratio,
                    0.0,
                    1.0,
                )
            )

            angles[name] = (
                g20_lo
                + ratio
                * (
                    g20_hi
                    - g20_lo
                )
            )

        return angles

    # ========================================================
    # Official angle -> raw20
    # ========================================================

    def angles_to_raw(
        self,
        angles,
    ):

        src_indices = [
            x
            for x
            in self.handcore
            .sourcedataindex_r
            if x is not None
        ]

        n = max(
            len(
                self.urdf_joint_names
            ),
            max(src_indices) + 1,
            25,
        )

        temp = np.zeros(
            n,
            dtype=np.float64,
        )

        # Critical:
        #
        # For motor i:
        #
        # URDF joint = urdfdataindex_r[i]
        # input slot = sourcedataindex_r[i]
        #
        # Therefore we map BY NAME instead of assuming q-array
        # ordering between our retargeter and official repo.
        for motor_i in range(
            self.handcore.hand_numjoints_r
        ):

            src_i = (
                self.handcore
                .sourcedataindex_r[
                    motor_i
                ]
            )

            urdf_i = (
                self.handcore
                .urdfdataindex_r[
                    motor_i
                ]
            )

            if (
                src_i is None
                or
                urdf_i is None
            ):
                continue

            name = (
                self.urdf_joint_names[
                    urdf_i
                ]
            )

            if name not in angles:
                continue

            temp[src_i] = (
                angles[name]
            )

        # OFFICIAL mapping from radians to command range.
        raw = (
            self.handcore
            .trans_to_motor_right(
                temp
            )
        )

        raw = [
            int(x)
            for x in raw
        ]

        # OFFICIAL LinkerForce motor constraints.
        raw = (
            self.hw_helper
            ._apply_motor_constraints(
                raw
            )
        )

        # Preserve any hardware channels that official mapping
        # explicitly marks as "no source".
        if (
            self.captured_raw
            is not None
        ):

            for i, src in enumerate(
                self.handcore
                .sourcedataindex_r
            ):

                if src is None:
                    raw[i] = int(
                        round(
                            self.captured_raw[i]
                        )
                    )

        return np.asarray(
            raw,
            dtype=np.float64,
        )

    # ========================================================
    # UDP
    # ========================================================

    def receive_udp(self):

        while True:

            try:

                data, _ = (
                    self.sock.recvfrom(
                        65535
                    )
                )

            except BlockingIOError:
                return

            try:

                pkt = json.loads(
                    data.decode(
                        "utf-8"
                    )
                )

                u = np.asarray(
                    pkt["u16"],
                    dtype=np.float64,
                )

                if u.shape != (16,):
                    continue

                if not np.all(
                    np.isfinite(u)
                ):
                    continue

                self.last_udp = pkt
                self.last_udp_rx = (
                    time.monotonic()
                )

            except Exception as e:

                print(
                    "[UDP ERROR]",
                    e,
                    flush=True,
                )

    # ========================================================
    # Terminal commands
    # ========================================================

    def stdin_worker(self):

        while rclpy.ok():

            try:
                line = input().strip()

            except EOFError:
                return

            if line:
                self.cmd_queue.put(
                    line.lower()
                )

    def other_publishers(
        self,
    ):

        # Our publisher itself counts as one.
        return max(
            0,
            self.count_publishers(
                TOPIC
            ) - 1,
        )

    def print_status(self):

        now = time.monotonic()

        udp_age = (
            now - self.last_udp_rx
            if self.last_udp is not None
            else float("inf")
        )

        capture_age = (
            now - self.captured_time
            if self.captured_raw is not None
            else float("inf")
        )

        print()
        print(
            "============================================"
        )
        print(
            "STATE:",
            "ARMED"
            if self.armed
            else "DISARMED",
        )

        print(
            "UDP:",
            "YES"
            if self.last_udp is not None
            else "NO",
            f"age={udp_age:.3f}s",
        )

        if self.last_udp is not None:

            print(
                "calib_done =",
                self.last_udp.get(
                    "calib_done"
                ),
            )

            print(
                "active_pinch =",
                self.last_udp.get(
                    "active_pinch"
                ),
            )

        print(
            "captured raw20 =",
            "YES"
            if self.captured_raw is not None
            else "NO",
            f"age={capture_age:.1f}s",
        )

        if self.captured_raw is not None:

            print(
                "captured =",
                np.round(
                    self.captured_raw
                ).astype(int).tolist(),
            )

        print(
            "other command publishers =",
            self.other_publishers(),
        )

        if (
            self.last_udp is not None
        ):

            try:

                angles = (
                    self.u16_to_angles(
                        self.last_udp[
                            "u16"
                        ]
                    )
                )

                raw = (
                    self.angles_to_raw(
                        angles
                    )
                )

                print(
                    "current target raw20 =",
                    np.round(
                        raw
                    ).astype(int).tolist(),
                )

            except Exception as e:

                print(
                    "target conversion ERROR:",
                    e,
                )

        print(
            "============================================"
        )
        print()

    def do_arm(self):

        now = time.monotonic()

        if self.last_udp is None:

            print(
                "[ARM REFUSED] no UDP u16"
            )
            return

        age = (
            now
            - self.last_udp_rx
        )

        if age > 0.25:

            print(
                "[ARM REFUSED] "
                f"UDP stale: {age:.3f}s"
            )
            return

        if not bool(
            self.last_udp.get(
                "calib_done",
                False,
            )
        ):

            print(
                "[ARM REFUSED] "
                "Wuji calibration not complete"
            )
            return

        if self.captured_raw is None:

            print(
                "[ARM REFUSED] "
                "没有捕获到有效 /cb_right_hand_state raw20。"
            )

            print(
                "确认 driver 使用 "
                "g20_fast_feedback:=true，"
                "并且 state 不是 -1。"
            )

            return

        state_age = now - self.captured_time
        if state_age > 0.25:
            print(
                "[ARM REFUSED] "
                f"hardware feedback stale: {state_age:.3f}s"
            )
            return

        others = (
            self.other_publishers()
        )

        if others > 0:

            print(
                "[ARM REFUSED] "
                f"还有 {others} 个其它 command publisher"
            )

            print(
                "先停止官方 GUI/旧 teleop publisher。"
            )

            return

        # Seamless takeover from fresh physical feedback.
        self.last_out = (
            self.captured_raw.copy()
        )

        self.hw_helper.smooth_positions = [
            float(x)
            for x in self.captured_raw
        ]

        self.armed = True
        self.auto_arm_pending = False

        print()
        print(
            "============================================"
        )
        print(
            "ARMED"
        )
        print(
            "Start command = captured raw20"
        )
        print(
            "Official max_step =",
            self.hw_helper.max_step,
            "raw counts/frame @ 120Hz",
        )
        print(
            "============================================"
        )
        print()

    def do_hold(self):

        self.auto_arm_pending = False

        if self.armed:

            self.armed = False

            print()
            print(
                "============================================"
            )
            print(
                "HOLD / DISARMED"
            )
            print(
                "Publishing stopped."
            )
            print(
                "No reset command was sent."
            )
            print(
                "Automatic re-arm disabled; use arm or restart."
            )
            print(
                "============================================"
            )
            print()

    def process_commands(self):

        while True:

            try:

                cmd = (
                    self.cmd_queue
                    .get_nowait()
                )

            except queue.Empty:
                return

            if cmd == "status":

                self.print_status()

            elif cmd == "arm":

                self.do_arm()

            elif cmd in (
                "hold",
                "disarm",
                "stop",
            ):

                self.do_hold()

            elif cmd in (
                "quit",
                "exit",
            ):

                self.do_hold()

                rclpy.shutdown()

                return

            else:

                print(
                    "Commands: "
                    "status | arm | hold | quit"
                )

    # ========================================================
    # Publish
    # ========================================================


    def apply_real_pinch_sim2real(
        self,
        raw_target,
        packet,
    ):
        """
        Physical-hand correction synchronized with V9.4.

        thumb_alpha:
            physical thumb 4 motors

        finger_alpha:
            active finger abduction + tip/PIP

        root_alpha:
            active finger MCP/base ONLY

        Inside ROOT_ONLY:
            thumb_alpha = 1
            finger_alpha = 1
            only root_alpha changes

        Therefore slightly reopening a pinch cannot produce
        thumb lateral sweep.
        """

        raw_target = np.asarray(
            raw_target,
            dtype=np.float64,
        ).copy()

        self._real_pinch_name = None
        self._real_thumb_alpha = 0.0
        self._real_finger_alpha = 0.0
        self._real_root_alpha = 0.0

        if not isinstance(packet, dict):
            return raw_target

        name = packet.get(
            "active_pinch"
        )

        if name not in REAL_PINCH_RAW20:
            return raw_target

        try:

            thumb_alpha = float(
                packet.get(
                    "pinch_thumb_strength",
                    0.0,
                )
            )

            finger_alpha = float(
                packet.get(
                    "pinch_finger_prep_strength",
                    packet.get(
                        "pinch_prep_strength",
                        0.0,
                    ),
                )
            )

            root_alpha = float(
                packet.get(
                    "pinch_root_strength",
                    packet.get(
                        "pinch_close_strength",
                        0.0,
                    ),
                )
            )

        except (
            TypeError,
            ValueError,
        ):
            return raw_target

        vals = np.asarray(
            [
                thumb_alpha,
                finger_alpha,
                root_alpha,
            ],
            dtype=np.float64,
        )

        if not np.all(
            np.isfinite(vals)
        ):
            return raw_target

        thumb_alpha = float(
            np.clip(
                thumb_alpha,
                0.0,
                1.0,
            )
        )

        finger_alpha = float(
            np.clip(
                finger_alpha,
                0.0,
                1.0,
            )
        )

        root_alpha = float(
            np.clip(
                root_alpha,
                0.0,
                1.0,
            )
        )

        if (
            thumb_alpha <= 0.0
            and
            finger_alpha <= 0.0
            and
            root_alpha <= 0.0
        ):
            return raw_target

        # =====================================================
        # Nominal contact raw20 under current normalized mapping
        # =====================================================
        if not hasattr(
            self,
            "_real_pinch_nominal_raw",
        ):

            nominal = {}

            for finger, q16 in (
                SIM_PINCH_U16.items()
            ):

                q16 = np.asarray(
                    q16,
                    dtype=np.float64,
                )

                angles = (
                    self.u16_to_angles(
                        q16
                    )
                )

                raw = np.asarray(
                    self.angles_to_raw(
                        angles
                    ),
                    dtype=np.float64,
                )

                if raw.shape != (20,):
                    raise RuntimeError(
                        f"{finger}: bad nominal raw shape "
                        f"{raw.shape}"
                    )

                nominal[finger] = raw

            self._real_pinch_nominal_raw = (
                nominal
            )

            print(
                "[REAL ROOT-CLOSE] "
                "contact calibration loaded",
                flush=True,
            )

        nominal = np.asarray(
            self._real_pinch_nominal_raw[
                name
            ],
            dtype=np.float64,
        )

        physical = np.asarray(
            REAL_PINCH_RAW20[
                name
            ],
            dtype=np.float64,
        )

        # =====================================================
        # Physical motor grouping
        # =====================================================
        thumb_channels = [
            0,
            5,
            10,
            15,
        ]

        finger_prepare_channels = {
            "index":  [6, 16],
            "middle": [7, 17],
            "ring":   [8, 18],
            "pinky":  [9, 19],
        }[name]

        root_channel = {
            "index":  1,
            "middle": 2,
            "ring":   3,
            "pinky":  4,
        }[name]

        # Thumb physical sim2real correction
        for ch in thumb_channels:

            raw_target[ch] += (
                thumb_alpha
                *
                (
                    physical[ch]
                    -
                    nominal[ch]
                )
            )

        # Target finger orientation / distal preparation
        for ch in finger_prepare_channels:

            raw_target[ch] += (
                finger_alpha
                *
                (
                    physical[ch]
                    -
                    nominal[ch]
                )
            )

        # MCP/root only final closing
        raw_target[root_channel] += (
            root_alpha
            *
            (
                physical[root_channel]
                -
                nominal[root_channel]
            )
        )

        self._real_pinch_name = name

        self._real_thumb_alpha = (
            thumb_alpha
        )

        self._real_finger_alpha = (
            finger_alpha
        )

        self._real_root_alpha = (
            root_alpha
        )

        return np.clip(
            raw_target,
            0.0,
            255.0,
        )


    def publish_raw(
        self,
        raw,
    ):

        msg = JointState()

        msg.header.stamp = (
            self.get_clock()
            .now()
            .to_msg()
        )

        # Match official publisher naming.
        msg.name = [
            f"joint{i + 1}"
            for i in range(20)
        ]

        msg.position = [
            float(
                int(
                    round(x)
                )
            )
            for x in raw
        ]

        msg.velocity = [
            255.0
        ] * 20

        self.publisher.publish(
            msg
        )

    # ========================================================
    # Main 120 Hz loop
    # ========================================================

    def loop(self):

        self.receive_udp()
        self.process_commands()

        # ====================================================
        # V9.4 AUTO ARM
        #
        # The bridge is launched with stdin disconnected.
        # Automatically ARM once at startup after all checks.
        # A HOLD must remain latched even after inputs recover.
        # ====================================================
        if not self.armed and self.auto_arm_pending:

            now = time.monotonic()

            _udp_fresh = (
                self.last_udp
                is not None
                and
                (
                    now
                    - self.last_udp_rx
                )
                <= 0.25
            )

            _calib_ok = (
                self.last_udp
                is not None
                and
                bool(
                    self.last_udp.get(
                        "calib_done",
                        False,
                    )
                )
            )

            _state_ok = (
                self.captured_raw
                is not None
                and
                now - self.captured_time <= 0.25
            )

            _publisher_ok = (
                self.other_publishers()
                == 0
            )

            if (
                _udp_fresh
                and
                _calib_ok
                and
                _state_ok
                and
                _publisher_ok
            ):
                print(
                    "[AUTO ARM] "
                    "UDP+CALIB+STATE+PUBLISHER checks passed",
                    flush=True,
                )

                self.do_arm()

        if not self.armed:
            return

        # Do not fight another controller.
        others = (
            self.other_publishers()
        )

        if others > 0:

            print(
                "[SAFETY] another command publisher "
                "appeared -> HOLD",
                flush=True,
            )

            self.do_hold()

            return

        now = time.monotonic()

        # UDP watchdog.
        age = (
            now
            - self.last_udp_rx
        )

        if (
            self.last_udp is None
            or
            age > 0.30
        ):

            print(
                "[WATCHDOG] Wuji/u16 lost -> HOLD",
                flush=True,
            )

            self.do_hold()

            return

        if not self.last_udp.get(
            "calib_done",
            False,
        ):

            print(
                "[WATCHDOG] calibration reset -> HOLD",
                flush=True,
            )

            self.do_hold()

            return

        # Real feedback watchdog, independent of fresh UDP commands.
        if (
            self.captured_raw is None
            or now - self.captured_time > 0.30
        ):
            print(
                "[WATCHDOG] hardware feedback lost -> HOLD",
                flush=True,
            )
            self.do_hold()
            return

        try:

            u = self.last_udp[
                "u16"
            ]

            angles = (
                self.u16_to_angles(
                    u
                )
            )

            raw_target = (
                self.angles_to_raw(
                    angles
                )
            )

            # Physical-hand pad-to-pad sim2real correction.
            # Keep the original normalized mapping and only add
            # the measured residual near an actual pinch.
            raw_target = (
                self.apply_real_pinch_sim2real(
                    raw_target,
                    self.last_udp,
                )
            )

            # Official EMA + max-step.
            raw_target = np.asarray(
                self.hw_helper
                ._apply_smooth(
                    raw_target
                    .round()
                    .astype(int)
                    .tolist()
                ),
                dtype=np.float64,
            )

            if self.last_out is None:

                self.last_out = (
                    self.captured_raw.copy()
                )

            # V9.3:
            # Do NOT apply a second bridge-side slew limiter.
            #
            # hw_helper._apply_smooth() above already provides:
            #   1. EMA
            #   2. max_step safety limiting
            #
            # A second raw_step limiter caused large tracking latency.
            out = np.clip(
                raw_target,
                0,
                255,
            )

            # Preserve unmapped/reserved channels.
            for i, src in enumerate(
                self.handcore
                .sourcedataindex_r
            ):

                if (
                    src is None
                    and
                    self.captured_raw
                    is not None
                ):

                    out[i] = (
                        self.captured_raw[i]
                    )

            self.publish_raw(
                out
            )

            self.last_out = (
                out.copy()
            )

            self.print_counter += 1

            if (
                self.print_counter
                % 60
                == 0
            ):

                print(
                    "[HW] "
                    f"pinch="
                    f"{self.last_udp.get('active_pinch')} "
                    f"real={getattr(self, '_real_pinch_name', None)} "
                    f"thumb={getattr(self, '_real_thumb_alpha', 0.0):.3f} "
                    f"finger={getattr(self, '_real_finger_alpha', 0.0):.3f} "
                    f"root={getattr(self, '_real_root_alpha', 0.0):.3f} "
                    f"raw="
                    f"{np.round(out).astype(int).tolist()}",
                    flush=True,
                )

        except Exception as e:

            print(
                "[SAFETY] conversion/publish error:",
                repr(e),
                flush=True,
            )

            self.do_hold()


def main():

    rclpy.init()

    node = None

    try:

        node = L20Bridge()

        rclpy.spin(
            node
        )

    except KeyboardInterrupt:
        pass

    finally:

        if node is not None:

            node.armed = False

            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
