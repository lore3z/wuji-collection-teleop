#!/usr/bin/env python3

import json
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
        self.hw_helper.smooth_alpha = 0.25
        self.hw_helper.max_step = 4

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

        # Listen while DISARMED so we can capture the LAST
        # official/safe command before takeover.
        self.subscription = (
            self.create_subscription(
                JointState,
                TOPIC,
                self.command_monitor_cb,
                10,
            )
        )

        # ------------------------------------------------------
        # State
        # ------------------------------------------------------

        self.armed = False

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
            1.0 / 60.0,
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
            "  3) captured previous raw20 command"
        )
        print(
            "  4) no other command publisher"
        )
        print(
            "============================================"
        )
        print()

    # ========================================================
    # Capture existing controller command
    # ========================================================

    def command_monitor_cb(
        self,
        msg,
    ):

        # Once armed, messages can be our own.
        if self.armed:
            return

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

        if (
            np.min(arr) < -0.5
            or
            np.max(arr) > 255.5
        ):
            return

        self.captured_raw = (
            arr.copy()
        )

        self.captured_time = (
            time.monotonic()
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

        # Clip again using OFFICIAL URDF limits.
        for name in list(
            angles.keys()
        ):

            joint = (
                self.handcore
                .RightHandId
                .joint_map[name]
            )

            if joint.limit is None:
                continue

            angles[name] = float(
                np.clip(
                    angles[name],

                    joint.limit.lower,
                    joint.limit.upper,
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
                "没有捕获到上一控制器的 raw20。"
            )

            print(
                "先在 DISARMED 状态下让官方 GUI/"
                "控制器发布当前安全姿态，"
                "然后停止那个控制器。"
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

        # Seamless takeover from previous command.
        self.last_out = (
            self.captured_raw.copy()
        )

        self.hw_helper.smooth_positions = [
            float(x)
            for x in self.captured_raw
        ]

        self.armed = True

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
            "Bridge slew limit =",
            self.raw_step,
            "raw counts/frame @ 60Hz",
        )
        print(
            "============================================"
        )
        print()

    def do_hold(self):

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
    # Main 60 Hz loop
    # ========================================================

    def loop(self):

        self.receive_udp()
        self.process_commands()

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

            # Additional conservative bridge slew.
            delta = (
                raw_target
                - self.last_out
            )

            delta = np.clip(
                delta,
                -self.raw_step,
                self.raw_step,
            )

            out = (
                self.last_out
                + delta
            )

            out = np.clip(
                out,
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
