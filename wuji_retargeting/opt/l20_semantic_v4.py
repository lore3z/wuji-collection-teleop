"""
L20 Semantic Retarget V4

No human/robot point-position matching.
No RobotMorphology target.
No fake DIP observation.

Four fingers:
    human proximal direction
        -> inverse robot proximal-direction map
        -> MCP roll + MCP pitch

    human total distal curl
        -> inverse robot curl map
        -> one L20 PIP actuator

Thumb:
    conservative first version:
    distal flex from human thumb bending
    CMC is held temporally until its semantic map is calibrated separately.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base import BaseOptimizer
from .l20_retarget_v2 import L20KinematicAdapter


class L20SemanticV4(BaseOptimizer):

    uses_raw_human_keypoints = True

    FINGERS = ["index", "middle", "ring", "pinky"]

    HUMAN = {
        "index":  (5, 6, 7, 8),
        "middle": (9, 10, 11, 12),
        "ring":   (13, 14, 15, 16),
        "pinky":  (17, 18, 19, 20),
    }

    THUMB = (1, 2, 3, 4)

    def __init__(self, config: dict):
        super().__init__(config)

        rcfg = config.get("retarget", {})

        self.adapter = L20KinematicAdapter(self.robot)

        self.num_control_dofs = 16
        self.control_dof_names = list(self.adapter.u_names)
        self.u_index = dict(self.adapter.u_index)

        self.debug_every = int(
            rcfg.get("v4_debug_every", 20)
        )

        self.dir_grid_roll_n = int(
            rcfg.get("v4_roll_samples", 41)
        )

        self.dir_grid_pitch_n = int(
            rcfg.get("v4_pitch_samples", 81)
        )

        self.curl_grid_n = int(
            rcfg.get("v4_curl_samples", 121)
        )

        self.last_u16 = None
        self.last_qpos = None
        self._frame_counter = 0

        # Frames, BaseOptimizer order:
        # thumb,index,middle,ring,pinky
        self.level1 = [
            self.robot.get_link_index(n)
            for n in self.link1_names
        ]

        self.level2 = [
            self.robot.get_link_index(n)
            for n in self.link3_names
        ]

        self.level3 = [
            self.robot.get_link_index(n)
            for n in self.link4_names
        ]

        self.tip = [
            self.robot.get_link_index(n)
            for n in self.task_link_names
        ]

        # ----------------------------------------------------------
        # IMPORTANT:
        # L20 URDF zero geometry itself shows:
        #
        # X ~ palm normal
        # Y ~ lateral / index-pinky direction
        # Z ~ finger-forward
        #
        # So do NOT apply MediaPipe frame estimation to the robot.
        #
        # Human data after apply_mediapipe_transformations() already
        # has the same semantic axis convention.
        # ----------------------------------------------------------

        self._build_finger_luts()

        print(
            "[L20SemanticV4] initialized"
        )
        print(
            "[L20SemanticV4] NO RobotMorphology"
        )
        print(
            "[L20SemanticV4] NO point-position matching"
        )
        print(
            "[L20SemanticV4] four fingers use robot-native direction/curl LUTs"
        )
        print(
            "[L20SemanticV4] thumb CMC temporarily held; no fake CMC inference"
        )

    # ============================================================
    # Geometry
    # ============================================================

    @staticmethod
    def _unit(v):
        v = np.asarray(v, dtype=np.float64)

        n = np.linalg.norm(v)

        if n < 1e-10:
            return np.zeros(3, dtype=np.float64)

        return v / n

    @classmethod
    def _angle(cls, a, b):
        a = cls._unit(a)
        b = cls._unit(b)

        if (
            np.linalg.norm(a) < 1e-8
            or np.linalg.norm(b) < 1e-8
        ):
            return 0.0

        return float(
            np.arccos(
                np.clip(
                    np.dot(a, b),
                    -1.0,
                    1.0,
                )
            )
        )

    # ============================================================
    # Robot features
    # ============================================================

    def _robot_digit_points(
        self,
        u,
        digit_index,
    ):
        q = self.adapter.expand(u)

        self.robot.compute_forward_kinematics(q)

        frame_ids = [
            self.level1[digit_index],
            self.level2[digit_index],
            self.level3[digit_index],
            self.tip[digit_index],
        ]

        return [
            self.robot.get_link_pose(fid)[:3, 3].copy()
            for fid in frame_ids
        ]

    def _robot_proximal_dir(
        self,
        u,
        digit_index,
    ):
        P = self._robot_digit_points(
            u,
            digit_index,
        )

        return self._unit(
            P[1] - P[0]
        )

    def _robot_total_curl(
        self,
        u,
        digit_index,
    ):
        """
        Robot curl feature using a long-baseline distal direction.

        proximal:
            level1 -> level2

        distal envelope:
            level2 -> tip
        """

        P = self._robot_digit_points(
            u,
            digit_index,
        )

        proximal = (
            P[1] - P[0]
        )

        distal_envelope = (
            P[3] - P[1]
        )

        return self._angle(
            proximal,
            distal_envelope,
        )

    # ============================================================
    # Human features
    # ============================================================

    def _human_finger_features(
        self,
        kp,
        finger,
    ):
        m, p, d, t = self.HUMAN[finger]

        proximal = self._unit(
            kp[p] - kp[m]
        )

        # Long baseline suppresses the unreliable Wuji DIP point.
        distal_envelope = (
            kp[t] - kp[p]
        )

        curl = self._angle(
            proximal,
            distal_envelope,
        )

        return proximal, curl

    def _human_thumb_flex(
        self,
        kp,
    ):
        c, m, i, t = self.THUMB

        s1 = kp[m] - kp[c]
        s2 = kp[i] - kp[m]
        s3 = kp[t] - kp[i]

        a1 = self._angle(s1, s2)
        a2 = self._angle(s2, s3)

        r = 0.8079

        # One actuator drives MCP + IP.
        q = (
            a1
            + r * a2
        ) / (
            1.0
            + r * r
        )

        return float(q)

    # ============================================================
    # Offline robot LUT
    # ============================================================

    def _build_finger_luts(self):

        self._direction_lut = {}
        self._curl_lut = {}

        base = np.clip(
            np.zeros(16),
            self.adapter.lower,
            self.adapter.upper,
        )

        for fi, finger in enumerate(
            self.FINGERS
        ):
            digit_index = fi + 1

            roll_i = self.u_index[
                f"{finger}_mcp_roll"
            ]

            pitch_i = self.u_index[
                f"{finger}_mcp_pitch"
            ]

            pip_i = self.u_index[
                f"{finger}_pip"
            ]

            rolls = np.linspace(
                self.adapter.lower[roll_i],
                self.adapter.upper[roll_i],
                self.dir_grid_roll_n,
            )

            pitches = np.linspace(
                self.adapter.lower[pitch_i],
                self.adapter.upper[pitch_i],
                self.dir_grid_pitch_n,
            )

            dirs = []
            commands = []

            # Build actual robot reachable direction manifold.
            for qr in rolls:
                for qp in pitches:

                    u = base.copy()

                    u[roll_i] = qr
                    u[pitch_i] = qp
                    u[pip_i] = 0.0

                    d = self._robot_proximal_dir(
                        u,
                        digit_index,
                    )

                    dirs.append(d)
                    commands.append(
                        [qr, qp]
                    )

            self._direction_lut[finger] = (
                np.asarray(
                    dirs,
                    dtype=np.float64,
                ),
                np.asarray(
                    commands,
                    dtype=np.float64,
                ),
            )

            # ----------------------------------------------------
            # Curl map
            # ----------------------------------------------------

            qs = np.linspace(
                self.adapter.lower[pip_i],
                self.adapter.upper[pip_i],
                self.curl_grid_n,
            )

            curls = []

            for qpip in qs:

                u = base.copy()

                # Use zero spread / zero MCP flex while calibrating
                # intrinsic distal curl.
                u[roll_i] = np.clip(
                    0.0,
                    self.adapter.lower[roll_i],
                    self.adapter.upper[roll_i],
                )

                u[pitch_i] = np.clip(
                    0.0,
                    self.adapter.lower[pitch_i],
                    self.adapter.upper[pitch_i],
                )

                u[pip_i] = qpip

                curls.append(
                    self._robot_total_curl(
                        u,
                        digit_index,
                    )
                )

            self._curl_lut[finger] = (
                np.asarray(
                    curls,
                    dtype=np.float64,
                ),
                qs,
            )

    # ============================================================
    # Inverse semantic maps
    # ============================================================

    def _inverse_direction(
        self,
        finger,
        human_dir,
    ):
        dirs, commands = (
            self._direction_lut[
                finger
            ]
        )

        h = self._unit(
            human_dir
        )

        # Angular nearest neighbour:
        #
        # maximizing dot product == minimizing angle.
        score = dirs @ h

        idx = int(
            np.argmax(score)
        )

        q = commands[idx]

        err = np.degrees(
            np.arccos(
                np.clip(
                    score[idx],
                    -1.0,
                    1.0,
                )
            )
        )

        return (
            float(q[0]),
            float(q[1]),
            float(err),
        )

    def _inverse_curl(
        self,
        finger,
        human_curl,
    ):
        curls, qs = self._curl_lut[
            finger
        ]

        idx = int(
            np.argmin(
                np.abs(
                    curls
                    - human_curl
                )
            )
        )

        return (
            float(qs[idx]),
            float(
                np.degrees(
                    curls[idx]
                    - human_curl
                )
            ),
        )

    # ============================================================
    # Solve
    # ============================================================

    def solve(
        self,
        mediapipe_keypoints: np.ndarray,
        last_qpos: Optional[np.ndarray] = None,
    ) -> np.ndarray:

        kp = np.asarray(
            mediapipe_keypoints,
            dtype=np.float64,
        )

        if kp.shape != (21, 3):
            raise ValueError(
                f"expected (21,3), got {kp.shape}"
            )

        if not np.all(
            np.isfinite(kp)
        ):
            raise ValueError(
                "keypoints contain NaN/Inf"
            )

        if (
            last_qpos is not None
            and np.asarray(last_qpos).shape == (21,)
        ):
            u = self.adapter.compress(
                np.asarray(
                    last_qpos,
                    dtype=np.float64,
                )
            )

        elif self.last_u16 is not None:
            u = self.last_u16.copy()

        else:
            u = np.clip(
                np.zeros(16),
                self.adapter.lower,
                self.adapter.upper,
            )

        debug = {}

        # --------------------------------------------------------
        # Four fingers
        # --------------------------------------------------------

        for finger in self.FINGERS:

            human_dir, human_curl = (
                self._human_finger_features(
                    kp,
                    finger,
                )
            )

            qroll, qpitch, dir_err = (
                self._inverse_direction(
                    finger,
                    human_dir,
                )
            )

            qpip, curl_err = (
                self._inverse_curl(
                    finger,
                    human_curl,
                )
            )

            u[
                self.u_index[
                    f"{finger}_mcp_roll"
                ]
            ] = qroll

            u[
                self.u_index[
                    f"{finger}_mcp_pitch"
                ]
            ] = qpitch

            u[
                self.u_index[
                    f"{finger}_pip"
                ]
            ] = qpip

            debug[finger] = {
                "dir": human_dir.copy(),
                "curl": human_curl,
                "dir_err": dir_err,
                "curl_err": curl_err,
                "roll": qroll,
                "pitch": qpitch,
                "pip": qpip,
            }

        # --------------------------------------------------------
        # Thumb distal flex:
        #
        # Do something we actually have an observable for.
        # --------------------------------------------------------

        thumb_flex = (
            self._human_thumb_flex(
                kp
            )
        )

        tf = self.u_index[
            "thumb_mcp"
        ]

        u[tf] = np.clip(
            thumb_flex,
            self.adapter.lower[tf],
            self.adapter.upper[tf],
        )

        # --------------------------------------------------------
        # Thumb CMC:
        #
        # IMPORTANT:
        # Do NOT invent a CMC solution yet.
        #
        # Hold previous CMC state.
        # On startup use zero-feasible pose.
        # --------------------------------------------------------

        for name in [
            "thumb_cmc_roll",
            "thumb_cmc_yaw",
            "thumb_cmc_pitch",
        ]:
            ui = self.u_index[name]

            u[ui] = np.clip(
                u[ui],
                self.adapter.lower[ui],
                self.adapter.upper[ui],
            )

        u = np.clip(
            u,
            self.adapter.lower,
            self.adapter.upper,
        )

        q = self.adapter.expand(u)

        self.last_u16 = u.copy()
        self.last_qpos = q.copy()

        self._frame_counter += 1

        if (
            self.debug_every > 0
            and self._frame_counter
            % self.debug_every
            == 0
        ):

            print(
                "[V4 SEMANTIC] ---------------------------"
            )

            for finger in self.FINGERS:

                d = debug[finger]

                print(
                    "[V4 FINGER] "
                    f"{finger:6s} "
                    "humanDir="
                    + np.array2string(
                        d["dir"],
                        precision=3,
                        separator=",",
                    )
                    + " "
                    f"humanCurl="
                    f"{np.degrees(d['curl']):5.1f}deg "
                    f"-> roll="
                    f"{np.degrees(d['roll']):6.1f} "
                    f"pitch="
                    f"{np.degrees(d['pitch']):6.1f} "
                    f"pip="
                    f"{np.degrees(d['pip']):6.1f} "
                    f"dirErr={d['dir_err']:5.1f}deg "
                    f"curlErr={d['curl_err']:5.1f}deg"
                )

            print(
                "[V4 THUMB] "
                "CMC=HELD "
                f"roll="
                f"{np.degrees(u[self.u_index['thumb_cmc_roll']]):.1f} "
                f"yaw="
                f"{np.degrees(u[self.u_index['thumb_cmc_yaw']]):.1f} "
                f"pitch="
                f"{np.degrees(u[self.u_index['thumb_cmc_pitch']]):.1f} "
                f"flex="
                f"{np.degrees(u[tf]):.1f}"
            )

            print(
                "[V4 SEMANTIC] ---------------------------"
            )

        return q

    def compute_cost(
        self,
        qpos: np.ndarray,
        mediapipe_keypoints: np.ndarray,
    ) -> float:

        # Semantic diagnostics only.
        # No point-position distance.
        q = np.asarray(
            qpos,
            dtype=np.float64,
        )

        u = self.adapter.compress(q)

        kp = np.asarray(
            mediapipe_keypoints,
            dtype=np.float64,
        )

        cost = 0.0

        for fi, finger in enumerate(
            self.FINGERS
        ):
            digit_index = fi + 1

            hd, hc = self._human_finger_features(
                kp,
                finger,
            )

            rd = self._robot_proximal_dir(
                u,
                digit_index,
            )

            rc = self._robot_total_curl(
                u,
                digit_index,
            )

            cost += (
                1.0
                - np.clip(
                    np.dot(
                        self._unit(hd),
                        self._unit(rd),
                    ),
                    -1.0,
                    1.0,
                )
            )

            cost += (
                hc - rc
            ) ** 2

        return float(cost)

    def reset(self):
        self.last_u16 = None
        self.last_qpos = None
        self._frame_counter = 0
