from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from .base import BaseOptimizer
from .l20_retarget_v2 import L20KinematicAdapter


class L20MultiPinchRetargeter(BaseOptimizer):
    """
    Wuji -> human semantic commands -> L20 u16 -> q21

    Human calibration:
        OPEN
        FIST
        THUMB-INDEX
        THUMB-MIDDLE
        THUMB-RING
        THUMB-PINKY

    Every calibration pose:
        movement -> discard current window
        continuously stable -> accept median

    Runtime:
        normal finger flex/spread
              +
        calibrated human pinch strength
              ->
        blend toward robot-native L20 pinch pose

    NO human/robot keypoint fitting.
    NO RobotMorphology.
    NO robot bone chasing.
    """

    uses_raw_human_keypoints = True

    FINGERS = [
        "index",
        "middle",
        "ring",
        "pinky",
    ]

    HUMAN = {
        "index":  (5, 6, 7, 8),
        "middle": (9, 10, 11, 12),
        "ring":   (13, 14, 15, 16),
        "pinky":  (17, 18, 19, 20),
    }

    TIP_INDEX = {
        "index": 8,
        "middle": 12,
        "ring": 16,
        "pinky": 20,
    }

    PHASES = [
        "OPEN",
        "FIST",
        "TI",
        "TM",
        "TR",
        "TP",
    ]

    PHASE_TO_FINGER = {
        "TI": "index",
        "TM": "middle",
        "TR": "ring",
        "TP": "pinky",
    }

    FINGER_TO_PHASE = {
        "index": "TI",
        "middle": "TM",
        "ring": "TR",
        "pinky": "TP",
    }

    PHASE_LABEL = {
        "OPEN": "五指完全伸直",
        "FIST": "完全握拳",
        "TI": "拇指 + 食指捏合",
        "TM": "拇指 + 中指捏合",
        "TR": "拇指 + 无名指捏合",
        "TP": "拇指 + 小指捏合",
    }

    # Robot nominal OPEN thumb command.
    OPEN_THUMB_ROLL = np.deg2rad(-10.0)
    OPEN_THUMB_YAW = np.deg2rad(15.0)
    OPEN_THUMB_PITCH = np.deg2rad(4.0)
    OPEN_THUMB_MCP = 0.0

    def __init__(self, config: dict):
        super().__init__(config)

        rcfg = config.get("retarget", {})

        self.adapter = L20KinematicAdapter(
            self.robot
        )

        self.num_control_dofs = 16
        self.control_dof_names = list(
            self.adapter.u_names
        )
        self.u_index = dict(
            self.adapter.u_index
        )

        # ======================================================
        # Stable calibration
        # ======================================================

        self.calib_stable_sec = float(
            rcfg.get(
                "cmd_calib_stable_sec",
                2.0,
            )
        )

        self.calib_min_samples = int(
            rcfg.get(
                "cmd_calib_min_samples",
                30,
            )
        )

        self.calib_frame_angle_deg = float(
            rcfg.get(
                "cmd_calib_frame_angle_deg",
                3.0,
            )
        )

        self.calib_frame_geom = float(
            rcfg.get(
                "cmd_calib_frame_thumb_pos",
                0.045,
            )
        )

        self.calib_anchor_angle_deg = float(
            rcfg.get(
                "cmd_calib_anchor_angle_deg",
                5.0,
            )
        )

        self.calib_anchor_geom = float(
            rcfg.get(
                "cmd_calib_anchor_thumb_pos",
                0.075,
            )
        )

        self.spread_full_deg = float(
            rcfg.get(
                "cmd_spread_full_deg",
                18.0,
            )
        )

        # ======================================================
        # Pinch state/hysteresis
        # ======================================================

        self.pinch_enter = float(
            rcfg.get(
                "cmd_pinch_enter",
                0.35,
            )
        )

        self.pinch_exit = float(
            rcfg.get(
                "cmd_pinch_exit",
                0.18,
            )
        )

        self.pinch_switch_margin = float(
            rcfg.get(
                "cmd_pinch_switch_margin",
                0.12,
            )
        )

        # ------------------------------------------------------
        # False-pinch suppression
        #
        # During a real fist the thumb naturally gets close to
        # middle/ring/index. Distance alone must NOT activate a
        # robot pinch pose, otherwise that selected finger will
        # be pulled back toward its pinch endpoint and appear to
        # "stick up".
        # ------------------------------------------------------

        self.fist_gate_finger = float(
            rcfg.get(
                "cmd_fist_gate_finger",
                0.68,
            )
        )

        self.fist_gate_count = int(
            rcfg.get(
                "cmd_fist_gate_count",
                3,
            )
        )

        self.fist_gate_mean = float(
            rcfg.get(
                "cmd_fist_gate_mean",
                0.62,
            )
        )

        # A new pinch must also clearly beat the second-best
        # candidate.  This prevents thumb-near-several-fingers
        # configurations from arbitrarily selecting one finger.
        self.pinch_dominance = float(
            rcfg.get(
                "cmd_pinch_dominance",
                0.12,
            )
        )

        self.debug_every = int(
            rcfg.get(
                "cmd_debug_every",
                10,
            )
        )

        self._active_pinch = None

        # ======================================================
        # Load ROBOT-native pinch poses
        # ======================================================

        repo_root = (
            Path(__file__)
            .resolve()
            .parents[2]
        )

        pinch_path = Path(
            rcfg.get(
                "robot_pinch_pose_file",
                repo_root
                / "example"
                / "config"
                / "l20_robot_pinch_poses.yaml",
            )
        )

        if not pinch_path.is_absolute():
            pinch_path = (
                repo_root
                / pinch_path
            )

        if not pinch_path.exists():
            raise FileNotFoundError(
                f"Robot pinch pose YAML not found: "
                f"{pinch_path}"
            )

        with open(
            pinch_path,
            "r",
        ) as f:
            pinch_yaml = yaml.safe_load(f)

        self.robot_pinch = {}

        for finger in self.FINGERS:

            key = f"thumb_{finger}"

            pose = pinch_yaml[
                "poses"
            ][key]

            controls = pose[
                "controls"
            ]

            self.robot_pinch[
                finger
            ] = {
                name: float(v["rad"])
                for name, v
                in controls.items()
            }

        print()
        print(
            "============================================"
        )
        print(
            "L20 MULTI-PINCH RETARGET"
        )
        print(
            "Human calibration:"
        )
        print(
            "OPEN -> FIST -> TI -> TM -> TR -> TP"
        )
        print(
            "Robot pinch poses loaded:"
        )

        for finger in self.FINGERS:
            print(
                "  thumb-",
                finger,
            )

        print(
            "NO human/robot point matching"
        )
        print(
            "============================================"
        )

        self.last_u16 = None
        self.last_qpos = None
        self._frame_counter = 0

        self._reset_calibration()

    # =========================================================
    # Geometry helpers
    # =========================================================

    @staticmethod
    def _unit(v):

        v = np.asarray(
            v,
            dtype=np.float64,
        )

        n = float(
            np.linalg.norm(v)
        )

        if n < 1e-10:
            return np.zeros(3)

        return v / n

    @classmethod
    def _angle(cls, a, b):

        a = cls._unit(a)
        b = cls._unit(b)

        if (
            np.linalg.norm(a) < 1e-8
            or
            np.linalg.norm(b) < 1e-8
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

    @staticmethod
    def _map01(x, a, b):

        d = b - a

        if abs(d) < 1e-6:
            return 0.0

        return float(
            np.clip(
                (x - a) / d,
                0.0,
                1.0,
            )
        )

    @staticmethod
    def _smooth01(x):

        x = float(
            np.clip(
                x,
                0.0,
                1.0,
            )
        )

        return (
            x * x
            * (3.0 - 2.0 * x)
        )

    # =========================================================
    # Human palm-local measurements
    # =========================================================

    def _palm_frame(self, kp):

        wrist = kp[0]

        mcps = np.stack([
            kp[5],
            kp[9],
            kp[13],
            kp[17],
        ])

        center = np.mean(
            mcps,
            axis=0,
        )

        lateral = self._unit(
            kp[5] - kp[17]
        )

        forward = (
            center - wrist
        )

        forward = (
            forward
            - lateral
            * np.dot(
                forward,
                lateral,
            )
        )

        forward = self._unit(
            forward
        )

        normal = self._unit(
            np.cross(
                lateral,
                forward,
            )
        )

        forward = self._unit(
            np.cross(
                normal,
                lateral,
            )
        )

        scale = max(
            float(
                np.linalg.norm(
                    center - wrist
                )
            ),
            1e-6,
        )

        return (
            lateral,
            forward,
            normal,
            center,
            scale,
        )

    def _measure_finger(
        self,
        kp,
        finger,
        lateral,
        forward,
        normal,
    ):

        m, p, d, t = self.HUMAN[
            finger
        ]

        proximal = self._unit(
            kp[p] - kp[m]
        )

        # TRUE PIP segment:
        # use PIP -> DIP, NOT PIP -> TIP.
        #
        # PIP -> TIP mixes PIP and DIP motion and destroys
        # independent MCP/PIP observability.
        middle = self._unit(
            kp[d] - kp[p]
        )

        # Root flex evidence relative to palm plane.
        root = float(
            np.degrees(
                np.arcsin(
                    np.clip(
                        abs(
                            np.dot(
                                proximal,
                                normal,
                            )
                        ),
                        0.0,
                        1.0,
                    )
                )
            )
        )

        # True human PIP flexion.
        #
        # proximal : MCP -> PIP
        # middle   : PIP -> DIP
        #
        # DIP is intentionally NOT used here because robot DIP
        # is mechanically coupled from robot PIP.
        curl = float(
            np.degrees(
                self._angle(
                    proximal,
                    middle,
                )
            )
        )

        spread = float(
            np.degrees(
                np.arctan2(
                    np.dot(
                        proximal,
                        lateral,
                    ),
                    np.dot(
                        proximal,
                        forward,
                    ),
                )
            )
        )

        return {
            "root": root,
            "curl": curl,
            "spread": spread,
        }

    def _measure_all(
        self,
        kp,
    ):

        (
            lateral,
            forward,
            normal,
            center,
            scale,
        ) = self._palm_frame(kp)

        fingers = {}

        for finger in self.FINGERS:
            fingers[finger] = (
                self._measure_finger(
                    kp,
                    finger,
                    lateral,
                    forward,
                    normal,
                )
            )

        # Thumb bend.
        s1 = kp[2] - kp[1]
        s2 = kp[3] - kp[2]
        s3 = kp[4] - kp[3]

        thumb_bend = float(
            np.degrees(
                self._angle(s1, s2)
                + self._angle(s2, s3)
            )
        )

        # Thumb position in HUMAN palm-local coordinates.
        thumb_v = (
            kp[4] - center
        ) / scale

        thumb_pos = np.array([
            np.dot(
                thumb_v,
                lateral,
            ),
            np.dot(
                thumb_v,
                forward,
            ),
            np.dot(
                thumb_v,
                normal,
            ),
        ])

        # Most important for multi-pinch:
        #
        # HUMAN thumb-tip -> HUMAN fingertip distance,
        # normalized by current palm size.
        #
        # This is a HUMAN semantic measurement only.
        pinch_dist = {}

        for finger in self.FINGERS:

            tip_idx = self.TIP_INDEX[
                finger
            ]

            pinch_dist[finger] = float(
                np.linalg.norm(
                    kp[4]
                    - kp[tip_idx]
                )
                / scale
            )

        return {
            "fingers": fingers,

            "thumb": {
                "bend": thumb_bend,
                "pos": thumb_pos,
            },

            "pinch_dist": pinch_dist,
        }

    # =========================================================
    # Stable-window calibration
    # =========================================================

    def _reset_calibration(self):

        self._calib_done = False
        self._phase_index = 0

        self._phase_start = None
        self._stable_anchor = None
        self._prev_measurement = None

        self._phase_samples = []
        self._last_status_print = 0.0

        self.calib = {}

        self._active_pinch = None

    def _median_measurements(
        self,
        samples,
    ):

        out = {
            "fingers": {},
            "thumb": {},
            "pinch_dist": {},
        }

        for finger in self.FINGERS:

            out["fingers"][
                finger
            ] = {}

            for key in [
                "root",
                "curl",
                "spread",
            ]:

                out["fingers"][
                    finger
                ][key] = float(
                    np.median([
                        s["fingers"]
                        [finger]
                        [key]
                        for s in samples
                    ])
                )

            out["pinch_dist"][
                finger
            ] = float(
                np.median([
                    s["pinch_dist"]
                    [finger]
                    for s in samples
                ])
            )

        out["thumb"]["bend"] = float(
            np.median([
                s["thumb"]["bend"]
                for s in samples
            ])
        )

        out["thumb"]["pos"] = np.median(
            np.stack([
                s["thumb"]["pos"]
                for s in samples
            ]),
            axis=0,
        )

        return out

    def _motion_metric(
        self,
        a,
        b,
    ):

        if a is None or b is None:
            return 0.0, 0.0

        angles = []

        for finger in self.FINGERS:

            for key in [
                "root",
                "curl",
                "spread",
            ]:

                angles.append(
                    abs(
                        a["fingers"]
                        [finger]
                        [key]
                        -
                        b["fingers"]
                        [finger]
                        [key]
                    )
                )

        angles.append(
            abs(
                a["thumb"]["bend"]
                -
                b["thumb"]["bend"]
            )
        )

        thumb_delta = float(
            np.linalg.norm(
                np.asarray(
                    a["thumb"]["pos"]
                )
                -
                np.asarray(
                    b["thumb"]["pos"]
                )
            )
        )

        pinch_delta = max(
            abs(
                a["pinch_dist"][f]
                -
                b["pinch_dist"][f]
            )
            for f in self.FINGERS
        )

        geom_delta = max(
            thumb_delta,
            pinch_delta,
        )

        return (
            float(max(angles)),
            float(geom_delta),
        )

    def _is_stable(
        self,
        m,
    ):

        frame_angle, frame_geom = (
            self._motion_metric(
                m,
                self._prev_measurement,
            )
        )

        anchor_angle, anchor_geom = (
            self._motion_metric(
                m,
                self._stable_anchor,
            )
        )

        ok = (
            frame_angle
            <= self.calib_frame_angle_deg

            and

            frame_geom
            <= self.calib_frame_geom

            and

            anchor_angle
            <= self.calib_anchor_angle_deg

            and

            anchor_geom
            <= self.calib_anchor_geom
        )

        return ok, {
            "frame_angle": frame_angle,
            "frame_geom": frame_geom,
            "anchor_angle": anchor_angle,
            "anchor_geom": anchor_geom,
        }

    def _print_phase(self):

        phase = self.PHASES[
            self._phase_index
        ]

        print()
        print(
            "============================================",
            flush=True,
        )

        print(
            f"[CALIB "
            f"{self._phase_index + 1}/"
            f"{len(self.PHASES)}] "
            f"{phase}",
            flush=True,
        )

        print(
            ">>> "
            + self.PHASE_LABEL[phase]
            + " <<<",
            flush=True,
        )

        print(
            f"连续稳定 "
            f"{self.calib_stable_sec:.1f}s "
            f"后自动完成",
            flush=True,
        )

        print(
            "检测到明显运动 -> 当前数据全部丢弃 -> 从0重新累计",
            flush=True,
        )

        print(
            "============================================",
            flush=True,
        )

    def _validate_phase(
        self,
        phase,
        candidate,
    ):

        # OPEN is the reference.
        if phase == "OPEN":
            return True, ""

        # Pinch phases must actually bring the requested HUMAN
        # fingertips substantially closer than OPEN.
        if phase in self.PHASE_TO_FINGER:

            finger = self.PHASE_TO_FINGER[
                phase
            ]

            open_d = self.calib[
                "OPEN"
            ]["pinch_dist"][finger]

            closed_d = candidate[
                "pinch_dist"
            ][finger]

            if closed_d >= (
                0.75 * open_d
            ):

                return (
                    False,
                    (
                        f"thumb-{finger} 没有明显闭合: "
                        f"OPEN={open_d:.3f}, "
                        f"PINCH={closed_d:.3f}"
                    ),
                )

        return True, ""

    def _calibration_step(
        self,
        measurement,
    ):

        phase = self.PHASES[
            self._phase_index
        ]

        now = time.monotonic()

        # First frame of this phase.
        if self._phase_start is None:

            self._phase_start = now
            self._stable_anchor = measurement
            self._prev_measurement = measurement

            self._phase_samples = [
                measurement
            ]

            self._last_status_print = now

            self._print_phase()

            return

        stable, diag = self._is_stable(
            measurement
        )

        if not stable:

            old_time = (
                now
                - self._phase_start
            )

            self._phase_start = now
            self._stable_anchor = measurement
            self._prev_measurement = measurement

            self._phase_samples = [
                measurement
            ]

            print(
                f"[CALIB {phase}] "
                f"MOVEMENT -> RESTART "
                f"(previous={old_time:.2f}s, "
                f"frameΔ={diag['frame_angle']:.1f}°, "
                f"anchorΔ={diag['anchor_angle']:.1f}°)",
                flush=True,
            )

            return

        self._prev_measurement = measurement

        self._phase_samples.append(
            measurement
        )

        stable_time = (
            now
            - self._phase_start
        )

        if (
            now
            - self._last_status_print
            >= 0.4
        ):

            self._last_status_print = now

            print(
                f"[CALIB {phase}] "
                f"STABLE "
                f"{stable_time:.2f}/"
                f"{self.calib_stable_sec:.2f}s "
                f"samples="
                f"{len(self._phase_samples)}",
                flush=True,
            )

        if (
            stable_time
            < self.calib_stable_sec
        ):
            return

        if (
            len(self._phase_samples)
            < self.calib_min_samples
        ):
            return

        candidate = (
            self._median_measurements(
                self._phase_samples
            )
        )

        ok, reason = self._validate_phase(
            phase,
            candidate,
        )

        if not ok:

            print()
            print(
                f"[CALIB {phase}] REJECTED",
                flush=True,
            )

            print(
                reason,
                flush=True,
            )

            print(
                "请重新保持正确姿态。",
                flush=True,
            )

            self._phase_start = None
            self._stable_anchor = None
            self._prev_measurement = None
            self._phase_samples = []

            return

        self.calib[
            phase
        ] = candidate

        print()
        print(
            f"[CALIB {phase}] COMPLETE "
            f"stable={stable_time:.2f}s "
            f"samples={len(self._phase_samples)}",
            flush=True,
        )

        if phase in self.PHASE_TO_FINGER:

            f = self.PHASE_TO_FINGER[
                phase
            ]

            print(
                f"human thumb-{f} distance: "
                f"OPEN="
                f"{self.calib['OPEN']['pinch_dist'][f]:.3f} "
                f"PINCH="
                f"{candidate['pinch_dist'][f]:.3f}",
                flush=True,
            )

        self._phase_index += 1

        self._phase_start = None
        self._stable_anchor = None
        self._prev_measurement = None
        self._phase_samples = []

        if (
            self._phase_index
            >= len(self.PHASES)
        ):

            self._calib_done = True

            print()
            print(
                "============================================"
            )
            print(
                "SIX-POSE CALIBRATION COMPLETE"
            )

            for f in self.FINGERS:

                phase_f = (
                    self.FINGER_TO_PHASE[f]
                )

                print(
                    f"thumb-{f:6s}: "
                    f"OPEN="
                    f"{self.calib['OPEN']['pinch_dist'][f]:.3f} "
                    f"PINCH="
                    f"{self.calib[phase_f]['pinch_dist'][f]:.3f}"
                )

            print(
                "NORMAL MULTI-PINCH TELEOP START"
            )
            print(
                "============================================"
            )
            print()

            return

        print()
        print(
            "下一步："
            + self.PHASE_LABEL[
                self.PHASES[
                    self._phase_index
                ]
            ],
            flush=True,
        )

        print(
            "移动过程中会不断重新计时，"
            "真正稳定以后才会采纳。",
            flush=True,
        )

    # =========================================================
    # Human commands after calibration
    # =========================================================

    def _finger_commands(
        self,
        measurement,
    ):

        cmd = {}

        for finger in self.FINGERS:

            raw = measurement[
                "fingers"
            ][finger]

            o = self.calib[
                "OPEN"
            ]["fingers"][finger]

            f = self.calib[
                "FIST"
            ]["fingers"][finger]

            root = self._map01(
                raw["root"],
                o["root"],
                f["root"],
            )

            curl = self._map01(
                raw["curl"],
                o["curl"],
                f["curl"],
            )

            spread = float(
                np.clip(
                    (
                        raw["spread"]
                        - o["spread"]
                    )
                    / self.spread_full_deg,
                    -1.0,
                    1.0,
                )
            )

            # Deep flexion -> spread becomes weakly observable.
            if (
                root > 0.5
                or
                curl > 0.5
            ):
                spread *= 0.25

            cmd[finger] = {
                "root": root,
                "curl": curl,
                "spread": spread,
            }

        return cmd

    def _pinch_alphas(
        self,
        measurement,
    ):

        alphas = {}

        for finger in self.FINGERS:

            phase = self.FINGER_TO_PHASE[
                finger
            ]

            open_d = self.calib[
                "OPEN"
            ]["pinch_dist"][finger]

            close_d = self.calib[
                phase
            ]["pinch_dist"][finger]

            current_d = measurement[
                "pinch_dist"
            ][finger]

            denom = (
                open_d
                - close_d
            )

            if denom <= 1e-5:

                alpha = 0.0

            else:

                alpha = (
                    open_d
                    - current_d
                ) / denom

            alphas[finger] = (
                self._smooth01(
                    alpha
                )
            )

        return alphas

    def _fist_state(
        self,
        finger_cmd,
    ):
        """
        Detect HUMAN four-finger fist state from calibrated
        OPEN->FIST semantic commands.

        This is only a gesture-priority gate.

        It does NOT change robot finger angles.
        It does NOT force fingers to move together.
        """

        scores = {}

        for finger in self.FINGERS:

            c = finger_cmd[
                finger
            ]

            # Keep the original independent root/curl channels.
            #
            # We only use their average here to ask:
            # "Is this finger broadly in a closed/fist state?"
            score = 0.5 * (
                float(c["root"])
                +
                float(c["curl"])
            )

            scores[finger] = float(
                np.clip(
                    score,
                    0.0,
                    1.0,
                )
            )

        values = np.array(
            [
                scores[f]
                for f in self.FINGERS
            ],
            dtype=np.float64,
        )

        mean_score = float(
            np.mean(values)
        )

        closed_count = int(
            np.sum(
                values
                >= self.fist_gate_finger
            )
        )

        is_fist = (
            closed_count
            >= self.fist_gate_count
            and
            mean_score
            >= self.fist_gate_mean
        )

        return (
            is_fist,
            mean_score,
            closed_count,
            scores,
        )

    def _select_pinch(
        self,
        alphas,
        finger_cmd,
    ):
        """
        Gesture priority:

            FIST
              >
            PINCH

        A closed fist must never be reinterpreted as a pinch
        merely because the thumb happens to lie near one finger.
        """

        (
            is_fist,
            fist_mean,
            fist_count,
            fist_scores,
        ) = self._fist_state(
            finger_cmd
        )

        # ------------------------------------------------------
        # HARD FIST PRIORITY
        #
        # Critical fix:
        # remove any active pinch immediately.
        # ------------------------------------------------------

        if is_fist:

            self._active_pinch = None

            return (
                None,
                {
                    "fist": True,
                    "fist_mean": fist_mean,
                    "fist_count": fist_count,
                    "fist_scores": fist_scores,
                    "best": None,
                    "best_alpha": 0.0,
                    "second_alpha": 0.0,
                },
            )

        ranked = sorted(
            self.FINGERS,
            key=lambda f: alphas[f],
            reverse=True,
        )

        best = ranked[0]
        second = ranked[1]

        best_a = float(
            alphas[best]
        )

        second_a = float(
            alphas[second]
        )

        dominance = (
            best_a
            - second_a
        )

        diagnostics = {
            "fist": False,
            "fist_mean": fist_mean,
            "fist_count": fist_count,
            "fist_scores": fist_scores,
            "best": best,
            "best_alpha": best_a,
            "second_alpha": second_a,
            "dominance": dominance,
        }

        # ------------------------------------------------------
        # No currently active pinch.
        #
        # Require BOTH:
        #
        #   enough closure
        #   enough superiority over second-best candidate
        # ------------------------------------------------------

        if self._active_pinch is None:

            if (
                best_a
                >= self.pinch_enter
                and
                dominance
                >= self.pinch_dominance
            ):

                self._active_pinch = best

            return (
                self._active_pinch,
                diagnostics,
            )

        current = self._active_pinch
        current_a = float(
            alphas[current]
        )

        # ------------------------------------------------------
        # Release
        # ------------------------------------------------------

        if current_a < self.pinch_exit:

            self._active_pinch = None

            # Allow direct acquisition of another clearly
            # dominant pinch.
            if (
                best_a
                >= self.pinch_enter
                and
                dominance
                >= self.pinch_dominance
            ):
                self._active_pinch = best

            return (
                self._active_pinch,
                diagnostics,
            )

        # ------------------------------------------------------
        # Switch
        # ------------------------------------------------------

        if (
            best != current
            and
            best_a >= self.pinch_enter
            and
            dominance >= self.pinch_dominance
            and
            best_a
            > (
                current_a
                + self.pinch_switch_margin
            )
        ):

            self._active_pinch = best

        return (
            self._active_pinch,
            diagnostics,
        )

    # =========================================================
    # Robot commands
    # =========================================================

    def _dynamic_thumb_cmc(
        self,
        measurement,
    ):
        """
        Human calibrated thumb-tip position
                    ↓
        L20 native thumb CMC 3-DOF

        Calibration anchors:
            OPEN
            TI
            TM
            TR
            TP

        The four robot pinch targets already contain valid
        robot-native thumb CMC roll/yaw/pitch.

        We fit one affine mapping:

            [human_x human_y human_z 1]
                        ->
            [roll yaw pitch]

        Therefore these three DOFs remain continuously
        controllable outside pinch as well.
        """

        names = (
            "thumb_cmc_roll",
            "thumb_cmc_yaw",
            "thumb_cmc_pitch",
        )

        X = []
        Y = []

        # ---------------------------------------------
        # OPEN anchor
        # ---------------------------------------------

        open_pos = np.asarray(
            self.calib[
                "OPEN"
            ][
                "thumb"
            ][
                "pos"
            ],
            dtype=np.float64,
        )

        X.append(
            np.r_[
                open_pos,
                1.0,
            ]
        )

        Y.append(
            [
                self.OPEN_THUMB_ROLL,
                self.OPEN_THUMB_YAW,
                self.OPEN_THUMB_PITCH,
            ]
        )


        # ---------------------------------------------
        # TI / TM / TR / TP
        # ---------------------------------------------

        for finger in self.FINGERS:

            phase = self.FINGER_TO_PHASE[
                finger
            ]

            human_pos = np.asarray(
                self.calib[
                    phase
                ][
                    "thumb"
                ][
                    "pos"
                ],
                dtype=np.float64,
            )

            target = (
                self._robot_pinch_target_u(
                    finger
                )
            )

            X.append(
                np.r_[
                    human_pos,
                    1.0,
                ]
            )

            Y.append(
                [
                    target[
                        self.u_index[
                            names[0]
                        ]
                    ],
                    target[
                        self.u_index[
                            names[1]
                        ]
                    ],
                    target[
                        self.u_index[
                            names[2]
                        ]
                    ],
                ]
            )


        X = np.asarray(
            X,
            dtype=np.float64,
        )

        Y = np.asarray(
            Y,
            dtype=np.float64,
        )


        # Least-squares affine mapping.
        coef, *_ = np.linalg.lstsq(
            X,
            Y,
            rcond=1e-6,
        )


        current_pos = np.asarray(
            measurement[
                "thumb"
            ][
                "pos"
            ],
            dtype=np.float64,
        )


        pred = (
            np.r_[
                current_pos,
                1.0,
            ]
            @ coef
        )


        # Clip in native L20 independent space.
        for j, name in enumerate(
            names
        ):

            i = self.u_index[
                name
            ]

            pred[j] = np.clip(
                pred[j],
                self.adapter.lower[i],
                self.adapter.upper[i],
            )


        return pred


    def _base_u16(
        self,
        finger_cmd,
        measurement,
    ):

        u = np.clip(
            np.zeros(16),
            self.adapter.lower,
            self.adapter.upper,
        )

        for finger in self.FINGERS:

            c = finger_cmd[
                finger
            ]

            ir = self.u_index[
                f"{finger}_mcp_roll"
            ]

            ip = self.u_index[
                f"{finger}_mcp_pitch"
            ]

            ic = self.u_index[
                f"{finger}_pip"
            ]

            spread_robot = (
                -c["spread"]
            )

            if spread_robot >= 0:

                u[ir] = (
                    spread_robot
                    * self.adapter.upper[ir]
                )

            else:

                u[ir] = (
                    -spread_robot
                    * self.adapter.lower[ir]
                )

            u[ip] = (
                self.adapter.lower[ip]
                + c["root"]
                * (
                    self.adapter.upper[ip]
                    - self.adapter.lower[ip]
                )
            )

            u[ic] = (
                self.adapter.lower[ic]
                + c["curl"]
                * (
                    self.adapter.upper[ic]
                    - self.adapter.lower[ic]
                )
            )

        # Normal thumb flex from OPEN -> FIST human bend.
        thumb_open = self.calib[
            "OPEN"
        ]["thumb"]["bend"]

        thumb_fist = self.calib[
            "FIST"
        ]["thumb"]["bend"]

        thumb_flex = self._map01(
            measurement[
                "thumb"
            ]["bend"],
            thumb_open,
            thumb_fist,
        )

        # --------------------------------------------------
        # Dynamic human thumb -> native L20 CMC 3DOF
        # --------------------------------------------------

        thumb_cmc = self._dynamic_thumb_cmc(
            measurement
        )

        u[
            self.u_index[
                "thumb_cmc_roll"
            ]
        ] = thumb_cmc[0]

        u[
            self.u_index[
                "thumb_cmc_yaw"
            ]
        ] = thumb_cmc[1]

        u[
            self.u_index[
                "thumb_cmc_pitch"
            ]
        ] = thumb_cmc[2]

        thumb_mcp_i = self.u_index[
            "thumb_mcp"
        ]

        u[thumb_mcp_i] = (
            thumb_flex
            * self.adapter.upper[
                thumb_mcp_i
            ]
        )

        return np.clip(
            u,
            self.adapter.lower,
            self.adapter.upper,
        )

    def _robot_pinch_target_u(
        self,
        finger,
    ):

        # Start at zero so unrelated fingers remain untouched.
        u = np.clip(
            np.zeros(16),
            self.adapter.lower,
            self.adapter.upper,
        )

        controls = self.robot_pinch[
            finger
        ]

        for name, value in controls.items():

            if name not in self.u_index:
                continue

            u[
                self.u_index[name]
            ] = value

        return np.clip(
            u,
            self.adapter.lower,
            self.adapter.upper,
        )

    def _apply_pinch_overlay(
        self,
        base_u,
        finger,
        alpha,
    ):

        if (
            finger is None
            or
            alpha <= 0.0
        ):
            return base_u

        u = base_u.copy()

        target = (
            self._robot_pinch_target_u(
                finger
            )
        )

        # Only the selected finger + thumb are blended.
        names = [
            f"{finger}_mcp_roll",
            f"{finger}_mcp_pitch",
            f"{finger}_pip",

            "thumb_cmc_roll",
            "thumb_cmc_yaw",
            "thumb_cmc_pitch",
            "thumb_mcp",
        ]

        for name in names:

            i = self.u_index[
                name
            ]

            u[i] = (
                (1.0 - alpha)
                * u[i]
                + alpha
                * target[i]
            )

        return np.clip(
            u,
            self.adapter.lower,
            self.adapter.upper,
        )

    def _calib_open_u16(self):

        u = np.clip(
            np.zeros(16),
            self.adapter.lower,
            self.adapter.upper,
        )

        u[
            self.u_index[
                "thumb_cmc_roll"
            ]
        ] = self.OPEN_THUMB_ROLL

        u[
            self.u_index[
                "thumb_cmc_yaw"
            ]
        ] = self.OPEN_THUMB_YAW

        u[
            self.u_index[
                "thumb_cmc_pitch"
            ]
        ] = self.OPEN_THUMB_PITCH

        return u

    # =========================================================
    # Solve
    # =========================================================

    def solve(
        self,
        mediapipe_keypoints: np.ndarray,
        last_qpos: Optional[np.ndarray] = None,
    ) -> np.ndarray:

        del last_qpos

        kp = np.asarray(
            mediapipe_keypoints,
            dtype=np.float64,
        )

        if kp.shape != (21, 3):
            raise ValueError(
                f"Expected (21,3), got {kp.shape}"
            )

        measurement = self._measure_all(
            kp
        )

        # Calibration.
        if not self._calib_done:

            self._calibration_step(
                measurement
            )

            u16 = self._calib_open_u16()

            q21 = self.adapter.expand(
                u16
            )

            self.last_u16 = u16.copy()
            self.last_qpos = q21.copy()

            return q21

        finger_cmd = self._finger_commands(
            measurement
        )

        alphas = self._pinch_alphas(
            measurement
        )

        active, pinch_diag = self._select_pinch(
            alphas,
            finger_cmd,
        )

        base_u = self._base_u16(
            finger_cmd,
            measurement,
        )

        if active is None:
            alpha = 0.0
        else:
            alpha = alphas[
                active
            ]

        u16 = self._apply_pinch_overlay(
            base_u,
            active,
            alpha,
        )

        q21 = self.adapter.expand(
            u16
        )

        self.last_u16 = u16.copy()
        self.last_qpos = q21.copy()

        self._frame_counter += 1

        if (
            self.debug_every > 0
            and
            self._frame_counter
            % self.debug_every
            == 0
        ):

            print(
                "[WUJI MULTI] ----------------------------"
            )

            print(
                "[WUJI PINCH] "
                f"active={active} "
                f"alpha={alpha:.3f} "
                f"I={alphas['index']:.2f} "
                f"M={alphas['middle']:.2f} "
                f"R={alphas['ring']:.2f} "
                f"P={alphas['pinky']:.2f} "
                f"fist={pinch_diag['fist']} "
                f"fist_mean={pinch_diag['fist_mean']:.2f} "
                f"fist_count={pinch_diag['fist_count']}"
            )

            for finger in self.FINGERS:

                c = finger_cmd[
                    finger
                ]

                print(
                    "[WUJI FINGER] "
                    f"{finger:6s} "
                    f"spread={c['spread']:+.2f} "
                    f"root={c['root']:.2f} "
                    f"curl={c['curl']:.2f}"
                )

            print(
                "[L20 CMD16] "
                + np.array2string(
                    u16,
                    precision=4,
                    separator=",",
                )
            )

            print(
                "[WUJI MULTI] ----------------------------"
            )

        return q21

    def compute_cost(
        self,
        qpos,
        mediapipe_keypoints,
    ):
        return 0.0

    def reset(self):

        self.last_u16 = None
        self.last_qpos = None
        self._frame_counter = 0

        self._reset_calibration()

        print()
        print(
            "MULTI-PINCH CALIBRATION RESET"
        )
