from __future__ import annotations

from typing import Optional
import time
import numpy as np

from .base import BaseOptimizer
from .l20_retarget_v2 import L20KinematicAdapter


class L20CommandRetargeter(BaseOptimizer):
    """
    Wuji -> semantic command -> L20 u16 -> q21

    Calibration:
        1. OPEN
        2. FIST
        3. O

    No RobotMorphology.
    No human/robot point matching.
    No IK bone chasing.
    """

    uses_raw_human_keypoints = True

    FINGERS = ["index", "middle", "ring", "pinky"]

    HUMAN = {
        "index":  (5, 6, 7, 8),
        "middle": (9, 10, 11, 12),
        "ring":   (13, 14, 15, 16),
        "pinky":  (17, 18, 19, 20),
    }

    THUMB = (1, 2, 3, 4)

    PHASES = ["OPEN", "FIST", "O"]

    def __init__(self, config: dict):
        super().__init__(config)

        rcfg = config.get("retarget", {})

        self.adapter = L20KinematicAdapter(self.robot)
        self.num_control_dofs = 16
        self.control_dof_names = list(self.adapter.u_names)
        self.u_index = dict(self.adapter.u_index)

        # --------------------------------------------------------
        # Calibration no longer uses fixed "settle then sample".
        #
        # A pose is accepted ONLY after a continuous stable window.
        # Any sufficiently large movement clears the entire current
        # window and restarts stability timing from zero.
        # --------------------------------------------------------

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

        # Per-frame motion thresholds.
        self.calib_frame_angle_deg = float(
            rcfg.get(
                "cmd_calib_frame_angle_deg",
                3.0,
            )
        )

        self.calib_frame_thumb_pos = float(
            rcfg.get(
                "cmd_calib_frame_thumb_pos",
                0.045,
            )
        )

        # Drift relative to the beginning of the current stable run.
        # This catches slow continuous movement that could otherwise
        # pass a frame-to-frame test.
        self.calib_anchor_angle_deg = float(
            rcfg.get(
                "cmd_calib_anchor_angle_deg",
                5.0,
            )
        )

        self.calib_anchor_thumb_pos = float(
            rcfg.get(
                "cmd_calib_anchor_thumb_pos",
                0.075,
            )
        )

        self.spread_full_deg = float(
            rcfg.get("cmd_spread_full_deg", 18.0)
        )

        self.debug_every = int(
            rcfg.get("cmd_debug_every", 10)
        )

        self.last_u16 = None
        self.last_qpos = None
        self._frame_counter = 0

        self._reset_calibration()

        print()
        print("============================================")
        print("L20 THREE-POSE CALIBRATION")
        print("1. OPEN  : 五指完全伸直")
        print("2. FIST  : 四指完全握紧")
        print("3. O     : 拇指+食指形成 O 型")
        print("============================================")
        print()
        self._print_phase_prompt()

    # ==========================================================
    # Helpers
    # ==========================================================

    @staticmethod
    def _unit(v):
        v = np.asarray(v, dtype=np.float64)
        n = np.linalg.norm(v)

        if n < 1e-10:
            return np.zeros(3)

        return v / n

    @classmethod
    def _angle(cls, a, b):
        a = cls._unit(a)
        b = cls._unit(b)

        if np.linalg.norm(a) < 1e-8 or np.linalg.norm(b) < 1e-8:
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
    def _linear01(x, a, b):
        d = b - a

        if abs(d) < 1e-5:
            return 0.0

        return float(
            np.clip(
                (x - a) / d,
                0.0,
                1.0,
            )
        )

    @staticmethod
    def _piecewise_half(x, open_v, mid_v, close_v):
        """
        OPEN -> 0
        O    -> 0.5
        FIST -> 1

        Used only for INDEX.
        """

        # Expected monotonic feature.
        monotonic_up = (
            open_v < mid_v < close_v
        )

        monotonic_down = (
            open_v > mid_v > close_v
        )

        if not (
            monotonic_up
            or monotonic_down
        ):
            d = close_v - open_v

            if abs(d) < 1e-5:
                return 0.0

            return float(
                np.clip(
                    (x - open_v) / d,
                    0.0,
                    1.0,
                )
            )

        if monotonic_up:

            if x <= mid_v:
                d = mid_v - open_v

                return float(
                    np.clip(
                        0.5
                        * (x - open_v)
                        / max(d, 1e-6),
                        0.0,
                        0.5,
                    )
                )

            d = close_v - mid_v

            return float(
                np.clip(
                    0.5
                    + 0.5
                    * (x - mid_v)
                    / max(d, 1e-6),
                    0.5,
                    1.0,
                )
            )

        # monotonic down
        if x >= mid_v:
            d = open_v - mid_v

            return float(
                np.clip(
                    0.5
                    * (open_v - x)
                    / max(d, 1e-6),
                    0.0,
                    0.5,
                )
            )

        d = mid_v - close_v

        return float(
            np.clip(
                0.5
                + 0.5
                * (mid_v - x)
                / max(d, 1e-6),
                0.5,
                1.0,
            )
        )

    # ==========================================================
    # Human palm frame
    # ==========================================================

    def _palm_frame(self, kp):

        wrist = kp[0]

        index_mcp = kp[5]
        middle_mcp = kp[9]
        ring_mcp = kp[13]
        pinky_mcp = kp[17]

        lateral = self._unit(
            index_mcp - pinky_mcp
        )

        center = np.mean(
            np.stack([
                index_mcp,
                middle_mcp,
                ring_mcp,
                pinky_mcp,
            ]),
            axis=0,
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

        forward = self._unit(forward)

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

        palm_scale = max(
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
            palm_scale,
        )

    # ==========================================================
    # Human semantic measurements
    # ==========================================================

    def _measure_finger(
        self,
        kp,
        finger,
        lateral,
        forward,
        normal,
    ):

        m, p, d, t = self.HUMAN[finger]

        proximal = self._unit(
            kp[p] - kp[m]
        )

        distal_long = self._unit(
            kp[t] - kp[p]
        )

        # MCP flex evidence.
        normal_component = np.clip(
            abs(
                np.dot(
                    proximal,
                    normal,
                )
            ),
            0.0,
            1.0,
        )

        root_deg = float(
            np.degrees(
                np.arcsin(
                    normal_component
                )
            )
        )

        # Overall distal curvature.
        curl_deg = float(
            np.degrees(
                self._angle(
                    proximal,
                    distal_long,
                )
            )
        )

        lat = float(
            np.dot(
                proximal,
                lateral,
            )
        )

        fwd = float(
            np.dot(
                proximal,
                forward,
            )
        )

        spread_deg = float(
            np.degrees(
                np.arctan2(
                    lat,
                    fwd,
                )
            )
        )

        return {
            "root": root_deg,
            "curl": curl_deg,
            "spread": spread_deg,
        }

    def _measure_thumb(
        self,
        kp,
        lateral,
        forward,
        normal,
        palm_center,
        palm_scale,
    ):

        c, m, i, t = self.THUMB

        s1 = kp[m] - kp[c]
        s2 = kp[i] - kp[m]
        s3 = kp[t] - kp[i]

        bend = float(
            np.degrees(
                self._angle(s1, s2)
                + self._angle(s2, s3)
            )
        )

        # Thumb-tip position in HUMAN palm-local coordinates.
        v = (
            kp[t]
            - palm_center
        ) / palm_scale

        local = np.array([
            np.dot(v, lateral),
            np.dot(v, forward),
            np.dot(v, normal),
        ])

        return {
            "bend": bend,
            "pos": local,
        }

    def _measure_all(self, kp):

        (
            lateral,
            forward,
            normal,
            center,
            scale,
        ) = self._palm_frame(kp)

        fingers = {}

        for f in self.FINGERS:
            fingers[f] = self._measure_finger(
                kp,
                f,
                lateral,
                forward,
                normal,
            )

        thumb = self._measure_thumb(
            kp,
            lateral,
            forward,
            normal,
            center,
            scale,
        )

        return {
            "fingers": fingers,
            "thumb": thumb,
        }

    # ==========================================================
    # Calibration
    # ==========================================================

    def _reset_calibration(self):

        self._calib_done = False
        self._phase_index = 0
        # IMPORTANT:
        # Do NOT start calibration timer during optimizer construction.
        # Wuji SDK initialization can take several seconds and would
        # otherwise consume the whole settle/sample window before the
        # first valid glove frame arrives.
        self._phase_start = None
        self._last_status_print = 0.0

        # Stable-window state.
        self._phase_samples = []
        self._stable_anchor = None
        self._prev_measurement = None

        self.calib = {}

    def _print_phase_prompt(self):

        if self._phase_index >= len(self.PHASES):
            return

        phase = self.PHASES[
            self._phase_index
        ]

        print()
        print("============================================")

        if phase == "OPEN":
            print("[CALIB 1/3] OPEN / 手掌张开标定")
            print("五指全部伸直。")
            print("拇指向后打开到舒适的最大位置。")

        elif phase == "FIST":
            print("[CALIB 2/3] FIST / 握拳标定")
            print("四指完全握紧。")
            print("保持自然完整握拳姿态。")

        elif phase == "O":
            print("[CALIB 3/3] O / O型标定")
            print("拇指和食指形成 O 型。")
            print("拇指第一关节尽量弯曲。")

        print(
            f"保持姿态不动：连续稳定 "
            f"{self.calib_stable_sec:.1f}s 后自动完成"
        )

        print(
            "只要检测到明显运动，当前稳定计时立即清零并重新开始"
        )

        print("============================================")
        print()

    @staticmethod
    def _median_measurements(samples):

        out = {
            "fingers": {},
            "thumb": {},
        }

        for finger in [
            "index",
            "middle",
            "ring",
            "pinky",
        ]:

            out["fingers"][finger] = {}

            for key in [
                "root",
                "curl",
                "spread",
            ]:

                vals = [
                    s["fingers"][finger][key]
                    for s in samples
                ]

                out["fingers"][finger][key] = float(
                    np.median(vals)
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

    def _calib_motion_metric(
        self,
        a,
        b,
    ):
        """
        Compare two HUMAN semantic measurements.

        Returns:
            angle_max_deg
            thumb_pos_dist

        This operates ONLY in Wuji/human semantic space.
        It has nothing to do with robot geometry.
        """

        if a is None or b is None:
            return 0.0, 0.0

        angle_diffs = []

        for finger in self.FINGERS:

            fa = a["fingers"][finger]
            fb = b["fingers"][finger]

            angle_diffs.extend([
                abs(fa["root"] - fb["root"]),
                abs(fa["curl"] - fb["curl"]),
                abs(fa["spread"] - fb["spread"]),
            ])

        angle_diffs.append(
            abs(
                a["thumb"]["bend"]
                - b["thumb"]["bend"]
            )
        )

        thumb_pos_dist = float(
            np.linalg.norm(
                np.asarray(
                    a["thumb"]["pos"],
                    dtype=np.float64,
                )
                -
                np.asarray(
                    b["thumb"]["pos"],
                    dtype=np.float64,
                )
            )
        )

        return (
            float(max(angle_diffs)),
            thumb_pos_dist,
        )

    def _calib_is_stable(
        self,
        measurement,
    ):
        """
        Two stability checks:

        1. current vs previous frame:
           catches sudden motion.

        2. current vs stable-window anchor:
           catches slow drift.

        Returns:
            stable: bool
            diagnostic: dict
        """

        frame_angle, frame_thumb = (
            self._calib_motion_metric(
                measurement,
                self._prev_measurement,
            )
        )

        anchor_angle, anchor_thumb = (
            self._calib_motion_metric(
                measurement,
                self._stable_anchor,
            )
        )

        frame_ok = (
            frame_angle
            <= self.calib_frame_angle_deg
            and
            frame_thumb
            <= self.calib_frame_thumb_pos
        )

        anchor_ok = (
            anchor_angle
            <= self.calib_anchor_angle_deg
            and
            anchor_thumb
            <= self.calib_anchor_thumb_pos
        )

        return (
            frame_ok and anchor_ok,
            {
                "frame_angle": frame_angle,
                "frame_thumb": frame_thumb,
                "anchor_angle": anchor_angle,
                "anchor_thumb": anchor_thumb,
            },
        )

    def _calibration_step(
        self,
        measurement,
    ):

        phase = self.PHASES[
            self._phase_index
        ]

        phase_num = (
            self._phase_index + 1
        )

        now = time.monotonic()

        # ------------------------------------------------------
        # First valid Wuji frame of this pose.
        # ------------------------------------------------------

        if self._phase_start is None:

            self._phase_start = now

            self._stable_anchor = measurement
            self._prev_measurement = measurement

            self._phase_samples = [
                measurement
            ]

            self._last_status_print = now

            print()
            print(
                "============================================",
                flush=True,
            )

            if phase == "OPEN":

                print(
                    "[CALIB 1/3] OPEN / 完全伸直",
                    flush=True,
                )

                print(
                    ">>> 五指完全伸直，拇指自然向外张开 <<<",
                    flush=True,
                )

            elif phase == "FIST":

                print(
                    "[CALIB 2/3] FIST / 完全握拳",
                    flush=True,
                )

                print(
                    ">>> 四指彻底握紧并保持 <<<",
                    flush=True,
                )

            elif phase == "O":

                print(
                    "[CALIB 3/3] O / O型",
                    flush=True,
                )

                print(
                    ">>> 拇指和食指形成 O 型并保持 <<<",
                    flush=True,
                )

            print(
                f"需要连续稳定 "
                f"{self.calib_stable_sec:.1f} 秒",
                flush=True,
            )

            print(
                "检测到明显运动后自动从 0 秒重新开始",
                flush=True,
            )

            print(
                "============================================",
                flush=True,
            )

            return

        # ------------------------------------------------------
        # Is the current frame still part of the same stable pose?
        # ------------------------------------------------------

        stable, diag = self._calib_is_stable(
            measurement
        )

        if not stable:

            # --------------------------------------------------
            # MOVEMENT:
            #
            # Throw away ALL samples collected so far.
            # Current frame becomes the new candidate anchor.
            # --------------------------------------------------

            old_elapsed = (
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
                f"[CALIB {phase_num}/3 {phase}] "
                f"MOVEMENT -> RESTART 0.0s "
                f"(之前稳定 {old_elapsed:.2f}s, "
                f"frameΔ={diag['frame_angle']:.1f}°, "
                f"anchorΔ={diag['anchor_angle']:.1f}°, "
                f"thumbΔ={diag['anchor_thumb']:.3f})",
                flush=True,
            )

            return

        # ------------------------------------------------------
        # Still stable.
        # ------------------------------------------------------

        self._prev_measurement = measurement

        self._phase_samples.append(
            measurement
        )

        stable_time = (
            now
            - self._phase_start
        )

        # Repeated visual status.
        if (
            now
            - self._last_status_print
            >= 0.4
        ):

            self._last_status_print = now

            progress = min(
                100.0,
                100.0
                * stable_time
                / max(
                    self.calib_stable_sec,
                    1e-6,
                ),
            )

            print(
                f"[CALIB {phase_num}/3 {phase}] "
                f"STABLE "
                f"{stable_time:.2f}/"
                f"{self.calib_stable_sec:.2f}s "
                f"({progress:5.1f}%) "
                f"samples={len(self._phase_samples)} "
                f"Δframe={diag['frame_angle']:.1f}° "
                f"Δanchor={diag['anchor_angle']:.1f}°",
                flush=True,
            )

        # ------------------------------------------------------
        # Need BOTH:
        #
        #   enough continuous stable time
        #   enough actual frames
        # ------------------------------------------------------

        if (
            stable_time
            < self.calib_stable_sec
        ):
            return

        if (
            len(self._phase_samples)
            < self.calib_min_samples
        ):

            if (
                now
                - self._last_status_print
                >= 0.4
            ):

                print(
                    f"[CALIB {phase}] "
                    f"时间已满足，等待足够帧数 "
                    f"{len(self._phase_samples)}/"
                    f"{self.calib_min_samples}",
                    flush=True,
                )

            return

        # ------------------------------------------------------
        # We now have one CONTINUOUS stable segment.
        # Median of that segment becomes this pose calibration.
        # ------------------------------------------------------

        self.calib[phase] = (
            self._median_measurements(
                self._phase_samples
            )
        )

        print()
        print(
            "============================================",
            flush=True,
        )

        print(
            f"[CALIB {phase}] COMPLETE",
            flush=True,
        )

        print(
            f"连续稳定 {stable_time:.2f}s, "
            f"有效样本 {len(self._phase_samples)} 帧",
            flush=True,
        )

        print(
            "============================================",
            flush=True,
        )

        # ------------------------------------------------------
        # Next pose.
        # ------------------------------------------------------

        self._phase_index += 1

        self._phase_start = None
        self._stable_anchor = None
        self._prev_measurement = None
        self._phase_samples = []
        self._last_status_print = 0.0

        if (
            self._phase_index
            >= len(self.PHASES)
        ):

            self._calib_done = True
            self._print_calibration_summary()
            return

        next_phase = self.PHASES[
            self._phase_index
        ]

        print()
        print(
            "--------------------------------------------",
            flush=True,
        )

        if next_phase == "FIST":

            print(
                "下一步：请完全握拳。",
                flush=True,
            )

        elif next_phase == "O":

            print(
                "下一步：请做拇指 + 食指 O 型。",
                flush=True,
            )

        print(
            "可以先随便移动到目标姿态；"
            "只有稳定下来后才会开始累计。",
            flush=True,
        )

        print(
            "--------------------------------------------",
            flush=True,
        )

    def _print_calibration_summary(self):

        print()
        print("============================================")
        print("THREE-POSE CALIBRATION COMPLETE")
        print("============================================")

        for finger in self.FINGERS:

            o = self.calib["OPEN"][
                "fingers"
            ][finger]

            f = self.calib["FIST"][
                "fingers"
            ][finger]

            oo = self.calib["O"][
                "fingers"
            ][finger]

            print(
                f"{finger:6s} | "
                f"OPEN root={o['root']:5.1f} "
                f"curl={o['curl']:5.1f} | "
                f"O root={oo['root']:5.1f} "
                f"curl={oo['curl']:5.1f} | "
                f"FIST root={f['root']:5.1f} "
                f"curl={f['curl']:5.1f}"
            )

        print(
            "thumb bend: "
            f"OPEN={self.calib['OPEN']['thumb']['bend']:.1f} "
            f"O={self.calib['O']['thumb']['bend']:.1f} "
            f"FIST={self.calib['FIST']['thumb']['bend']:.1f}"
        )

        print("============================================")
        print("NORMAL TELEOP START")
        print("============================================")
        print()

    # ==========================================================
    # Semantic normalization
    # ==========================================================

    def _finger_command(
        self,
        finger,
        raw,
    ):

        open_v = self.calib[
            "OPEN"
        ]["fingers"][finger]

        fist_v = self.calib[
            "FIST"
        ]["fingers"][finger]

        o_v = self.calib[
            "O"
        ]["fingers"][finger]

        # INDEX:
        # OPEN -> 0
        # O    -> 0.5
        # FIST -> 1
        if finger == "index":

            root = self._piecewise_half(
                raw["root"],
                open_v["root"],
                o_v["root"],
                fist_v["root"],
            )

            curl = self._piecewise_half(
                raw["curl"],
                open_v["curl"],
                o_v["curl"],
                fist_v["curl"],
            )

        else:

            # Other fingers:
            # OPEN -> 0
            # FIST -> 1
            root = self._linear01(
                raw["root"],
                open_v["root"],
                fist_v["root"],
            )

            curl = self._linear01(
                raw["curl"],
                open_v["curl"],
                fist_v["curl"],
            )

        # Spread zero from OPEN.
        spread = float(
            np.clip(
                (
                    raw["spread"]
                    - open_v["spread"]
                )
                / self.spread_full_deg,
                -1.0,
                1.0,
            )
        )

        # Don't let unreliable spread explode while curled.
        if root > 0.45 or curl > 0.45:
            spread *= 0.25

        return {
            "root": root,
            "curl": curl,
            "spread": spread,
        }

    def _thumb_command(
        self,
        raw,
    ):

        open_t = self.calib[
            "OPEN"
        ]["thumb"]

        o_t = self.calib[
            "O"
        ]["thumb"]

        # O calibration defines thumb maximum useful flex.
        flex = self._linear01(
            raw["bend"],
            open_t["bend"],
            o_t["bend"],
        )

        # ------------------------------------------------------
        # Thumb opposition is measured as progress from:
        #
        # OPEN thumb-tip palm-local position
        #       ->
        # O thumb-tip palm-local position
        #
        # It does NOT depend on robot geometry.
        # It also does not require human/robot bone matching.
        # ------------------------------------------------------

        p0 = np.asarray(
            open_t["pos"],
            dtype=np.float64,
        )

        p1 = np.asarray(
            o_t["pos"],
            dtype=np.float64,
        )

        p = np.asarray(
            raw["pos"],
            dtype=np.float64,
        )

        axis = p1 - p0

        denom = float(
            np.dot(
                axis,
                axis,
            )
        )

        if denom < 1e-8:
            opposition = 0.0
        else:
            opposition = float(
                np.clip(
                    np.dot(
                        p - p0,
                        axis,
                    ) / denom,
                    0.0,
                    1.0,
                )
            )

        return {
            "flex": flex,
            "opposition": opposition,
        }

    # ==========================================================
    # Command -> L20 u16
    # ==========================================================

    # ==========================================================
    # L20 ROBOT-NATIVE O POSE
    #
    # Obtained entirely from L20 kinematics:
    #
    #     robot thumb tip -> robot index tip
    #
    # NO Wuji/human point fitting was involved.
    # ==========================================================

    O_INDEX_MCP_PITCH = 0.74875164
    O_INDEX_PIP       = 0.86603117

    O_THUMB_ROLL      = -0.01983000
    O_THUMB_YAW       = 1.08779804
    O_THUMB_PITCH     = 0.23976201
    O_THUMB_MCP       = 0.49701970

    # Robot nominal open-thumb command.
    OPEN_THUMB_ROLL   = np.deg2rad(-10.0)
    OPEN_THUMB_YAW    = np.deg2rad(15.0)
    OPEN_THUMB_PITCH  = np.deg2rad(4.0)
    OPEN_THUMB_MCP    = 0.0

    @staticmethod
    def _robot_three_point_map(
        x,
        mid_value,
        max_value,
    ):
        """
        Human semantic command:
            0.0 = OPEN
            0.5 = O
            1.0 = FIST

        Robot command:
            0.0       -> robot OPEN
            mid_value -> robot-native O pose
            max_value -> robot maximum fist command
        """

        x = float(
            np.clip(
                x,
                0.0,
                1.0,
            )
        )

        if x <= 0.5:

            return (
                (x / 0.5)
                * mid_value
            )

        alpha = (
            (x - 0.5)
            / 0.5
        )

        return (
            mid_value
            + alpha
            * (
                max_value
                - mid_value
            )
        )

    def _command_to_u16(
        self,
        finger_cmd,
        thumb_cmd,
    ):

        u = np.clip(
            np.zeros(16),
            self.adapter.lower,
            self.adapter.upper,
        )

        # ======================================================
        # FOUR FINGERS
        # ======================================================

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

            # --------------------------------------------------
            # SPREAD
            # --------------------------------------------------

            spread_robot = (
                -c["spread"]
            )

            if spread_robot >= 0.0:

                u[ir] = (
                    spread_robot
                    * self.adapter.upper[
                        ir
                    ]
                )

            else:

                u[ir] = (
                    -spread_robot
                    * self.adapter.lower[
                        ir
                    ]
                )

            # --------------------------------------------------
            # INDEX:
            #
            # human calibration already defines:
            #
            # OPEN -> 0
            # O    -> 0.5
            # FIST -> 1
            #
            # Therefore map those semantic anchors onto the
            # ROBOT'S OWN native poses:
            #
            # 0   -> L20 open
            # .5  -> solved L20 O pose
            # 1   -> L20 maximum curl
            # --------------------------------------------------

            if finger == "index":

                u[ip] = self._robot_three_point_map(
                    c["root"],
                    self.O_INDEX_MCP_PITCH,
                    self.adapter.upper[ip],
                )

                u[ic] = self._robot_three_point_map(
                    c["curl"],
                    self.O_INDEX_PIP,
                    self.adapter.upper[ic],
                )

            else:

                # Other fingers only use OPEN -> FIST calibration.
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

        # ======================================================
        # THUMB
        # ======================================================

        flex = float(
            np.clip(
                thumb_cmd["flex"],
                0.0,
                1.0,
            )
        )

        opp = float(
            np.clip(
                thumb_cmd["opposition"],
                0.0,
                1.0,
            )
        )

        tmcp = self.u_index[
            "thumb_mcp"
        ]

        troll = self.u_index[
            "thumb_cmc_roll"
        ]

        tyaw = self.u_index[
            "thumb_cmc_yaw"
        ]

        tpitch = self.u_index[
            "thumb_cmc_pitch"
        ]

        # ------------------------------------------------------
        # CMC:
        #
        # Human opposition command is purely semantic:
        #
        #   0 = OPEN
        #   1 = O
        #
        # Robot executes its OWN solved O trajectory.
        # ------------------------------------------------------

        u[troll] = (
            (1.0 - opp)
            * self.OPEN_THUMB_ROLL
            + opp
            * self.O_THUMB_ROLL
        )

        u[tyaw] = (
            (1.0 - opp)
            * self.OPEN_THUMB_YAW
            + opp
            * self.O_THUMB_YAW
        )

        u[tpitch] = (
            (1.0 - opp)
            * self.OPEN_THUMB_PITCH
            + opp
            * self.O_THUMB_PITCH
        )

        # ------------------------------------------------------
        # Thumb distal flex:
        #
        # flex=1 corresponds exactly to the thumb flex observed
        # during Wuji O calibration, therefore robot flex=1 goes
        # exactly to the solved L20 O thumb MCP value.
        #
        # It is intentionally NOT driven to the URDF maximum.
        # ------------------------------------------------------

        u[tmcp] = (
            (1.0 - flex)
            * self.OPEN_THUMB_MCP
            + flex
            * self.O_THUMB_MCP
        )

        return np.clip(
            u,
            self.adapter.lower,
            self.adapter.upper,
        )

    def _calib_pose_u16(self):

        # During human calibration the robot stays in its nominal
        # open pose.  This does NOT influence human measurements.

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

        u[
            self.u_index[
                "thumb_mcp"
            ]
        ] = self.OPEN_THUMB_MCP

        return np.clip(
            u,
            self.adapter.lower,
            self.adapter.upper,
        )

    # ==========================================================
    # Solve
    # ==========================================================

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

        raw = self._measure_all(kp)

        if not self._calib_done:

            self._calibration_step(raw)

            u16 = self._calib_pose_u16()
            q21 = self.adapter.expand(u16)

            self.last_u16 = u16.copy()
            self.last_qpos = q21.copy()

            return q21

        finger_cmd = {}

        for finger in self.FINGERS:

            finger_cmd[finger] = (
                self._finger_command(
                    finger,
                    raw["fingers"][finger],
                )
            )

        thumb_cmd = self._thumb_command(
            raw["thumb"]
        )

        u16 = self._command_to_u16(
            finger_cmd,
            thumb_cmd,
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
            % self.debug_every == 0
        ):

            print(
                "[WUJI CMD3] -----------------------------"
            )

            for finger in self.FINGERS:

                c = finger_cmd[finger]

                print(
                    "[WUJI CMD3] "
                    f"{finger:6s} "
                    f"spread={c['spread']:+.2f} "
                    f"root={c['root']:.2f} "
                    f"curl={c['curl']:.2f}"
                )

            print(
                "[WUJI CMD3] "
                f"thumb "
                f"flex={thumb_cmd['flex']:.2f} "
                f"opp={thumb_cmd['opposition']:.2f}"
            )

            print(
                "[L20 CMD16] "
                + np.array2string(
                    u16,
                    precision=3,
                    separator=",",
                )
            )

            print(
                "[WUJI CMD3] -----------------------------"
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
        print("CALIBRATION RESET")
        self._print_phase_prompt()
