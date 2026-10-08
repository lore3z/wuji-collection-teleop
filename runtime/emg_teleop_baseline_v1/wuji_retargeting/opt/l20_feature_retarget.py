"""
Feature-space retargeting for LinkerHand L20.

IMPORTANT
---------
This retargeter does NOT match human keypoint positions to robot
keypoint positions.

It extracts human joint-motion features and maps those features onto
the 16 independently controllable L20 coordinates.

Four fingers:
    MCP ab/adduction
    MCP flexion
    PIP flexion
    DIP flexion

Thumb:
    CMC orientation from segment directions
    MCP flexion
    IP flexion

Mechanical coupling is handled only by L20KinematicAdapter.
"""

from __future__ import annotations

from typing import Optional

import nlopt
import numpy as np

from .base import BaseOptimizer
from .l20_retarget_v2 import L20KinematicAdapter
from ..mediapipe import (
    OPERATOR2MANO_RIGHT,
    OPERATOR2MANO_LEFT,
)


class L20FeatureRetargeter(BaseOptimizer):

    # Tell Retargeter explicitly:
    #
    # DO NOT pass the RobotMorphology target into this optimizer.
    #
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

    THUMB = (1, 2, 3, 4)

    DIP_RATIO = 0.80790960
    THUMB_IP_RATIO = 0.8079

    def __init__(self, config: dict):
        super().__init__(config)

        rcfg = config.get(
            "retarget",
            {},
        )

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

        self.q_index = dict(
            self.adapter.q_index
        )

        # --------------------------------------------------------
        # Only thumb CMC uses a tiny numerical direction optimizer.
        #
        # This is a 3-variable problem, NOT full-hand IK.
        # --------------------------------------------------------

        self.thumb_dir_w1 = float(
            rcfg.get(
                "feature_thumb_dir_w1",
                4.0,
            )
        )

        self.thumb_dir_w2 = float(
            rcfg.get(
                "feature_thumb_dir_w2",
                2.0,
            )
        )

        self.thumb_temporal = float(
            rcfg.get(
                "feature_thumb_temporal",
                0.03,
            )
        )

        self.thumb_maxeval = int(
            rcfg.get(
                "feature_thumb_maxeval",
                35,
            )
        )

        self.debug_every = int(
            rcfg.get(
                "feature_debug_every",
                20,
            )
        )

        self._frame_counter = 0
        self.last_u16 = None
        self.last_qpos = None

        # --------------------------------------------------------
        # Robot frames used only for SEMANTIC DIRECTION features.
        #
        # No robot landmark position is ever compared with a human
        # landmark position.
        # --------------------------------------------------------

        self.palm_frame = self.robot.get_link_index(
            self.origin_link_name
        )

        self.level1_frames = [
            self.robot.get_link_index(n)
            for n in self.link1_names
        ]

        self.level2_frames = [
            self.robot.get_link_index(n)
            for n in self.link3_names
        ]

        self.level3_frames = [
            self.robot.get_link_index(n)
            for n in self.link4_names
        ]

        self.tip_frames = [
            self.robot.get_link_index(n)
            for n in self.task_link_names
        ]

        # order from BaseOptimizer is:
        # thumb,index,middle,ring,pinky
        self.thumb_level1 = self.level1_frames[0]
        self.thumb_level2 = self.level2_frames[0]
        self.thumb_level3 = self.level3_frames[0]

        # --------------------------------------------------------
        # Build a ROBOT canonical palm coordinate frame once.
        #
        # Human keypoints have already been transformed to the same
        # MANO-style convention by apply_mediapipe_transformations().
        #
        # We construct the analogous robot frame from:
        #
        # palm + index base + middle base
        #
        # This is an orientation normalization only.
        # No scale or point matching is involved.
        # --------------------------------------------------------

        self.robot_to_canonical = (
            self._build_robot_canonical_rotation()
        )

        # --------------------------------------------------------
        # Determine the sign/scale of each robot MCP-roll joint
        # automatically from the URDF.
        #
        # We do not assume that a positive URDF joint angle has the
        # same semantic spread sign for every finger.
        # --------------------------------------------------------

        self._roll_feature_gain = (
            self._calibrate_roll_feature_gain()
        )

        # --------------------------------------------------------
        # Tiny 3D thumb-CMC optimizer.
        # --------------------------------------------------------

        self._thumb_u_idx = np.array(
            [
                self.u_index["thumb_cmc_roll"],
                self.u_index["thumb_cmc_yaw"],
                self.u_index["thumb_cmc_pitch"],
            ],
            dtype=np.int64,
        )

        self.thumb_opt = nlopt.opt(
            nlopt.LN_BOBYQA,
            3,
        )

        self.thumb_opt.set_lower_bounds(
            self.adapter.lower[
                self._thumb_u_idx
            ].tolist()
        )

        self.thumb_opt.set_upper_bounds(
            self.adapter.upper[
                self._thumb_u_idx
            ].tolist()
        )

        self.thumb_opt.set_maxeval(
            self.thumb_maxeval
        )

        self.thumb_opt.set_xtol_rel(
            1e-4
        )

        print(
            "[L20FeatureRetargeter] "
            "feature-space retargeting enabled"
        )

        print(
            "[L20FeatureRetargeter] "
            "RobotMorphology/keypoint-position matching is NOT used"
        )

        print(
            "[L20FeatureRetargeter] "
            "native control dimension = 16"
        )

        print(
            "[L20FeatureRetargeter] MCP-roll semantic gains:"
        )

        for name, gain in zip(
            self.FINGERS,
            self._roll_feature_gain,
        ):
            print(
                f"[L20FeatureRetargeter] "
                f"  {name:6s}: {gain:+.3f} qrad/feature-rad"
            )

    # ============================================================
    # Basic geometry
    # ============================================================

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
            return np.zeros(
                3,
                dtype=np.float64,
            )

        return v / n

    @classmethod
    def _angle(cls, a, b):
        ua = cls._unit(a)
        ub = cls._unit(b)

        if (
            np.linalg.norm(ua) < 1e-8
            or np.linalg.norm(ub) < 1e-8
        ):
            return 0.0

        c = np.clip(
            np.dot(ua, ub),
            -1.0,
            1.0,
        )

        return float(
            np.arccos(c)
        )

    @staticmethod
    def _signed_angle_yz(a, b):
        """
        Signed in-palm-plane angle around canonical +X.

        Canonical coordinates after mediapipe.py are approximately:

            X : palm normal
            Y : index-side direction
            Z : wrist -> middle MCP
        """

        a = np.asarray(
            [a[1], a[2]],
            dtype=np.float64,
        )

        b = np.asarray(
            [b[1], b[2]],
            dtype=np.float64,
        )

        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)

        if na < 1e-8 or nb < 1e-8:
            return 0.0

        a /= na
        b /= nb

        cross_x = (
            a[0] * b[1]
            - a[1] * b[0]
        )

        dot = np.clip(
            np.dot(a, b),
            -1.0,
            1.0,
        )

        return float(
            np.arctan2(
                cross_x,
                dot,
            )
        )

    # ============================================================
    # Human semantic features
    # ============================================================

    def _finger_features(
        self,
        kp,
        finger,
    ):
        """
        Returns:

            spread
            mcp_flex
            pip_flex
            dip_flex

        All are ANGLES.

        No lengths are used.
        """

        m, p, d, t = self.HUMAN[
            finger
        ]

        metacarpal = (
            kp[m]
            - kp[0]
        )

        proximal = (
            kp[p]
            - kp[m]
        )

        middle = (
            kp[d]
            - kp[p]
        )

        distal = (
            kp[t]
            - kp[d]
        )

        # --------------------------------------------------------
        # MCP spread:
        #
        # in-plane angular change from the metacarpal ray to the
        # proximal phalanx.
        # --------------------------------------------------------

        spread = self._signed_angle_yz(
            metacarpal,
            proximal,
        )

        # --------------------------------------------------------
        # MCP flex:
        #
        # total angular deflection contains spread + flex.
        #
        # Use the orthogonal-angle approximation:
        #
        #     total^2 ~= spread^2 + flex^2
        #
        # We only need non-negative flex because the L20 MCP pitch
        # range is [0, positive].
        # --------------------------------------------------------

        total = self._angle(
            metacarpal,
            proximal,
        )

        mcp_flex = np.sqrt(
            max(
                total * total
                - spread * spread,
                0.0,
            )
        )

        # Direct anatomical bending angles.
        pip_flex = self._angle(
            proximal,
            middle,
        )

        dip_flex = self._angle(
            middle,
            distal,
        )

        return (
            float(spread),
            float(mcp_flex),
            float(pip_flex),
            float(dip_flex),
        )

    def _thumb_features(
        self,
        kp,
    ):
        """
        Human thumb features.

        The CMC itself is represented by segment DIRECTIONS,
        not by robot/human landmark positions.
        """

        c, m, i, t = self.THUMB

        seg1 = (
            kp[m]
            - kp[c]
        )

        seg2 = (
            kp[i]
            - kp[m]
        )

        seg3 = (
            kp[t]
            - kp[i]
        )

        d1 = self._unit(
            seg1
        )

        d2 = self._unit(
            seg2
        )

        mcp_flex = self._angle(
            seg1,
            seg2,
        )

        ip_flex = self._angle(
            seg2,
            seg3,
        )

        # --------------------------------------------------------
        # Observability of axial CMC orientation.
        #
        # If seg1 and seg2 are nearly parallel, the thumb-chain
        # plane is ill-defined.  In that state CMC roll cannot be
        # honestly reconstructed from point positions.
        #
        # 0 -> nearly unobservable
        # 1 -> strongly observable
        # --------------------------------------------------------

        observability = float(
            np.clip(
                np.sin(mcp_flex) ** 2,
                0.0,
                1.0,
            )
        )

        return (
            d1,
            d2,
            float(mcp_flex),
            float(ip_flex),
            observability,
        )

    # ============================================================
    # Robot canonical palm frame
    # ============================================================

    @staticmethod
    def _estimate_frame(
        wrist,
        index,
        middle,
    ):
        points = np.stack(
            [
                wrist,
                index,
                middle,
            ],
            axis=0,
        )

        x_vector = (
            points[0]
            - points[2]
        )

        centered = (
            points
            - np.mean(
                points,
                axis=0,
                keepdims=True,
            )
        )

        _, _, v = np.linalg.svd(
            centered
        )

        normal = v[2]

        x = (
            x_vector
            - np.sum(
                x_vector
                * normal
            ) * normal
        )

        x /= (
            np.linalg.norm(x)
            + 1e-12
        )

        z = np.cross(
            x,
            normal,
        )

        if (
            np.sum(
                z
                * (
                    points[1]
                    - points[2]
                )
            )
            < 0
        ):
            normal *= -1
            z *= -1

        return np.stack(
            [
                x,
                normal,
                z,
            ],
            axis=1,
        )

    def _build_robot_canonical_rotation(
        self,
    ):
        q0 = self.adapter.expand(
            np.clip(
                np.zeros(16),
                self.adapter.lower,
                self.adapter.upper,
            )
        )

        self.robot.compute_forward_kinematics(
            q0
        )

        palm = self.robot.get_link_pose(
            self.palm_frame
        )[:3, 3]

        # index/middle level1 frames
        index = self.robot.get_link_pose(
            self.level1_frames[1]
        )[:3, 3]

        middle = self.robot.get_link_pose(
            self.level1_frames[2]
        )[:3, 3]

        # Fallback if level1 frame origins are degenerate.
        if (
            np.linalg.norm(index - middle) < 1e-6
            or np.linalg.norm(middle - palm) < 1e-6
        ):
            index = self.robot.get_link_pose(
                self.level2_frames[1]
            )[:3, 3]

            middle = self.robot.get_link_pose(
                self.level2_frames[2]
            )[:3, 3]

        frame = self._estimate_frame(
            palm,
            index,
            middle,
        )

        hand_side = getattr(
            self.robot,
            "hand_side",
            "right",
        )

        operator = (
            OPERATOR2MANO_RIGHT
            if hand_side == "right"
            else OPERATOR2MANO_LEFT
        )

        # For a row vector:
        #
        #     canonical = world_vec @ robot_to_canonical
        #
        return (
            frame @ operator
        )

    def _to_canonical_vec(
        self,
        v,
    ):
        return (
            np.asarray(
                v,
                dtype=np.float64,
            )
            @ self.robot_to_canonical
        )

    # ============================================================
    # Robot semantic features
    # ============================================================

    def _robot_finger_spread(
        self,
        u16,
        finger_index,
    ):
        """
        Robot semantic MCP spread feature used ONLY to discover
        the URDF joint sign/scale.

        finger_index:
            0 index
            1 middle
            2 ring
            3 pinky
        """

        robot_digit = (
            finger_index + 1
        )

        q = self.adapter.expand(
            u16
        )

        self.robot.compute_forward_kinematics(
            q
        )

        palm = self.robot.get_link_pose(
            self.palm_frame
        )[:3, 3]

        p1 = self.robot.get_link_pose(
            self.level1_frames[
                robot_digit
            ]
        )[:3, 3]

        p2 = self.robot.get_link_pose(
            self.level2_frames[
                robot_digit
            ]
        )[:3, 3]

        a = self._to_canonical_vec(
            p1 - palm
        )

        b = self._to_canonical_vec(
            p2 - p1
        )

        return self._signed_angle_yz(
            a,
            b,
        )

    def _calibrate_roll_feature_gain(
        self,
    ):
        gains = []

        q0 = np.clip(
            np.zeros(16),
            self.adapter.lower,
            self.adapter.upper,
        )

        eps = np.deg2rad(
            3.0
        )

        for finger_index, finger in enumerate(
            self.FINGERS
        ):
            ui = self.u_index[
                f"{finger}_mcp_roll"
            ]

            f0 = self._robot_finger_spread(
                q0,
                finger_index,
            )

            u1 = q0.copy()

            if (
                q0[ui] + eps
                <= self.adapter.upper[ui]
            ):
                dq = eps
            else:
                dq = -eps

            u1[ui] += dq

            f1 = self._robot_finger_spread(
                u1,
                finger_index,
            )

            # Wrap angular difference.
            df = np.arctan2(
                np.sin(f1 - f0),
                np.cos(f1 - f0),
            )

            slope = (
                df / dq
                if abs(dq) > 1e-12
                else 0.0
            )

            if abs(slope) < 0.20:
                gain = 1.0
            else:
                gain = 1.0 / slope

            # Avoid pathological amplification caused by a badly
            # chosen diagnostic frame.
            gain = float(
                np.clip(
                    gain,
                    -2.0,
                    2.0,
                )
            )

            gains.append(
                gain
            )

        return np.asarray(
            gains,
            dtype=np.float64,
        )

    # ============================================================
    # Direct finger projection
    # ============================================================

    def _set_four_fingers(
        self,
        u,
        kp,
    ):
        debug = {}

        r = self.DIP_RATIO

        for fi, finger in enumerate(
            self.FINGERS
        ):
            (
                spread,
                mcp_flex,
                pip_flex,
                dip_flex,
            ) = self._finger_features(
                kp,
                finger,
            )

            roll_i = self.u_index[
                f"{finger}_mcp_roll"
            ]

            pitch_i = self.u_index[
                f"{finger}_mcp_pitch"
            ]

            pip_i = self.u_index[
                f"{finger}_pip"
            ]

            # ----------------------------------------------------
            # MCP roll:
            #
            # semantic spread angle -> robot roll,
            # with URDF sign automatically identified.
            # ----------------------------------------------------

            q_roll = (
                self._roll_feature_gain[fi]
                * spread
            )

            # ----------------------------------------------------
            # MCP pitch:
            #
            # direct human flex angle.
            # ----------------------------------------------------

            q_pitch = mcp_flex

            # ----------------------------------------------------
            # Distal actuator:
            #
            # solve
            #
            #   min_q
            #       (q - human_PIP)^2
            #     + (r*q - human_DIP)^2
            #
            # analytically.
            # ----------------------------------------------------

            q_distal = (
                pip_flex
                + r * dip_flex
            ) / (
                1.0
                + r * r
            )

            u[roll_i] = np.clip(
                q_roll,
                self.adapter.lower[roll_i],
                self.adapter.upper[roll_i],
            )

            u[pitch_i] = np.clip(
                q_pitch,
                self.adapter.lower[pitch_i],
                self.adapter.upper[pitch_i],
            )

            u[pip_i] = np.clip(
                q_distal,
                self.adapter.lower[pip_i],
                self.adapter.upper[pip_i],
            )

            debug[finger] = {
                "spread": spread,
                "mcp": mcp_flex,
                "pip": pip_flex,
                "dip": dip_flex,
            }

        return debug

    # ============================================================
    # Thumb distal actuator
    # ============================================================

    def _set_thumb_flex(
        self,
        u,
        mcp_flex,
        ip_flex,
    ):
        r = self.THUMB_IP_RATIO

        # ========================================================
        # Thumb proximal flex
        #
        # Human:
        #   mcp_flex = angle(seg1, seg2)
        #   ip_flex  = angle(seg2, seg3)
        #
        # Robot r2 used by the CMC direction solver is the
        # MCP->IP segment.  Its direction is controlled by
        # thumb_mcp, NOT by thumb_ip.
        #
        # Therefore thumb_mcp must first reproduce HUMAN MCP
        # flex directly.  Mixing human IP flex into this value
        # corrupts r2 before the CMC optimizer even starts.
        #
        # Robot thumb_ip remains mechanically coupled through
        # L20KinematicAdapter / URDF mimic.
        # ========================================================

        q_flex = mcp_flex

        ui = self.u_index[
            "thumb_mcp"
        ]

        u[ui] = np.clip(
            q_flex,
            self.adapter.lower[ui],
            self.adapter.upper[ui],
        )

    # ============================================================
    # Thumb CMC direction matching
    # ============================================================

    def _robot_thumb_dirs(
        self,
        u,
    ):
        q = self.adapter.expand(
            u
        )

        self.robot.compute_forward_kinematics(
            q
        )

        p1 = self.robot.get_link_pose(
            self.thumb_level1
        )[:3, 3]

        p2 = self.robot.get_link_pose(
            self.thumb_level2
        )[:3, 3]

        p3 = self.robot.get_link_pose(
            self.thumb_level3
        )[:3, 3]

        d1 = self._unit(
            self._to_canonical_vec(
                p2 - p1
            )
        )

        d2 = self._unit(
            self._to_canonical_vec(
                p3 - p2
            )
        )

        return d1, d2

    def _solve_thumb_cmc(
        self,
        u,
        human_d1,
        human_d2,
        observability,
    ):
        base_u = u.copy()

        if self.last_u16 is not None:
            x0 = self.last_u16[
                self._thumb_u_idx
            ].copy()
        else:
            x0 = base_u[
                self._thumb_u_idx
            ].copy()

        lo = self.adapter.lower[
            self._thumb_u_idx
        ]

        hi = self.adapter.upper[
            self._thumb_u_idx
        ]

        x0 = np.clip(
            x0,
            lo,
            hi,
        )

        span = np.maximum(
            hi - lo,
            1e-6,
        )

        # --------------------------------------------------------
        # Important:
        #
        # second segment contributes strongly only when thumb bend
        # makes axial orientation observable.
        #
        # When the thumb is almost straight, roll is not pretended
        # to be observable from points; temporal minimum-motion
        # regularization resolves the null space instead.
        # --------------------------------------------------------

        # Diagnostic:
        # Keep the second thumb segment fully active.
        #
        # Previously, when the human thumb MCP was nearly straight,
        # observability ~= 0 and d2 was reduced to only ~10%.
        # That leaves the 3-DOF CMC orientation poorly constrained.
        w2 = self.thumb_dir_w2

        reference = x0.copy()

        def objective(x, grad):
            del grad

            test_u = base_u.copy()

            test_u[
                self._thumb_u_idx
            ] = x

            r1, r2 = self._robot_thumb_dirs(
                test_u
            )

            d1_cost = (
                1.0
                - np.clip(
                    np.dot(
                        r1,
                        human_d1,
                    ),
                    -1.0,
                    1.0,
                )
            )

            d2_cost = (
                1.0
                - np.clip(
                    np.dot(
                        r2,
                        human_d2,
                    ),
                    -1.0,
                    1.0,
                )
            )

            dx = (
                x - reference
            ) / span

            temporal = float(
                dx @ dx
            )

            return float(
                self.thumb_dir_w1
                * d1_cost
                + w2 * d2_cost
                + self.thumb_temporal
                * temporal
            )

        self.thumb_opt.set_min_objective(
            objective
        )

        # ========================================================
        # Thumb CMC multi-start branch search
        #
        # The L20 thumb CMC behaves like an X-Z-X serial
        # orientation chain.  At yaw ~= 0, roll and pitch become
        # directionally degenerate, so a local optimizer starting
        # from the previous frame can remain trapped on that
        # singular branch.
        #
        # Try several LEGAL positive-yaw branches and keep the
        # solution with the lowest ORIGINAL objective.
        #
        # No gesture calibration.
        # No pinch pose.
        # No modification of the objective.
        # ========================================================

        seeds = [
            x0.copy(),
        ]

        yaw_i = 1
        pitch_i = 2

        # Only perform branch exploration while we are close to
        # the yaw lower bound.  Once a valid branch is found,
        # following frames revert naturally to the previous result.
        near_yaw_singularity = (
            abs(x0[yaw_i] - lo[yaw_i])
            < np.deg2rad(2.0)
        )

        if near_yaw_singularity:

            seed_specs = [
                # yaw, pitch
                (15.0,  0.0),
                (25.0,  5.0),
                (35.0, 10.0),
                (45.0, 15.0),
                (60.0, 20.0),
                (75.0, 25.0),
            ]

            for yaw_deg, pitch_deg in seed_specs:

                seed = x0.copy()

                seed[yaw_i] = np.deg2rad(
                    yaw_deg
                )

                seed[pitch_i] = np.deg2rad(
                    pitch_deg
                )

                seed = np.clip(
                    seed,
                    lo,
                    hi,
                )

                seeds.append(seed)

        best_result = None
        best_cost = np.inf

        debug_candidates = []

        for seed in seeds:

            try:
                candidate = np.asarray(
                    self.thumb_opt.optimize(
                        seed
                    ),
                    dtype=np.float64,
                )

                candidate = np.clip(
                    candidate,
                    lo,
                    hi,
                )

                cost = float(
                    objective(
                        candidate,
                        None,
                    )
                )

                debug_candidates.append(
                    (
                        candidate.copy(),
                        cost,
                    )
                )

                if cost < best_cost:
                    best_cost = cost
                    best_result = candidate.copy()

            except Exception as exc:
                print(
                    "[L20FeatureRetargeter] "
                    "thumb optimizer candidate warning:",
                    repr(exc),
                )

        if best_result is None:
            result = x0
        else:
            result = best_result

        if not hasattr(
            self,
            "_thumb_multistart_counter",
        ):
            self._thumb_multistart_counter = 0

        self._thumb_multistart_counter += 1

        if (
            near_yaw_singularity
            and
            self._thumb_multistart_counter % 20 == 0
        ):
            print(
                "\n[THUMB MULTISTART]"
            )

            for cand, cost in debug_candidates:
                print(
                    "  "
                    + np.array2string(
                        np.degrees(cand),
                        precision=1,
                        separator=",",
                    )
                    + f"  cost={cost:.6f}"
                )

            print(
                "  BEST="
                + np.array2string(
                    np.degrees(result),
                    precision=1,
                    separator=",",
                )
                + f" cost={best_cost:.6f}"
            )

        result = np.clip(
            result,
            lo,
            hi,
        )

        # ============================================================
        # THUMB CMC LOCAL SENSITIVITY DIAGNOSTIC
        #
        # Diagnostic only.
        # Does NOT modify result / u16.
        #
        # For roll/yaw/pitch independently, evaluate:
        #
        #     current
        #     current - 5 deg
        #     current + 5 deg
        #
        # This tells us whether each DOF is trying to move outside
        # its legal bound, is locally useful, or is almost invisible.
        # ============================================================

        if not hasattr(
            self,
            "_thumb_sens_counter",
        ):
            self._thumb_sens_counter = 0

        self._thumb_sens_counter += 1

        if (
            self._thumb_sens_counter
            % 30
            == 0
        ):

            def _diag_score(x):

                x = np.asarray(
                    x,
                    dtype=np.float64,
                )

                test_u = base_u.copy()

                test_u[
                    self._thumb_u_idx
                ] = x

                rr1, rr2 = (
                    self._robot_thumb_dirs(
                        test_u
                    )
                )

                dot1 = float(
                    np.clip(
                        np.dot(
                            rr1,
                            human_d1,
                        ),
                        -1.0,
                        1.0,
                    )
                )

                dot2 = float(
                    np.clip(
                        np.dot(
                            rr2,
                            human_d2,
                        ),
                        -1.0,
                        1.0,
                    )
                )

                c1 = 1.0 - dot1
                c2 = 1.0 - dot2

                dx = (
                    x - reference
                ) / span

                temporal = float(
                    dx @ dx
                )

                total = float(
                    self.thumb_dir_w1
                    * c1
                    + w2
                    * c2
                    + self.thumb_temporal
                    * temporal
                )

                e1 = float(
                    np.degrees(
                        self._angle(
                            rr1,
                            human_d1,
                        )
                    )
                )

                e2 = float(
                    np.degrees(
                        self._angle(
                            rr2,
                            human_d2,
                        )
                    )
                )

                return (
                    total,
                    e1,
                    e2,
                    temporal,
                )

            names = [
                "roll",
                "yaw",
                "pitch",
            ]

            print(
                "\n"
                "============================================",
                flush=True,
            )

            print(
                "[THUMB SENS]",
                flush=True,
            )

            print(
                "lo(deg)   =",
                np.round(
                    np.degrees(lo),
                    2,
                ),
                flush=True,
            )

            print(
                "hi(deg)   =",
                np.round(
                    np.degrees(hi),
                    2,
                ),
                flush=True,
            )

            print(
                "x0(deg)   =",
                np.round(
                    np.degrees(x0),
                    2,
                ),
                flush=True,
            )

            print(
                "result(deg)=",
                np.round(
                    np.degrees(result),
                    2,
                ),
                flush=True,
            )

            print(
                "weights: "
                f"w1={self.thumb_dir_w1:.4f} "
                f"w2={w2:.4f} "
                f"temporal={self.thumb_temporal:.6f}",
                flush=True,
            )

            base_score = _diag_score(
                result
            )

            print(
                "CURRENT"
                f" total={base_score[0]:.6f}"
                f" d1={base_score[1]:6.2f}deg"
                f" d2={base_score[2]:6.2f}deg"
                f" temp={base_score[3]:.6f}",
                flush=True,
            )

            delta = np.deg2rad(
                5.0
            )

            for j, name in enumerate(
                names
            ):

                for sign, label in [
                    (-1.0, "-5"),
                    (+1.0, "+5"),
                ]:

                    xx = result.copy()

                    wanted = (
                        xx[j]
                        + sign * delta
                    )

                    xx[j] = np.clip(
                        wanted,
                        lo[j],
                        hi[j],
                    )

                    actual_delta = float(
                        np.degrees(
                            xx[j]
                            - result[j]
                        )
                    )

                    score = _diag_score(
                        xx
                    )

                    clipped = (
                        abs(
                            xx[j]
                            - wanted
                        )
                        > 1e-9
                    )

                    print(
                        f"{name:5s} {label:>2s}: "
                        f"actual={actual_delta:+6.2f}deg "
                        f"total={score[0]:.6f} "
                        f"d1={score[1]:6.2f}deg "
                        f"d2={score[2]:6.2f}deg "
                        f"temp={score[3]:.6f}"
                        + (
                            "  [BOUND]"
                            if clipped
                            else ""
                        ),
                        flush=True,
                    )

            print(
                "============================================"
                "\n",
                flush=True,
            )

        # ========================================================
        # ONE-SHOT THUMB 4DOF JOINT-OPTIMIZATION PROBE
        #
        # Diagnostic only.
        #
        # Jointly optimize:
        #   CMC roll / yaw / pitch + thumb_mcp
        #
        # This tests whether the large d2 error is caused by the
        # architecture fixing thumb_mcp BEFORE solving CMC.
        #
        # No calibration.
        # No pinch pose.
        # Does NOT change the actual output.
        # ========================================================

        if not hasattr(self, "_thumb_4d_probe_done"):
            self._thumb_4d_probe_done = False

        # Run once only when the human thumb is substantially bent,
        # so the probe is informative.
        if (
            not self._thumb_4d_probe_done
            and observability > 0.70
        ):
            self._thumb_4d_probe_done = True

            mcp_ui = self.u_index[
                "thumb_mcp"
            ]

            idx4 = np.array(
                [
                    self.u_index["thumb_cmc_roll"],
                    self.u_index["thumb_cmc_yaw"],
                    self.u_index["thumb_cmc_pitch"],
                    mcp_ui,
                ],
                dtype=np.int64,
            )

            lo4 = self.adapter.lower[idx4].copy()
            hi4 = self.adapter.upper[idx4].copy()

            current4 = np.array(
                [
                    result[0],
                    result[1],
                    result[2],
                    base_u[mcp_ui],
                ],
                dtype=np.float64,
            )

            current4 = np.clip(
                current4,
                lo4,
                hi4,
            )

            def _score4(x):
                test_u = base_u.copy()

                test_u[idx4] = np.asarray(
                    x,
                    dtype=np.float64,
                )

                rr1, rr2 = self._robot_thumb_dirs(
                    test_u
                )

                e1 = float(
                    self._angle(
                        rr1,
                        human_d1,
                    )
                )

                e2 = float(
                    self._angle(
                        rr2,
                        human_d2,
                    )
                )

                cost = float(
                    self.thumb_dir_w1
                    * (
                        1.0
                        - np.clip(
                            np.dot(
                                rr1,
                                human_d1,
                            ),
                            -1.0,
                            1.0,
                        )
                    )
                    + self.thumb_dir_w2
                    * (
                        1.0
                        - np.clip(
                            np.dot(
                                rr2,
                                human_d2,
                            ),
                            -1.0,
                            1.0,
                        )
                    )
                )

                return (
                    cost,
                    np.degrees(e1),
                    np.degrees(e2),
                )

            seeds = [
                current4.copy(),
                np.array([
                    np.deg2rad(-25),
                    np.deg2rad(15),
                    np.deg2rad(5),
                    np.deg2rad(20),
                ]),
                np.array([
                    np.deg2rad(-20),
                    np.deg2rad(30),
                    np.deg2rad(10),
                    np.deg2rad(35),
                ]),
                np.array([
                    np.deg2rad(-10),
                    np.deg2rad(45),
                    np.deg2rad(15),
                    np.deg2rad(50),
                ]),
                np.array([
                    0.0,
                    np.deg2rad(60),
                    np.deg2rad(20),
                    np.deg2rad(60),
                ]),
                np.array([
                    np.deg2rad(15),
                    np.deg2rad(75),
                    np.deg2rad(25),
                    np.deg2rad(65),
                ]),
            ]

            best = None
            best_score = None
            candidates = []

            for seed in seeds:
                seed = np.clip(
                    seed,
                    lo4,
                    hi4,
                )

                opt4 = nlopt.opt(
                    nlopt.LN_BOBYQA,
                    4,
                )

                opt4.set_lower_bounds(
                    lo4.tolist()
                )

                opt4.set_upper_bounds(
                    hi4.tolist()
                )

                opt4.set_maxeval(150)
                opt4.set_xtol_rel(1e-5)

                def _obj4(x, grad):
                    del grad
                    return _score4(x)[0]

                opt4.set_min_objective(
                    _obj4
                )

                try:
                    cand = np.asarray(
                        opt4.optimize(seed),
                        dtype=np.float64,
                    )

                    cand = np.clip(
                        cand,
                        lo4,
                        hi4,
                    )

                    score = _score4(cand)

                    candidates.append(
                        (
                            cand.copy(),
                            score,
                        )
                    )

                    if (
                        best_score is None
                        or score[0] < best_score[0]
                    ):
                        best = cand.copy()
                        best_score = score

                except Exception as exc:
                    print(
                        "[THUMB 4D PROBE ERROR]",
                        repr(exc),
                        flush=True,
                    )

            cur_score = _score4(
                current4
            )

            print(
                "\n"
                "============================================",
                flush=True,
            )

            print(
                "[THUMB 4D PROBE]",
                flush=True,
            )

            print(
                "human MCP/IP = "
                f"{np.degrees(self._angle(human_d1, human_d2)):.2f}deg"
                " / "
                "(IP not independently actuated)",
                flush=True,
            )

            print(
                "CURRENT "
                + np.array2string(
                    np.degrees(current4),
                    precision=2,
                    separator=",",
                )
                + (
                    f"  cost={cur_score[0]:.6f}"
                    f" d1={cur_score[1]:.2f}"
                    f" d2={cur_score[2]:.2f}"
                ),
                flush=True,
            )

            for cand, score in candidates:
                print(
                    "CAND    "
                    + np.array2string(
                        np.degrees(cand),
                        precision=2,
                        separator=",",
                    )
                    + (
                        f"  cost={score[0]:.6f}"
                        f" d1={score[1]:.2f}"
                        f" d2={score[2]:.2f}"
                    ),
                    flush=True,
                )

            if best is not None:
                print(
                    "BEST    "
                    + np.array2string(
                        np.degrees(best),
                        precision=2,
                        separator=",",
                    )
                    + (
                        f"  cost={best_score[0]:.6f}"
                        f" d1={best_score[1]:.2f}"
                        f" d2={best_score[2]:.2f}"
                    ),
                    flush=True,
                )

            print(
                "order = [roll, yaw, pitch, thumb_mcp]",
                flush=True,
            )

            print(
                "============================================"
                "\n",
                flush=True,
            )

        # IMPORTANT:
        # Probe is diagnostic only.
        # Actual controller still uses the original 3DOF result.
        u[
            self._thumb_u_idx
        ] = result

        r1, r2 = self._robot_thumb_dirs(
            u
        )

        err1 = np.degrees(
            self._angle(
                r1,
                human_d1,
            )
        )

        err2 = np.degrees(
            self._angle(
                r2,
                human_d2,
            )
        )

        return (
            float(err1),
            float(err2),
        )

    # ============================================================
    # Public API
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
                f"Expected (21,3), got {kp.shape}"
            )

        if not np.all(
            np.isfinite(kp)
        ):
            raise ValueError(
                "Human keypoints contain NaN/Inf"
            )

        # --------------------------------------------------------
        # Start directly in TRUE independent control space.
        # --------------------------------------------------------

        if last_qpos is not None:
            qlast = np.asarray(
                last_qpos,
                dtype=np.float64,
            )

            if qlast.shape == (21,):
                u = self.adapter.compress(
                    qlast
                )
            else:
                u = None

        else:
            u = None

        if u is None:

            if self.last_u16 is not None:
                u = self.last_u16.copy()
            else:
                u = np.clip(
                    np.zeros(16),
                    self.adapter.lower,
                    self.adapter.upper,
                )

        # --------------------------------------------------------
        # Four fingers:
        # geometry -> anatomical ANGLES -> L20 actuator coordinates.
        # --------------------------------------------------------

        finger_debug = self._set_four_fingers(
            u,
            kp,
        )

        # --------------------------------------------------------
        # Thumb:
        #
        # distal flex is directly projected from human joint angles;
        # CMC is solved only from segment directions.
        # --------------------------------------------------------

        (
            human_d1,
            human_d2,
            thumb_mcp,
            thumb_ip,
            observability,
        ) = self._thumb_features(
            kp
        )

        self._set_thumb_flex(
            u,
            thumb_mcp,
            thumb_ip,
        )

        err1, err2 = self._solve_thumb_cmc(
            u,
            human_d1,
            human_d2,
            observability,
        )

        u = np.clip(
            u,
            self.adapter.lower,
            self.adapter.upper,
        )

        q = self.adapter.expand(
            u
        )

        self.last_u16 = u.copy()
        self.last_qpos = q.copy()

        # --------------------------------------------------------
        # Transparent diagnostics.
        # --------------------------------------------------------

        self._frame_counter += 1

        if (
            self.debug_every > 0
            and self._frame_counter
            % self.debug_every
            == 0
        ):
            # ==================================================
            # OBSERVABILITY AUDIT
            #
            # Diagnostic only.
            # Does NOT participate in control output.
            # ==================================================

            print(
                "[OBS AUDIT] ------------------------------"
            )

            for _finger in self.FINGERS:

                _m, _p, _d, _t = self.HUMAN[
                    _finger
                ]

                _meta = self._unit(
                    kp[_m] - kp[0]
                )

                _prox = self._unit(
                    kp[_p] - kp[_m]
                )

                _mid = self._unit(
                    kp[_d] - kp[_p]
                )

                _dist = self._unit(
                    kp[_t] - kp[_d]
                )

                # How much of proximal phalanx remains observable
                # in the palm plane.
                _plane_obs = float(
                    np.linalg.norm(
                        _prox[1:]
                    )
                )

                # Direct elevation out of the palm YZ plane.
                #
                # This is a much cleaner MCP-flex candidate than:
                #
                # sqrt(total_angle^2 - spread^2)
                _plane_flex = float(
                    np.arctan2(
                        abs(_prox[0]),
                        max(
                            _plane_obs,
                            1e-8,
                        ),
                    )
                )

                # Existing spread candidate.
                _spread_old = self._signed_angle_yz(
                    _meta,
                    _prox,
                )

                # Existing PIP/DIP observables.
                _pip = self._angle(
                    _prox,
                    _mid,
                )

                _dip = self._angle(
                    _mid,
                    _dist,
                )

                # Combined distal curvature candidate.
                #
                # Uses a longer baseline:
                # proximal phalanx versus PIP->TIP.
                #
                # This does NOT match robot bone positions.
                _pip_to_tip = self._unit(
                    kp[_t] - kp[_p]
                )

                _distal_total = self._angle(
                    _prox,
                    _pip_to_tip,
                )

                print(
                    "[OBS FINGER] "
                    f"{_finger:6s} "
                    f"prox="
                    + np.array2string(
                        _prox,
                        precision=3,
                        separator=",",
                    )
                    + " "
                    f"planeObs={_plane_obs:.3f} "
                    f"planeFlex={np.degrees(_plane_flex):6.1f} "
                    f"spreadOld={np.degrees(_spread_old):7.1f} "
                    f"PIP={np.degrees(_pip):6.1f} "
                    f"DIP={np.degrees(_dip):6.1f} "
                    f"distalTotal={np.degrees(_distal_total):6.1f}"
                )

            (
                _td1,
                _td2,
                _tmcp,
                _tip,
                _tobs,
            ) = self._thumb_features(
                kp
            )

            print(
                "[OBS THUMB] "
                "d1="
                + np.array2string(
                    _td1,
                    precision=3,
                    separator=",",
                )
                + " d2="
                + np.array2string(
                    _td2,
                    precision=3,
                    separator=",",
                )
                + " "
                f"MCP={np.degrees(_tmcp):.1f} "
                f"IP={np.degrees(_tip):.1f} "
                f"obs={_tobs:.3f}"
            )

            print(
                "[OBS AUDIT] ------------------------------"
            )

            print(
                "[L20 FEATURE] human angles deg:"
            )

            for finger in self.FINGERS:
                f = finger_debug[
                    finger
                ]

                print(
                    "[L20 FEATURE] "
                    f"{finger:6s} "
                    f"spread={np.degrees(f['spread']):6.1f} "
                    f"mcp={np.degrees(f['mcp']):6.1f} "
                    f"pip={np.degrees(f['pip']):6.1f} "
                    f"dip={np.degrees(f['dip']):6.1f}"
                )

            print(
                "[L20 FEATURE] "
                f"thumb human "
                f"mcp={np.degrees(thumb_mcp):.1f} "
                f"ip={np.degrees(thumb_ip):.1f} "
                f"obs={observability:.2f}"
            )

            print(
                "[L20 FEATURE] "
                f"thumb direction error "
                f"d1={err1:.1f}deg "
                f"d2={err2:.1f}deg"
            )

            print(
                "[L20 CTRL16] "
                + np.array2string(
                    u,
                    precision=3,
                    separator=",",
                )
            )

            print(
                "[L20 THUMB] "
                f"roll={np.degrees(u[self.u_index['thumb_cmc_roll']]):.1f} "
                f"yaw={np.degrees(u[self.u_index['thumb_cmc_yaw']]):.1f} "
                f"pitch={np.degrees(u[self.u_index['thumb_cmc_pitch']]):.1f} "
                f"flex={np.degrees(u[self.u_index['thumb_mcp']]):.1f}"
            )

        return q

    def compute_cost(
        self,
        qpos: np.ndarray,
        mediapipe_keypoints: np.ndarray,
    ) -> float:
        """
        Diagnostic semantic cost.

        This deliberately does NOT calculate human-vs-robot point
        distance.
        """

        q = np.asarray(
            qpos,
            dtype=np.float64,
        )

        u = self.adapter.compress(
            q
        )

        kp = np.asarray(
            mediapipe_keypoints,
            dtype=np.float64,
        )

        cost = 0.0

        r = self.DIP_RATIO

        for fi, finger in enumerate(
            self.FINGERS
        ):
            spread, mcp, pip, dip = (
                self._finger_features(
                    kp,
                    finger,
                )
            )

            roll_i = self.u_index[
                f"{finger}_mcp_roll"
            ]

            pitch_i = self.u_index[
                f"{finger}_mcp_pitch"
            ]

            pip_i = self.u_index[
                f"{finger}_pip"
            ]

            expected_roll = (
                self._roll_feature_gain[fi]
                * spread
            )

            cost += (
                u[roll_i]
                - expected_roll
            ) ** 2

            cost += (
                u[pitch_i]
                - mcp
            ) ** 2

            cost += (
                u[pip_i]
                - pip
            ) ** 2

            cost += (
                r * u[pip_i]
                - dip
            ) ** 2

        (
            h1,
            h2,
            tmcp,
            tip,
            obs,
        ) = self._thumb_features(
            kp
        )

        del obs

        tf = self.u_index[
            "thumb_mcp"
        ]

        cost += (
            u[tf] - tmcp
        ) ** 2

        cost += (
            self.THUMB_IP_RATIO
            * u[tf]
            - tip
        ) ** 2

        r1, r2 = self._robot_thumb_dirs(
            u
        )

        cost += (
            1.0
            - np.clip(
                np.dot(r1, h1),
                -1.0,
                1.0,
            )
        )

        cost += (
            1.0
            - np.clip(
                np.dot(r2, h2),
                -1.0,
                1.0,
            )
        )

        return float(cost)

    def reset(self):
        self.last_u16 = None
        self.last_qpos = None
        self._frame_counter = 0
