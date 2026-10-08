"""
L20 native 16-DOF retargeter.

Design goals
------------
1. Optimize the 16 independently controllable L20 coordinates directly.
2. Expand u16 -> q21 through exactly one kinematic adapter.
3. Keep all optimization results mechanically feasible.
4. Constrain the COMPLETE thumb chain, especially its proximal/C MC pose,
   instead of allowing fingertip distance alone to select an IK branch.
5. Avoid gesture-specific thumb hacks, dynamic CMC bounds and branch locks.

The optimizer returns q21 for compatibility with the existing Retargeter /
MuJoCo viewer, while ``last_u16`` contains the actual independent L20
control coordinates.
"""

from __future__ import annotations

from typing import Optional

import nlopt
import numpy as np

from .base import (
    BaseOptimizer,
    M_TO_CM,
    TimingStats,
)


# ======================================================================
# L20 16DOF <-> 21 mechanical-coordinate adapter
# ======================================================================

class L20KinematicAdapter:
    """
    Map 16 independently controllable L20 coordinates to the 21
    mechanical joint coordinates contained in the URDF.

    q21 = E @ u16

    The five dependent coordinates are:

        index_dip  = 0.80790960 * index_pip
        middle_dip = 0.80790960 * middle_pip
        ring_dip   = 0.80790960 * ring_pip
        pinky_dip  = 0.80790960 * pinky_pip
        thumb_ip   = 0.8079 * thumb_mcp

    The mapping is isolated in this class intentionally.  If later
    hardware calibration shows a different transmission law, only this
    adapter has to change.
    """

    CONTROL_NAMES = [
        # index
        "index_mcp_roll",
        "index_mcp_pitch",
        "index_pip",

        # middle
        "middle_mcp_roll",
        "middle_mcp_pitch",
        "middle_pip",

        # ring
        "ring_mcp_roll",
        "ring_mcp_pitch",
        "ring_pip",

        # pinky
        "pinky_mcp_roll",
        "pinky_mcp_pitch",
        "pinky_pip",

        # thumb
        "thumb_cmc_roll",
        "thumb_cmc_yaw",
        "thumb_cmc_pitch",
        "thumb_mcp",
    ]

    COUPLINGS = [
        ("index_dip",  "index_pip",  0.80790960),
        ("middle_dip", "middle_pip", 0.80790960),
        ("ring_dip",   "ring_pip",   0.80790960),
        ("pinky_dip",  "pinky_pip",  0.80790960),
        ("thumb_ip",   "thumb_mcp",  0.8079),
    ]

    def __init__(self, robot):
        self.robot = robot

        self.q_names = list(
            robot.dof_joint_names
        )

        self.nq = len(
            self.q_names
        )

        if self.nq != 21:
            raise RuntimeError(
                "L20RetargetV2 expects 21 URDF mechanical "
                f"coordinates, got nq={self.nq}"
            )

        self.q_index = {
            name: i
            for i, name in enumerate(
                self.q_names
            )
        }

        required = set(
            self.CONTROL_NAMES
        )

        for child, parent, _ in self.COUPLINGS:
            required.add(child)
            required.add(parent)

        missing = sorted(
            required - set(self.q_names)
        )

        if missing:
            raise RuntimeError(
                "L20 URDF is missing required joints: "
                f"{missing}"
            )

        self.u_names = list(
            self.CONTROL_NAMES
        )

        self.nu = len(
            self.u_names
        )

        if self.nu != 16:
            raise RuntimeError(
                f"Internal error: expected 16 controls, got {self.nu}"
            )

        self.u_index = {
            name: i
            for i, name in enumerate(
                self.u_names
            )
        }

        # --------------------------------------------------------------
        # Expansion matrix:
        #
        #              q21 = E @ u16
        #
        # --------------------------------------------------------------

        self.E = np.zeros(
            (self.nq, self.nu),
            dtype=np.float64,
        )

        for name in self.u_names:
            qi = self.q_index[name]
            ui = self.u_index[name]

            self.E[qi, ui] = 1.0

        for child, parent, ratio in self.COUPLINGS:
            child_qi = self.q_index[child]
            parent_ui = self.u_index[parent]

            self.E[
                child_qi,
                parent_ui,
            ] = ratio

        # Verify every q21 coordinate is represented exactly once.
        covered_rows = (
            np.max(
                np.abs(self.E),
                axis=1,
            )
            > 0
        )

        if not np.all(covered_rows):
            uncovered = [
                self.q_names[i]
                for i in np.where(
                    ~covered_rows
                )[0]
            ]

            raise RuntimeError(
                "L20 adapter does not cover URDF joints: "
                f"{uncovered}"
            )

        # --------------------------------------------------------------
        # Independent-coordinate bounds.
        #
        # Start from the parent URDF limit and intersect it with the
        # child limit induced by each mechanical coupling.
        # --------------------------------------------------------------

        q_limits = np.asarray(
            robot.joint_limits,
            dtype=np.float64,
        )

        self.lower = np.empty(
            self.nu,
            dtype=np.float64,
        )

        self.upper = np.empty(
            self.nu,
            dtype=np.float64,
        )

        for name in self.u_names:
            qi = self.q_index[name]
            ui = self.u_index[name]

            self.lower[ui] = q_limits[qi, 0]
            self.upper[ui] = q_limits[qi, 1]

        for child, parent, ratio in self.COUPLINGS:

            if abs(ratio) < 1e-12:
                raise RuntimeError(
                    f"Invalid zero coupling ratio for {child}"
                )

            child_qi = self.q_index[child]
            parent_ui = self.u_index[parent]

            child_lo = q_limits[
                child_qi,
                0,
            ]

            child_hi = q_limits[
                child_qi,
                1,
            ]

            if ratio > 0.0:
                induced_lo = (
                    child_lo / ratio
                )

                induced_hi = (
                    child_hi / ratio
                )
            else:
                induced_lo = (
                    child_hi / ratio
                )

                induced_hi = (
                    child_lo / ratio
                )

            self.lower[parent_ui] = max(
                self.lower[parent_ui],
                induced_lo,
            )

            self.upper[parent_ui] = min(
                self.upper[parent_ui],
                induced_hi,
            )

        if np.any(
            self.lower > self.upper
        ):
            bad = []

            for i, name in enumerate(
                self.u_names
            ):
                if self.lower[i] > self.upper[i]:
                    bad.append(
                        (
                            name,
                            self.lower[i],
                            self.upper[i],
                        )
                    )

            raise RuntimeError(
                "Invalid L20 independent bounds: "
                f"{bad}"
            )

    def expand(
        self,
        u16: np.ndarray,
    ) -> np.ndarray:
        """Expand independent u16 into URDF q21."""

        u = np.asarray(
            u16,
            dtype=np.float64,
        )

        if u.shape != (self.nu,):
            raise ValueError(
                f"Expected u16 shape {(self.nu,)}, got {u.shape}"
            )

        u = np.clip(
            u,
            self.lower,
            self.upper,
        )

        return self.E @ u

    def compress(
        self,
        q21: np.ndarray,
    ) -> np.ndarray:
        """
        Extract independent coordinates from q21.

        This deliberately uses the independent parent coordinates;
        dependent child coordinates do not influence the result.
        """

        q = np.asarray(
            q21,
            dtype=np.float64,
        )

        if q.shape != (self.nq,):
            raise ValueError(
                f"Expected q21 shape {(self.nq,)}, got {q.shape}"
            )

        u = np.empty(
            self.nu,
            dtype=np.float64,
        )

        for name in self.u_names:
            u[
                self.u_index[name]
            ] = q[
                self.q_index[name]
            ]

        return np.clip(
            u,
            self.lower,
            self.upper,
        )

    def project_q21(
        self,
        q21: np.ndarray,
    ) -> np.ndarray:
        """Project an arbitrary q21 pose to the exact L20 manifold."""

        return self.expand(
            self.compress(q21)
        )

    def jacobian21_to_16(
        self,
        J21: np.ndarray,
    ) -> np.ndarray:
        """
        Chain rule:

            q21 = E u16

            J16 = J21 E
        """

        return J21 @ self.E


# ======================================================================
# Native 16DOF optimizer
# ======================================================================

class L20RetargetV2(BaseOptimizer):
    """
    L20-specific retargeter operating directly in the robot's
    16 independently controllable coordinates.

    The geometric objective uses all four landmarks of every digit:

        base/proximal
        middle
        distal
        fingertip

    and all four chain-segment directions.

    This is especially important for the thumb: its proximal geometry
    now participates explicitly in the optimization, preventing the
    solver from satisfying a fingertip-distance objective with an
    incorrect CMC branch.
    """

    # Standard 21-point hand layout.
    #
    # thumb:  1,2,3,4
    # index:  5,6,7,8
    # middle: 9,10,11,12
    # ring:   13,14,15,16
    # pinky:  17,18,19,20

    KP_LEVEL_1 = np.array(
        [1, 5, 9, 13, 17],
        dtype=np.int64,
    )

    KP_LEVEL_2 = np.array(
        [2, 6, 10, 14, 18],
        dtype=np.int64,
    )

    KP_LEVEL_3 = np.array(
        [3, 7, 11, 15, 19],
        dtype=np.int64,
    )

    KP_TIP = np.array(
        [4, 8, 12, 16, 20],
        dtype=np.int64,
    )

    def __init__(
        self,
        config: dict,
    ):
        super().__init__(config)

        rcfg = config.get(
            "retarget",
            {},
        )

        # --------------------------------------------------------------
        # True L20 16-DOF representation
        # --------------------------------------------------------------

        self.adapter = L20KinematicAdapter(
            self.robot
        )

        self.num_control_dofs = (
            self.adapter.nu
        )

        self.control_dof_names = list(
            self.adapter.u_names
        )

        # --------------------------------------------------------------
        # Loss configuration
        # --------------------------------------------------------------

        self.v2_huber_delta_cm = float(
            rcfg.get(
                "v2_huber_delta_cm",
                1.5,
            )
        )

        self.v2_w_landmark = float(
            rcfg.get(
                "v2_w_landmark",
                1.0,
            )
        )

        self.v2_w_direction = float(
            rcfg.get(
                "v2_w_direction",
                1.0,
            )
        )

        # Strong extra weight on thumb proximal geometry.
        self.v2_w_thumb_proximal = float(
            rcfg.get(
                "v2_w_thumb_proximal",
                5.0,
            )
        )

        self.v2_w_temporal = float(
            rcfg.get(
                "v2_w_temporal",
                0.03,
            )
        )

        self.v2_maxeval = int(
            rcfg.get(
                "v2_maxeval",
                45,
            )
        )

        self.v2_ftol_abs = float(
            rcfg.get(
                "v2_ftol_abs",
                1e-5,
            )
        )

        self.v2_xtol_rel = float(
            rcfg.get(
                "v2_xtol_rel",
                1e-5,
            )
        )

        self.v2_debug_every = int(
            rcfg.get(
                "v2_debug_every",
                20,
            )
        )

        # --------------------------------------------------------------
        # Compatibility with RobotMorphology.
        # --------------------------------------------------------------

        segment_scaling_config = rcfg.get(
            "segment_scaling",
            {},
        )

        finger_names = [
            "thumb",
            "index",
            "middle",
            "ring",
            "pinky",
        ]

        self.segment_scaling = np.ones(
            (5, 3),
            dtype=np.float64,
        )

        self.segment_scaling_full = np.ones(
            (5, 4),
            dtype=np.float64,
        )

        for i, finger_name in enumerate(
            finger_names
        ):
            if finger_name not in segment_scaling_config:
                continue

            scale = np.asarray(
                segment_scaling_config[
                    finger_name
                ],
                dtype=np.float64,
            )

            if scale.shape == (4,):
                self.segment_scaling_full[
                    i
                ] = scale

                self.segment_scaling[
                    i
                ] = scale[1:]

            elif scale.shape == (3,):
                self.segment_scaling_full[
                    i
                ] = np.array(
                    [
                        1.0,
                        scale[0],
                        scale[1],
                        scale[2],
                    ]
                )

                self.segment_scaling[
                    i
                ] = scale

        # --------------------------------------------------------------
        # Resolve frames.
        #
        # BaseOptimizer already resolves:
        #
        #   origin_link_name
        #   link1_names
        #   link3_names
        #   link4_names
        #   task_link_names
        #
        # --------------------------------------------------------------

        self._palm_frame = self.robot.get_link_index(
            self.origin_link_name
        )

        self._level1_frames = [
            self.robot.get_link_index(n)
            for n in self.link1_names
        ]

        self._level2_frames = [
            self.robot.get_link_index(n)
            for n in self.link3_names
        ]

        self._level3_frames = [
            self.robot.get_link_index(n)
            for n in self.link4_names
        ]

        self._tip_frames = [
            self.robot.get_link_index(n)
            for n in self.task_link_names
        ]

        self._frame_ids = (
            [self._palm_frame]
            + self._level1_frames
            + self._level2_frames
            + self._level3_frames
            + self._tip_frames
        )

        # Indices inside self._frame_ids.
        self._palm_i = 0

        self._level1_i = np.arange(
            1,
            6,
            dtype=np.int64,
        )

        self._level2_i = np.arange(
            6,
            11,
            dtype=np.int64,
        )

        self._level3_i = np.arange(
            11,
            16,
            dtype=np.int64,
        )

        self._tip_i = np.arange(
            16,
            21,
            dtype=np.int64,
        )

        # --------------------------------------------------------------
        # Landmark weights.
        #
        # rows:
        #   proximal / middle / distal / tip
        #
        # cols:
        #   thumb / index / middle / ring / pinky
        # --------------------------------------------------------------

        self._landmark_weights = np.array(
            [
                [2.5, 1.2, 1.2, 1.2, 1.2],
                [3.5, 1.5, 1.5, 1.5, 1.5],
                [2.5, 1.5, 1.5, 1.5, 1.5],
                [2.5, 2.0, 2.0, 2.0, 2.0],
            ],
            dtype=np.float64,
        )

        # Segment direction weights.
        #
        # Thumb proximal direction receives deliberately strong
        # weight because this is exactly what the previous retarget
        # objective was missing.
        self._direction_weights = np.array(
            [
                [2.0, 0.5, 0.5, 0.5, 0.5],
                [6.0, 1.0, 1.0, 1.0, 1.0],
                [3.0, 1.0, 1.0, 1.0, 1.0],
                [2.0, 1.0, 1.0, 1.0, 1.0],
            ],
            dtype=np.float64,
        )

        # --------------------------------------------------------------
        # True 16-dimensional NLopt instance.
        # --------------------------------------------------------------

        self.opt = nlopt.opt(
            nlopt.LD_SLSQP,
            self.num_control_dofs,
        )

        self.opt.set_lower_bounds(
            self.adapter.lower.tolist()
        )

        self.opt.set_upper_bounds(
            self.adapter.upper.tolist()
        )

        self.opt.set_maxeval(
            self.v2_maxeval
        )

        self.opt.set_ftol_abs(
            self.v2_ftol_abs
        )

        self.opt.set_xtol_rel(
            self.v2_xtol_rel
        )

        # --------------------------------------------------------------
        # State
        # --------------------------------------------------------------

        self.last_u16: Optional[np.ndarray] = None
        self.last_qpos: Optional[np.ndarray] = None

        self._pinch_source_keypoints = None

        self._frame_counter = 0

        self._timing = TimingStats()
        self._enable_timing = False

        # Span used for dimensionless temporal regularization.
        self._u_span = np.maximum(
            self.adapter.upper
            - self.adapter.lower,
            1e-6,
        )

        print(
            "[L20RetargetV2] native optimizer enabled: "
            "16 control DOFs -> 21 mechanical coordinates"
        )

        print(
            "[L20RetargetV2] controls:"
        )

        for i, name in enumerate(
            self.control_dof_names
        ):
            lo = np.degrees(
                self.adapter.lower[i]
            )

            hi = np.degrees(
                self.adapter.upper[i]
            )

            print(
                f"[L20RetargetV2] "
                f"  u[{i:2d}] {name:20s} "
                f"{lo:7.2f} .. {hi:7.2f} deg"
            )

        print(
            "[L20RetargetV2] thumb proximal geometry "
            "is explicitly constrained."
        )

    # ==================================================================
    # Compatibility hooks
    # ==================================================================

    def set_pinch_source_keypoints(
        self,
        keypoints: np.ndarray,
    ):
        """
        Retargeter V1 may provide the original human skeleton through
        this method.  V2 currently keeps it only for compatibility /
        future contact-state extensions.
        """

        kp = np.asarray(
            keypoints,
            dtype=np.float64,
        )

        if kp.shape == (21, 3):
            self._pinch_source_keypoints = (
                kp.copy()
            )

    def reset(self):
        self.last_u16 = None
        self.last_qpos = None
        self._frame_counter = 0

    def get_timing_stats(self):
        return self._timing

    def reset_timing_stats(self):
        self._timing = TimingStats()

    def set_timing_enabled(
        self,
        enabled: bool,
    ):
        self._enable_timing = bool(
            enabled
        )

    # ==================================================================
    # Small numerical helpers
    # ==================================================================

    def _huber_vector(
        self,
        diff: np.ndarray,
        J_diff: np.ndarray,
        weight: float,
    ):
        """
        Huber loss on ||diff|| with analytical gradient.

        diff:
            (3,)

        J_diff:
            (3, 16)
        """

        d = float(
            np.linalg.norm(diff)
        )

        delta = self.v2_huber_delta_cm

        if d < 1e-10:
            return 0.0, np.zeros(
                self.num_control_dofs,
                dtype=np.float64,
            )

        if d <= delta:
            base_loss = 0.5 * d * d
            dloss_dd = d
        else:
            base_loss = (
                delta
                * (
                    d
                    - 0.5 * delta
                )
            )

            dloss_dd = delta

        unit = diff / d

        grad = (
            weight
            * dloss_dd
            * (
                unit @ J_diff
            )
        )

        return (
            weight * base_loss,
            grad,
        )

    def _direction_loss(
        self,
        robot_vec: np.ndarray,
        target_vec: np.ndarray,
        J_vec: np.ndarray,
        weight: float,
    ):
        """
        Direction-only segment loss:

            1/2 * w * || normalize(v_r) - normalize(v_h) ||^2
        """

        nr = float(
            np.linalg.norm(robot_vec)
        )

        nt = float(
            np.linalg.norm(target_vec)
        )

        if nr < 1e-8 or nt < 1e-8:
            return (
                0.0,
                np.zeros(
                    self.num_control_dofs,
                    dtype=np.float64,
                ),
            )

        ur = robot_vec / nr
        ut = target_vec / nt

        diff = ur - ut

        loss = (
            0.5
            * weight
            * float(
                diff @ diff
            )
        )

        J_norm = (
            np.eye(3)
            - np.outer(ur, ur)
        ) / nr

        grad = (
            weight
            * (
                diff
                @ J_norm
                @ J_vec
            )
        )

        return loss, grad

    # ==================================================================
    # FK / Jacobian
    # ==================================================================

    def _fk_with_u_jacobian(
        self,
        u16: np.ndarray,
    ):
        """
        Compute:

            u16
             |
             v
            q21
             |
             +--> FK positions
             |
             +--> J21
                    |
                    v
                  J16 = J21 E
        """

        q21 = self.adapter.expand(
            u16
        )

        self.robot.compute_forward_kinematics(
            q21
        )

        positions = np.array(
            [
                self.robot.get_link_pose(
                    frame_id
                )[:3, 3]
                for frame_id in self._frame_ids
            ],
            dtype=np.float64,
        ) * M_TO_CM

        J21 = (
            self.robot.compute_all_jacobians_batch(
                q21,
                self._frame_ids,
            )
            * M_TO_CM
        )

        # (frames, 3, 21) @ (21, 16)
        J16 = np.einsum(
            "fij,jk->fik",
            J21,
            self.adapter.E,
        )

        return (
            q21,
            positions,
            J16,
        )

    # ==================================================================
    # Target construction
    # ==================================================================

    def _target_levels(
        self,
        keypoints: np.ndarray,
    ):
        kp = np.asarray(
            keypoints,
            dtype=np.float64,
        )

        if kp.shape != (21, 3):
            raise ValueError(
                "L20RetargetV2 expects keypoints shape "
                f"(21,3), got {kp.shape}"
            )

        if not np.all(
            np.isfinite(kp)
        ):
            raise ValueError(
                "L20RetargetV2 received NaN/Inf keypoints"
            )

        # Everything is represented relative to the human wrist.
        rel = (
            kp
            - kp[0]
        ) * M_TO_CM

        levels = [
            rel[
                self.KP_LEVEL_1
            ],
            rel[
                self.KP_LEVEL_2
            ],
            rel[
                self.KP_LEVEL_3
            ],
            rel[
                self.KP_TIP
            ],
        ]

        return levels

    # ==================================================================
    # Objective
    # ==================================================================

    def _loss_and_grad(
        self,
        u16: np.ndarray,
        keypoints: np.ndarray,
        reference_u16: Optional[np.ndarray],
    ):
        (
            q21,
            P,
            J,
        ) = self._fk_with_u_jacobian(
            u16
        )

        del q21

        palm_P = P[
            self._palm_i
        ]

        palm_J = J[
            self._palm_i
        ]

        robot_levels_P = [
            P[
                self._level1_i
            ],
            P[
                self._level2_i
            ],
            P[
                self._level3_i
            ],
            P[
                self._tip_i
            ],
        ]

        robot_levels_J = [
            J[
                self._level1_i
            ],
            J[
                self._level2_i
            ],
            J[
                self._level3_i
            ],
            J[
                self._tip_i
            ],
        ]

        target_levels = self._target_levels(
            keypoints
        )

        total_loss = 0.0

        total_grad = np.zeros(
            self.num_control_dofs,
            dtype=np.float64,
        )

        # --------------------------------------------------------------
        # 1. Full-chain landmark vectors relative to palm/wrist.
        #
        # Unlike V1 this includes the proximal landmark for every digit,
        # especially the thumb.
        # --------------------------------------------------------------

        for level in range(4):

            RP = (
                robot_levels_P[level]
                - palm_P
            )

            RJ = (
                robot_levels_J[level]
                - palm_J[None, :, :]
            )

            TP = target_levels[level]

            for finger in range(5):

                weight = (
                    self.v2_w_landmark
                    * self._landmark_weights[
                        level,
                        finger,
                    ]
                )

                # Extra thumb proximal emphasis.
                if (
                    finger == 0
                    and level <= 1
                ):
                    weight *= (
                        self.v2_w_thumb_proximal
                    )

                diff = (
                    RP[finger]
                    - TP[finger]
                )

                loss_i, grad_i = (
                    self._huber_vector(
                        diff,
                        RJ[finger],
                        weight,
                    )
                )

                total_loss += loss_i
                total_grad += grad_i

        # --------------------------------------------------------------
        # 2. Segment-direction loss.
        #
        # Human chain:
        #
        # wrist -> L1 -> L2 -> L3 -> TIP
        #
        # Robot chain:
        #
        # palm  -> L1 -> L2 -> L3 -> TIP
        #
        # The thumb's first two directions receive much larger weight.
        # This explicitly constrains CMC/opposition geometry instead of
        # only asking the thumb tip to arrive at some location.
        # --------------------------------------------------------------

        robot_nodes_P = [
            np.repeat(
                palm_P[None, :],
                5,
                axis=0,
            ),
            robot_levels_P[0],
            robot_levels_P[1],
            robot_levels_P[2],
            robot_levels_P[3],
        ]

        robot_nodes_J = [
            np.repeat(
                palm_J[None, :, :],
                5,
                axis=0,
            ),
            robot_levels_J[0],
            robot_levels_J[1],
            robot_levels_J[2],
            robot_levels_J[3],
        ]

        zero_target = np.zeros(
            (5, 3),
            dtype=np.float64,
        )

        target_nodes = [
            zero_target,
            target_levels[0],
            target_levels[1],
            target_levels[2],
            target_levels[3],
        ]

        for segment in range(4):

            robot_vec = (
                robot_nodes_P[segment + 1]
                - robot_nodes_P[segment]
            )

            target_vec = (
                target_nodes[segment + 1]
                - target_nodes[segment]
            )

            J_vec = (
                robot_nodes_J[segment + 1]
                - robot_nodes_J[segment]
            )

            for finger in range(5):

                weight = (
                    self.v2_w_direction
                    * self._direction_weights[
                        segment,
                        finger,
                    ]
                )

                loss_i, grad_i = (
                    self._direction_loss(
                        robot_vec[finger],
                        target_vec[finger],
                        J_vec[finger],
                        weight,
                    )
                )

                total_loss += loss_i
                total_grad += grad_i

        # --------------------------------------------------------------
        # 3. Small temporal regularization in TRUE independent space.
        #
        # This does not define a thumb branch.  It only damps frame-to-
        # frame noise and is normalized by each actuator's usable range.
        # --------------------------------------------------------------

        if (
            reference_u16 is not None
            and self.v2_w_temporal > 0.0
        ):
            du_norm = (
                (
                    u16
                    - reference_u16
                )
                / self._u_span
            )

            total_loss += (
                0.5
                * self.v2_w_temporal
                * float(
                    du_norm
                    @ du_norm
                )
            )

            total_grad += (
                self.v2_w_temporal
                * du_norm
                / self._u_span
            )

        return (
            float(total_loss),
            total_grad,
        )

    # ==================================================================
    # Solve
    # ==================================================================

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
                f"Expected keypoints (21,3), got {kp.shape}"
            )

        # --------------------------------------------------------------
        # Warm start
        # --------------------------------------------------------------

        reference_u = None

        if last_qpos is not None:

            last_q = np.asarray(
                last_qpos,
                dtype=np.float64,
            )

            if last_q.shape == (
                self.adapter.nq,
            ):
                reference_u = (
                    self.adapter.compress(
                        last_q
                    )
                )

        if reference_u is None:
            reference_u = (
                None
                if self.last_u16 is None
                else self.last_u16.copy()
            )

        if reference_u is None:

            # Natural zero pose, clipped to feasible independent range.
            x0 = np.clip(
                np.zeros(
                    self.num_control_dofs,
                    dtype=np.float64,
                ),
                self.adapter.lower,
                self.adapter.upper,
            )

        else:
            x0 = np.clip(
                reference_u.copy(),
                self.adapter.lower,
                self.adapter.upper,
            )

        # Freeze target/reference for all objective evaluations of
        # this frame.
        target_kp = kp.copy()

        reference_for_objective = (
            None
            if reference_u is None
            else reference_u.copy()
        )

        def objective(
            x,
            grad,
        ):
            loss, g = self._loss_and_grad(
                np.asarray(
                    x,
                    dtype=np.float64,
                ),
                target_kp,
                reference_for_objective,
            )

            if grad.size > 0:
                grad[:] = g

            return loss

        self.opt.set_min_objective(
            objective
        )

        # --------------------------------------------------------------
        # Optimize in 16D.
        # --------------------------------------------------------------

        try:
            result_u = np.asarray(
                self.opt.optimize(
                    x0
                ),
                dtype=np.float64,
            )

        except Exception as exc:

            print(
                "[L20RetargetV2] NLopt warning:",
                repr(exc),
            )

            result_u = x0.copy()

        result_u = np.clip(
            result_u,
            self.adapter.lower,
            self.adapter.upper,
        )

        result_q = self.adapter.expand(
            result_u
        )

        self.last_u16 = (
            result_u.copy()
        )

        self.last_qpos = (
            result_q.copy()
        )

        # --------------------------------------------------------------
        # Debug
        # --------------------------------------------------------------

        self._frame_counter += 1

        if (
            self.v2_debug_every > 0
            and self._frame_counter
            % self.v2_debug_every
            == 0
        ):

            ui = self.adapter.u_index

            print(
                "[L20 V2] "
                f"thumb="
                f"[roll "
                f"{np.degrees(result_u[ui['thumb_cmc_roll']]):.1f}, "
                f"yaw "
                f"{np.degrees(result_u[ui['thumb_cmc_yaw']]):.1f}, "
                f"pitch "
                f"{np.degrees(result_u[ui['thumb_cmc_pitch']]):.1f}, "
                f"flex "
                f"{np.degrees(result_u[ui['thumb_mcp']]):.1f}]"
            )

            print(
                "[L20 V2] "
                "finger flex="
                f"I({np.degrees(result_u[ui['index_mcp_pitch']]):.1f},"
                f"{np.degrees(result_u[ui['index_pip']]):.1f}) "
                f"M({np.degrees(result_u[ui['middle_mcp_pitch']]):.1f},"
                f"{np.degrees(result_u[ui['middle_pip']]):.1f}) "
                f"R({np.degrees(result_u[ui['ring_mcp_pitch']]):.1f},"
                f"{np.degrees(result_u[ui['ring_pip']]):.1f}) "
                f"P({np.degrees(result_u[ui['pinky_mcp_pitch']]):.1f},"
                f"{np.degrees(result_u[ui['pinky_pip']]):.1f})"
            )

        return result_q

    # ==================================================================
    # Diagnostics
    # ==================================================================

    def compute_cost(
        self,
        qpos: np.ndarray,
        mediapipe_keypoints: np.ndarray,
    ) -> float:

        q = self.adapter.project_q21(
            np.asarray(
                qpos,
                dtype=np.float64,
            )
        )

        u = self.adapter.compress(
            q
        )

        loss, _ = self._loss_and_grad(
            u,
            np.asarray(
                mediapipe_keypoints,
                dtype=np.float64,
            ),
            None,
        )

        return float(loss)
