"""Adaptive optimizer with analytical gradients for hand retargeting."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np

from .base import BaseOptimizer, M_TO_CM, TimingStats, huber_loss_np, huber_loss_grad_np
from ..robot import RobotWrapper


class AdaptiveOptimizerAnalytical(BaseOptimizer):
    """Adaptive optimizer with analytical (hand-written) gradients.

    Same loss function as AdaptiveOptimizer but uses hand-written gradients
    instead of autograd for faster performance.
    """

    def __init__(self, config: dict):
        """Initialize AdaptiveOptimizerAnalytical."""
        super().__init__(config)

        # Initialize timing stats
        self._timing = TimingStats()
        self._enable_timing = True

        retarget_config = config.get('retarget', {})

        # TipDirVec parameters
        self.huber_delta_dir = retarget_config.get('huber_delta_dir', 0.5)
        self.w_pos = retarget_config.get('w_pos', 1.0)
        self.w_dir = retarget_config.get('w_dir', 10.0)
        self.scaling = retarget_config.get('scaling', 1.0)
        self.project_tip_dir = retarget_config.get('project_tip_dir', False)

        # FullHandVec parameters
        self.w_full_hand = retarget_config.get('w_full_hand', 1.0)
        segment_scaling_config = retarget_config.get('segment_scaling', {})
        finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
        # For optimization: (5, 3) - PIP, DIP, TIP only
        self.segment_scaling = np.ones((5, 3), dtype=np.float64)
        # For visualization: (5, 4) - MCP, PIP, DIP, TIP (full version)
        self.segment_scaling_full = np.ones((5, 4), dtype=np.float64)
        for i, finger_name in enumerate(finger_names):
            if finger_name in segment_scaling_config:
                scales = np.array(segment_scaling_config[finger_name])
                if len(scales) == 4:
                    # 4-param format: [MCP, PIP, DIP, TIP]
                    self.segment_scaling_full[i] = scales
                    self.segment_scaling[i] = scales[1:4]  # PIP, DIP, TIP for optimization
                elif len(scales) == 3:
                    # 3-param format: [PIP, DIP, TIP] - assume MCP scale = 1.0
                    self.segment_scaling_full[i] = np.array([1.0, scales[0], scales[1], scales[2]])
                    self.segment_scaling[i] = scales

        # Pinch thresholds
        pinch_config = retarget_config.get('pinch_thresholds', {})
        self.d1 = np.array([
            pinch_config.get('index', {}).get('d1', 2.0),
            pinch_config.get('middle', {}).get('d1', 2.0),
            pinch_config.get('ring', {}).get('d1', 2.0),
            pinch_config.get('pinky', {}).get('d1', 2.0),
        ], dtype=np.float64)
        self.d2 = np.array([
            pinch_config.get('index', {}).get('d2', 4.0),
            pinch_config.get('middle', {}).get('d2', 4.0),
            pinch_config.get('ring', {}).get('d2', 4.0),
            pinch_config.get('pinky', {}).get('d2', 4.0),
        ], dtype=np.float64)

        # link1 (finger-plane) names come from BaseOptimizer._resolve_link_names,
        # so a custom optimizer.link_naming applies to them too.
        all_link_names = (
            [self.origin_link_name] +
            self.task_link_names +
            self.link3_names +
            self.link4_names +
            self.link1_names
        )
        self.computed_link_names = list(dict.fromkeys(all_link_names))
        self.computed_link_indices = [
            self.robot.get_link_index(name) for name in self.computed_link_names
        ]
        self.link1_indices = [
            self.computed_link_names.index(name) for name in self.link1_names
        ]

        # Optional extensions (default: inactive, equivalent to the base loss).
        # Thumb-specific: drop wrist->PIP loss term from FullHand on thumb (use DIP+TIP only).
        self.thumb_skip_pip = retarget_config.get('thumb_skip_pip', False)
        # Hyperextension soft constraint on PIP/DIP joints.
        self.w_hyper = retarget_config.get('w_hyper', 0.0)
        self.soft_min = retarget_config.get('soft_min', 0.0)
        # DIP<->PIP biomechanical coupling soft constraint.
        self.w_couple = retarget_config.get('w_couple', 0.0)
        self.couple_ratio = retarget_config.get('couple_ratio', 0.7)

        # Explicit fingertip-to-fingertip pinch distance constraint.
        # Active only when pinch alpha > 0.
        self.w_pinch = retarget_config.get('w_pinch', 8.0)

        # Pinch hysteresis:
        # enter follows d2; release occurs later at d2 + margin.
        self.pinch_release_margin = retarget_config.get(
            'pinch_release_margin',
            1.5,
        )

        # Small residual alpha while a contact is latched but the
        # fingertip distance has temporarily moved outside d2.
        self.pinch_hold_alpha = retarget_config.get(
            'pinch_hold_alpha',
            0.15,
        )

        # Robot fingertip centre-distance target at full contact.
        self.pinch_contact_target_cm = retarget_config.get(
            'pinch_contact_target_cm',
            0.5,
        )

        # --------------------------------------------------------
        # Thumb branch continuity
        #
        # During a deep grasp the optimizer may otherwise jump to
        # another CMC IK branch (typically a large negative
        # thumb_cmc_roll), making the thumb open outward even though
        # the fingers are closing.
        #
        # These weights are activated smoothly according to the
        # PREVIOUS frame's four-finger flexion, so the objective
        # keeps a clean analytical gradient.
        # --------------------------------------------------------

        self.w_thumb_branch_roll = retarget_config.get(
            'w_thumb_branch_roll',
            8.0,
        )

        self.w_thumb_branch_yaw = retarget_config.get(
            'w_thumb_branch_yaw',
            2.0,
        )

        self.thumb_branch_grip_on = retarget_config.get(
            'thumb_branch_grip_on',
            0.45,
        )

        self.thumb_branch_grip_full = retarget_config.get(
            'thumb_branch_grip_full',
            0.75,
        )


        # --------------------------------------------------------
        # Persistent thumb-roll branch lock.
        #
        # Unlike previous-frame continuity, this freezes a stable
        # roll reference when a multi-finger grasp begins, so the
        # optimizer cannot slowly drift toward the wrong CMC branch
        # over many frames.
        # --------------------------------------------------------

        self.thumb_roll_lock_on = retarget_config.get(
            'thumb_roll_lock_on',
            0.15,
        )

        self.thumb_roll_lock_off = retarget_config.get(
            'thumb_roll_lock_off',
            0.08,
        )

        self.thumb_roll_window_deg = retarget_config.get(
            'thumb_roll_window_deg',
            12.0,
        )

        self._thumb_roll_lock_active = False
        self._thumb_roll_anchor = None
        self._thumb_roll_lock_grip = 0.0

        # --------------------------------------------------------
        # Persistent grasp/contact model
        #
        # Pinch distance alone is insufficient for a closed fist:
        # during a deep grasp the thumb-to-fingertip Euclidean
        # distance may increase again even though the thumb should
        # remain opposed to the fingers.
        #
        # Therefore explicit thumb contact uses:
        #
        #   1) ordinary pinch alpha
        #   2) relative closure from an automatically learned
        #      open-hand thumb-to-finger distance
        #   3) one persistent active finger with switching hysteresis
        # --------------------------------------------------------

        self.grasp_contact_grip_on = retarget_config.get(
            'grasp_contact_grip_on',
            0.30,
        )

        self.grasp_contact_grip_full = retarget_config.get(
            'grasp_contact_grip_full',
            0.65,
        )

        self.grasp_contact_strength = retarget_config.get(
            'grasp_contact_strength',
            0.80,
        )

        self.grasp_contact_switch_margin = retarget_config.get(
            'grasp_contact_switch_margin',
            0.20,
        )

        self.human_open_update_alpha = retarget_config.get(
            'human_open_update_alpha',
            0.05,
        )


        # --------------------------------------------------------
        # Deep-grasp thumb CMC feasible region
        #
        # This is NOT a hardware joint limit.
        #
        # It is a grasp-state-dependent behavioral constraint used
        # only to eliminate the outward CMC branch observed during
        # a nearly closed fist.
        #
        # Moderate grasp / pinch remains completely unconstrained.
        # --------------------------------------------------------

        self.deep_grasp_cmc_on = retarget_config.get(
            'deep_grasp_cmc_on',
            0.85,
        )

        self.deep_grasp_cmc_full = retarget_config.get(
            'deep_grasp_cmc_full',
            0.95,
        )

        self.deep_grasp_roll_floor_deg = retarget_config.get(
            'deep_grasp_roll_floor_deg',
            -20.0,
        )

        self.deep_grasp_yaw_floor_deg = retarget_config.get(
            'deep_grasp_yaw_floor_deg',
            82.0,
        )

        self._deep_grasp_cmc_gate = 0.0
        self._deep_grasp_roll_lower = None
        self._deep_grasp_yaw_lower = None

        self._human_open_pinch_distances = None
        self._explicit_pinch_active_idx = -1
        self._explicit_pinch_alpha_4 = np.zeros(
            4,
            dtype=np.float64,
        )

        self._current_human_grip = 0.0
        self._current_relative_contact = np.zeros(
            4,
            dtype=np.float64,
        )
        self._pinch_latched = np.zeros(
            4,
            dtype=bool,
        )

        # The optimizer.urdf_path override (e.g. a Wuji Hand 2 model) is loaded up front
        # by BaseOptimizer, so self.robot is already the final hand here.

        # Resolve PIP/DIP qpos indices from the (possibly overridden) URDF.
        self._resolve_flex_indices()
        self._init_robot_pinch_reference()

        # Constrain the 21 URDF joint coordinates to the
        # 16 independently controllable L20 hardware DOFs.
        self._install_l20_hardware_constraints()
        self._init_thumb_branch_indices()


    def _install_l20_hardware_constraints(self):
        """
        Constrain the 21-joint L20 URDF representation to the
        16 independently controllable hardware coordinates.

        The five dependent URDF joints are:

            index_dip  = 0.80790960 * index_pip
            middle_dip = 0.80790960 * middle_pip
            ring_dip   = 0.80790960 * ring_pip
            pinky_dip  = 0.80790960 * pinky_pip

            thumb_ip   = 0.8079 * thumb_mcp

        NLopt still sees a 21-dimensional q vector, but five
        independent equality constraints reduce the feasible
        manifold to exactly 16 DOFs.

        This keeps the rest of the retargeting/FK code unchanged.
        """

        names = list(
            self.robot.dof_joint_names
        )

        idx = {
            name: i
            for i, name in enumerate(names)
        }

        coupling_names = [
            (
                "index_dip",
                "index_pip",
                0.80790960,
            ),
            (
                "middle_dip",
                "middle_pip",
                0.80790960,
            ),
            (
                "ring_dip",
                "ring_pip",
                0.80790960,
            ),
            (
                "pinky_dip",
                "pinky_pip",
                0.80790960,
            ),
            (
                "thumb_ip",
                "thumb_mcp",
                0.8079,
            ),
        ]

        required = {
            name
            for child, parent, _ in coupling_names
            for name in (child, parent)
        }

        missing = sorted(
            required - set(names)
        )

        if missing:
            raise RuntimeError(
                "L20 hardware constraints cannot be installed; "
                f"missing joints: {missing}"
            )

        self._l20_hw_couplings = [
            (
                idx[child],
                idx[parent],
                ratio,
                child,
                parent,
            )
            for child, parent, ratio
            in coupling_names
        ]

        # ----------------------------------------------------
        # Strict L20 hardware-feasible bounds
        #
        # A feasible pose must satisfy BOTH:
        #
        #   child = ratio * parent
        #
        # and the URDF limits of parent and child.
        #
        # Example:
        #
        #   index_dip = 0.80790960 * index_pip
        #
        # Therefore:
        #
        #   index_pip <= index_dip_max / 0.80790960
        #
        # This keeps every optimized pose inside the actual
        # 16-DOF manifold represented by the current L20 URDF.
        # ----------------------------------------------------

        limits = np.asarray(
            self.robot.joint_limits,
            dtype=np.float64,
        )

        lower = limits[:, 0].copy()
        upper = limits[:, 1].copy()

        for (
            child_i,
            parent_i,
            ratio,
            child_name,
            parent_name,
        ) in self._l20_hw_couplings:

            if ratio <= 0.0:
                raise RuntimeError(
                    "Current L20 implementation expects "
                    "positive mimic multipliers, got "
                    f"{child_name}={ratio}*{parent_name}"
                )

            # child = ratio * parent
            #
            # Child joint limits induce an additional valid
            # interval on the independently controlled parent.
            parent_lo_from_child = (
                lower[child_i] / ratio
            )

            parent_hi_from_child = (
                upper[child_i] / ratio
            )

            lower[parent_i] = max(
                lower[parent_i],
                parent_lo_from_child,
            )

            upper[parent_i] = min(
                upper[parent_i],
                parent_hi_from_child,
            )

            if lower[parent_i] > upper[parent_i]:
                raise RuntimeError(
                    "No feasible range for "
                    f"{child_name}={ratio}*{parent_name}"
                )

        self._l20_hw_lower = lower
        self._l20_hw_upper = upper

        self.opt.set_lower_bounds(
            lower.tolist()
        )

        self.opt.set_upper_bounds(
            upper.tolist()
        )

        # Keep callbacks alive explicitly.
        self._l20_constraint_callbacks = []

        for (
            child_i,
            parent_i,
            ratio,
            child_name,
            parent_name,
        ) in self._l20_hw_couplings:

            def constraint(
                x,
                grad,
                ci=child_i,
                pi=parent_i,
                r=ratio,
            ):
                if grad.size > 0:
                    grad[:] = 0.0
                    grad[ci] = 1.0
                    grad[pi] = -r

                return float(
                    x[ci]
                    - r * x[pi]
                )

            self.opt.add_equality_constraint(
                constraint,
                1e-8,
            )

            self._l20_constraint_callbacks.append(
                constraint
            )

        print(
            "[AdaptiveOptimizer] "
            "L20 hardware manifold enabled: "
            "21 URDF joints -> 16 controllable DOFs"
        )

        for (
            child_i,
            parent_i,
            ratio,
            child_name,
            parent_name,
        ) in self._l20_hw_couplings:

            print(
                "[AdaptiveOptimizer]   "
                f"{child_name} = "
                f"{ratio:.4f} * "
                f"{parent_name}"
            )

        print(
            "[AdaptiveOptimizer] "
            "effective independent flex ranges:"
        )

        for parent_name in [
            "index_pip",
            "middle_pip",
            "ring_pip",
            "pinky_pip",
            "thumb_mcp",
        ]:
            i = idx[parent_name]

            print(
                "[AdaptiveOptimizer]   "
                f"{parent_name}: "
                f"{np.degrees(lower[i]):.2f} deg "
                f".. "
                f"{np.degrees(upper[i]):.2f} deg"
            )


    def _project_l20_hardware_constraints(
        self,
        qpos: np.ndarray,
    ) -> np.ndarray:
        """
        Numerically project qpos onto the same L20 hardware
        manifold used by the NLopt equality constraints.

        Used for initialization, regularization references and
        final cleanup of numerical optimizer tolerance.
        """

        q = np.asarray(
            qpos,
            dtype=np.float64,
        ).copy()

        if q.shape != (self.num_joints,):
            raise ValueError(
                "Expected qpos shape "
                f"({self.num_joints},), "
                f"got {q.shape}"
            )

        q = np.clip(
            q,
            self._l20_hw_lower,
            self._l20_hw_upper,
        )

        for (
            child_i,
            parent_i,
            ratio,
            _,
            _,
        ) in self._l20_hw_couplings:

            # Parent has already been restricted so the dependent
            # joint automatically remains inside its URDF limit.
            q[child_i] = (
                ratio * q[parent_i]
            )

        return q



    def _init_thumb_branch_indices(self):
        """Resolve L20 thumb CMC and four-finger flex qpos indices."""

        names = list(
            self.robot.dof_joint_names
        )

        idx = {
            name: i
            for i, name in enumerate(names)
        }

        required = [
            "thumb_cmc_roll",
            "thumb_cmc_yaw",

            "index_mcp_pitch",
            "index_pip",

            "middle_mcp_pitch",
            "middle_pip",

            "ring_mcp_pitch",
            "ring_pip",

            "pinky_mcp_pitch",
            "pinky_pip",
        ]

        missing = [
            n for n in required
            if n not in idx
        ]

        if missing:
            raise RuntimeError(
                "Cannot initialize L20 thumb branch indices; "
                f"missing joints: {missing}"
            )

        self._thumb_cmc_roll_idx = idx[
            "thumb_cmc_roll"
        ]

        self._thumb_cmc_yaw_idx = idx[
            "thumb_cmc_yaw"
        ]

        # Use both MCP pitch and the independently controlled
        # distal-flex coordinate of every non-thumb finger.
        self._grip_qpos_idx = np.array([
            idx["index_mcp_pitch"],
            idx["index_pip"],

            idx["middle_mcp_pitch"],
            idx["middle_pip"],

            idx["ring_mcp_pitch"],
            idx["ring_pip"],

            idx["pinky_mcp_pitch"],
            idx["pinky_pip"],
        ], dtype=np.int64)

        self._grip_lower = np.asarray(
            self._l20_hw_lower[
                self._grip_qpos_idx
            ],
            dtype=np.float64,
        )

        self._grip_upper = np.asarray(
            self._l20_hw_upper[
                self._grip_qpos_idx
            ],
            dtype=np.float64,
        )


    def _compute_l20_grip(self, qpos):
        """
        Return normalized four-finger grasp amount in [0, 1].

        0 = extended
        1 = deeply flexed

        This deliberately excludes the thumb, because the thumb
        branch penalty must be driven by the state of the other
        four fingers.
        """

        q = np.asarray(
            qpos,
            dtype=np.float64,
        )

        values = q[
            self._grip_qpos_idx
        ]

        normalized = (
            values - self._grip_lower
        ) / (
            self._grip_upper
            - self._grip_lower
            + 1e-8
        )

        normalized = np.clip(
            normalized,
            0.0,
            1.0,
        )

        return float(
            np.mean(normalized)
        )


    def _thumb_branch_gate(self, grip):
        """
        Smoothly activate branch continuity:

            grip <= grip_on    -> 0
            grip >= grip_full  -> 1
        """

        denom = (
            self.thumb_branch_grip_full
            - self.thumb_branch_grip_on
        )

        t = np.clip(
            (
                grip
                - self.thumb_branch_grip_on
            )
            / (
                denom + 1e-8
            ),
            0.0,
            1.0,
        )

        # Smoothstep.
        return float(
            t * t * (3.0 - 2.0 * t)
        )



    def _prepare_thumb_roll_lock(
        self,
        init_qpos: np.ndarray,
        reference_qpos: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Configure per-frame NLopt bounds for thumb_cmc_roll.

        reference_qpos is the PREVIOUS hardware-feasible solution.

        State machine:

            grip < lock_on:
                thumb is free;
                continuously remember the current healthy roll.

            grip >= lock_on:
                freeze that remembered roll as the branch anchor.

            while locked:
                thumb_cmc_roll is constrained to
                anchor +/- thumb_roll_window_deg.

            grip <= lock_off:
                release the lock.

        This prevents accumulated per-frame drift toward the
        -45.8 degree CMC-roll boundary during a sustained fist.
        """

        q_init = np.asarray(
            init_qpos,
            dtype=np.float64,
        ).copy()

        lower = np.asarray(
            self._l20_hw_lower,
            dtype=np.float64,
        ).copy()

        upper = np.asarray(
            self._l20_hw_upper,
            dtype=np.float64,
        ).copy()

        roll_i = self._thumb_cmc_roll_idx

        # No previous solution yet:
        # do not lock on the artificial optimizer initial pose.
        if reference_qpos is None:

            self._thumb_roll_lock_active = False
            self._thumb_roll_anchor = None
            self._thumb_roll_lock_grip = 0.0

            self.opt.set_lower_bounds(
                lower.tolist()
            )
            self.opt.set_upper_bounds(
                upper.tolist()
            )

            return np.clip(
                q_init,
                lower,
                upper,
            )

        q_ref = np.asarray(
            reference_qpos,
            dtype=np.float64,
        )

        grip = self._compute_l20_grip(
            q_ref
        )

        self._thumb_roll_lock_grip = grip

        # ----------------------------------------------------
        # Unlocked state
        # ----------------------------------------------------

        if not self._thumb_roll_lock_active:

            # While the hand is sufficiently open, continuously
            # track the latest valid thumb branch.
            if grip < self.thumb_roll_lock_on:

                self._thumb_roll_anchor = float(
                    q_ref[roll_i]
                )

            else:
                # IMPORTANT:
                #
                # Do NOT overwrite the anchor here.
                #
                # The stored value comes from the last frame below
                # lock_on, i.e. immediately before the grasp entered
                # the branch-sensitive region.
                if self._thumb_roll_anchor is None:
                    self._thumb_roll_anchor = float(
                        q_ref[roll_i]
                    )

                self._thumb_roll_lock_active = True

        # ----------------------------------------------------
        # Locked state
        # ----------------------------------------------------

        else:
            if grip <= self.thumb_roll_lock_off:

                self._thumb_roll_lock_active = False

                self._thumb_roll_anchor = float(
                    q_ref[roll_i]
                )

        # ----------------------------------------------------
        # Dynamic roll bounds
        # ----------------------------------------------------

        if (
            self._thumb_roll_lock_active
            and self._thumb_roll_anchor is not None
        ):

            half_window = np.deg2rad(
                self.thumb_roll_window_deg
            )

            lock_lo = (
                self._thumb_roll_anchor
                - half_window
            )

            lock_hi = (
                self._thumb_roll_anchor
                + half_window
            )

            lower[roll_i] = max(
                lower[roll_i],
                lock_lo,
            )

            upper[roll_i] = min(
                upper[roll_i],
                lock_hi,
            )

            if lower[roll_i] > upper[roll_i]:
                raise RuntimeError(
                    "Invalid thumb roll lock interval: "
                    f"anchor={np.degrees(self._thumb_roll_anchor):.2f} deg"
                )

        # Save current dynamic bounds for diagnostics.
        self._thumb_roll_dynamic_lower = lower
        self._thumb_roll_dynamic_upper = upper

        # NLopt supports changing bounds between solves.
        self.opt.set_lower_bounds(
            lower.tolist()
        )

        self.opt.set_upper_bounds(
            upper.tolist()
        )

        # Warm start must also satisfy the current dynamic bound.
        q_init = np.clip(
            q_init,
            lower,
            upper,
        )

        return q_init



    def _apply_deep_grasp_cmc_bounds(
        self,
        init_qpos: np.ndarray,
    ) -> np.ndarray:
        """
        Progressively narrow thumb CMC roll/yaw only during a
        DEEP human grasp.

        HG <= on:
            original full L20 CMC ranges

        HG >= full:
            roll >= configured roll floor
            yaw  >= configured yaw floor

        Between on/full the lower bounds are smoothly interpolated.

        This removes the outward CMC branch without using an
        initial-pose anchor or constraining normal pinch motion.
        """

        q_init = np.asarray(
            init_qpos,
            dtype=np.float64,
        ).copy()

        lower = np.asarray(
            self._l20_hw_lower,
            dtype=np.float64,
        ).copy()

        upper = np.asarray(
            self._l20_hw_upper,
            dtype=np.float64,
        ).copy()

        hg = float(
            getattr(
                self,
                "_current_human_grip",
                0.0,
            )
        )

        t = (
            hg
            - self.deep_grasp_cmc_on
        ) / (
            self.deep_grasp_cmc_full
            - self.deep_grasp_cmc_on
            + 1e-8
        )

        gate = self._smoothstep01(t)

        roll_i = self._thumb_cmc_roll_idx
        yaw_i = self._thumb_cmc_yaw_idx

        roll_floor = np.deg2rad(
            self.deep_grasp_roll_floor_deg
        )

        yaw_floor = np.deg2rad(
            self.deep_grasp_yaw_floor_deg
        )

        # Smoothly interpolate from the ORIGINAL L20 lower bound
        # toward the deep-grasp behavioral lower bound.
        roll_lower = (
            (1.0 - gate)
            * lower[roll_i]
            +
            gate
            * max(
                lower[roll_i],
                roll_floor,
            )
        )

        yaw_lower = (
            (1.0 - gate)
            * lower[yaw_i]
            +
            gate
            * max(
                lower[yaw_i],
                yaw_floor,
            )
        )

        lower[roll_i] = roll_lower
        lower[yaw_i] = yaw_lower

        if lower[roll_i] > upper[roll_i]:
            raise RuntimeError(
                "deep-grasp roll bound invalid"
            )

        if lower[yaw_i] > upper[yaw_i]:
            raise RuntimeError(
                "deep-grasp yaw bound invalid"
            )

        self._deep_grasp_cmc_gate = gate
        self._deep_grasp_roll_lower = roll_lower
        self._deep_grasp_yaw_lower = yaw_lower

        # IMPORTANT:
        # reset ALL bounds here so there is no residual per-frame
        # bound from any previous state.
        self.opt.set_lower_bounds(
            lower.tolist()
        )

        self.opt.set_upper_bounds(
            upper.tolist()
        )

        # Warm start must satisfy the same frame-level bounds.
        return np.clip(
            q_init,
            lower,
            upper,
        )


    def _init_robot_pinch_reference(self):
        """Compute robot-native thumb-to-finger distances at q=0."""

        q0 = np.zeros(
            self.num_joints,
            dtype=np.float64,
        )

        limits = np.asarray(
            self.robot.joint_limits,
            dtype=np.float64,
        )

        q0 = np.clip(
            q0,
            limits[:, 0],
            limits[:, 1],
        )

        self.robot.compute_forward_kinematics(
            q0
        )

        positions = np.array([
            self.robot.get_link_pose(idx)[:3, 3]
            for idx in self.computed_link_indices
        ], dtype=np.float64) * M_TO_CM

        task_pos = positions[
            self.task_indices
        ]

        thumb = task_pos[0]
        fingers = task_pos[1:]

        self._robot_open_pinch_distances = (
            np.linalg.norm(
                fingers - thumb,
                axis=1,
            )
        )

        print(
            "[AdaptiveOptimizer] robot open pinch distances "
            "(TI/TM/TR/TP cm):",
            np.round(
                self._robot_open_pinch_distances,
                2,
            ),
        )

    def _resolve_flex_indices(self):
        """Resolve PIP/DIP qpos indices dynamically from the URDF kinematic chain.

        Each finger's PIP/DIP joint is mapped to its qpos slot via the PIP/DIP
        link's parent joint, so the mapping follows the actual URDF instead of a
        fixed index layout. This keeps the hyperextension (``w_hyper``) and
        DIP<->PIP coupling (``w_couple``) soft constraints on the correct qpos
        dimensions even when a custom URDF (``optimizer.urdf_path``, e.g. Wuji Hand 2)
        declares joints in a different order or with a non-uniform DOF layout. A
        missing finger link or a duplicate resolved index raises immediately at
        load time instead of corrupting the optimization silently.
        """
        pip_idx = [self.robot.get_actuated_qpos_index(n) for n in self.link3_names]
        dip_idx = [self.robot.get_actuated_qpos_index(n) for n in self.link4_names]

        combined = pip_idx + dip_idx
        if len(set(combined)) != len(combined):
            raise RuntimeError(
                "Resolved PIP/DIP qpos indices contain duplicates "
                f"(pip={pip_idx}, dip={dip_idx}); check the URDF finger chain "
                "(finger{i}_link3 = PIP, finger{i}_link4 = DIP)."
            )

        self._pip_idx = np.array(pip_idx, dtype=np.int64)
        self._dip_idx = np.array(dip_idx, dtype=np.int64)
        # flex = PIP ∪ DIP, sorted (order is irrelevant: the penalty is an
        # elementwise sum and its gradient scatters back to the same indices).
        self._flex_idx = np.array(sorted(combined), dtype=np.int64)

    def set_pinch_source_keypoints(self, keypoints: np.ndarray):
        """Set original transformed human skeleton used for pinch detection."""
        kp = np.asarray(keypoints, dtype=np.float64)

        if kp.shape != (21, 3):
            raise ValueError(
                f"pinch source expected (21,3), got {kp.shape}"
            )

        self._pinch_source_keypoints = kp.copy()

    def _get_pinch_source_keypoints(
        self,
        fallback_keypoints: np.ndarray,
    ) -> np.ndarray:
        kp = getattr(
            self,
            "_pinch_source_keypoints",
            None,
        )

        if kp is not None:
            kp = np.asarray(kp, dtype=np.float64)

            if kp.shape == (21, 3):
                return kp

        return np.asarray(
            fallback_keypoints,
            dtype=np.float64,
        )


    def _compute_human_grip(
        self,
        mediapipe_keypoints: np.ndarray,
    ) -> float:
        """
        Estimate four-finger grasp amount directly from the Wuji
        skeleton, independent of the robot optimizer result.

        Uses PIP + DIP bending angles of index/middle/ring/pinky.

        0 -> straight/open
        1 -> strongly curled
        """

        kp = self._get_pinch_source_keypoints(
            mediapipe_keypoints
        )

        mcp_idx = [5, 9, 13, 17]
        pip_idx = [6, 10, 14, 18]
        dip_idx = [7, 11, 15, 19]
        tip_idx = [8, 12, 16, 20]

        curls = []

        for m, p, d, t in zip(
            mcp_idx,
            pip_idx,
            dip_idx,
            tip_idx,
        ):
            v1 = kp[p] - kp[m]
            v2 = kp[d] - kp[p]
            v3 = kp[t] - kp[d]

            def angle(a, b):
                na = np.linalg.norm(a)
                nb = np.linalg.norm(b)

                if na < 1e-8 or nb < 1e-8:
                    return 0.0

                c = np.dot(a, b) / (na * nb)
                c = np.clip(c, -1.0, 1.0)

                return float(
                    np.arccos(c)
                )

            a1 = angle(v1, v2)
            a2 = angle(v2, v3)

            # About 120 degrees of accumulated PIP+DIP bending is
            # treated as a fully curled finger.
            curl = np.clip(
                (a1 + a2) / np.deg2rad(120.0),
                0.0,
                1.0,
            )

            curls.append(curl)

        return float(
            np.mean(curls)
        )


    @staticmethod
    def _smoothstep01(x):
        x = float(
            np.clip(x, 0.0, 1.0)
        )
        return x * x * (3.0 - 2.0 * x)


    def _update_explicit_pinch_contact(
        self,
        mediapipe_keypoints: np.ndarray,
        alphas: np.ndarray,
        update_state: bool = True,
    ) -> np.ndarray:
        """
        Build ONE persistent explicit thumb-contact target.

        Ordinary pinch remains fully supported.

        During a grasp, an additional relative-contact score is
        computed from:

            learned open distance -> current distance

        so a deep fist does not lose thumb opposition merely because
        thumb-to-fingertip Euclidean distance increases again.
        """

        pinch_kp = self._get_pinch_source_keypoints(
            mediapipe_keypoints
        )

        thumb = pinch_kp[
            self.MP_TIP_INDICES[0]
        ]

        fingers = pinch_kp[
            self.MP_TIP_INDICES[1:]
        ]

        distances = (
            np.linalg.norm(
                fingers - thumb,
                axis=1,
            )
            * M_TO_CM
        )

        human_grip = self._compute_human_grip(
            mediapipe_keypoints
        )

        absolute_alpha = np.asarray(
            alphas[1:],
            dtype=np.float64,
        )

        # ----------------------------------------------------
        # Automatically learn open-hand thumb/finger distances.
        #
        # Update only while the four fingers are clearly open
        # and there is no strong pinch.
        # ----------------------------------------------------

        can_update_open = (
            human_grip < 0.15
            and np.max(absolute_alpha) < 0.20
        )

        if update_state and can_update_open:

            if self._human_open_pinch_distances is None:
                self._human_open_pinch_distances = (
                    distances.copy()
                )
            else:
                a = self.human_open_update_alpha

                self._human_open_pinch_distances = (
                    (1.0 - a)
                    * self._human_open_pinch_distances
                    +
                    a * distances
                )

        relative_alpha = np.zeros(
            4,
            dtype=np.float64,
        )

        if self._human_open_pinch_distances is not None:

            open_d = np.asarray(
                self._human_open_pinch_distances,
                dtype=np.float64,
            )

            # d1 is used only as a reasonable "strong closure"
            # reference.  This is relative to each user's learned
            # natural open geometry rather than relying on absolute
            # human/robot scale matching.
            denom = np.maximum(
                open_d - self.d1,
                1.0,
            )

            relative_alpha = np.clip(
                (
                    open_d
                    - distances
                )
                / denom,
                0.0,
                1.0,
            )

        # Smooth activation of the grasp-specific component.
        grip_t = (
            human_grip
            - self.grasp_contact_grip_on
        ) / (
            self.grasp_contact_grip_full
            - self.grasp_contact_grip_on
            + 1e-8
        )

        grip_gate = self._smoothstep01(
            grip_t
        )

        grasp_alpha = (
            grip_gate
            * self.grasp_contact_strength
            * relative_alpha
        )

        # Ordinary pinch always wins if stronger.
        score = np.maximum(
            absolute_alpha,
            grasp_alpha,
        )

        active = int(
            self._explicit_pinch_active_idx
        )

        best = int(
            np.argmax(score)
        )

        best_score = float(
            score[best]
        )

        if update_state:

            if active < 0:

                if best_score > 0.05:
                    active = best

            else:
                current_score = float(
                    score[active]
                )

                # If current contact has almost disappeared,
                # allow a new meaningful contact immediately.
                if (
                    current_score < 0.05
                    and best_score > 0.08
                ):
                    active = best

                # Otherwise require a clear advantage before
                # switching contact identity.
                elif (
                    best != active
                    and best_score
                    > current_score
                    + self.grasp_contact_switch_margin
                ):
                    active = best

                # Release only when the HAND itself is no longer
                # grasping and all contact evidence has vanished.
                elif (
                    human_grip < 0.20
                    and best_score < 0.03
                ):
                    active = -1

            self._explicit_pinch_active_idx = active

        else:
            # Cost evaluation must not modify persistent state.
            if active < 0 and best_score > 0.05:
                active = best

        explicit = np.zeros(
            4,
            dtype=np.float64,
        )

        if active >= 0:
            explicit[active] = score[active]

        if update_state:
            self._explicit_pinch_alpha_4 = (
                explicit.copy()
            )

            self._current_human_grip = (
                human_grip
            )

            self._current_relative_contact = (
                relative_alpha.copy()
            )

            self._current_human_pinch_distances = (
                distances.copy()
            )

        return explicit


    def _compute_pinch_alpha(
        self,
        mediapipe_keypoints: np.ndarray,
        update_state: bool = True,
    ) -> np.ndarray:
        """
        Compute per-finger pinch/contact alpha.

        d1..d2:
            normal smooth activation.

        d2..release:
            if the contact was already active, retain a small
            hysteresis alpha instead of instantly dropping to zero.

        This prevents a deep fist / sliding contact from causing
        the thumb IK branch to disappear in one frame.
        """

        pinch_kp = self._get_pinch_source_keypoints(
            mediapipe_keypoints
        )

        thumb_tip = pinch_kp[
            self.MP_TIP_INDICES[0]
        ]

        finger_tips = pinch_kp[
            self.MP_TIP_INDICES[1:]
        ]

        distances = (
            np.linalg.norm(
                finger_tips - thumb_tip,
                axis=1,
            )
            * M_TO_CM
        )

        # Original activation.
        raw_alpha = np.clip(
            (
                self.d2
                - distances
            )
            / (
                self.d2
                - self.d1
                + 1e-8
            ),
            0.0,
            1.0,
        )

        release_distance = (
            self.d2
            + self.pinch_release_margin
        )

        if update_state:

            enter = distances < self.d2

            leave = (
                distances
                > release_distance
            )

            self._pinch_latched = (
                self._pinch_latched
                | enter
            )

            self._pinch_latched[
                leave
            ] = False

        latched = np.asarray(
            self._pinch_latched,
            dtype=bool,
        )

        # Hysteresis tail only matters between d2 and release.
        hold = (
            self.pinch_hold_alpha
            * np.clip(
                (
                    release_distance
                    - distances
                )
                / (
                    self.pinch_release_margin
                    + 1e-8
                ),
                0.0,
                1.0,
            )
        )

        alphas_4 = np.where(
            latched,
            np.maximum(
                raw_alpha,
                hold,
            ),
            raw_alpha,
        )

        alpha_thumb = np.max(
            alphas_4
        )

        # Keep diagnostics accessible.
        self._last_pinch_distances = (
            distances.copy()
        )

        self._last_pinch_alphas_4 = (
            alphas_4.copy()
        )

        return np.concatenate([
            [alpha_thumb],
            alphas_4,
        ])

    def solve(
        self,
        mediapipe_keypoints: np.ndarray,
        last_qpos: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Solve for joint angles."""
        if self._enable_timing:
            t_total_start = time.perf_counter()
            t_preprocess_start = time.perf_counter()
            self._timing.start_frame()

        mediapipe_keypoints = np.asarray(mediapipe_keypoints, dtype=np.float64)
        if mediapipe_keypoints.shape != (21, 3):
            raise ValueError(f"Expected shape (21, 3), got {mediapipe_keypoints.shape}")

        reg_qpos = self._get_reg_qpos(last_qpos)
        init_qpos = self._get_init_qpos(last_qpos)

        # Both the optimization initial state and temporal
        # regularization reference must be hardware-feasible.
        init_qpos = self._project_l20_hardware_constraints(
            init_qpos
        )

        if reg_qpos is not None:
            reg_qpos = self._project_l20_hardware_constraints(
                reg_qpos
            )

        # Configure persistent thumb-roll branch lock from the
        # PREVIOUS hardware-feasible pose.
        #
        # This happens once per frame, before NLopt starts.
        # The bounds therefore remain fixed during all SLSQP
        # objective evaluations of this frame.
        init_qpos = self._prepare_thumb_roll_lock(
            init_qpos,
            reg_qpos,
        )

        alphas = self._compute_pinch_alpha(mediapipe_keypoints)

        # Build a stable single-contact target.  This is separate
        # from the ordinary per-finger alpha array so the existing
        # tip/full-hand blending remains unchanged.
        self._update_explicit_pinch_contact(
            mediapipe_keypoints,
            alphas,
            update_state=True,
        )

        # Current-frame Wuji human grasp is now available.
        # Apply deep-grasp CMC bounds AFTER contact preprocessing
        # and BEFORE NLopt starts.
        init_qpos = self._apply_deep_grasp_cmc_bounds(
            init_qpos
        )

        # Human thumb-to-fingertip distances, in cm.
        pinch_kp = self._get_pinch_source_keypoints(
            mediapipe_keypoints
        )
        thumb_tip_human = pinch_kp[self.MP_TIP_INDICES[0]]
        finger_tips_human = pinch_kp[self.MP_TIP_INDICES[1:]]
        target_pinch_distances = (
            np.linalg.norm(
                finger_tips_human - thumb_tip_human,
                axis=1,
            ) * M_TO_CM
        )

        target_tip_vectors = self._compute_tip_vectors(
            mediapipe_keypoints,
            self.segment_scaling[:, 2:3],
        )
        target_tip_dirs = self._compute_tip_dirs(mediapipe_keypoints)
        target_full_hand_vectors = self._compute_full_hand_vectors(
            mediapipe_keypoints, self.segment_scaling
        )

        if self._enable_timing:
            self._timing.preprocess_ms += (time.perf_counter() - t_preprocess_start) * 1000
            t_nlopt_start = time.perf_counter()

        objective_fn = self._get_objective_analytical(
            target_tip_vectors,
            target_tip_dirs,
            target_full_hand_vectors,
            target_pinch_distances,
            alphas,
            reg_qpos,
        )
        result = self._run_optimization(
            objective_fn,
            init_qpos,
        )

        # SLSQP satisfies the equality constraints numerically.
        # Project once more so downstream simulation/hardware sees
        # the relation exactly, not merely within solver tolerance.
        result = self._project_l20_hardware_constraints(
            result
        )

        # BaseOptimizer._run_optimization stored the raw numerical
        # result. Keep the exact hardware-feasible result as the
        # warm start for the next frame.
        self.last_qpos = result.astype(
            np.float64
        )

        # ===== TEMP PINCH DEBUG =====
        # Print human target vs optimized robot thumb-index distance.
        if not hasattr(self, "_pinch_debug_counter"):
            self._pinch_debug_counter = 0

        self._pinch_debug_counter += 1

        if self._pinch_debug_counter % 20 == 0:
            self.robot.compute_forward_kinematics(result)

            _dbg_positions = np.array([
                self.robot.get_link_pose(idx)[:3, 3]
                for idx in self.computed_link_indices
            ], dtype=np.float64) * M_TO_CM

            _dbg_task = _dbg_positions[self.task_indices]

            _robot_distances = np.linalg.norm(
                _dbg_task[1:]
                - _dbg_task[0],
                axis=1,
            )

            _human_distances = np.asarray(
                target_pinch_distances,
                dtype=np.float64,
            )

            _a = np.asarray(
                alphas[1:],
                dtype=np.float64,
            )

            print(
                "[PINCH DEBUG] "
                "H="
                + np.array2string(
                    _human_distances,
                    precision=2,
                    separator=",",
                )
                + " | R="
                + np.array2string(
                    _robot_distances,
                    precision=2,
                    separator=",",
                )
                + " | A="
                + np.array2string(
                    _a,
                    precision=2,
                    separator=",",
                )
                + f" | At={alphas[0]:.2f}"
            )

            _grip = self._compute_l20_grip(
                result
            )

            _gate = self._thumb_branch_gate(
                _grip
            )

            _contact_a = np.asarray(
                alphas[1:],
                dtype=np.float64,
            )

            if np.max(_contact_a) > 1e-8:
                _active_i = int(
                    np.argmax(_contact_a)
                )

                _active_name = [
                    "index",
                    "middle",
                    "ring",
                    "pinky",
                ][_active_i]
            else:
                _active_name = "none"

            _anchor = getattr(
                self,
                "_thumb_roll_anchor",
                None,
            )

            if _anchor is None:
                _anchor_text = "none"
            else:
                _anchor_text = (
                    f"{np.degrees(_anchor):.1f}deg"
                )

            print(
                "[THUMB BRANCH] "
                f"G={_grip:.2f} "
                f"gate={_gate:.2f} "
                f"active={_active_name} "
                f"lock={int(self._thumb_roll_lock_active)} "
                f"anchor={_anchor_text} "
                f"roll={np.degrees(result[self._thumb_cmc_roll_idx]):.1f}deg "
                f"yaw={np.degrees(result[self._thumb_cmc_yaw_idx]):.1f}deg"
            )

            _exp = np.asarray(
                getattr(
                    self,
                    "_explicit_pinch_alpha_4",
                    np.zeros(4),
                ),
                dtype=np.float64,
            )

            _exp_i = int(
                getattr(
                    self,
                    "_explicit_pinch_active_idx",
                    -1,
                )
            )

            if _exp_i >= 0:
                _exp_name = [
                    "index",
                    "middle",
                    "ring",
                    "pinky",
                ][_exp_i]
            else:
                _exp_name = "none"

            _rel = np.asarray(
                getattr(
                    self,
                    "_current_relative_contact",
                    np.zeros(4),
                ),
                dtype=np.float64,
            )

            print(
                "[GRASP CONTACT] "
                f"HG={getattr(self, '_current_human_grip', 0.0):.2f} "
                f"active={_exp_name} "
                "E="
                + np.array2string(
                    _exp,
                    precision=2,
                    separator=",",
                )
                + " REL="
                + np.array2string(
                    _rel,
                    precision=2,
                    separator=",",
                )
            )

            print(
                "[CMC BOUNDS] "
                f"HG={getattr(self, '_current_human_grip', 0.0):.2f} "
                f"gate={getattr(self, '_deep_grasp_cmc_gate', 0.0):.2f} "
                f"roll_min={np.degrees(getattr(self, '_deep_grasp_roll_lower', self._l20_hw_lower[self._thumb_cmc_roll_idx])):.1f}deg "
                f"yaw_min={np.degrees(getattr(self, '_deep_grasp_yaw_lower', self._l20_hw_lower[self._thumb_cmc_yaw_idx])):.1f}deg"
            )


        # ===== END TEMP PINCH DEBUG =====

        if self._enable_timing:
            self._timing.nlopt_ms += (time.perf_counter() - t_nlopt_start) * 1000
            self._timing.total_ms += (time.perf_counter() - t_total_start) * 1000
            self._timing.call_count += 1
            self._timing.end_frame(self.opt.get_numevals())

        return result

    def compute_cost(
        self,
        qpos: np.ndarray,
        mediapipe_keypoints: np.ndarray,
    ) -> float:
        """Compute cost for given joint angles."""

        qpos = self._project_l20_hardware_constraints(
            qpos
        )

        alphas = self._compute_pinch_alpha(
            mediapipe_keypoints,
            update_state=False,
        )

        self._update_explicit_pinch_contact(
            mediapipe_keypoints,
            alphas,
            update_state=False,
        )

        pinch_kp = self._get_pinch_source_keypoints(
            mediapipe_keypoints
        )
        thumb_tip_human = pinch_kp[self.MP_TIP_INDICES[0]]
        finger_tips_human = pinch_kp[self.MP_TIP_INDICES[1:]]
        target_pinch_distances = (
            np.linalg.norm(
                finger_tips_human - thumb_tip_human,
                axis=1,
            ) * M_TO_CM
        )

        target_tip_vectors = self._compute_tip_vectors(
            mediapipe_keypoints,
            self.segment_scaling[:, 2:3],
        )
        target_tip_dirs = self._compute_tip_dirs(mediapipe_keypoints)
        target_full_hand_vectors = self._compute_full_hand_vectors(
            mediapipe_keypoints, self.segment_scaling
        )
        loss, _ = self._loss_and_grad_analytical(
            qpos,
            target_tip_vectors,
            target_tip_dirs,
            target_full_hand_vectors,
            target_pinch_distances,
            alphas,
            None,
        )
        return float(loss)

    def _loss_and_grad_analytical(
        self,
        qpos: np.ndarray,
        target_tip_vectors: np.ndarray,
        target_tip_dirs: np.ndarray,
        target_full_hand_vectors: np.ndarray,
        target_pinch_distances: np.ndarray,
        alphas: np.ndarray,
        last_qpos: Optional[np.ndarray],
    ) -> tuple[float, np.ndarray]:
        """Compute loss and gradient analytically."""
        qpos = np.asarray(qpos, dtype=np.float64)

        # Forward kinematics
        if self._enable_timing:
            t_fk_start = time.perf_counter()

        self.robot.compute_forward_kinematics(qpos)
        positions = np.array([
            self.robot.get_link_pose(idx)[:3, 3] for idx in self.computed_link_indices
        ], dtype=np.float64) * M_TO_CM

        if self._enable_timing:
            self._timing.fk_ms += (time.perf_counter() - t_fk_start) * 1000
            t_jac_start = time.perf_counter()

        # Get Jacobians (num_links, 3, nq) - already in world frame
        Js = self.robot.compute_all_jacobians_batch(qpos, self.computed_link_indices) * M_TO_CM

        if self._enable_timing:
            self._timing.jacobian_ms += (time.perf_counter() - t_jac_start) * 1000
            t_grad_start = time.perf_counter()

        # Extract positions
        origin_pos = positions[self.origin_indices]  # (5, 3)
        task_pos = positions[self.task_indices]  # (5, 3)
        link3_pos = positions[self.link3_indices]  # (5, 3)
        link4_pos = positions[self.link4_indices]  # (5, 3)
        wrist_pos = positions[self.origin_indices[0]]  # (3,)

        # Get Jacobians for each link type
        J_origin = Js[self.origin_indices]  # (5, 3, nq)
        J_task = Js[self.task_indices]  # (5, 3, nq)
        J_link3 = Js[self.link3_indices]  # (5, 3, nq)
        J_link4 = Js[self.link4_indices]  # (5, 3, nq)
        J_wrist = Js[self.origin_indices[0]]  # (3, nq)

        total_loss = 0.0
        total_grad = np.zeros(self.num_joints, dtype=np.float64)

        # === Tip Position Loss ===
        # robot_tip_vec = task_pos - origin_pos
        # diff = robot_tip_vec - target_tip_vectors
        # dist = ||diff||
        # loss = huber(dist)
        robot_tip_vec = task_pos - origin_pos  # (5, 3)
        diff_pos = robot_tip_vec - target_tip_vectors  # (5, 3)
        dist_pos = np.linalg.norm(diff_pos, axis=1)  # (5,)
        loss_tip_pos = huber_loss_np(dist_pos, self.huber_delta)  # (5,)

        # Gradient: d(huber(dist))/dq = huber'(dist) * d(dist)/dq
        # d(dist)/dq = (diff / dist) @ (J_task - J_origin)
        huber_grad_pos = huber_loss_grad_np(dist_pos, self.huber_delta)  # (5,)
        diff_normed_pos = diff_pos / (dist_pos[:, None] + 1e-8)  # (5, 3)
        for i in range(5):
            grad_coeff = alphas[i] * self.w_pos * huber_grad_pos[i]
            # d(pos)/dq for task - origin
            J_diff = J_task[i] - J_origin[i]  # (3, nq)
            total_grad += grad_coeff * (diff_normed_pos[i] @ J_diff)

        # === Tip Direction Loss ===
        # robot_tip_dir_vec = task_pos - link4_pos
        # robot_tip_dir = normalized(robot_tip_dir_vec)
        # diff = robot_tip_dir - target_tip_dirs
        # dist = ||diff||
        # loss = huber(dist)
        robot_tip_dir_vec = task_pos - link4_pos  # (5, 3)
        robot_tip_dir_norm = np.linalg.norm(robot_tip_dir_vec, axis=1, keepdims=True)  # (5, 1)
        robot_tip_dirs = robot_tip_dir_vec / (robot_tip_dir_norm + 1e-8)  # (5, 3)

        diff_dir = robot_tip_dirs - target_tip_dirs  # (5, 3)
        dist_dir = np.linalg.norm(diff_dir, axis=1)  # (5,)
        loss_tip_dir = huber_loss_np(dist_dir, self.huber_delta_dir)  # (5,)

        # Gradient for normalized direction is more complex
        # Let v = task_pos - link4_pos, n = ||v||, u = v/n
        # du/dq = (I - u*u^T) / n @ (J_task - J_link4)
        huber_grad_dir = huber_loss_grad_np(dist_dir, self.huber_delta_dir)  # (5,)
        diff_normed_dir = diff_dir / (dist_dir[:, None] + 1e-8)  # (5, 3)
        for i in range(5):
            grad_coeff = alphas[i] * self.w_dir * huber_grad_dir[i]
            u = robot_tip_dirs[i]  # (3,)
            n = robot_tip_dir_norm[i, 0]  # scalar
            # Jacobian of normalization: (I - u*u^T) / n
            J_norm = (np.eye(3) - np.outer(u, u)) / (n + 1e-8)  # (3, 3)
            J_diff = J_task[i] - J_link4[i]  # (3, nq)
            # Chain rule: diff_normed_dir @ J_norm @ J_diff
            total_grad += grad_coeff * (diff_normed_dir[i] @ J_norm @ J_diff)

        # === Full Hand Vec Loss ===
        # PIP: link3 - wrist
        # DIP: link4 - wrist
        # TIP: task - wrist
        robot_pip_vec = link3_pos - wrist_pos  # (5, 3)
        robot_dip_vec = link4_pos - wrist_pos  # (5, 3)
        robot_tip_vec_full = task_pos - wrist_pos  # (5, 3)

        target_pip = target_full_hand_vectors[:5]
        target_dip = target_full_hand_vectors[5:10]
        target_tip = target_full_hand_vectors[10:15]

        diff_pip = robot_pip_vec - target_pip
        diff_dip = robot_dip_vec - target_dip
        diff_tip = robot_tip_vec_full - target_tip

        dist_pip = np.linalg.norm(diff_pip, axis=1)
        dist_dip = np.linalg.norm(diff_dip, axis=1)
        dist_tip = np.linalg.norm(diff_tip, axis=1)

        loss_pip = huber_loss_np(dist_pip, self.huber_delta)
        loss_dip = huber_loss_np(dist_dip, self.huber_delta)
        loss_tip_full = huber_loss_np(dist_tip, self.huber_delta)

        # Per-finger PIP mask + divisor. When thumb_skip_pip=False (default) this
        # is mask=1/n=3 for all fingers => identical to (loss_pip+loss_dip+loss_tip)/3.0.
        pip_mask = np.ones(5, dtype=np.float64)
        n_terms = np.full(5, 3.0, dtype=np.float64)
        if self.thumb_skip_pip:
            pip_mask[0] = 0.0
            n_terms[0] = 2.0

        loss_full_hand = (pip_mask * loss_pip + loss_dip + loss_tip_full) / n_terms  # (5,)

        # Gradients for full hand vectors
        huber_grad_pip = huber_loss_grad_np(dist_pip, self.huber_delta)
        huber_grad_dip = huber_loss_grad_np(dist_dip, self.huber_delta)
        huber_grad_tip = huber_loss_grad_np(dist_tip, self.huber_delta)

        diff_normed_pip = diff_pip / (dist_pip[:, None] + 1e-8)
        diff_normed_dip = diff_dip / (dist_dip[:, None] + 1e-8)
        diff_normed_tip = diff_tip / (dist_tip[:, None] + 1e-8)

        for i in range(5):
            grad_coeff = (1.0 - alphas[i]) * self.w_full_hand / n_terms[i]
            # PIP gradient (skipped only when thumb_skip_pip masks finger i)
            if pip_mask[i] != 0.0:
                total_grad += grad_coeff * huber_grad_pip[i] * (diff_normed_pip[i] @ (J_link3[i] - J_wrist))
            # DIP gradient
            total_grad += grad_coeff * huber_grad_dip[i] * (diff_normed_dip[i] @ (J_link4[i] - J_wrist))
            # TIP gradient
            total_grad += grad_coeff * huber_grad_tip[i] * (diff_normed_tip[i] @ (J_task[i] - J_wrist))

        # === Explicit Pinch Distance Loss ===
        #
        # Match the robot thumb-to-finger fingertip distance to the
        # human thumb-to-finger fingertip distance.
        #
        # task_pos order:
        #   0 thumb
        #   1 index
        #   2 middle
        #   3 ring
        #   4 pinky
        #
        # All values here are in centimeters.

        if self.w_pinch != 0.0:

            thumb_robot = task_pos[0]
            finger_robot = task_pos[1:]

            robot_pinch_vec = finger_robot - thumb_robot
            robot_pinch_dist = np.linalg.norm(
                robot_pinch_vec,
                axis=1,
            )

            # Explicit contact identity/strength is frozen for
            # the whole NLopt solve.  It was selected once during
            # preprocessing using pinch + human grasp state.
            pinch_alpha = np.asarray(
                getattr(
                    self,
                    "_explicit_pinch_alpha_4",
                    alphas[1:],
                ),
                dtype=np.float64,
            )

            # -------------------------------------------------
            # IMPORTANT:
            #
            # Human and L20 absolute fingertip distances are not
            # on the same scale.
            #
            # Use the human distance only to obtain contact
            # progress (alpha), then map that progress onto the
            # robot's OWN natural-open geometry.
            #
            # alpha = 0:
            #   robot-native open distance
            #
            # alpha = 1:
            #   near-contact target
            # -------------------------------------------------

            robot_target_pinch_dist = (
                (1.0 - pinch_alpha)
                * self._robot_open_pinch_distances
                +
                pinch_alpha
                * self.pinch_contact_target_cm
            )

            pinch_diff = (
                robot_pinch_dist
                - robot_target_pinch_dist
            )

            # 1/2 * w * alpha * e^2
            total_loss += (
                0.5
                * self.w_pinch
                * np.sum(
                    pinch_alpha
                    * pinch_diff ** 2
                )
            )

            # Analytical gradient.
            for i in range(4):

                if pinch_alpha[i] <= 1e-8:
                    continue

                dist = robot_pinch_dist[i]

                if dist <= 1e-8:
                    continue

                direction = (
                    robot_pinch_vec[i] / dist
                )

                # finger tip Jacobian - thumb tip Jacobian
                J_rel = (
                    J_task[i + 1]
                    - J_task[0]
                )

                total_grad += (
                    self.w_pinch
                    * pinch_alpha[i]
                    * pinch_diff[i]
                    * (direction @ J_rel)
                )

        # === Total Loss ===
        loss_tip_dir_vec = self.w_pos * loss_tip_pos + self.w_dir * loss_tip_dir
        loss_full = self.w_full_hand * loss_full_hand
        loss_per_finger = alphas * loss_tip_dir_vec + (1.0 - alphas) * loss_full
        # IMPORTANT:
        # Explicit pinch loss may already have been accumulated above.
        # Do NOT overwrite total_loss here.
        total_loss += np.sum(loss_per_finger)

        # === Regularization ===
        if last_qpos is not None:
            total_loss += self.norm_delta * np.sum((qpos - last_qpos) ** 2)
            total_grad += 2.0 * self.norm_delta * (qpos - last_qpos)

        # === Hyperextension penalty (PIP/DIP only, gated by w_hyper) ===
        if self.w_hyper != 0.0:
            flex_qpos = qpos[self._flex_idx]
            penalty = np.maximum(self.soft_min - flex_qpos, 0.0)
            total_loss += self.w_hyper * np.sum(penalty ** 2)
            total_grad[self._flex_idx] += self.w_hyper * (-2.0 * penalty)

        # === Coupling penalty (DIP toward couple_ratio * PIP, gated by w_couple) ===
        if self.w_couple != 0.0:
            pip_q = qpos[self._pip_idx]
            dip_q = qpos[self._dip_idx]
            diff = dip_q - self.couple_ratio * pip_q
            total_loss += self.w_couple * np.sum(diff ** 2)
            total_grad[self._dip_idx] += self.w_couple * (2.0 * diff)
            total_grad[self._pip_idx] += self.w_couple * (-2.0 * self.couple_ratio * diff)

        # === Deep-Grip Thumb CMC Branch Continuity ===
        #
        # The problematic failure mode is not normal thumb motion;
        # it is a sudden CMC branch switch while the other four
        # fingers are already closing deeply.
        #
        # Therefore:
        #
        #   open hand  -> almost no extra constraint
        #   half grasp -> gradually activate
        #   deep grasp -> strongly prefer previous roll/yaw branch
        #
        # Only roll/yaw are regularized here.  Pitch and MCP/IP
        # remain free to follow the retargeting objective.

        if (
            last_qpos is not None
            and (
                self.w_thumb_branch_roll != 0.0
                or self.w_thumb_branch_yaw != 0.0
            )
        ):
            last_q = np.asarray(
                last_qpos,
                dtype=np.float64,
            )

            grip_prev = self._compute_l20_grip(
                last_q
            )

            branch_gate = self._thumb_branch_gate(
                grip_prev
            )

            if branch_gate > 1e-8:

                roll_i = self._thumb_cmc_roll_idx
                yaw_i = self._thumb_cmc_yaw_idx

                d_roll = (
                    qpos[roll_i]
                    - last_q[roll_i]
                )

                d_yaw = (
                    qpos[yaw_i]
                    - last_q[yaw_i]
                )

                total_loss += (
                    0.5
                    * branch_gate
                    * (
                        self.w_thumb_branch_roll
                        * d_roll * d_roll
                        +
                        self.w_thumb_branch_yaw
                        * d_yaw * d_yaw
                    )
                )

                total_grad[roll_i] += (
                    branch_gate
                    * self.w_thumb_branch_roll
                    * d_roll
                )

                total_grad[yaw_i] += (
                    branch_gate
                    * self.w_thumb_branch_yaw
                    * d_yaw
                )

        if self._enable_timing:
            self._timing.gradient_ms += (time.perf_counter() - t_grad_start) * 1000

        return total_loss, total_grad

    def get_timing_stats(self) -> TimingStats:
        """Get timing statistics."""
        return self._timing

    def reset_timing_stats(self):
        """Reset timing statistics."""
        self._timing.reset()

    def set_timing_enabled(self, enabled: bool):
        """Enable or disable timing instrumentation."""
        self._enable_timing = enabled

    def _get_objective_analytical(
        self,
        target_tip_vectors: np.ndarray,
        target_tip_dirs: np.ndarray,
        target_full_hand_vectors: np.ndarray,
        target_pinch_distances: np.ndarray,
        alphas: np.ndarray,
        last_qpos: Optional[np.ndarray],
    ):
        """Create NLopt objective function with analytical gradient."""
        target_tip_vectors = np.asarray(target_tip_vectors, dtype=np.float64)
        target_tip_dirs = np.asarray(target_tip_dirs, dtype=np.float64)
        target_full_hand_vectors = np.asarray(target_full_hand_vectors, dtype=np.float64)
        target_pinch_distances = np.asarray(target_pinch_distances, dtype=np.float64)
        alphas = np.asarray(alphas, dtype=np.float64)
        if last_qpos is not None:
            last_qpos = np.asarray(last_qpos, dtype=np.float64)

        def objective(x, grad_out):
            qpos = np.asarray(x, dtype=np.float64)
            loss, grad = self._loss_and_grad_analytical(
                qpos,
                target_tip_vectors,
                target_tip_dirs,
                target_full_hand_vectors,
                target_pinch_distances,
                alphas,
                last_qpos,
            )
            if grad_out.size > 0:
                grad_out[:] = grad
            # Record iteration loss for plotting
            if self._enable_timing:
                self._timing.record_iter_loss(float(loss))
            return float(loss)

        return objective
