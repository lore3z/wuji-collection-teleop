"""Pure validation around RealMan's ordinary IK/FK algorithms.

The adapter passed here exposes algorithm calls only.  This module has no ROS
dependency and no controller command path.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .pose_mapper import (
    InvalidPose,
    normalize_quaternion_xyzw,
    quaternion_angle_rad,
    quaternion_wxyz_to_xyzw,
    quaternion_xyzw_to_wxyz,
)


def validate_safety_mode(dry_run: bool, enable_motion: bool) -> None:
    if dry_run is not True or enable_motion is not False:
        raise RuntimeError(
            "Safety lock requires dry_run=true and enable_motion=false"
        )


@dataclass(frozen=True)
class IKDryRunResult:
    accepted: bool
    reason: str
    ik_return_code: int
    target_joints_deg: Optional[np.ndarray] = None
    max_joint_delta_deg: float = float("nan")
    arm_angle_deg: float = float("nan")
    arm_angle_delta_deg: float = float("nan")
    fk_position_error_m: float = float("nan")
    fk_orientation_error_rad: float = float("nan")
    near_joint_limit: bool = False
    safety_action: str = "fault"


class IKDryRunEvaluator:
    """Run ordinary IK and reject unsafe/inconsistent solutions."""

    def __init__(self, algorithm, initial_joints_deg, joint_min_deg,
                 joint_max_deg, max_joint_jump_deg,
                 joint_limit_margin_deg, max_fk_position_error_m,
                 max_fk_orientation_error_rad, fk_position_warn_m=None,
                 fk_position_fault_m=None):
        self.algorithm = algorithm
        self.reference_joints = self._seven_finite(
            initial_joints_deg, "initial joints")
        self.joint_min = self._seven_finite(joint_min_deg, "joint minimum")
        self.joint_max = self._seven_finite(joint_max_deg, "joint maximum")
        if np.any(self.joint_min >= self.joint_max):
            raise ValueError("joint minimum must be below joint maximum")
        self.max_joint_jump_deg = self._positive(
            max_joint_jump_deg, "max_joint_jump_deg")
        self.joint_limit_margin_deg = self._nonnegative(
            joint_limit_margin_deg, "joint_limit_margin_deg")
        legacy_position_limit = self._positive(
            max_fk_position_error_m, "max_fk_position_error_m")
        self.fk_position_warn_m = self._positive(
            legacy_position_limit if fk_position_warn_m is None
            else fk_position_warn_m, "fk_position_warn_m")
        self.fk_position_fault_m = self._positive(
            self.fk_position_warn_m if fk_position_fault_m is None
            else fk_position_fault_m, "fk_position_fault_m")
        if self.fk_position_fault_m < self.fk_position_warn_m:
            raise ValueError(
                "fk_position_fault_m must be >= fk_position_warn_m")
        self.max_fk_orientation_error_rad = self._positive(
            max_fk_orientation_error_rad, "max_fk_orientation_error_rad")
        self.previous_arm_angle_deg: Optional[float] = None

    @staticmethod
    def _seven_finite(value, label):
        array = np.asarray(value, dtype=float)
        if array.shape != (7,) or not np.all(np.isfinite(array)):
            raise ValueError(f"{label} must contain seven finite values")
        return array.copy()

    @staticmethod
    def _positive(value, label):
        result = float(value)
        if not np.isfinite(result) or result <= 0.0:
            raise ValueError(f"{label} must be finite and positive")
        return result

    @staticmethod
    def _nonnegative(value, label):
        result = float(value)
        if not np.isfinite(result) or result < 0.0:
            raise ValueError(f"{label} must be finite and nonnegative")
        return result

    def evaluate(self, target_position, target_quaternion_xyzw,
                 target_arm_angle_deg=None,
                 joint_continuity_reference_deg=None):
        try:
            position = np.asarray(target_position, dtype=float)
            if position.shape != (3,) or not np.all(np.isfinite(position)):
                raise InvalidPose("IK target position must contain three finite values")
            quaternion = normalize_quaternion_xyzw(target_quaternion_xyzw)
        except (InvalidPose, ValueError) as exc:
            return IKDryRunResult(False, str(exc), -999)

        # The adapter accepts the official pose order [x,y,z,w,x,y,z].
        target_wxyz = quaternion_xyzw_to_wxyz(quaternion)
        target_pose = np.concatenate([position, target_wxyz]).tolist()
        if target_arm_angle_deg is None:
            ik_code, joints = self.algorithm.inverse(
                self.reference_joints.tolist(), target_pose)
        else:
            ik_code, joints = self.algorithm.inverse_for_arm_angle(
                self.reference_joints.tolist(), target_pose,
                float(target_arm_angle_deg))
        if ik_code != 0:
            return IKDryRunResult(False, f"ordinary IK failed: code {ik_code}", ik_code)
        try:
            joints = self._seven_finite(joints, "IK joints")
        except ValueError as exc:
            return IKDryRunResult(False, str(exc), ik_code)

        if np.any(joints < self.joint_min) or np.any(joints > self.joint_max):
            return IKDryRunResult(False, "IK solution exceeds joint limits", ik_code,
                                  target_joints_deg=joints)
        continuity_reference = (
            self.reference_joints if joint_continuity_reference_deg is None
            else self._seven_finite(
                joint_continuity_reference_deg,
                "joint continuity reference"))
        joint_delta = float(np.max(np.abs(joints - continuity_reference)))
        if joint_delta > self.max_joint_jump_deg:
            return IKDryRunResult(
                False,
                f"joint jump {joint_delta:.6f} deg exceeds limit",
                ik_code, target_joints_deg=joints,
                max_joint_delta_deg=joint_delta,
            )
        margin = np.minimum(joints - self.joint_min, self.joint_max - joints)
        near_limit = bool(np.any(margin <= self.joint_limit_margin_deg))

        arm_code, arm_angle = self.algorithm.arm_angle(joints.tolist())
        arm_angle = float(arm_angle) if arm_code == 0 else float("nan")
        arm_delta = (float("nan") if self.previous_arm_angle_deg is None or
                     not np.isfinite(arm_angle) else
                     abs((arm_angle - self.previous_arm_angle_deg + 180.0)
                         % 360.0 - 180.0))

        fk_pose = np.asarray(self.algorithm.forward(joints.tolist()), dtype=float)
        if fk_pose.shape != (7,) or not np.all(np.isfinite(fk_pose)):
            return IKDryRunResult(False, "FK returned invalid pose", ik_code,
                                  target_joints_deg=joints,
                                  max_joint_delta_deg=joint_delta,
                                  arm_angle_deg=arm_angle,
                                  arm_angle_delta_deg=arm_delta,
                                  near_joint_limit=near_limit)
        fk_position = fk_pose[:3]
        try:
            fk_quaternion = quaternion_wxyz_to_xyzw(fk_pose[3:7])
        except InvalidPose as exc:
            return IKDryRunResult(False, f"FK quaternion invalid: {exc}", ik_code)
        position_error = float(np.linalg.norm(fk_position - position))
        orientation_error = quaternion_angle_rad(fk_quaternion, quaternion)
        if position_error > self.fk_position_fault_m:
            reason = (
                f"FK position residual {position_error:.6f} m exceeds "
                f"fault threshold {self.fk_position_fault_m:.6f} m")
            accepted = False
            safety_action = "fault"
        elif position_error > self.fk_position_warn_m:
            reason = (
                f"FK position residual {position_error:.6f} m exceeds "
                f"warning threshold {self.fk_position_warn_m:.6f} m; "
                "hold previous target")
            accepted = False
            safety_action = "hold"
        elif orientation_error > self.max_fk_orientation_error_rad:
            reason = f"FK orientation residual {orientation_error:.6f} rad exceeds limit"
            accepted = False
            safety_action = "fault"
        else:
            reason = "accepted dry-run IK solution"
            accepted = True
            safety_action = "accept"

        result = IKDryRunResult(
            accepted, reason, ik_code, joints, joint_delta, arm_angle,
            arm_delta, position_error, orientation_error, near_limit,
            safety_action,
        )
        if accepted:
            self.reference_joints = joints.copy()
            if np.isfinite(arm_angle):
                self.previous_arm_angle_deg = arm_angle
        return result
