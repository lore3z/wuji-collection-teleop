"""Read-only RM75 configuration audit and Base-X ordinary-IK preflight.

This module intentionally contains no robot motion, stop, enable, or
configuration calls.  Controller queries and local algorithm-library calls are
the only operations permitted here.
"""

from dataclasses import asdict, dataclass
import math

import numpy as np
from Robotic_Arm.rm_robot_interface import rm_inverse_kinematics_params_t

from .ik_dry_run import IKDryRunEvaluator
from .pose_mapper import quaternion_wxyz_to_xyzw


class _AlgorithmAdapter:
    """Algorithm-only adapter kept independent of ROS imports."""

    def __init__(self, robot):
        self.robot = robot

    def inverse(self, reference_joints_deg, target_pose_wxyz):
        params = rm_inverse_kinematics_params_t(
            q_in=reference_joints_deg, q_pose=target_pose_wxyz, flag=0)
        return self.robot.rm_algo_inverse_kinematics(params)

    def forward(self, joints_deg):
        return self.robot.rm_algo_forward_kinematics(joints_deg, flag=0)

    def arm_angle(self, joints_deg):
        return self.robot.rm_algo_calculate_arm_angle_from_config_rm75(joints_deg)


READ_ONLY_QUERY_NAMES = (
    "rm_get_arm_software_info",
    "rm_get_current_arm_state",
    "rm_get_controller_state",
    "rm_get_arm_current_trajectory",
    "rm_get_joint_err_flag",
    "rm_get_current_tool_frame",
    "rm_get_current_work_frame",
    "rm_get_total_tool_frame",
    "rm_get_total_work_frame",
    "rm_get_collision_stage",
    "rm_get_collision_detection",
    "rm_get_avoid_singularity_mode",
    "rm_get_self_collision_enable",
    "rm_get_self_endeffector_collision_enable",
    "rm_get_electronic_fence_enable",
    "rm_get_electronic_fence_config",
    "rm_get_electronic_fence_list_infos",
    "rm_get_realtime_push",
)


@dataclass(frozen=True)
class PathPointResult:
    offset_m: float
    accepted: bool
    reason: str
    joints_deg: list[float] | None
    max_joint_delta_deg: float
    minimum_joint_margin_deg: float
    fk_position_error_m: float
    fk_orientation_error_rad: float
    local_self_collision: bool | None


def base_x_offsets(max_offset_m=0.005, sample_step_m=0.0005):
    """Return current->-limit->current->+limit offsets without duplicates."""
    limit = float(max_offset_m)
    step = float(sample_step_m)
    if not np.isfinite(limit) or limit <= 0.0:
        raise ValueError("max_offset_m must be finite and positive")
    if not np.isfinite(step) or step <= 0.0 or step > limit:
        raise ValueError("sample_step_m must be positive and no larger than max_offset_m")

    count = int(math.ceil(limit / step))
    negative = np.linspace(0.0, -limit, count + 1)
    return_to_zero = np.linspace(-limit, 0.0, count + 1)[1:]
    positive = np.linspace(0.0, limit, count + 1)[1:]
    return np.concatenate((negative, return_to_zero, positive))


def _return_code(value):
    if isinstance(value, tuple):
        return value[0]
    if isinstance(value, dict):
        return value.get("return_code")
    return None


def collect_read_only_audit(robot):
    """Execute the explicit query allowlist and fail on any query error."""
    audit = {}
    for name in READ_ONLY_QUERY_NAMES:
        result = getattr(robot, name)()
        audit[name] = result
        code = _return_code(result)
        if code != 0:
            raise RuntimeError(f"{name} failed: return code {code}")
    audit["rm_algo_version"] = robot.rm_algo_version()
    audit["algorithm_singularity_thresholds"] = (
        robot.rm_algo_kin_get_singularity_thresholds())
    audit["algorithm_joint_min_deg"] = robot.rm_algo_get_joint_min_limit()
    audit["algorithm_joint_max_deg"] = robot.rm_algo_get_joint_max_limit()
    return audit


def run_base_x_preflight(robot, state, max_offset_m=0.005,
                         sample_step_m=0.0005, joint_limit_margin_deg=5.0,
                         max_joint_jump_deg=12.0,
                         max_fk_position_error_m=0.002,
                         max_fk_orientation_error_deg=1.0):
    """Sample Base X and evaluate it using only the local ordinary IK/FK API."""
    pose = np.asarray(state.get("pose", []), dtype=float)
    joints = np.asarray(state.get("joint", []), dtype=float)
    if pose.shape != (6,) or not np.all(np.isfinite(pose)):
        raise ValueError("current pose must contain six finite values")
    if joints.shape != (7,) or not np.all(np.isfinite(joints)):
        raise ValueError("current joints must contain seven finite values")

    quaternion = quaternion_wxyz_to_xyzw(
        robot.rm_algo_euler2quaternion(pose[3:6].tolist()))
    joint_min = np.asarray(robot.rm_algo_get_joint_min_limit(), dtype=float)
    joint_max = np.asarray(robot.rm_algo_get_joint_max_limit(), dtype=float)
    evaluator = IKDryRunEvaluator(
        _AlgorithmAdapter(robot), joints, joint_min, joint_max,
        max_joint_jump_deg=max_joint_jump_deg,
        joint_limit_margin_deg=joint_limit_margin_deg,
        max_fk_position_error_m=max_fk_position_error_m,
        max_fk_orientation_error_rad=math.radians(max_fk_orientation_error_deg),
    )

    points = []
    for offset in base_x_offsets(max_offset_m, sample_step_m):
        target = pose[:3].copy()
        target[0] += offset
        ik = evaluator.evaluate(target, quaternion)
        solution = ik.target_joints_deg
        margin = float("nan")
        collision = None
        if solution is not None:
            margin = float(np.min(np.minimum(solution - joint_min,
                                             joint_max - solution)))
            collision = bool(
                robot.rm_algo_safety_robot_self_collision_detection(
                    solution.tolist()))
        accepted = bool(ik.accepted and not ik.near_joint_limit and
                        collision is not True)
        reason = ik.reason
        if ik.accepted and ik.near_joint_limit:
            reason = "IK solution is inside configured joint-limit margin"
        elif ik.accepted and collision:
            reason = "local algorithm reports self-collision"
        points.append(PathPointResult(
            offset_m=float(offset), accepted=accepted, reason=reason,
            joints_deg=None if solution is None else solution.tolist(),
            max_joint_delta_deg=float(ik.max_joint_delta_deg),
            minimum_joint_margin_deg=margin,
            fk_position_error_m=float(ik.fk_position_error_m),
            fk_orientation_error_rad=float(ik.fk_orientation_error_rad),
            local_self_collision=collision,
        ))

    return {
        "disclaimer": (
            "Local ordinary IK/FK preflight only; the controller's internal "
            "pose-pass-through IK may choose a different solution."
        ),
        "path_frame": "RM Base X",
        "initial_pose_m_rad": pose.tolist(),
        "initial_joints_deg": joints.tolist(),
        "max_offset_m": float(max_offset_m),
        "sample_step_m": float(sample_step_m),
        "passed": all(point.accepted for point in points),
        "points": [asdict(point) for point in points],
    }
