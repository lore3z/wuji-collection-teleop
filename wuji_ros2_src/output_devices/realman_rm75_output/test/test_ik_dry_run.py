import math
from pathlib import Path

import numpy as np
import pytest

from realman_rm75_output.ik_dry_run import (
    IKDryRunEvaluator,
    validate_safety_mode,
)
from realman_rm75_output.pose_mapper import (
    RelativePoseMapper,
    matrix_to_quaternion_xyzw,
    quaternion_to_matrix_xyzw,
    quaternion_wxyz_to_xyzw,
    quaternion_xyzw_to_wxyz,
)


class FakeAlgorithm:
    def __init__(self, ik_code=0, joints=None, fk_pose=None,
                 arm_code=0, arm_angle=12.0):
        self.ik_code = ik_code
        self.joints = [0.0] * 7 if joints is None else joints
        self.fk_pose = ([0.2, -0.1, 0.3, 1.0, 0.0, 0.0, 0.0]
                        if fk_pose is None else fk_pose)
        self.arm_code = arm_code
        self.arm_angle_value = arm_angle
        self.last_reference = None
        self.last_target = None
        self.last_arm_angle = None

    def inverse(self, reference, target_wxyz):
        self.last_reference = reference
        self.last_target = target_wxyz
        return self.ik_code, self.joints

    def inverse_for_arm_angle(self, reference, target_wxyz, arm_angle):
        self.last_reference = reference
        self.last_target = target_wxyz
        self.last_arm_angle = arm_angle
        return self.ik_code, self.joints

    def forward(self, joints):
        return self.fk_pose

    def arm_angle(self, joints):
        return self.arm_code, self.arm_angle_value


def evaluator(fake, initial=None, max_jump=12.0, fk_warn=0.002,
              fk_fault=None):
    return IKDryRunEvaluator(
        fake,
        [0.0] * 7 if initial is None else initial,
        [-180.0] * 7,
        [180.0] * 7,
        max_joint_jump_deg=max_jump,
        joint_limit_margin_deg=5.0,
        max_fk_position_error_m=0.002,
        max_fk_orientation_error_rad=math.radians(1.0),
        fk_position_warn_m=fk_warn,
        fk_position_fault_m=fk_fault,
    )


def mapper(axis=np.eye(3)):
    result = RelativePoseMapper(
        axis_mapping=axis,
        position_scale=1.0,
        max_position_jump_m=0.20,
        max_rotation_jump_rad=math.radians(90),
        workspace_min=[-2, -2, -2],
        workspace_max=[2, 2, 2],
        tracker_timeout_sec=0.5,
    )
    result.set_rm_initial_pose([0.2, -0.1, 0.3], [0, 0, 0, 1])
    return result


@pytest.mark.parametrize("axis,sign", [(0, 1), (0, -1), (1, 1), (1, -1),
                                        (2, 1), (2, -1)])
def test_all_six_translations_follow_axis_mapping(axis, sign):
    transform = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=float)
    m = mapper(transform)
    m.process([0, 0, 0], [0, 0, 0, 1])
    position = np.zeros(3)
    position[axis] = sign * 0.01
    result = m.process(position, [0, 0, 0, 1])
    np.testing.assert_allclose(result.mapped_translation,
                               transform @ position, atol=1e-12)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_single_axis_rotation_uses_left_composition_formula(axis):
    transform = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=float)
    m = RelativePoseMapper(
        transform, 1.0, 0.2, math.radians(90), [-2]*3, [2]*3, 0.5)
    rm_initial = matrix_to_quaternion_xyzw(
        np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float))
    m.set_rm_initial_pose([0, 0, 0], rm_initial)
    m.process([0, 0, 0], [0, 0, 0, 1])
    rotvec = np.zeros(3)
    rotvec[axis] = math.radians(20)
    angle = np.linalg.norm(rotvec)
    unit = rotvec / angle
    x, y, z = unit * math.sin(angle / 2)
    current_q = np.array([x, y, z, math.cos(angle / 2)])
    result = m.process([0, 0, 0], current_q)
    source_rotation = quaternion_to_matrix_xyzw(current_q)
    expected = transform @ source_rotation @ transform.T
    expected = expected @ quaternion_to_matrix_xyzw(rm_initial)
    np.testing.assert_allclose(
        quaternion_to_matrix_xyzw(result.target_quaternion_xyzw),
        expected, atol=1e-10)


def test_quaternion_order_round_trip():
    wxyz = np.array([0.5, 0.5, -0.5, 0.5])
    xyzw = quaternion_wxyz_to_xyzw(wxyz)
    np.testing.assert_allclose(xyzw, [0.5, -0.5, 0.5, 0.5])
    np.testing.assert_allclose(quaternion_xyzw_to_wxyz(xyzw), wxyz)


def test_ik_failure_produces_no_accepted_output():
    result = evaluator(FakeAlgorithm(ik_code=1)).evaluate(
        [0.2, -0.1, 0.3], [0, 0, 0, 1])
    assert not result.accepted
    assert result.target_joints_deg is None
    assert "IK failed" in result.reason


def test_joint_jump_is_rejected_and_reference_is_not_advanced():
    fake = FakeAlgorithm(joints=[20.0, 0, 0, 0, 0, 0, 0])
    check = evaluator(fake, max_jump=5.0)
    result = check.evaluate([0.2, -0.1, 0.3], [0, 0, 0, 1])
    assert not result.accepted
    assert "joint jump" in result.reason
    np.testing.assert_allclose(check.reference_joints, [0] * 7)


def test_arm_angle_observation_failure_does_not_constrain_ordinary_ik():
    fake = FakeAlgorithm(arm_code=-1)
    result = evaluator(fake).evaluate([0.2, -0.1, 0.3], [0, 0, 0, 1])
    assert result.accepted
    assert math.isnan(result.arm_angle_deg)
    assert math.isnan(result.arm_angle_delta_deg)


def test_fk_position_residual_at_or_below_warning_is_accepted():
    fake = FakeAlgorithm(fk_pose=[0.2029, -0.1, 0.3, 1, 0, 0, 0])
    result = evaluator(fake, fk_warn=0.003, fk_fault=0.005).evaluate(
        [0.2, -0.1, 0.3], [0, 0, 0, 1])
    assert result.accepted
    assert result.safety_action == "accept"


def test_fk_position_warning_holds_and_does_not_advance_reference():
    fake = FakeAlgorithm(
        joints=[1.0] * 7,
        fk_pose=[0.204, -0.1, 0.3, 1, 0, 0, 0])
    check = evaluator(fake, fk_warn=0.003, fk_fault=0.005)
    result = check.evaluate([0.2, -0.1, 0.3], [0, 0, 0, 1])
    assert not result.accepted
    assert result.safety_action == "hold"
    np.testing.assert_allclose(check.reference_joints, [0] * 7)


def test_fk_position_above_fault_threshold_is_hard_fault():
    fake = FakeAlgorithm(fk_pose=[0.2051, -0.1, 0.3, 1, 0, 0, 0])
    result = evaluator(fake, fk_warn=0.003, fk_fault=0.005).evaluate(
        [0.2, -0.1, 0.3], [0, 0, 0, 1])
    assert not result.accepted
    assert result.safety_action == "fault"


def test_arm_angle_delta_wraps_across_180_degrees():
    fake = FakeAlgorithm(arm_angle=179.0)
    check = evaluator(fake)
    assert check.evaluate([0.2, -0.1, 0.3], [0, 0, 0, 1]).accepted
    fake.arm_angle_value = -179.0
    second = check.evaluate([0.2, -0.1, 0.3], [0, 0, 0, 1])
    assert second.accepted
    assert second.arm_angle_delta_deg == pytest.approx(2.0)


def test_ordinary_ik_receives_previous_accepted_solution_and_wxyz_pose():
    fake = FakeAlgorithm(joints=[1.0] * 7)
    check = evaluator(fake)
    result = check.evaluate([0.2, -0.1, 0.3], [0, 0, 0, 1])
    assert result.accepted
    np.testing.assert_allclose(fake.last_reference, [0] * 7)
    np.testing.assert_allclose(fake.last_target, [0.2, -0.1, 0.3, 1, 0, 0, 0])
    np.testing.assert_allclose(check.reference_joints, [1] * 7)


def test_arm_angle_ik_receives_previous_solution_pose_and_target_angle():
    fake = FakeAlgorithm(joints=[1.0] * 7, arm_angle=23.0)
    check = evaluator(fake)
    result = check.evaluate(
        [0.2, -0.1, 0.3], [0, 0, 0, 1], target_arm_angle_deg=23.0)
    assert result.accepted
    np.testing.assert_allclose(fake.last_reference, [0] * 7)
    assert fake.last_arm_angle == 23.0
    np.testing.assert_allclose(check.reference_joints, [1] * 7)


def test_continuity_check_can_be_separate_from_ik_seed():
    fake = FakeAlgorithm(joints=[20.5] * 7)
    check = evaluator(fake, initial=[0] * 7, max_jump=1.0)
    result = check.evaluate(
        [0.2, -0.1, 0.3], [0, 0, 0, 1],
        joint_continuity_reference_deg=[20.0] * 7)
    assert result.accepted
    np.testing.assert_allclose(fake.last_reference, [0] * 7)
    assert result.max_joint_delta_deg == pytest.approx(0.5)


@pytest.mark.parametrize("dry_run,enable_motion", [(False, False), (True, True),
                                                     (False, True)])
def test_unsafe_mode_is_rejected(dry_run, enable_motion):
    with pytest.raises(RuntimeError, match="Safety lock"):
        validate_safety_mode(dry_run, enable_motion)


def test_only_safe_mode_is_accepted():
    validate_safety_mode(True, False)


def test_real_motion_apis_are_confined_to_narrow_backend_files():
    package_root = Path(__file__).resolve().parents[1]
    sources = {
        path: path.read_text(encoding="utf-8")
        for path in package_root.rglob("*.py")
        if path != Path(__file__).resolve()
    }
    source = "\n".join(sources.values())
    forbidden = [
        "rm_" + "movel(", "rm_" + "movep(", "rm_" + "movej_follow",
        "rm_" + "move_stop",
    ]
    assert not any(name in source for name in forbidden)
    pose_api = "rm_" + "movep_canfd"
    containing = [path for path, text in sources.items() if pose_api in text]
    assert [path.name for path in containing] == ["movep_backend.py"]
    assert sources[containing[0]].count(pose_api) == 1
    joint_api = "rm_" + "movej_canfd"
    joint_containing = [path for path, text in sources.items()
                        if joint_api in text]
    assert [path.name for path in joint_containing] == ["movej_backend.py"]
    assert sources[joint_containing[0]].count(joint_api) == 1
    planned_joint_api = "rm_" + "movej("
    planned_joint_containing = [
        path for path, text in sources.items()
        if planned_joint_api in text]
    assert [path.name for path in planned_joint_containing] == [
        "movej_backend.py"]
    assert sources[planned_joint_containing[0]].count(planned_joint_api) == 1
