import math
from pathlib import Path

import numpy as np
import pinocchio as pin
import pytest

from realman_rm75_output.external_pose_follower import (
    ElbowAssistController,
    OperatorPoseMapper,
    interpolate_pose,
    solve_with_boundary_fallback,
)
from realman_rm75_output.offline_trajectory import (
    DEFAULT_HOME_JOINTS_RAD,
    RM75Kinematics,
)


URDF = (Path(__file__).resolve().parents[3] /
        "rm_description" / "urdf" / "rm_75.urdf")


def test_absolute_operator_pose_is_used_directly():
    home = RM75Kinematics(URDF).forward(DEFAULT_HOME_JOINTS_RAD)
    mapper = OperatorPoseMapper(home, input_mode="absolute")
    target = mapper.process(
        [0.30, -0.20, 0.50], [0.0, 0.0, 0.0, 2.0])
    np.testing.assert_allclose(target.translation, [0.30, -0.20, 0.50])
    np.testing.assert_allclose(target.rotation, np.eye(3), atol=1e-12)


def test_relative_operator_pose_rebases_and_maps_position_and_rotation():
    home = RM75Kinematics(URDF).forward(DEFAULT_HOME_JOINTS_RAD)
    # Source X -> base -Y, source Y -> base X, source Z -> base Z.
    axes = np.array([[0.0, 1.0, 0.0],
                     [-1.0, 0.0, 0.0],
                     [0.0, 0.0, 1.0]])
    mapper = OperatorPoseMapper(
        home, input_mode="relative", source_to_base=axes,
        position_scale=0.5)
    baseline = mapper.process([1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(baseline.translation, home.translation)
    np.testing.assert_allclose(baseline.rotation, home.rotation)

    angle = math.pi / 6.0
    source_quaternion = [0.0, 0.0, math.sin(angle / 2.0),
                         math.cos(angle / 2.0)]
    target = mapper.process([1.10, 2.20, 3.30], source_quaternion)
    expected_offset = 0.5 * axes @ np.array([0.10, 0.20, 0.30])
    expected_delta = axes @ pin.rpy.rpyToMatrix(0.0, 0.0, angle) @ axes.T
    np.testing.assert_allclose(
        target.translation, home.translation + expected_offset, atol=1e-12)
    np.testing.assert_allclose(
        target.rotation, expected_delta @ home.rotation, atol=1e-12)


def test_rebase_keeps_current_robot_pose_on_next_operator_sample():
    home = RM75Kinematics(URDF).forward(DEFAULT_HOME_JOINTS_RAD)
    mapper = OperatorPoseMapper(home, input_mode="relative")
    mapper.process([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
    moved = mapper.process([0.05, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
    mapper.rebase(moved)
    rebased = mapper.process([9.0, 8.0, 7.0], [0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(rebased.translation, moved.translation)
    np.testing.assert_allclose(rebased.rotation, moved.rotation)


def test_pose_interpolation_preserves_half_translation_and_rotation():
    start = pin.SE3.Identity()
    end = pin.SE3(
        pin.rpy.rpyToMatrix(math.radians(60.0), 0.0, 0.0),
        np.array([1.0, 0.0, 0.0]))
    halfway = interpolate_pose(start, end, 0.5)

    np.testing.assert_allclose(halfway.translation, [0.5, 0.0, 0.0])
    np.testing.assert_allclose(
        pin.log3(halfway.rotation), [math.radians(30.0), 0.0, 0.0],
        atol=1e-12)


class _ThresholdKinematics:
    """Accept targets up to x=0.6 for deterministic boundary-search tests."""

    def solve(self, target, seed, nominal=None):
        del nominal
        x = float(target.translation[0])
        if x <= 0.6:
            return np.asarray(seed) + x, True, 3, np.zeros(6)
        error = np.array([x - 0.6, 0.0, 0.0, 0.0, 0.0, 0.0])
        return np.asarray(seed), False, 60, error


def test_unreachable_target_falls_back_to_nearest_reachable_path_point():
    start = pin.SE3.Identity()
    requested = pin.SE3(np.eye(3), np.array([1.0, 0.0, 0.0]))
    result = solve_with_boundary_fallback(
        _ThresholdKinematics(), requested, start, np.zeros(7), np.zeros(7),
        search_iterations=8)

    assert result.accepted
    assert result.status == "boundary"
    assert 0.59 <= result.fraction <= 0.60
    assert float(result.target.translation[0]) == result.fraction


def test_boundary_fallback_can_be_disabled():
    start = pin.SE3.Identity()
    requested = pin.SE3(np.eye(3), np.array([1.0, 0.0, 0.0]))
    result = solve_with_boundary_fallback(
        _ThresholdKinematics(), requested, start, np.zeros(7), np.zeros(7),
        enabled=False)

    assert not result.accepted
    assert result.status == "rejected"
    assert result.fraction == 0.0


def test_mapper_exposes_each_orientation_stage_for_diagnostics():
    home = pin.SE3.Identity()
    axes = np.array([[0.0, 0.0, -1.0],
                     [-1.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0]])
    mapper = OperatorPoseMapper(
        home, input_mode="relative", source_to_base=axes,
        position_scale=0.5)
    mapper.process([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
    angle = math.radians(30.0)
    mapper.process(
        [0.0, 0.1, 0.0],
        [math.sin(angle / 2.0), 0.0, 0.0, math.cos(angle / 2.0)])

    diagnostic = mapper.last_diagnostics
    np.testing.assert_allclose(diagnostic["relative_position"], [0.0, 0.1, 0.0])
    np.testing.assert_allclose(diagnostic["mapped_position"], [0.0, 0.0, 0.1])
    np.testing.assert_allclose(
        pin.log3(diagnostic["relative_rotation"]), [angle, 0.0, 0.0],
        atol=1e-12)
    np.testing.assert_allclose(
        pin.log3(diagnostic["mapped_rotation"]), [0.0, -angle, 0.0],
        atol=1e-12)


def test_real_rm75_safe_home_retreats_at_upper_workspace_boundary():
    kinematics = RM75Kinematics(URDF)
    safe_home_joints = np.array(
        [-math.pi / 2.0, 0.4, 0.0, 1.5, 0.0,
         -0.3292036732, math.radians(10.0)])
    safe_home = kinematics.forward(safe_home_joints)
    np.testing.assert_allclose(
        safe_home.rotation[:, 2], [0.0, -1.0, 0.0], atol=1e-5)
    requested = safe_home.copy()
    requested.translation[2] += 0.20

    result = solve_with_boundary_fallback(
        kinematics, requested, safe_home, safe_home_joints, safe_home_joints,
        search_iterations=7)

    assert result.accepted
    assert result.status == "boundary"
    applied_up = float(result.target.translation[2] - safe_home.translation[2])
    assert 0.10 < applied_up < 0.20


def test_elbow_assist_ramps_in_and_fails_back_to_zero_on_invalid_quality():
    controller = ElbowAssistController(
        mode="assist_low", configured_weight=0.02,
        timeout_sec=10.0, ramp_up_sec=2.0, ramp_down_sec=0.5)
    controller.receive_direction([0.0, -1.0, 0.0], now=0.0)
    controller.set_quality(True)
    assert controller.update(0.0)[1:] == (0.0, "RAMPING_IN")
    assert controller.update(1.0)[1] == pytest.approx(0.01)
    assert controller.update(2.0)[1:] == (pytest.approx(0.02), "ACTIVE")

    controller.set_quality(False)
    assert controller.update(2.25)[1] == pytest.approx(0.01)
    direction, weight, status = controller.update(2.5)
    assert direction is None
    assert weight == pytest.approx(0.0)
    assert status == "INVALID"


def test_observe_mode_never_generates_ik_weight():
    controller = ElbowAssistController(mode="observe")
    controller.receive_direction([0.0, -1.0, 0.0], now=0.0)
    controller.set_quality(True)
    direction, weight, status = controller.update(5.0)
    assert direction is None
    assert weight == 0.0
    assert status == "OBSERVE"
