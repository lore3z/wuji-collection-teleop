import math
from pathlib import Path

import numpy as np
import pinocchio as pin
import pytest

from realman_rm75_output.offline_trajectory import (
    DEFAULT_HOME_JOINTS_RAD,
    RM75Kinematics,
    elbow_direction_error_rad,
    elbow_nullspace_velocity,
    singularity_adaptive_damping,
    target_pose_at_phase,
)


URDF = (Path(__file__).resolve().parents[3] /
        "rm_description" / "urdf" / "rm_75.urdf")


def test_adaptive_damping_preserves_legacy_value_outside_singularity():
    assert singularity_adaptive_damping(0.0779) == pytest.approx(1e-5)
    assert singularity_adaptive_damping(0.03) == pytest.approx(1e-5)
    assert 1e-5 < singularity_adaptive_damping(0.015) < 0.08 ** 2
    assert singularity_adaptive_damping(0.0) == pytest.approx(0.08 ** 2)


def test_adaptive_mode_matches_legacy_solver_in_normal_workspace():
    legacy = RM75Kinematics(URDF)
    adaptive = RM75Kinematics(URDF)
    joints = np.array([
        -math.pi / 2.0, 0.4, 0.0, 1.5, 0.0,
        -0.3292036732, math.radians(10.0)])
    target = legacy.forward(joints)
    target.translation[0] += 0.005

    legacy_joints, legacy_ok, _, legacy_error = legacy.solve(
        target, joints, nominal=joints)
    adaptive_joints, adaptive_ok, _, adaptive_error = adaptive.solve(
        target, joints, nominal=joints,
        adaptive_singularity_damping=True)

    assert legacy_ok, legacy_error
    assert adaptive_ok, adaptive_error
    assert not adaptive.adaptive_damping_was_active
    assert adaptive.last_minimum_singular_value >= 0.03
    np.testing.assert_allclose(adaptive_joints, legacy_joints, atol=1e-12)


def test_cartesian_path_is_closed_and_nontrivial():
    kinematics = RM75Kinematics(URDF)
    home = kinematics.forward(DEFAULT_HOME_JOINTS_RAD)
    start = target_pose_at_phase(home, 0.0, 0.08, 0.06, 0.04)
    middle = target_pose_at_phase(home, math.pi, 0.08, 0.06, 0.04)
    finish = target_pose_at_phase(home, 2.0 * math.pi, 0.08, 0.06, 0.04)

    np.testing.assert_allclose(start.translation, home.translation, atol=1e-12)
    np.testing.assert_allclose(finish.translation, home.translation, atol=1e-12)
    assert np.linalg.norm(middle.translation - home.translation) > 0.1
    np.testing.assert_allclose(start.rotation, finish.rotation, atol=1e-12)


def test_rm75_ik_tracks_one_complete_path_without_joint_drift():
    kinematics = RM75Kinematics(URDF)
    home_joints = DEFAULT_HOME_JOINTS_RAD.copy()
    home = kinematics.forward(home_joints)
    joints = home_joints.copy()
    maximum_position_error = 0.0
    maximum_rotation_error = 0.0

    for index in range(1, 181):
        phase = 2.0 * math.pi * index / 180
        target = target_pose_at_phase(home, phase, 0.08, 0.06, 0.04)
        joints, accepted, _iterations, error = kinematics.solve(
            target, joints, nominal=home_joints)
        assert accepted, f"IK failed at sample {index}: {error}"
        actual = kinematics.forward(joints)
        residual = actual.actInv(target)
        residual_vector = pin.log6(residual).vector
        maximum_position_error = max(
            maximum_position_error,
            float(np.linalg.norm(residual_vector[:3])))
        maximum_rotation_error = max(
            maximum_rotation_error,
            float(np.linalg.norm(residual_vector[3:])))

    assert maximum_position_error < 3e-5
    assert maximum_rotation_error < 3e-4
    assert np.max(np.abs(joints - home_joints)) < 1e-3


def test_signed_elbow_error_and_nullspace_velocity_follow_requested_swivel():
    axis = np.array([1.0, 0.0, 0.0])
    current = np.array([0.0, -1.0, 0.0])
    desired = np.array([0.0, 0.0, -1.0])
    assert math.degrees(elbow_direction_error_rad(
        axis, current, desired)) == pytest.approx(90.0)
    jacobian = np.zeros((3, 7))
    jacobian[2, 0] = -0.2
    velocity, clipped_error = elbow_nullspace_velocity(
        jacobian, np.eye(7), axis, current, desired,
        radius_m=0.2, weight=0.02,
        max_error_rad=math.radians(10.0))
    assert math.degrees(clipped_error) == pytest.approx(10.0)
    assert velocity[0] > 0.0
    np.testing.assert_allclose(velocity[1:], 0.0)


def test_low_elbow_assist_reduces_swivel_error_without_losing_tcp_pose():
    kinematics = RM75Kinematics(URDF)
    joints = DEFAULT_HOME_JOINTS_RAD.copy()
    target = kinematics.forward(joints)
    _, _, _, axis, current, _, _ = kinematics.elbow_geometry(joints)
    desired = pin.exp3(math.radians(8.0) * axis) @ current
    initial_error = abs(elbow_direction_error_rad(axis, current, desired))

    solved, accepted, _iterations, residual = kinematics.solve(
        target, joints, nominal=joints,
        elbow_direction=desired, elbow_weight=0.02,
        max_elbow_error_rad=math.radians(10.0))
    assert accepted, residual
    _, _, _, final_axis, final_direction, _, _ = (
        kinematics.elbow_geometry(solved))
    final_error = abs(elbow_direction_error_rad(
        final_axis, final_direction, desired))
    assert final_error < initial_error
    actual = kinematics.forward(solved)
    tcp_error = pin.log6(actual.actInv(target)).vector
    assert np.linalg.norm(tcp_error[:3]) < 3e-5
    assert np.linalg.norm(tcp_error[3:]) < 3e-4
