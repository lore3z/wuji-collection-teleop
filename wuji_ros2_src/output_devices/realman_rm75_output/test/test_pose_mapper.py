import math

import numpy as np
import pytest

from realman_rm75_output.pose_mapper import InvalidPose, RelativePoseMapper


def quat_z(degrees):
    half = math.radians(degrees) / 2.0
    return np.array([0.0, 0.0, math.sin(half), math.cos(half)])


def quat_axis(axis, degrees):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    half = math.radians(degrees) / 2.0
    return np.r_[axis * math.sin(half), math.cos(half)]


def mapper(axis=None, clock=lambda: 0.0, max_jump=0.10,
           translation_only=False, freeze_timeout=0.75,
           rotation_only=False, rotation_scale=1.0):
    result = RelativePoseMapper(
        axis_mapping=np.eye(3) if axis is None else axis,
        position_scale=1.0,
        max_position_jump_m=max_jump,
        max_rotation_jump_rad=math.radians(45.0),
        tracker_timeout_sec=0.5,
        workspace_min=[-1.0, -1.0, -1.0],
        workspace_max=[1.0, 1.0, 1.0],
        clock=clock,
        translation_only=translation_only,
        freeze_timeout_sec=freeze_timeout,
        rotation_only=rotation_only,
        rotation_scale=rotation_scale,
    )
    result.set_rm_initial_pose([0.2, -0.1, 0.3], [0.0, 0.0, 0.0, 1.0])
    return result


def test_stationary_pico_keeps_rm_initial_pose():
    m = mapper()
    first = m.process([2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 1.0])
    second = m.process([2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(first.target_position, [0.2, -0.1, 0.3])
    np.testing.assert_allclose(second.target_quaternion_xyzw, [0, 0, 0, 1])


@pytest.mark.parametrize("source_axis,target_axis", [(0, 0), (1, 1), (2, 2)])
def test_one_centimeter_changes_only_mapped_axis(source_axis, target_axis):
    m = mapper()
    initial = np.array([0.0, 0.0, 0.0])
    m.process(initial, [0, 0, 0, 1])
    moved = initial.copy()
    moved[source_axis] = 0.01
    result = m.process(moved, [0, 0, 0, 1])
    expected = np.zeros(3)
    expected[target_axis] = 0.01
    np.testing.assert_allclose(result.mapped_translation, expected, atol=1e-12)


def test_rotation_does_not_change_position():
    m = mapper()
    m.process([0, 0, 0], [0, 0, 0, 1])
    result = m.process([0, 0, 0], quat_z(20))
    np.testing.assert_allclose(result.target_position, [0.2, -0.1, 0.3])
    assert not np.allclose(result.target_quaternion_xyzw, [0, 0, 0, 1])


def test_translation_only_locks_rm_initial_orientation():
    m = mapper(translation_only=True)
    m.process([0, 0, 0], [0, 0, 0, 1], received_at=0.0)
    result = m.process([0.01, 0, 0], quat_z(120), received_at=0.1)
    np.testing.assert_allclose(result.target_position, [0.21, -0.1, 0.3])
    np.testing.assert_allclose(result.target_quaternion_xyzw, [0, 0, 0, 1])


def test_rotation_only_locks_position_and_uniformly_scales_rotation():
    m = mapper(rotation_only=True, rotation_scale=0.2)
    m.process([0, 0, 0], [0, 0, 0, 1], received_at=0.0)
    qx_20 = np.array([math.sin(math.radians(10)), 0, 0,
                      math.cos(math.radians(10))])
    result = m.process([0.05, 0.02, -0.01], qx_20, received_at=0.1)
    np.testing.assert_allclose(result.target_position, [0.2, -0.1, 0.3])
    assert quaternion_angle(result.target_quaternion_xyzw) == pytest.approx(
        math.radians(4.0))


def quaternion_angle(q):
    q = np.asarray(q)
    return 2.0 * math.acos(abs(float(q[3])))


RM_FROM_WUJI = np.array([
    [0.0, 1.0, 0.0],
    [-1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
])


@pytest.mark.parametrize("source_axis", range(3))
def test_xyz_dry_run_uses_the_same_rigid_basis_mapping(source_axis):
    m = mapper(axis=RM_FROM_WUJI)
    origin = np.zeros(3)
    m.process(origin, [0, 0, 0, 1], received_at=0.0)
    moved = origin.copy()
    moved[source_axis] = 0.01
    result = m.process(moved, [0, 0, 0, 1], received_at=0.1)
    np.testing.assert_allclose(
        result.mapped_translation, 0.01 * RM_FROM_WUJI[:, source_axis],
        atol=1e-12)


@pytest.mark.parametrize("source_axis", range(3))
def test_roll_pitch_yaw_dry_run_uses_rigid_rotation_conjugation(source_axis):
    m = mapper(axis=RM_FROM_WUJI)
    m.process([0, 0, 0], [0, 0, 0, 1], received_at=0.0)
    source_vector = np.eye(3)[source_axis]
    result = m.process(
        [0, 0, 0], quat_axis(source_vector, 20.0), received_at=0.1)
    expected = quat_axis(RM_FROM_WUJI @ source_vector, 20.0)
    assert abs(float(np.dot(result.mapped_relative_rotation_xyzw, expected))) == \
        pytest.approx(1.0)


def test_reflection_is_rejected_as_a_coordinate_mapping():
    with pytest.raises(ValueError, match="determinant \\+1"):
        mapper(axis=np.diag([1.0, 1.0, -1.0]))


def test_translation_and_rotation_only_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        mapper(translation_only=True, rotation_only=True)


def test_q_and_negative_q_are_equivalent():
    q = quat_z(15)
    a = mapper()
    b = mapper()
    a.process([0, 0, 0], [0, 0, 0, 1])
    b.process([0, 0, 0], [0, 0, 0, 1])
    result_a = a.process([0, 0, 0], q)
    result_b = b.process([0, 0, 0], -q)
    assert abs(float(np.dot(result_a.target_quaternion_xyzw,
                            result_b.target_quaternion_xyzw))) == pytest.approx(1.0)


def test_abnormal_position_jump_is_rejected_without_state_update():
    m = mapper(max_jump=0.02)
    accepted = m.process([0, 0, 0], [0, 0, 0, 1], received_at=1.0)
    with pytest.raises(InvalidPose, match="position jump"):
        m.process([0.10, 0, 0], [0, 0, 0, 1], received_at=1.1)
    assert m.last_result is accepted
    assert m.last_message_time == 1.0


def test_reset_tracker_reference_accepts_relocalized_pose_as_new_baseline():
    m = mapper(max_jump=0.02)
    m.process([0, 0, 0], [0, 0, 0, 1], received_at=1.0)
    with pytest.raises(InvalidPose, match="position jump"):
        m.process([0.10, 0, 0], [0, 0, 0, 1], received_at=1.1)
    m.reset_tracker_reference()
    result = m.process([0.10, 0, 0], [0, 0, 0, 1], received_at=1.2)
    np.testing.assert_allclose(result.target_position, [0.2, -0.1, 0.3])


def test_abnormal_rotation_jump_is_rejected():
    m = mapper()
    m.process([0, 0, 0], [0, 0, 0, 1])
    with pytest.raises(InvalidPose, match="rotation jump"):
        m.process([0, 0, 0], quat_z(60))


def test_tracker_timeout_stops_target_updates():
    now = [10.0]
    m = mapper(clock=lambda: now[0])
    accepted = m.process([0, 0, 0], [0, 0, 0, 1])
    now[0] = 10.51
    assert m.tracker_timed_out()
    assert m.last_result is accepted


def test_identical_pose_stream_is_rejected_as_frozen():
    m = mapper(freeze_timeout=0.5)
    m.process([0, 0, 0], [0, 0, 0, 1], received_at=1.0)
    m.process([0, 0, 0], [0, 0, 0, 1], received_at=1.4)
    with pytest.raises(InvalidPose, match="pose frozen"):
        m.process([0, 0, 0], [0, 0, 0, -1], received_at=1.51)


def test_pose_change_resets_freeze_timer():
    m = mapper(freeze_timeout=0.5)
    m.process([0, 0, 0], [0, 0, 0, 1], received_at=1.0)
    m.process([0.001, 0, 0], [0, 0, 0, 1], received_at=1.4)
    m.process([0.001, 0, 0], [0, 0, 0, 1], received_at=1.8)


def test_non_finite_input_is_rejected():
    m = mapper()
    with pytest.raises(InvalidPose, match="finite"):
        m.process([float("nan"), 0, 0], [0, 0, 0, 1])
