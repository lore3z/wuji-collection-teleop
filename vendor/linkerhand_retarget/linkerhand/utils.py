import enum
import copy
import yaml
import math
import os
import numpy as np
from transforms3d.quaternions import axangle2quat, qmult
from transforms3d.quaternions import mat2quat
from transforms3d.euler import mat2euler, quat2mat, euler2mat
from scipy.spatial.transform import Rotation as R


class DataSource(enum.Enum):
    motion = enum.auto()
    video = enum.auto()
    vr = enum.auto()


def read_yaml(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def extract_dataset_folder_last_two_digits(dir_name):
    # Extract the last two characters, ensure they are digits, and convert to integer
    last_two = dir_name[-2:]  # Get the last two characters
    if last_two.isdigit():
        return int(last_two)
    else:
        return -1  # Return -1 (or some other value) if there are no digits


def _back_project_batch(points, intrinsics):
    """ Back-project a batch of points from 3D to 2D image space using vectorized operations """
    points = np.array(points)

    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    u = (x * fx / z) + cx
    v = (y * fy / z) + cy

    projected_points = np.vstack((u, v)).T.astype(int)

    # return projected_points


def translate_wrist_to_origin(joint_positions):
    wrist_position = joint_positions[0]
    updated_positions = joint_positions - wrist_position
    return updated_positions


def apply_pose_matrix(joint_positions, pose_matrix):
    homogeneous_joint_positions = np.hstack([joint_positions, np.ones((joint_positions.shape[0], 1))])
    transformed_positions = np.dot(homogeneous_joint_positions, pose_matrix.T)
    transformed_positions_3d = transformed_positions[:, :3]
    return transformed_positions_3d


def inverse_transformation(matrix):
    # Assuming matrix is a 4x4 numpy array
    R = matrix[:3, :3]
    T = matrix[:3, 3]

    R_inv = np.linalg.inv(R)
    T_inv = -np.dot(R_inv, T)

    inverse_matrix = np.eye(4)  # Create a 4x4 identity matrix
    inverse_matrix[:3, :3] = R_inv
    inverse_matrix[:3, 3] = T_inv

    return inverse_matrix


def update_R_delta_init(frame_0_eef_pos, frame_0_eef_quat):
    global R_delta_init

    frame_0_pose = np.eye(4)
    frame_0_pose[:3, :3] = quat2mat(frame_0_eef_quat)
    frame_0_pose[:3, 3] = frame_0_eef_pos

    pose_ori_matirx = frame_0_pose[:3, :3]
    pose_ori_correction_matrix = np.dot(np.array([[0, -1, 0],
                                                  [0, 0, 1],
                                                  [1, 0, 0]]), euler2mat(0, 0, 0))
    pose_ori_matirx = np.dot(pose_ori_matirx, pose_ori_correction_matrix)

    canonical_t265_ori = np.array([[1, 0, 0],
                                   [0, -1, 0],
                                   [0, 0, -1]])
    x_angle, y_angle, z_angle = mat2euler(frame_0_pose[:3, :3])
    canonical_t265_ori = np.dot(canonical_t265_ori, euler2mat(-z_angle, x_angle + 0.3, y_angle))

    R_delta_init = np.dot(canonical_t265_ori, pose_ori_matirx.T)


def switch_axis(quaternion_xyzw, i, j):
    q1 = np.array(
        [quaternion_xyzw[3], quaternion_xyzw[0], quaternion_xyzw[1], quaternion_xyzw[2]]
    )
    rot_mat = quat2mat(q1)
    rot_mat_copy = copy.deepcopy(rot_mat)
    rot_mat[i] = rot_mat_copy[j]
    rot_mat[j] = rot_mat_copy[i]
    import pdb

    pdb.set_trace()
    q2 = mat2quat(rot_mat)
    q3 = np.array([q2[1], q2[2], q2[3], q2[0]])
    return q3


def swap_quaternion_axes(quaternion, axis1, axis2):
    """
    Swap two axes in a quaternion without converting to Euler angles.

    Args:
        quaternion (list or np.ndarray): The input quaternion [x, y, z, w].
        axis1 (int): The index of the first axis to swap (0 for X, 1 for Y, 2 for Z).
        axis2 (int): The index of the second axis to swap (0 for X, 1 for Y, 2 for Z).

    Returns:
        np.ndarray: The new quaternion with swapped axes [x', y', z', w'].
    """
    if axis1 < 0 or axis1 > 2 or axis2 < 0 or axis2 > 2:
        raise ValueError("Axis indices must be 0, 1, or 2.")

    # Create a copy of the input quaternion
    new_quaternion = quaternion.copy()

    # Swap the elements corresponding to the specified axes
    new_quaternion[axis1], new_quaternion[axis2] = quaternion[axis2], quaternion[axis1]

    return new_quaternion


def trans_xyzwori_to_wxyzori(ori):
    return ((ori[3], ori[0], ori[1], ori[2]))


def trans_wxyzori_to_xyzwori(ori):
    return ((ori[1], ori[2], ori[3], ori[0]))


def scale_value(original_value, a_min, a_max, b_min, b_max):
    return (original_value - a_min) * (b_max - b_min) / (a_max - a_min) + b_min


def is_within_range(value, min_value, max_value):
    return min(max_value, max(min_value, value))


def extend_line(point1, point2, distance):
    vector = np.array(point2) - np.array(point1)
    vector_magnitude = np.linalg.norm(vector)
    unit_vector = vector / vector_magnitude
    extended_point = point2 + unit_vector * distance
    return extended_point


def poseture_to_matrix(position, ori):
    matrix = np.eye(4)
    rotation_matrix = quat2mat(ori)
    matrix[:3, :3] = rotation_matrix
    matrix[:3, 3] = position
    return matrix


def make_reference_matrix(position, ori, lenpose, lenori):
    rotated_quaternion_wxyz = np.array([ori[3], ori[0], ori[1], ori[2]])
    rotated_quaternion_wxyz_len = np.array([lenori[3], lenori[0], lenori[1], lenori[2]])
    base_matrix = poseture_to_matrix(position, rotated_quaternion_wxyz)
    adder_matrix = poseture_to_matrix(lenpose, rotated_quaternion_wxyz_len)
    return base_matrix @ adder_matrix


def change_orientation(ori):
    return np.array([ori[3], ori[0], ori[1], ori[2]])


def cal_distance(point_s, point_t):
    vector = [point_s[i] - point_t[i] for i in range(3)]
    vector_magnitude = math.sqrt(sum(x ** 2 for x in vector))
    return vector_magnitude


def exponential_growth(minvalue, maxvalue, growth_factor, num_points):
    valuelist = np.linspace(minvalue, maxvalue, num_points)
    x = valuelist.astype(int)
    y = num_points - np.exp(growth_factor * x)
    min_y = np.min(y)
    y -= min_y
    y = y / np.max(y) * maxvalue
    lookup_table = dict(zip(x, y))
    return lookup_table


def change_list(q):
    converted_list = [None if value == 'None' else value for value in q]
    return converted_list


def quaternion_conjugate(q):
    x, y, z, w = q
    return np.array([-x, -y, -z, w])


def quaternion_norm_squared(q):
    return np.dot(q, q)


def quaternion_inverse(q):
    q_conjugate = quaternion_conjugate(q)
    norm_sq = quaternion_norm_squared(q)
    return q_conjugate / norm_sq


def quaternion_multiply(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,  # x
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,  # y
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,  # z
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2  # w
    ])


def exponential_growth_fun(x_values, c, a_min, a_max):
    # 计算 k，使得 a(1) = a_max
    k = a_max / (np.exp(c) - 1)

    # 计算指数增长值
    a_values = k * (np.exp(c * x_values) - 1)

    # 将 a_values 映射到 [a_min, a_max] 范围
    a_min_original = 0  # 原有公式的最小值
    a_max_original = k * (np.exp(c) - 1)  # 原有公式的最大值
    a_values_mapped = a_min + (a_values - a_min_original) * (a_max - a_min) / (a_max_original - a_min_original)

    return a_values_mapped


def quaternion_matrixinv(q):
    qw, qx, qy, qz = q
    rotation_matrix = np.array([[2 * qw ** 2 + 2 * qx ** 2 - 1, 2 * qx * qy - 2 * qw * qz, 2 * qw * qy + 2 * qx * qz],
                                [2 * qw * qz + 2 * qx * qy, 2 * qw ** 2 + 2 * qy ** 2 - 1, 2 * qy * qz - 2 * qw * qx],
                                [2 * qx * qz - 2 * qw * qy, 2 * qw * qx + 2 * qy * qz, 2 * qw ** 2 + 2 * qz ** 2 - 1]])

    return np.linalg.inv(rotation_matrix)


def unitydata_to_worldspacedata(initial_positions):
    new_positions = []
    for position in initial_positions:
        new_positions.append([position[0], position[2], position[1]])
    return new_positions


def get_quaternion_relative(ori, targetori):
    q_target = R.from_quat(targetori)
    q_parent = R.from_quat(ori)
    q_parent_inv = q_parent.inv().as_quat() * -1
    q_parent_inv = R.from_quat(q_parent_inv)
    q_relative = q_parent_inv * q_target
    return q_relative.as_quat()


def get_child_quaternion(ori, ori_relative):
    q_child = R.from_quat(ori)
    q_relative = R.from_quat(ori_relative)
    q_result = q_relative * q_child
    return q_result.as_quat()


def quat2handposition(quat, bone):
    root = bone[0]
    fn = np.array([0, 0, 1, 2, 3, 0, 5, 6, 7, 8, 0, 10, 11, 12, 13, 0, 15, 16, 17, 18, 0, 20, 21, 22, 23])
    orin = np.array([0, 0, 1, 2, 3, 0, 4, 5, 6, 7, 0, 8, 9, 10, 11, 0, 12, 13, 14, 15, 0, 16, 17, 18, 19])
    boneVer = bone[:25] - bone[fn]
    pos = np.ones((bone.shape[0], 1)) * root
    for i in range(1, 25):
        qt = quat[orin[i]]
        pos[i] = boneVer[i] @ quaternion_matrixinv(qt) + pos[fn[i]]
    return pos


def rotate_matrix_x(radians):
    return np.array([
        [1, 0, 0],
        [0, np.cos(radians), -np.sin(radians)],
        [0, np.sin(radians), np.cos(radians)]
    ])


def rotate_matrix_y(radians):
    return np.array([
        [np.cos(radians), 0, np.sin(radians)],
        [0, 1, 0],
        [-np.sin(radians), 0, np.cos(radians)]
    ])


def rotate_matrix_z(radians):
    return np.array([
        [np.cos(radians), -np.sin(radians), 0],
        [np.sin(radians), np.cos(radians), 0],
        [0, 0, 1]
    ])


def rotate_quaternion(original_quat, roll, pitch, yaw):
    """应用绕X, Y, Z轴的旋转到原始四元数。

    参数:
    original_quat (array_like): 原始四元数 [x, y, z, w] 格式。
    roll (float): 绕X轴旋转的角度（度）。
    pitch (float): 绕Y轴旋转的角度（度）。
    yaw (float): 绕Z轴旋转的角度（度）。

    返回:
    np.ndarray: 旋转后的四元数 [x, y, z, w]
    """
    # 原始四元数转换为旋转对象
    original_rotation = R.from_quat(original_quat)

    # 将角度转换为弧度
    roll_rad = np.radians(roll)
    pitch_rad = np.radians(pitch)
    yaw_rad = np.radians(yaw)

    # 创建旋转对象，从给定的欧拉角创建一个新的旋转对象
    rotation = R.from_euler('xyz', [roll_rad, pitch_rad, yaw_rad])

    # 组合旋转，先应用原始旋转，再应用新旋转
    new_rotation = original_rotation * rotation

    # 返回结果四元数，转换为 [x, y, z, w] 形式
    return new_rotation.as_quat()


def cubic_model(x, a, b, c, d):
    """Cubic model for curve fitting"""
    return a * x ** 3 + b * x ** 2 + c * x + d
