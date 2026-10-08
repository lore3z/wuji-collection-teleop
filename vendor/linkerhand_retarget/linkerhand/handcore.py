from .config import *
from .yourdfpy import URDF
from .constants import RobotName,ROBOT_LEN,ROBOT_LEN_MAP
import threading
from pathlib import Path

class HandCore():
    def __init__(self, hand_config: HandConfig):
        handconfig = hand_config.handconfig
        self.baseconfig = hand_config.baseconfig
        self.retagetconfig = hand_config.retagetconfig
        self.modelconfig = hand_config.modelconfig
        robot_dir = hand_config.robot_dir
        targetpose = hand_config.targetpose
        self.robot_name_str_r = self.baseconfig["system"]["robotname_r"]
        self.robot_name_str_l = self.baseconfig["system"]["robotname_l"]

        self.righturdfpath = os.path.join(robot_dir, 'linker_hand', f'{self.robot_name_str_r}_right', f'linkerhand_{self.robot_name_str_r}_right.urdf')
        urdf_path = Path(self.righturdfpath)
        if not urdf_path.exists():
            raise ValueError(f"URDF path {urdf_path} does not exist")
        self.dataminvalue_r = change_list(handconfig[f'commandlower_right_{self.robot_name_str_r}'])
        self.datamaxvalue_r = change_list(handconfig[f'commandupper_right_{self.robot_name_str_r}'])
        self.sourcedataindex_r = change_list(handconfig[f'commandsourcedataindex_right_{self.robot_name_str_r}'])
        self.urdfdataindex_r = change_list(handconfig[f'urdfdataindex_right_{self.robot_name_str_r}'])
        self.RightHandId = URDF.load(self.righturdfpath)


        self.lefturdfpath = os.path.join(robot_dir, 'linker_hand', f'{self.robot_name_str_l}_left', f'linkerhand_{self.robot_name_str_l}_left.urdf')
        urdf_path = Path(self.lefturdfpath)
        if not urdf_path.exists():
            raise ValueError(f"URDF path {urdf_path} does not exist")
        self.dataminvalue_l = change_list(handconfig[f'commandlower_left_{self.robot_name_str_l}'])
        self.datamaxvalue_l = change_list(handconfig[f'commandupper_left_{self.robot_name_str_l}'])
        self.sourcedataindex_l = change_list(handconfig[f'commandsourcedataindex_left_{self.robot_name_str_l}'])
        self.urdfdataindex_l = change_list(handconfig[f'urdfdataindex_left_{self.robot_name_str_l}'])
        self.LeftHandId = URDF.load(self.lefturdfpath)

        self.hand_lower_limits_r, self.hand_upper_limits_r, self.hand_joint_ranges_r = self.get_joint_limits(
            self.RightHandId)
        self.hand_lower_limits_l, self.hand_upper_limits_l, self.hand_joint_ranges_l = self.get_joint_limits(
            self.LeftHandId)

        self.hand_numjoints_r = ROBOT_LEN_MAP[RobotName[self.robot_name_str_r]]
        self.hand_numjoints_l = ROBOT_LEN_MAP[RobotName[self.robot_name_str_l]]

        if "human" in self.baseconfig["humanset"]["targethandfile"]:
            self.right_hand_targetpose = np.array(targetpose['initial_positions']['right_hand'])
            self.left_hand_targetpose = np.array(targetpose['initial_positions']['left_hand'])
        else:
            self.right_hand_targetpose = np.array(targetpose['initial_positions']['right_hand'][f'{self.robot_name_str_r}'])
            self.left_hand_targetpose = np.array(targetpose['initial_positions']['left_hand'][f'{self.robot_name_str_l}'])
        self.debugcount = 0
        self.multi_target_kf_r = MultiTargetKalman(self.hand_numjoints_r)
        self.multi_target_kf_l = MultiTargetKalman(self.hand_numjoints_l)

        # 四元数测试用
        self.angle = 0
        self.counter = 0

        # 共享数据
        self.lock = threading.Lock()  # 线程锁
        self.right_joint_angles = []  # 存储关节角度等数据
        self.left_joint_angles = []   # 存储关节角度等数据
        self.last_qpos_r = None
        self.last_qpos_l = None

    @staticmethod
    def get_joint_limits(robot):
        joint_lower_limits = []
        joint_upper_limits = []
        joint_ranges = []
        # 遍历所有关节
        for joint_name, joint in robot.joint_map.items():
            # 跳过固定关节
            if joint.type == "fixed":
                continue
            # 获取关节限位值
            if joint.limit is not None:
                lower = joint.limit.lower
                upper = joint.limit.upper
            else:
                # 对于没有明确限位的关节，使用默认值
                # 连续旋转关节使用 ±π
                if joint.type == "revolute":
                    lower = -3.1415926535  # -180°
                    upper = 3.1415926535  # +180°
                # 平移关节使用 ±1m
                elif joint.type == "prismatic":
                    lower = -1.0
                    upper = 1.0
                # 其他类型关节使用 ±∞
                else:
                    lower = float('-inf')
                    upper = float('inf')
            # 添加到结果列表
            joint_lower_limits.append(lower)
            joint_upper_limits.append(upper)
            joint_ranges.append(upper - lower)

        return joint_lower_limits, joint_upper_limits, joint_ranges

    @staticmethod
    def projection_process(hand_position):
        qpos = [0.0] * 30
        cos_theta = 0
        # 处理拇指部分,占用5个数据位
        trumb_a = hand_position[0, :]  # 对应MATLAB的 position_rightHand(1,:)
        trumb_b = hand_position[1, :]
        trumb_c = hand_position[2, :]
        trumb_d = hand_position[3, :]  # 夹角顶点
        trumb_e = hand_position[4, :]
        # 拇指侧摆部分处理成YZ平面
        A_proj = np.array([trumb_a[1], trumb_a[2]])
        B_proj = np.array([trumb_b[1], trumb_b[2]])
        C_proj = np.array([trumb_c[1], trumb_c[2]])
        vec_BA_proj = A_proj - B_proj
        vec_BC_proj = C_proj - B_proj
        dot_product = np.dot(vec_BA_proj, vec_BC_proj)
        norm_AB = np.linalg.norm(vec_BA_proj)
        norm_BC = np.linalg.norm(vec_BC_proj)
        if norm_AB != 0 and norm_BC != 0:
            cos_theta = np.arccos(dot_product / (norm_AB * norm_BC))
        angle_deg = np.pi - cos_theta
        # 拇指旋转
        qpos[0] = angle_deg

        # 拇指侧摆部分处理成XY平面
        A_proj = trumb_a[:2]  # 提取[X, Y]
        B_proj = trumb_b[:2]
        C_proj = trumb_c[:2]
        vec_BA_proj = A_proj - B_proj
        vec_BC_proj = C_proj - B_proj
        dot_product = np.dot(vec_BA_proj, vec_BC_proj)
        norm_AB = np.linalg.norm(vec_BA_proj)
        norm_BC = np.linalg.norm(vec_BC_proj)
        if norm_AB != 0 and norm_BC != 0:
            cos_theta = np.arccos(dot_product / (norm_AB * norm_BC))
        angle_deg = np.pi - cos_theta
        # 拇指侧摆
        qpos[1] = angle_deg

        vecDC = trumb_c - trumb_d  # 对应MATLAB的 vecBA = C - D
        vecDE = trumb_e - trumb_d  # 对应MATLAB的 vecBC = E - D
        dot_product = np.dot(vecDC, vecDE)
        norm_AB = np.linalg.norm(vecDC)
        norm_BC = np.linalg.norm(vecDE)
        if norm_AB != 0 and norm_BC != 0:
            cos_theta = np.pi - np.arccos(dot_product / (norm_AB * norm_BC))
        # 拇指末端夹角
        qpos[4] = cos_theta

        vecBC = trumb_b - trumb_c  # 对应MATLAB的 vecBA = C - D
        vecDC = trumb_d - trumb_c  # 对应MATLAB的 vecBC = E - D
        dot_product = np.dot(vecDC, vecBC)
        norm_AB = np.linalg.norm(vecBC)
        norm_BC = np.linalg.norm(vecDC)
        if norm_AB != 0 and norm_BC != 0:
            cos_theta = np.pi - np.arccos(dot_product / (norm_AB * norm_BC))
        # 拇指中部夹角
        qpos[3] = cos_theta

        vecAB = trumb_a - trumb_b  # 对应MATLAB的 vecBA = C - D
        vecBC = trumb_c - trumb_b  # 对应MATLAB的 vecBC = E - D
        dot_product = np.dot(vecAB, vecBC)
        norm_AB = np.linalg.norm(vecAB)
        norm_BC = np.linalg.norm(vecBC)
        if norm_AB != 0 and norm_BC != 0:
            cos_theta = np.pi - np.arccos(dot_product / (norm_AB * norm_BC))
        # 拇指根部夹角
        qpos[2] = cos_theta

        # 处理四指部分,占用5个数据位
        # 食指侧摆部分处理成YZ平面
        for i in range(4):
            other_a = hand_position[5 + 5 * i, :]
            other_b = hand_position[6 + 5 * i, :]
            other_c = hand_position[7 + 5 * i, :]
            other_d = hand_position[8 + 5 * i, :]  # 夹角顶点
            other_e = hand_position[9 + 5 * i, :]

            A_proj = np.array([other_b[1], other_b[2] + 0.1])
            B_proj = np.array([other_b[1], other_b[2]])
            C_proj = np.array([other_c[1], other_b[2] + 0.1])
            vecAB = A_proj - B_proj
            vecBC = C_proj - B_proj
            dot_product = np.dot(vecAB, vecBC)
            norm_AB = np.linalg.norm(vecAB)
            norm_BC = np.linalg.norm(vecBC)
            if norm_AB != 0 and norm_BC != 0:
                cos_theta = np.arccos(dot_product / (norm_AB * norm_BC))
            # 侧摆
            qpos[5 + 5 * i] = cos_theta

            # 其余四指部分处理成XZ平面
            # 处理末端CDE3点
            A_proj = np.array([other_c[0], other_c[2]])
            B_proj = np.array([other_d[0], other_d[2]])
            C_proj = np.array([other_e[0], other_e[2]])
            vecAB = A_proj - B_proj
            vecBC = C_proj - B_proj
            dot_product = np.dot(vecAB, vecBC)
            norm_AB = np.linalg.norm(vecAB)
            norm_BC = np.linalg.norm(vecBC)
            if norm_AB != 0 and norm_BC != 0:
                cos_theta = np.pi - np.arccos(dot_product / (norm_AB * norm_BC))
            # 末端夹角
            qpos[9 + 5 * i] = cos_theta

            # 处理中部BCD3点
            A_proj = np.array([other_b[0], other_b[2]])
            B_proj = np.array([other_c[0], other_c[2]])
            C_proj = np.array([other_d[0], other_d[2]])
            vecAB = A_proj - B_proj
            vecBC = C_proj - B_proj
            dot_product = np.dot(vecAB, vecBC)
            norm_AB = np.linalg.norm(vecAB)
            norm_BC = np.linalg.norm(vecBC)
            cos_theta = np.pi - np.arccos(dot_product / (norm_AB * norm_BC))
            # 中部夹角
            qpos[8 + 5 * i] = cos_theta

            # 处理根部ABC3点
            A_proj = np.array([other_a[0], other_a[2]])
            B_proj = np.array([other_b[0], other_b[2]])
            C_proj = np.array([other_c[0], other_c[2]])
            vecAB = A_proj - B_proj
            vecBC = C_proj - B_proj
            dot_product = np.dot(vecAB, vecBC)
            norm_AB = np.linalg.norm(vecAB)
            norm_BC = np.linalg.norm(vecBC)
            if norm_AB != 0 and norm_BC != 0:
                cos_theta = np.pi - np.arccos(dot_product / (norm_AB * norm_BC))
            # 根部夹角
            qpos[7 + 5 * i] = cos_theta
        return qpos

    def trans_to_motor_left(self, temp_l):
        self.last_qpos_l = list(temp_l)
        jointpositions_l = [255.0] * self.hand_numjoints_l
        for i in range(self.hand_numjoints_l):
            if self.sourcedataindex_l[i] is not None:
                val_l = temp_l[self.sourcedataindex_l[i]]
                val_l = is_within_range(val_l,
                                        self.hand_lower_limits_l[self.urdfdataindex_l[i]],
                                        self.hand_upper_limits_l[self.urdfdataindex_l[i]])
                jointpositions_l[i] = int(scale_value(val_l,
                                                      self.hand_lower_limits_l[self.urdfdataindex_l[i]],
                                                      self.hand_upper_limits_l[self.urdfdataindex_l[i]],
                                                      self.dataminvalue_l[i],
                                                      self.datamaxvalue_l[i]))
        return jointpositions_l

    def trans_to_motor_right(self, temp_r):
        self.last_qpos_r = list(temp_r)
        jointpositions_r = [255.0] * self.hand_numjoints_r
        for i in range(self.hand_numjoints_r):
            if self.sourcedataindex_r[i] is not None:
                val_r = temp_r[self.sourcedataindex_r[i]]
                val_r = is_within_range(val_r,
                                        self.hand_lower_limits_r[self.urdfdataindex_r[i]],
                                        self.hand_upper_limits_r[self.urdfdataindex_r[i]])

                jointpositions_r[i] = int(scale_value(val_r,
                                                      self.hand_lower_limits_r[self.urdfdataindex_r[i]],
                                                      self.hand_upper_limits_r[self.urdfdataindex_r[i]],
                                                      self.dataminvalue_r[i],
                                                      self.datamaxvalue_r[i]))
        return jointpositions_r


    def generate_position(self, quaternion_r, quaternion_l):
        rootorin_correct_r = get_quaternion_relative(trans_wxyzori_to_xyzwori(quaternion_r[0]),
                                                     [0, 0, 0, 1])
        rootorin_correct_l = get_quaternion_relative(trans_wxyzori_to_xyzwori(quaternion_l[0]),
                                                     [0, 0, 0, 1])
        handorin_correct_r = []
        handorin_correct_l = []
        for i in range(20):
            handorin_correct_r.append(
                trans_xyzwori_to_wxyzori(get_child_quaternion(trans_wxyzori_to_xyzwori(quaternion_r[i]),
                                                              rootorin_correct_r)))
        for i in range(20):
            handorin_correct_l.append(
                trans_xyzwori_to_wxyzori(get_child_quaternion(trans_wxyzori_to_xyzwori(quaternion_l[i]),
                                                              rootorin_correct_l)))

        right_hand_pose = quat2handposition(handorin_correct_r, self.right_hand_targetpose)
        left_hand_pose = quat2handposition(handorin_correct_l, self.left_hand_targetpose)

        # 绕 Y 轴的旋转-90度
        Ry = rotate_matrix_y(np.radians(-90))
        right_hand_pose = np.dot(Ry, right_hand_pose.T).T
        # 先绕 Z 轴的旋转180度再绕Y轴旋转-90度
        Rz = rotate_matrix_z(np.radians(-180))
        left_hand_pose = np.dot(Rz, left_hand_pose.T).T
        Ry = rotate_matrix_y(np.radians(-90))
        left_hand_pose = np.dot(Ry, left_hand_pose.T).T
        return right_hand_pose, left_hand_pose

    def update_angles(self, rightangles, leftangle):
        with self.lock:
            self.right_joint_angles = rightangles
            self.left_joint_angles = leftangle

    def get_angles(self):
        with self.lock:
            return self.right_joint_angles.copy(), self.left_joint_angles.copy()

class KalmanFilter:
    def __init__(self, process_variance, measurement_variance, estimated_error, initial_value):
        self.process_variance = process_variance  # 过程噪声
        self.measurement_variance = measurement_variance  # 测量噪声
        self.estimated_error = estimated_error  # 初始估计误差
        self.current_estimate = initial_value  # 初始值

    def update(self, measurement):
        # 预测更新
        self.estimated_error += self.process_variance

        # 计算卡尔曼增益
        kalman_gain = self.estimated_error / (self.estimated_error + self.measurement_variance)

        # 更新估计值
        self.current_estimate += kalman_gain * (measurement - self.current_estimate)
        # 更新误差
        self.estimated_error *= (1 - kalman_gain)

        return self.current_estimate


class MultiTargetKalman:
    def __init__(self, num_targets, process_variance=0.01, measurement_variance=0.1, estimated_error=1,
                 initial_value=255):
        self.kalman_filters = [KalmanFilter(process_variance, measurement_variance, estimated_error, initial_value) for
                               _ in range(num_targets)]
        self.num_targets = num_targets
        self.smoothed_data = [[] for _ in range(num_targets)]

    def update(self, measurements, index):
        smooth_value = self.kalman_filters[index].update(measurements)
        return smooth_value
