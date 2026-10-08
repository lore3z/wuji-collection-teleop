"""
LinkerForce O30i 手型映射模块 - ROS2版本
支持基于标定数据的精确映射
v2.8.0升级了映射器算法
"""
import numpy as np
import copy
import xml.etree.ElementTree as ET
from pathlib import Path
from linkerhand.handcore import HandCore
from ..config.o30i_config import FINGER_CONFIGS, MAPPING_ORDER, ROBOT_OPOSE_RIGHT, ROBOT_OPOSE_LEFT, ROBOT_ORIGINAL_LEFT, ROBOT_ORIGINAL_RIGHT, ROBOT_FIST_LEFT, ROBOT_FIST_RIGHT, MULTI_SEGMENT_CONFIG_FROZEN, MOTOR_CONSTRAINTS
from linkerhand.handcoreex import DynamicWeightMultiStateLinearMapper

O30I_MUJOCO_JOINT_ARC_INDICES = (
    0, 1, 2, 3,
    4, 5, 6, 6,
    8, 9, 10, 10,
    12, 13, 14, 14,
    16, 17, 18, 18,
)
O30I_MUJOCO_JOINT_ARC_SIGNS = (1.0,) * len(O30I_MUJOCO_JOINT_ARC_INDICES)

O30I_TOPIC_JOINT_NAMES = (
    "拇指横滚",
    "拇指侧摆",
    "食指侧摆",
    "中指侧摆",
    "无名指侧摆",
    "小指侧摆",
    "拇指指根",
    "食指指根",
    "中指指根",
    "无名指指根",
    "小指指根",
    "食指指中",
    "中指指中",
    "无名指指中",
    "小指指中",
    "拇指指尖",
    "食指指尖",
    "中指指尖",
    "无名指指尖",
    "小指指尖",
)

O30I_MOTOR_QPOS_INDICES = (
    16, 17, 0, 8, 12, 4,
    18, 1, 9, 13, 5,
    2, 10, 14, 6,
    19, 2, 10, 14, 6,
)

O30I_MOTOR_ROBOT_INDICES = (
    0, 1, 4, 8, 12, 16,
    2, 5, 9, 13, 17,
    6, 10, 14, 18,
    3, 7, 11, 15, 19,
)
O30I_MOTOR_ARC_INDICES = O30I_MOTOR_ROBOT_INDICES
O30I_REVERSED_MOTOR_ROBOT_INDICES = {0, 4, 8, 12, 16}


def _clamp(value, lower, upper):
    return min(upper, max(lower, value))


def _scale_value(value, source_min, source_max, target_min, target_max):
    if abs(source_max - source_min) < 1e-9:
        return float(target_max)
    ratio = (value - source_min) / (source_max - source_min)
    ratio = max(0.0, min(1.0, ratio))
    return float(target_min + ratio * (target_max - target_min))


def _load_urdf_joint_limits(hand: str):
    package_dir = Path(__file__).resolve().parents[3]
    urdf_path = (
        package_dir
        / "assets"
        / "robots"
        / "hands"
        / "linker_hand"
        / f"o30i_{hand}"
        / f"linkerhand_o30i_{hand}.urdf"
    )
    root = ET.parse(urdf_path).getroot()
    limits = []
    for joint in root.findall("joint"):
        if joint.attrib.get("type") == "fixed":
            continue
        limit = joint.find("limit")
        if limit is None:
            limits.append((float("-inf"), float("inf")))
            continue
        limits.append((float(limit.attrib["lower"]), float(limit.attrib["upper"])))
    return tuple(limits)


O30I_URDF_JOINT_LIMITS_RIGHT = _load_urdf_joint_limits("right")
O30I_URDF_JOINT_LIMITS_LEFT = _load_urdf_joint_limits("left")


def _map_o30i_qpos_to_motor(qpos, hand: str):
    joint_limits = O30I_URDF_JOINT_LIMITS_LEFT if hand == "left" else O30I_URDF_JOINT_LIMITS_RIGHT
    jointpositions = [255.0] * len(O30I_MOTOR_QPOS_INDICES)
    for index, (source_idx, robot_idx) in enumerate(zip(O30I_MOTOR_QPOS_INDICES, O30I_MOTOR_ROBOT_INDICES)):
        if source_idx is None or robot_idx is None:
            continue
        if source_idx >= len(qpos) or robot_idx >= len(joint_limits):
            continue
        lower, upper = joint_limits[robot_idx]
        value = _clamp(qpos[source_idx], lower, upper)
        target_min, target_max = (255, 0) if robot_idx in O30I_REVERSED_MOTOR_ROBOT_INDICES else (0, 255)
        jointpositions[index] = int(round(_scale_value(value, lower, upper, target_min, target_max)))
    return jointpositions


def _map_o30i_arc_to_motor(arc_values, hand: str):
    joint_limits = O30I_URDF_JOINT_LIMITS_LEFT if hand == "left" else O30I_URDF_JOINT_LIMITS_RIGHT
    jointpositions = [255.0] * len(O30I_MOTOR_ARC_INDICES)
    for index, arc_idx in enumerate(O30I_MOTOR_ARC_INDICES):
        if arc_idx is None:
            continue
        if arc_idx >= len(arc_values) or arc_idx >= len(joint_limits):
            continue
        lower, upper = joint_limits[arc_idx]
        value = _clamp(arc_values[arc_idx], lower, upper)
        target_min, target_max = (255, 0) if arc_idx in O30I_REVERSED_MOTOR_ROBOT_INDICES else (0, 255)
        jointpositions[index] = int(round(_scale_value(value, lower, upper, target_min, target_max)))
    return jointpositions

def _build_o30i_right_mujoco_joint_arc_remaps(robot_original, robot_opose, robot_fist):
    _ = (robot_original, robot_opose, robot_fist)
    return (None,) * len(O30I_MUJOCO_JOINT_ARC_INDICES)


def _apply_o30i_arc_values(qpos, arc_storage, arc_value):
    qpos[16] = arc_storage[0] = arc_value[0]
    qpos[17] = arc_storage[1] = arc_value[1]
    qpos[18] = arc_storage[2] = arc_value[2]
    qpos[19] = arc_storage[3] = arc_value[3]

    qpos[0] = arc_storage[4] = arc_value[4]
    qpos[1] = arc_storage[5] = arc_value[5]
    qpos[2] = arc_storage[6] = arc_value[6]
    arc_storage[7] = arc_value[6]

    qpos[8] = arc_storage[8] = arc_value[8]
    qpos[9] = arc_storage[9] = arc_value[9]
    qpos[10] = arc_storage[10] = arc_value[10]
    arc_storage[11] = arc_value[10]

    qpos[12] = arc_storage[12] = arc_value[12]
    qpos[13] = arc_storage[13] = arc_value[13]
    qpos[14] = arc_storage[14] = arc_value[14]
    arc_storage[15] = arc_value[14]

    qpos[4] = arc_storage[16] = arc_value[16]
    qpos[5] = arc_storage[17] = arc_value[17]
    qpos[6] = arc_storage[18] = arc_value[18]
    arc_storage[19] = arc_value[18]


def _map_piecewise_linear(value, source_open, source_opose, source_fist,
                          target_open, target_opose, target_fist):
    def interpolate(x0, y0, x1, y1):
        if abs(x1 - x0) < 1e-9:
            return y1
        ratio = (value - x0) / (x1 - x0)
        ratio = max(0.0, min(1.0, ratio))
        return y0 + ratio * (y1 - y0)

    open_to_opose = (value - source_open) * (value - source_opose) <= 0
    opose_to_fist = (value - source_opose) * (value - source_fist) <= 0

    if open_to_opose:
        return interpolate(source_open, target_open, source_opose, target_opose)
    if opose_to_fist:
        return interpolate(source_opose, target_opose, source_fist, target_fist)

    anchors = (
        (source_open, target_open),
        (source_opose, target_opose),
        (source_fist, target_fist),
    )
    return min(anchors, key=lambda anchor: abs(value - anchor[0]))[1]


def _interpolate_clamped(value, source_start, source_end, target_start, target_end):
    if abs(source_end - source_start) < 1e-9:
        return float(target_end)

    ratio = (value - source_start) / (source_end - source_start)
    ratio = max(0.0, min(1.0, ratio))
    return float(target_start + ratio * (target_end - target_start))


def _matches_calibration_pose(joint_arc, calibration_pose) -> bool:
    if calibration_pose is None or len(joint_arc) != len(calibration_pose):
        return False
    return bool(np.allclose(joint_arc, calibration_pose, rtol=0.0, atol=1e-9))


def _resolve_version_config(configs: dict, version: str) -> dict:
    """
    解析版本配置，将字典格式的 weights/reverse_motion 转换为具体值
    """
    resolved = copy.deepcopy(configs)
    for finger_name, config in resolved.items():
        if 'weights' in config and isinstance(config['weights'], dict):
            config['weights'] = config['weights'].get(version, config['weights'].get('v2', [0.5, 0, 0.5]))
        if 'reverse_motion' in config and isinstance(config['reverse_motion'], dict):
            config['reverse_motion'] = config['reverse_motion'].get(version, config['reverse_motion'].get('v2', False))
    return resolved


class O30iStableDynamicWeightMultiStateLinearMapper(DynamicWeightMultiStateLinearMapper):
    def __init__(
        self,
        finger_configs,
        mapping_order,
        is_debug: bool = False,
        thumb_output_alpha: float = 0.35,
        thumb_output_max_step: float = 0.12,
        thumb_output_deadband: float = 0.02,
    ):
        super().__init__(finger_configs, mapping_order, is_debug=is_debug)
        self.thumb_output_alpha = thumb_output_alpha
        self.thumb_output_max_step = thumb_output_max_step
        self.thumb_output_deadband = thumb_output_deadband
        self._thumb_output_state = None

    def reset_thumb_filter(self, current=None):
        self._thumb_output_state = None if current is None else np.array(current, dtype=float)

    def map_glove_to_robot(self, source_current):
        source_current_array = np.array(source_current, dtype=float)
        for state_name, glove_state in self.glove_states.items():
            if np.allclose(source_current_array, glove_state, rtol=0.0, atol=1e-9):
                robot_angles = self.robot_states[state_name].copy()
                self.reset_thumb_filter(robot_angles)
                return robot_angles

        robot_angles = np.array(super().map_glove_to_robot(source_current), dtype=float)
        if self._thumb_output_state is None or len(self._thumb_output_state) != len(robot_angles):
            self._thumb_output_state = robot_angles.copy()
            return robot_angles

        for robot_idx in (0, 1):
            target = float(robot_angles[robot_idx])
            previous = float(self._thumb_output_state[robot_idx])
            delta = target - previous
            if abs(delta) <= self.thumb_output_deadband:
                filtered = previous
            else:
                step = np.clip(
                    delta * self.thumb_output_alpha,
                    -self.thumb_output_max_step,
                    self.thumb_output_max_step,
                )
                filtered = previous + float(step)
            robot_angles[robot_idx] = filtered
            self._thumb_output_state[robot_idx] = filtered

        return robot_angles

class RightHand:
    def __init__(self, handcore: HandCore, length=20, is_debug: bool = False):
        self.handcore = handcore
        self.hand_side = "right"
        self.g_jointpositions = [255] * length
        self.topic_joint_names = O30I_TOPIC_JOINT_NAMES
        self.g_jointvelocity = [255] * length
        self.last_jointpositions = [255] * length
        self.last_jointvelocity = [255] * length
        self.g_jointpositions_arc = [0] * length
        self.mujoco_joint_arc_indices = O30I_MUJOCO_JOINT_ARC_INDICES
        self.mujoco_joint_arc_signs = O30I_MUJOCO_JOINT_ARC_SIGNS
        self.g_jointvelocity_arc = [0] * length
        self.handstate = [0] * length
        self.calibrationoriginal = None    # 五指张开标定值 (对应255)
        self.calibrationfistpose = None    # 握拳标定值 (对应0)
        self.calibrationopose = None       # O型标定值 (对应中间值)
        self.glove_version = 'v2'
        
        # ========== 平滑滤波参数 ==========
        self.smooth_enabled = True
        self.smooth_alpha = 0.5  # 平滑系数：越小越平滑，范围 0.05-0.3
        self.smooth_positions = [255.0] * length  # 平滑后的位置（浮点）
        self.max_step = 20  # 每帧最大变化量，防止跳变

        # 目标机械手预设姿势，数值从URDF获取数据集，
        # 张开手的时候对应最小角度，
        # 握拳的时候对应最大角度
        # O型手势的时候，用工具驱动URDF去驱动目标机械手达到期望姿势，也可以调整这些参数使得实物更加达到期望角度
        # 其他手势也类似，也可以增加多个手势来实现多模态的映射器（后期陆续开发）
        self.robot_original = ROBOT_ORIGINAL_RIGHT
        self.robot_opose = ROBOT_OPOSE_RIGHT
        self.robot_fist = ROBOT_FIST_RIGHT
        self.mujoco_joint_arc_remaps = _build_o30i_right_mujoco_joint_arc_remaps(
            self.robot_original,
            self.robot_opose,
            self.robot_fist,
        )
        self.mujoco_joint_arc_mirrors = (None,) * len(O30I_MUJOCO_JOINT_ARC_INDICES)

        finger_configs = _resolve_version_config(FINGER_CONFIGS, self.glove_version)
        self.multi_state_mapper = O30iStableDynamicWeightMultiStateLinearMapper(
            finger_configs,
            MAPPING_ORDER,
            is_debug=is_debug,
        )

        for config_name, config in FINGER_CONFIGS.items():
            if config.get('dynamic_weight'):
                self.multi_state_mapper.set_dynamic_weight_config(config_name, config['dynamic_weight'])

        self.motor_constraints = MOTOR_CONSTRAINTS['right']

    def _apply_motor_constraints(self):
        for i, constraint in enumerate(self.motor_constraints):
            if constraint.get('enabled', False):
                min_val = constraint.get('min', 0)
                max_val = constraint.get('max', 255)
                self.g_jointpositions[i] = int(max(min_val, min(max_val, self.g_jointpositions[i])))

    def _set_g_jointpositions_from_qpos(self, qpos):
        self.g_jointpositions = _map_o30i_qpos_to_motor(qpos, self.hand_side)
        return self.g_jointpositions

    def _set_g_jointpositions_from_arc(self):
        self.g_jointpositions = _map_o30i_arc_to_motor(self.g_jointpositions_arc, self.hand_side)
        return self.g_jointpositions

    def set_glove_version(self, version: str):
        if not version:
            return
        
        major_version = version.split('.')[0]
        version_key = f'v{major_version}'
        
        if version_key == self.glove_version:
            return
            
        self.glove_version = version_key
        
        for finger_name, config in FINGER_CONFIGS.items():
            if 'weights' in config and isinstance(config['weights'], dict):
                if version_key in config['weights']:
                    self.multi_state_mapper.finger_configs[finger_name]['weights'] = config['weights'][version_key]
            
            if 'reverse_motion' in config and isinstance(config['reverse_motion'], dict):
                if version_key in config['reverse_motion']:
                    self.multi_state_mapper.finger_configs[finger_name]['reverse_motion'] = config['reverse_motion'][version_key]

    def initialize_mapper(self) -> bool:
        """
        初始化映射器

        将三种人手标定数据和三种机械手标定数据加载到映射器中
        分别是original,opose,fist

        人手是glove_前缀,机械手是robot_前缀
        """

        glove_original = self._to_list(self.calibrationoriginal)
        glove_fist = self._to_list(self.calibrationfistpose)
        glove_opose = self._to_list(self.calibrationopose)
        
        self.multi_state_mapper.add_state('original', glove_original, self.robot_original)
        self.multi_state_mapper.add_state('opose', glove_opose, self.robot_opose)
        self.multi_state_mapper.add_state('fist', glove_fist, self.robot_fist)

        self.multi_state_mapper.set_state_order(list(MULTI_SEGMENT_CONFIG_FROZEN))

    def _to_list(self, data):
        """转换为列表"""
        if hasattr(data, 'tolist'):
            return data.tolist()
        elif isinstance(data, np.ndarray):
            # print(111)
            return data.tolist()
        else:
            return list(data)

    # V2.8.0 本函数作废
    # def _linear_map_diff(self, current_diff, fist_diff, extend_ratio=1.2):
    #     """
    #     基于差值的线性映射到0-255
        
    #     注意: 传入joint_update的是差值 (当前值 - 张开值)
        
    #     参数:
    #         current_diff: 当前传感器差值 (当前值 - 张开值)
    #         fist_diff: 握拳时的差值 (握拳值 - 张开值)
    #         extend_ratio: 缩放比例，>1.0 使映射更容易到达0/255边界
        
    #     映射逻辑：
    #         - 差值为0（张开）→ 255
    #         - 差值为fist_diff（握拳）→ 0
    #     """
    #     if abs(fist_diff) < 0.01:
    #         return 128  # 变化太小，返回中值
        
    #     # 缩小fist_diff使得更容易到达0边界
    #     effective_fist_diff = fist_diff / extend_ratio
        
    #     # 计算比例: 差值0→比例0, 差值fist_diff→比例1
    #     ratio = current_diff / effective_fist_diff
    #     ratio = max(0.0, min(1.0, ratio))  # 限制在0-1之间
        
    #     # 映射: 比例0→255, 比例1→0
    #     return int((1 - ratio) * 255)

    def _apply_smooth(self, raw_positions):
        """
        对电机输出应用平滑滤波，防止跳变
        
        使用指数移动平均(EMA) + 最大步长限制
        """
        if not self.smooth_enabled:
            return raw_positions
        
        smoothed = []
        for i, raw in enumerate(raw_positions):
            # 指数移动平均
            target = self.smooth_alpha * raw + (1 - self.smooth_alpha) * self.smooth_positions[i]
            
            # 最大步长限制，防止大幅跳变
            diff = target - self.smooth_positions[i]
            if abs(diff) > self.max_step:
                target = self.smooth_positions[i] + (self.max_step if diff > 0 else -self.max_step)
            
            self.smooth_positions[i] = target
            smoothed.append(int(round(target)))
        
        return smoothed

    def joint_update(self, joint_arc):
        """
        右手映射 - 基于标定数据和预期机械手动作的映射器完成
        """
        qpos = np.zeros(25)
        # ========== 使用映射器进行精确映射 ==========
        if self.calibrationoriginal is not None and self.calibrationfistpose is not None and self.calibrationopose is not None:
            arc_value = self.multi_state_mapper.map_glove_to_robot(joint_arc)
        # ========== 没有标定数据时使用手动映射 ==========
        else:
            arc_value = None
        
        if arc_value is not None:
            _apply_o30i_arc_values(qpos, self.g_jointpositions_arc, arc_value)
        else:
            # 手动映射备用
            qpos[20] = joint_arc[4] * 2.2
            qpos[17] = joint_arc[2] * -2.5
            qpos[1] = joint_arc[6] * 0.1 + joint_arc[8] * 0.7
            qpos[9] = joint_arc[10] * 0.1 + joint_arc[12] * 0.7
            qpos[13] = joint_arc[14] * 0.1 + joint_arc[16] * 0.7
            qpos[5] = joint_arc[18] * 0.1 + joint_arc[20] * 0.7
        
        # ========== 应用平滑滤波 ==========
        if arc_value is not None:
            self._set_g_jointpositions_from_arc()
        else:
            self._set_g_jointpositions_from_qpos(qpos)
        self._apply_motor_constraints()
        self.g_jointpositions = self._apply_smooth(self.g_jointpositions)

    def speed_update(self):
        for i in range(len(self.g_jointpositions)):
            lastpos = self.last_jointpositions[i]
            position_error = int(abs(self.g_jointpositions[i] - lastpos))
            position_derict = 1 if self.g_jointpositions[i] - lastpos > 0 else -1
            slow_limit = 4
            fast_limit = 10
            max_vel = int(self.last_jointvelocity[i] * 2)
            mid_vel = int(self.last_jointvelocity[i] * 0.7)
            min_vel = int(self.last_jointvelocity[i] * 0.5)
            target_vel = self.last_jointvelocity[i]
            if self.handstate[i] == 0:  # stop
                if 0 < position_error:
                    target_vel = position_error * 5 + 30
                    self.handstate[i] = 1
            elif self.handstate[i] == 1:  # slow
                if position_error >= fast_limit:
                    target_vel = position_error * 5 + 50
                    if target_vel > mid_vel:
                        target_vel = mid_vel
                    self.handstate[i] = 2
                elif position_error == 0:
                    self.handstate[i] = 0
                    target_vel = position_error * 5 + 100
                else:
                    target_vel = position_error * 5 + 100
            else:  # fast
                if position_error >= fast_limit:
                    target_vel = position_error * 5 + 90
                    if target_vel > max_vel:
                        target_vel = max_vel
                elif slow_limit < position_error < fast_limit:
                    target_vel = position_error * 5 + 60
                    if target_vel < mid_vel:
                        target_vel = mid_vel
                    self.handstate[i] = 3
                elif 0 < position_error <= slow_limit:
                    target_vel = position_error * 5 + 40
                    if target_vel < min_vel:
                        target_vel = min_vel
                    self.handstate[i] = 1
            self.g_jointvelocity[i] = int(target_vel * 1)
            if self.g_jointvelocity[i] > 255:
                self.g_jointvelocity[i] = 255
            self.g_jointvelocity[i] = 255
            self.last_jointvelocity[i] = self.g_jointvelocity[i]
            self.last_jointpositions[i] = self.g_jointpositions[i]


class LeftHand:
    def __init__(self, handcore: HandCore, length=20, is_debug: bool = False):
        self.handcore = handcore
        self.hand_side = "left"
        self.g_jointpositions = [255] * length
        self.topic_joint_names = O30I_TOPIC_JOINT_NAMES
        self.g_jointvelocity = [255] * length
        self.last_jointpositions = [255] * length
        self.last_jointvelocity = [255] * length
        self.g_jointpositions_arc = [0] * length
        self.mujoco_joint_arc_indices = O30I_MUJOCO_JOINT_ARC_INDICES
        self.mujoco_joint_arc_signs = O30I_MUJOCO_JOINT_ARC_SIGNS
        self.g_jointvelocity_arc = [0] * length
        self.handstate = [0] * length
        self.calibrationoriginal = None
        self.calibrationfistpose = None
        self.calibrationopose = None
        self.glove_version = 'v2'
        
        self.smooth_enabled = True
        self.smooth_alpha = 0.5
        self.smooth_positions = [255.0] * length
        self.max_step = 20

        self.robot_original = ROBOT_ORIGINAL_LEFT
        self.robot_opose = ROBOT_OPOSE_LEFT
        self.robot_fist = ROBOT_FIST_LEFT
        self.mujoco_joint_arc_remaps = (None,) * len(O30I_MUJOCO_JOINT_ARC_INDICES)
        self.mujoco_joint_arc_mirrors = (None,) * len(O30I_MUJOCO_JOINT_ARC_INDICES)

        finger_configs = _resolve_version_config(FINGER_CONFIGS, self.glove_version)
        self.multi_state_mapper = O30iStableDynamicWeightMultiStateLinearMapper(
            finger_configs,
            MAPPING_ORDER,
            is_debug=is_debug,
        )

        for config_name, config in FINGER_CONFIGS.items():
            if config.get('dynamic_weight'):
                self.multi_state_mapper.set_dynamic_weight_config(config_name, config['dynamic_weight'])

        self.motor_constraints = MOTOR_CONSTRAINTS['left']

    def _apply_motor_constraints(self):
        for i, constraint in enumerate(self.motor_constraints):
            if constraint.get('enabled', False):
                min_val = constraint.get('min', 0)
                max_val = constraint.get('max', 255)
                self.g_jointpositions[i] = int(max(min_val, min(max_val, self.g_jointpositions[i])))

    def _set_g_jointpositions_from_qpos(self, qpos):
        self.g_jointpositions = _map_o30i_qpos_to_motor(qpos, self.hand_side)
        return self.g_jointpositions

    def _set_g_jointpositions_from_arc(self):
        self.g_jointpositions = _map_o30i_arc_to_motor(self.g_jointpositions_arc, self.hand_side)
        return self.g_jointpositions

    def set_glove_version(self, version: str):
        if not version:
            return
        
        major_version = version.split('.')[0]
        version_key = f'v{major_version}'
        
        if version_key == self.glove_version:
            return
            
        self.glove_version = version_key
        
        for finger_name, config in FINGER_CONFIGS.items():
            if 'weights' in config and isinstance(config['weights'], dict):
                if version_key in config['weights']:
                    self.multi_state_mapper.finger_configs[finger_name]['weights'] = config['weights'][version_key]
            
            if 'reverse_motion' in config and isinstance(config['reverse_motion'], dict):
                if version_key in config['reverse_motion']:
                    self.multi_state_mapper.finger_configs[finger_name]['reverse_motion'] = config['reverse_motion'][version_key]

    def initialize_mapper(self) -> bool:
        """
        初始化映射器

        将三种人手标定数据和三种机械手标定数据加载到映射器中
        分别是original,opose,fist

        人手是glove_前缀,机械手是robot_前缀
        """

        glove_original = self._to_list(self.calibrationoriginal)
        glove_fist = self._to_list(self.calibrationfistpose)
        glove_opose = self._to_list(self.calibrationopose)
        
        self.multi_state_mapper.add_state('original', glove_original, self.robot_original)
        self.multi_state_mapper.add_state('opose', glove_opose, self.robot_opose)
        self.multi_state_mapper.add_state('fist', glove_fist, self.robot_fist)

        self.multi_state_mapper.set_state_order(list(MULTI_SEGMENT_CONFIG_FROZEN))

        state_info = self.multi_state_mapper.get_state_info()

    def _to_list(self, data):
        """转换为列表"""
        if hasattr(data, 'tolist'):
            return data.tolist()
        elif isinstance(data, np.ndarray):
            # print(111)
            return data.tolist()
        else:
            return list(data)

    # V2.8.0 本函数作废
    # def _linear_map_diff(self, current_diff, fist_diff, extend_ratio=1.2):
    #     """
    #     基于差值的线性映射到0-255
        
    #     注意: 传入joint_update的是差值 (当前值 - 张开值)
        
    #     参数:
    #         current_diff: 当前传感器差值 (当前值 - 张开值)
    #         fist_diff: 握拳时的差值 (握拳值 - 张开值)
    #         extend_ratio: 缩放比例，>1.0 使映射更容易到达0/255边界
        
    #     映射逻辑：
    #         - 差值为0（张开）→ 255
    #         - 差值为fist_diff（握拳）→ 0
    #     """
    #     if abs(fist_diff) < 0.01:
    #         return 128  # 变化太小，返回中值
        
    #     # 缩小fist_diff使得更容易到达0边界
    #     effective_fist_diff = fist_diff / extend_ratio
        
    #     # 计算比例: 差值0→比例0, 差值fist_diff→比例1
    #     ratio = current_diff / effective_fist_diff
    #     ratio = max(0.0, min(1.0, ratio))  # 限制在0-1之间
        
    #     # 映射: 比例0→255, 比例1→0
    #     return int((1 - ratio) * 255)

    def _apply_smooth(self, raw_positions):
        """
        对电机输出应用平滑滤波，防止跳变
        
        使用指数移动平均(EMA) + 最大步长限制
        """
        if not self.smooth_enabled:
            return raw_positions
        
        smoothed = []
        for i, raw in enumerate(raw_positions):
            # 指数移动平均
            target = self.smooth_alpha * raw + (1 - self.smooth_alpha) * self.smooth_positions[i]
            
            # 最大步长限制，防止大幅跳变
            diff = target - self.smooth_positions[i]
            if abs(diff) > self.max_step:
                target = self.smooth_positions[i] + (self.max_step if diff > 0 else -self.max_step)
            
            self.smooth_positions[i] = target
            smoothed.append(int(round(target)))
        
        return smoothed

    def joint_update(self, joint_arc):
        """
        右手映射 - 基于标定数据和预期机械手动作的映射器完成
        """
        qpos = np.zeros(25)
        # ========== 使用映射器进行精确映射 ==========
        if self.calibrationoriginal is not None and self.calibrationfistpose is not None and self.calibrationopose is not None:
            is_exact_calibration_pose = (
                _matches_calibration_pose(joint_arc, self.calibrationoriginal)
                or _matches_calibration_pose(joint_arc, self.calibrationopose)
                or _matches_calibration_pose(joint_arc, self.calibrationfistpose)
            )
            if is_exact_calibration_pose and hasattr(self.multi_state_mapper, "reset_thumb_output"):
                self.multi_state_mapper.reset_thumb_output()
            arc_value = self.multi_state_mapper.map_glove_to_robot(joint_arc)
        # ========== 没有标定数据时使用手动映射 ==========
        else:
            arc_value = None
        
        if arc_value is not None:
            _apply_o30i_arc_values(qpos, self.g_jointpositions_arc, arc_value)
        else:
            # 手动映射备用
            qpos[20] = joint_arc[4] * 2.2
            qpos[17] = joint_arc[2] * -2.5
            qpos[1] = joint_arc[6] * 0.1 + joint_arc[8] * 0.7
            qpos[9] = joint_arc[10] * 0.1 + joint_arc[12] * 0.7
            qpos[13] = joint_arc[14] * 0.1 + joint_arc[16] * 0.7
            qpos[5] = joint_arc[18] * 0.1 + joint_arc[20] * 0.7
        
        if arc_value is not None:
            self._set_g_jointpositions_from_arc()
        else:
            self._set_g_jointpositions_from_qpos(qpos)

        # print(qpos[4],arc_value[17],self.g_jointpositions[9])
        self._apply_motor_constraints()

    def speed_update(self):
        for i in range(len(self.g_jointpositions)):
            lastpos = self.last_jointpositions[i]
            position_error = int(abs(self.g_jointpositions[i] - lastpos))
            position_derict = 1 if self.g_jointpositions[i] - lastpos > 0 else -1
            slow_limit = 4
            fast_limit = 10
            max_vel = int(self.last_jointvelocity[i] * 2)
            mid_vel = int(self.last_jointvelocity[i] * 0.7)
            min_vel = int(self.last_jointvelocity[i] * 0.5)
            target_vel = self.last_jointvelocity[i]
            if self.handstate[i] == 0:  # stop
                if 0 < position_error:
                    target_vel = position_error * 5 + 30
                    self.handstate[i] = 1
            elif self.handstate[i] == 1:  # slow
                if position_error >= fast_limit:
                    target_vel = position_error * 5 + 50
                    if target_vel > mid_vel:
                        target_vel = mid_vel
                    self.handstate[i] = 2
                elif position_error == 0:
                    self.handstate[i] = 0
                    target_vel = position_error * 5 + 100
                else:
                    target_vel = position_error * 5 + 100
            else:  # fast
                if position_error >= fast_limit:
                    target_vel = position_error * 5 + 90
                    if target_vel > max_vel:
                        target_vel = max_vel
                elif slow_limit < position_error < fast_limit:
                    target_vel = position_error * 5 + 60
                    if target_vel < mid_vel:
                        target_vel = mid_vel
                    self.handstate[i] = 3
                elif 0 < position_error <= slow_limit:
                    target_vel = position_error * 5 + 40
                    if target_vel < min_vel:
                        target_vel = min_vel
                    self.handstate[i] = 1
            self.g_jointvelocity[i] = int(target_vel * 1)
            if self.g_jointvelocity[i] > 255:
                self.g_jointvelocity[i] = 255
            self.g_jointvelocity[i] = 255
            self.last_jointvelocity[i] = self.g_jointvelocity[i]
            self.last_jointpositions[i] = self.g_jointpositions[i]
