"""
LinkerForce O20 手型映射模块 - ROS2版本
支持基于标定数据的精确映射
v2.8.0升级了映射器算法
"""
import numpy as np
import copy
import xml.etree.ElementTree as ET
from pathlib import Path
from linkerhand.handcore import HandCore
from ..config.o20_config import FINGER_CONFIGS, MAPPING_ORDER, ROBOT_OPOSE_RIGHT, ROBOT_OPOSE_LEFT, ROBOT_ORIGINAL_LEFT, ROBOT_ORIGINAL_RIGHT, ROBOT_FIST_LEFT, ROBOT_FIST_RIGHT, MULTI_SEGMENT_CONFIG, MOTOR_CONSTRAINTS
from linkerhand.handcoreex import DynamicWeightMultiStateLinearMapper

O20_THUMB_MOTOR_CALIBRATION = {
    5: {"qpos_index": 16, "robot_idx": 0, "opose_motor": 165},
    10: {"qpos_index": 17, "robot_idx": 1, "opose_motor": 138},
}

O20_MOTOR_COUNT = 20
O20_MOTOR_QPOS_INDICES = (
    18, 1, 9, 13, 5,
    16, 0, 8, 12, 4,
    17, None, None, None, None,
    19, 2, 10, 14, 6,
)
O20_MOTOR_ROBOT_INDICES = (
    2, 5, 8, 11, 14,
    0, 4, 7, 10, 13,
    1, None, None, None, None,
    3, 6, 9, 12, 15,
)
O20_MUJOCO_JOINT_ARC_INDICES = (0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19)
O20_MUJOCO_JOINT_ARC_SIGNS = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)


def _load_o20_urdf_joint_limits(hand: str):
    package_dir = Path(__file__).resolve().parents[3]
    urdf_path = (
        package_dir
        / "assets"
        / "robots"
        / "hands"
        / "linker_hand"
        / f"o20_{hand}"
        / f"linkerhand_o20_{hand}.urdf"
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


O20_URDF_JOINT_LIMITS_RIGHT = _load_o20_urdf_joint_limits("right")
O20_URDF_JOINT_LIMITS_LEFT = _load_o20_urdf_joint_limits("left")


def _piecewise_motor_value(value, lower, opose, upper, motor_lower, motor_opose, motor_upper):
    return _map_piecewise_linear(
        value,
        lower,
        opose,
        upper,
        motor_lower,
        motor_opose,
        motor_upper,
    )


def _clamp(value, lower, upper):
    return min(upper, max(lower, value))


def _clamp_to_range(value, endpoint_a, endpoint_b):
    return _clamp(value, min(endpoint_a, endpoint_b), max(endpoint_a, endpoint_b))


def _reverse_between(value, source_open, source_fist, target_open, target_fist):
    if abs(source_fist - source_open) < 1e-9:
        return target_fist
    ratio = (value - source_open) / (source_fist - source_open)
    ratio = _clamp(ratio, 0.0, 1.0)
    return target_open + ratio * (target_fist - target_open)


def _resolve_o20_urdf_joint_limits(handcore: HandCore, hand: str):
    lower_attr = "hand_lower_limits_l" if hand == "left" else "hand_lower_limits_r"
    upper_attr = "hand_upper_limits_l" if hand == "left" else "hand_upper_limits_r"
    lower_limits = getattr(handcore, lower_attr, None) if handcore is not None else None
    upper_limits = getattr(handcore, upper_attr, None) if handcore is not None else None
    if lower_limits is not None and upper_limits is not None:
        limits = tuple(zip(lower_limits, upper_limits))
        if len(limits) >= len(O20_MUJOCO_JOINT_ARC_INDICES):
            return limits
    return O20_URDF_JOINT_LIMITS_LEFT if hand == "left" else O20_URDF_JOINT_LIMITS_RIGHT


def _build_o20_mujoco_joint_arc_mirrors(joint_limits):
    return tuple([None] * len(O20_MUJOCO_JOINT_ARC_INDICES))


def _build_o20_mujoco_joint_arc_remaps(
        joint_limits,
        robot_original,
        robot_opose,
        robot_fist):
    remaps = [None] * len(O20_MUJOCO_JOINT_ARC_INDICES)
    if len(joint_limits) > 0:
        remaps[0] = (
            (robot_original[0], robot_original[0]),
            (robot_opose[0], robot_opose[0]),
            (robot_fist[0], robot_fist[0]),
        )
    if len(joint_limits) > 1:
        lower, upper = joint_limits[1]
        remaps[1] = (
            (lower, lower),
            (upper, upper),
        )
    return tuple(remaps)


class O20DynamicWeightMultiStateLinearMapper(DynamicWeightMultiStateLinearMapper):
    def __init__(self, finger_configs, mapping_order, is_debug=False, filter_enabled=False):
        super().__init__(finger_configs, mapping_order, is_debug=is_debug)
        self.finger_configs = copy.deepcopy(finger_configs)
        self.filter_enabled = filter_enabled
        self.last_raw_glove = None
        self.last_filtered_glove = None
        self.last_fused_values = {}

    def reset_filter(self, current=None):
        if current is None:
            initial_values = [0.0] * len(self.filters.filters)
        else:
            initial_values = np.array(current, dtype=float).tolist()

        self.filters.reset(initial_values)
        current_array = np.array(initial_values, dtype=float)
        self.last_raw_glove = current_array.copy()
        self.last_filtered_glove = current_array.copy()
        self.raw_history.append(current_array.copy())
        self.filtered_history.append(current_array.copy())
        max_history = 100
        if len(self.raw_history) > max_history:
            self.raw_history = self.raw_history[-max_history:]
            self.filtered_history = self.filtered_history[-max_history:]

    def _fused_value(self, data, config):
        weights = self._normalize_weights(config["weights"])
        return self._calculate_fused_value(np.array(data, dtype=float), config["joints"], weights)

    def _calculate_fused_value(self, data: np.ndarray, joints, weights) -> float:
        weights = np.array(weights, dtype=float)
        if len(weights) != len(joints):
            raise ValueError("weights 长度必须和 joints 长度一致")
        if abs(float(weights.sum())) > 1e-12:
            weights = weights / weights.sum()
        return float(sum(float(data[joint]) * weight for joint, weight in zip(joints, weights)))

    @staticmethod
    def _normalize_weights(weights):
        values = np.array(weights, dtype=float)
        total = float(values.sum())
        if abs(total) < 1e-12:
            return [0.0] * len(values)
        return (values / total).tolist()

    def map_glove_to_robot(self, source_current, use_filter=None):
        source_current_array = np.array(source_current, dtype=float)
        self.last_raw_glove = source_current_array.copy()
        if len(source_current_array) > 1:
            self.debug_value[3] = float(source_current_array[1])

        for state_name, glove_state in self.glove_states.items():
            if np.allclose(source_current_array, glove_state, rtol=0.0, atol=1e-9):
                self.last_filtered_glove = source_current_array.copy()
                if len(source_current_array) > 1:
                    self.debug_value[4] = float(source_current_array[1])
                return self.robot_states[state_name].copy()

        filter_enabled = self.filter_enabled if use_filter is None else use_filter
        if filter_enabled:
            if self.last_filtered_glove is None:
                self.reset_filter(source_current_array)
                glove_current = source_current_array.copy()
            else:
                glove_current = np.array(self.filters.update(source_current_array.tolist()), dtype=float)
            self.last_filtered_glove = glove_current.copy()
        else:
            glove_current = source_current_array.copy()
            self.last_filtered_glove = glove_current.copy()

        self.raw_history.append(source_current_array.copy())
        self.filtered_history.append(np.array(glove_current, dtype=float).copy())
        max_history = 100
        if len(self.raw_history) > max_history:
            self.raw_history = self.raw_history[-max_history:]
            self.filtered_history = self.filtered_history[-max_history:]

        if len(glove_current) > 1:
            self.debug_value[4] = float(glove_current[1])

        if isinstance(glove_current, np.ndarray):
            glove_current = glove_current.tolist()
        elif isinstance(glove_current, list):
            glove_current = glove_current
        else:
            glove_current = list(glove_current)

        if len(self.state_order) < 2:
            raise ValueError("请至少设置两个状态")

        if 'original' not in self.glove_states:
            raise ValueError("必须包含 'original' 状态作为基准")

        self.cached_mapped_values = {}
        glove_current_arr = np.array(glove_current)
        robot_angles = self.robot_states['original'].copy()

        for config_name in self.mapping_order:
            if config_name in self.dynamic_weight_configs:
                trigger_finger = self.dynamic_weight_configs[config_name]['trigger_finger']
                if trigger_finger not in self.cached_mapped_values:
                    trigger_value = self._calculate_trigger_value(
                        glove_current_arr, trigger_finger
                    )
                    self.cached_mapped_values[trigger_finger] = trigger_value

        for config_name in self.mapping_order:
            dynamic_config = self.dynamic_weight_configs.get(config_name)
            config = self.finger_configs[config_name]
            if dynamic_config:
                angle = self._map_finger_dynamic_weight(
                    glove_current_arr, config_name, dynamic_config, config
                )
            else:
                angle = self._map_finger_multi_state(glove_current_arr, config)
            robot_idx = self.finger_configs[config_name]['robot_idx']
            robot_angles[robot_idx] = angle

        return robot_angles


def _clamp_robot_arc_values(values, joint_limits):
    clipped = np.array(values, dtype=float).copy()
    for robot_idx, (lower, upper) in enumerate(joint_limits):
        if robot_idx >= len(clipped):
            break
        clipped[robot_idx] = _clamp(float(clipped[robot_idx]), float(lower), float(upper))
    return clipped


def _map_piecewise_linear(value, source_open, source_opose, source_fist,
                          target_open, target_opose, target_fist):
    def interpolate(x0, y0, x1, y1):
        if abs(x1 - x0) < 1e-9:
            return y1
        ratio = (value - x0) / (x1 - x0)
        ratio = _clamp(ratio, 0.0, 1.0)
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


def _scale_value(value, source_min, source_max, target_min, target_max):
    if abs(source_max - source_min) < 1e-9:
        return target_min
    return (value - source_min) * (target_max - target_min) / (source_max - source_min) + target_min


def _mirrored_value(value, lower, upper):
    return upper - (_clamp(value, lower, upper) - lower)


def _uses_reversed_output(config):
    return bool(config.get("reverse_output_direction", False) or config.get("reverse_motion", False))


def _build_effective_robot_states(finger_configs, robot_original, robot_opose, robot_fist):
    base_states = {
        "original": list(robot_original),
        "opose": list(robot_opose),
        "fist": list(robot_fist),
    }
    effective_states = copy.deepcopy(base_states)

    for config in finger_configs.values():
        if not _uses_reversed_output(config):
            continue

        robot_idx = config["robot_idx"]
        range_states = config.get("range_states") or ("original", "opose", "fist")
        range_values = [
            base_states[state_name][robot_idx]
            for state_name in range_states
            if state_name in base_states
        ]
        if not range_values:
            continue

        lower = min(range_values)
        upper = max(range_values)
        for state_name, values in effective_states.items():
            values[robot_idx] = _mirrored_value(base_states[state_name][robot_idx], lower, upper)

    return (
        effective_states["original"],
        effective_states["opose"],
        effective_states["fist"],
    )


def _map_o20_qpos_to_motor(handcore: HandCore, qpos, hand: str,
                           robot_original=None, robot_fist=None):
    if hand == 'left':
        if handcore is not None:
            setattr(handcore, 'last_qpos_l', list(qpos))
        robot_original = ROBOT_ORIGINAL_LEFT if robot_original is None else robot_original
        robot_fist = ROBOT_FIST_LEFT if robot_fist is None else robot_fist
    else:
        if handcore is not None:
            setattr(handcore, 'last_qpos_r', list(qpos))
        robot_original = ROBOT_ORIGINAL_RIGHT if robot_original is None else robot_original
        robot_fist = ROBOT_FIST_RIGHT if robot_fist is None else robot_fist

    jointpositions = [255.0] * O20_MOTOR_COUNT
    for index in range(O20_MOTOR_COUNT):
        source_idx = O20_MOTOR_QPOS_INDICES[index]
        robot_idx = O20_MOTOR_ROBOT_INDICES[index]
        if source_idx is None or robot_idx is None:
            continue
        if source_idx >= len(qpos):
            continue

        open_angle = robot_original[robot_idx]
        fist_angle = robot_fist[robot_idx]
        value = _clamp_to_range(qpos[source_idx], open_angle, fist_angle)
        motor_value = _scale_value(
            value,
            open_angle,
            fist_angle,
            0,
            255,
        )
        jointpositions[index] = int(round(_clamp(motor_value, 0, 255)))

    return jointpositions


def _matches_calibration_pose(joint_arc, calibration_pose) -> bool:
    if calibration_pose is None:
        return False
    if len(joint_arc) != len(calibration_pose):
        return False
    return bool(np.allclose(joint_arc, calibration_pose, rtol=0.0, atol=1e-9))


def _map_config_from_calibration(mapper, config_name, current,
                                 calibrationoriginal, calibrationopose, calibrationfistpose,
                                 robot_original, robot_opose, robot_fist):
    config = mapper.finger_configs.get(config_name)
    if not config:
        return None

    robot_idx = config["robot_idx"]
    current_value = mapper._fused_value(current, config)
    return _map_piecewise_linear(
        current_value,
        mapper._fused_value(calibrationoriginal, config),
        mapper._fused_value(calibrationopose, config),
        mapper._fused_value(calibrationfistpose, config),
        robot_original[robot_idx],
        robot_opose[robot_idx],
        robot_fist[robot_idx],
    )


class RightHand:
    def __init__(self, handcore: HandCore, length=20, is_debug: bool = False):
        self.handcore = handcore
        self.g_jointpositions = [0] * length
        self.g_jointvelocity = [255] * length
        self.last_jointpositions = [0] * length
        self.last_jointvelocity = [255] * length
        self.g_jointpositions_arc = [0] * length
        self.mujoco_joint_arc_indices = O20_MUJOCO_JOINT_ARC_INDICES
        self.mujoco_joint_arc_signs = O20_MUJOCO_JOINT_ARC_SIGNS
        self.g_jointvelocity_arc = [0] * length
        self.handstate = [0] * length
        self._debug_roll_counter = 0
        self.calibrationoriginal = None    # 五指张开标定值 (对应0)
        self.calibrationfistpose = None    # 握拳标定值 (对应255)
        self.calibrationopose = None       # O型标定值 (对应中间值)
        self.glove_version = 'v2'

        # ========== 平滑滤波参数 ==========
        self.smooth_enabled = True
        self.smooth_alpha = 0.5  # 平滑系数：越小越平滑，范围 0.05-0.3
        self.smooth_positions = [0.0] * length  # 平滑后的位置（浮点）
        self.smooth_positions_arc = [0.0] * length
        self.max_step = 20  # 每帧最大变化量，防止跳变

        # 目标机械手预设姿势，数值从URDF获取数据集，
        # 张开手的时候对应最小角度，
        # 握拳的时候对应最大角度
        # O型手势的时候，用工具驱动URDF去驱动目标机械手达到期望姿势，也可以调整这些参数使得实物更加达到期望角度
        # 其他手势也类似，也可以增加多个手势来实现多模态的映射器（后期陆续开发）
        self.robot_original = ROBOT_ORIGINAL_RIGHT
        self.robot_opose = ROBOT_OPOSE_RIGHT
        self.robot_fist = ROBOT_FIST_RIGHT
        self.urdf_joint_limits = _resolve_o20_urdf_joint_limits(self.handcore, "right")
        self.mujoco_joint_arc_mirrors = _build_o20_mujoco_joint_arc_mirrors(self.urdf_joint_limits)
        self.mujoco_joint_arc_remaps = _build_o20_mujoco_joint_arc_remaps(
            self.urdf_joint_limits,
            self.robot_original,
            self.robot_opose,
            self.robot_fist,
        )

        finger_configs = _resolve_version_config(FINGER_CONFIGS, self.glove_version)
        self.effective_robot_original, self.effective_robot_opose, self.effective_robot_fist = (
            _build_effective_robot_states(
                finger_configs,
                self.robot_original,
                self.robot_opose,
                self.robot_fist,
            )
        )
        self.multi_state_mapper = O20DynamicWeightMultiStateLinearMapper(
            finger_configs,
            MAPPING_ORDER,
            is_debug=is_debug,
        )

        self.motor_constraints = MOTOR_CONSTRAINTS['right']

    def _apply_motor_constraints(self):
        for i, constraint in enumerate(self.motor_constraints):
            if constraint.get('enabled', False):
                min_val = constraint.get('min', 0)
                max_val = constraint.get('max', 255)
                self.g_jointpositions[i] = int(max(min_val, min(max_val, self.g_jointpositions[i])))

    def _apply_thumb_motor_calibration(self, qpos):
        for motor_idx, calibration in O20_THUMB_MOTOR_CALIBRATION.items():
            robot_idx = calibration["robot_idx"]
            lower = self.effective_robot_original[robot_idx]
            upper = self.effective_robot_fist[robot_idx]
            value = _clamp_to_range(qpos[calibration["qpos_index"]], lower, upper)
            opose = self.effective_robot_opose[calibration["robot_idx"]]
            self.g_jointpositions[motor_idx] = int(round(_piecewise_motor_value(
                value,
                lower,
                opose,
                upper,
                0,
                calibration["opose_motor"],
                255,
            )))

    def _map_thumb_cmc_yaw(self, joint_arc):
        if (self.calibrationoriginal is None or self.calibrationopose is None
                or self.calibrationfistpose is None):
            return None

        return _map_piecewise_linear(
            joint_arc[1],
            self.calibrationoriginal[1],
            self.calibrationopose[1],
            self.calibrationfistpose[1],
            self.effective_robot_original[1],
            self.effective_robot_opose[1],
            self.effective_robot_fist[1],
        )

    def _map_thumb_cmc_roll(self, filtered_joint_arc, raw_joint_arc):
        if (self.calibrationoriginal is None or self.calibrationopose is None
                or self.calibrationfistpose is None):
            return None
        if _matches_calibration_pose(raw_joint_arc, self.calibrationoriginal):
            return self.robot_original[0]
        if _matches_calibration_pose(raw_joint_arc, self.calibrationopose):
            return self.robot_opose[0]
        if _matches_calibration_pose(raw_joint_arc, self.calibrationfistpose):
            return self.robot_fist[0]

        config = self.multi_state_mapper.finger_configs.get("thumb_rotate")
        if not config:
            return None

        current = np.array(filtered_joint_arc, dtype=float)
        original = np.array(self.calibrationoriginal, dtype=float)
        opose = np.array(self.calibrationopose, dtype=float)
        fist = np.array(self.calibrationfistpose, dtype=float)
        value = self.multi_state_mapper._fused_value(current, config)

        return _map_piecewise_linear(
            value,
            self.multi_state_mapper._fused_value(original, config),
            self.multi_state_mapper._fused_value(opose, config),
            self.multi_state_mapper._fused_value(fist, config),
            self.robot_original[0],
            self.robot_opose[0],
            self.robot_fist[0],
        )

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

        self.multi_state_mapper.set_state_order(MULTI_SEGMENT_CONFIG['states'])

    def _to_list(self, data):
        """转换为列表"""
        if hasattr(data, 'tolist'):
            return data.tolist()
        elif isinstance(data, np.ndarray):
            # print(111)
            return data.tolist()
        else:
            return list(data)

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

    def _apply_arc_smooth(self, raw_positions):
        if not self.smooth_enabled:
            return raw_positions

        smoothed = []
        for i, raw in enumerate(raw_positions):
            target = self.smooth_alpha * raw + (1 - self.smooth_alpha) * self.smooth_positions_arc[i]

            diff = target - self.smooth_positions_arc[i]
            if abs(diff) > self.max_step:
                target = self.smooth_positions_arc[i] + (self.max_step if diff > 0 else -self.max_step)

            self.smooth_positions_arc[i] = target
            smoothed.append(float(target))

        return smoothed

    def _is_exact_fist_pose(self, joint_arc) -> bool:
        return _matches_calibration_pose(joint_arc, self.calibrationfistpose)

    def _is_exact_calibration_pose(self, joint_arc) -> bool:
        return (
            _matches_calibration_pose(joint_arc, self.calibrationoriginal)
            or _matches_calibration_pose(joint_arc, self.calibrationopose)
            or _matches_calibration_pose(joint_arc, self.calibrationfistpose)
        )

    def joint_update(self, joint_arc):
        """
        右手映射 - 基于标定数据和预期机械手动作的映射器完成
        """
        qpos = np.zeros(25)
        is_exact_fist_pose = self._is_exact_fist_pose(joint_arc)
        is_exact_calibration_pose = self._is_exact_calibration_pose(joint_arc)
        # ========== 使用映射器进行精确映射 ==========
        if self.calibrationoriginal is not None and self.calibrationfistpose is not None and self.calibrationopose is not None:
            if is_exact_calibration_pose and self.smooth_enabled:
                self.multi_state_mapper.reset_filter(joint_arc)
            arc_value = self.multi_state_mapper.map_glove_to_robot(
                joint_arc,
                use_filter=self.smooth_enabled and not is_exact_calibration_pose,
            )
        # ========== 没有标定数据时使用手动映射 ==========
        else:
            arc_value = None
        if arc_value is not None:
            filtered_joint_arc = self.multi_state_mapper.last_filtered_glove
            if filtered_joint_arc is None:
                filtered_joint_arc = joint_arc
            thumb_cmc_roll = self._map_thumb_cmc_roll(filtered_joint_arc, joint_arc)
            if thumb_cmc_roll is not None:
                arc_value[0] = thumb_cmc_roll
            thumb_cmc_yaw = self._map_thumb_cmc_yaw(filtered_joint_arc)
            if thumb_cmc_yaw is not None:
                arc_value[1] = thumb_cmc_yaw
            pinky_roll = _map_config_from_calibration(
                self.multi_state_mapper,
                "pinky_roll",
                filtered_joint_arc,
                self.calibrationoriginal,
                self.calibrationopose,
                self.calibrationfistpose,
                self.robot_original,
                self.robot_opose,
                self.robot_fist,
            )
            if pinky_roll is not None:
                arc_value[13] = pinky_roll
            arc_value = _clamp_robot_arc_values(arc_value, self.urdf_joint_limits)

            qpos[16] = self.g_jointpositions_arc[0] = arc_value[0]
            qpos[17] = self.g_jointpositions_arc[1] = arc_value[1]
            qpos[18] = self.g_jointpositions_arc[2] = arc_value[2]
            qpos[19] = self.g_jointpositions_arc[3] = arc_value[3]

            qpos[0] = self.g_jointpositions_arc[5] = arc_value[4]
            qpos[1] = self.g_jointpositions_arc[6] = arc_value[5]
            qpos[2] = self.g_jointpositions_arc[7] = arc_value[6]
            qpos[3] = self.g_jointpositions_arc[8] = 0

            qpos[4] = arc_value[13]
            self.g_jointpositions_arc[17] = _reverse_between(
                arc_value[13],
                self.robot_original[13],
                self.robot_fist[13],
                0.0,
                self.robot_original[13],
            )
            qpos[5] = self.g_jointpositions_arc[18] = arc_value[14]
            qpos[6] = self.g_jointpositions_arc[19] = arc_value[15]
            qpos[7] = self.g_jointpositions_arc[4] = 0

            qpos[8] = self.g_jointpositions_arc[9] = arc_value[7]
            qpos[9] = self.g_jointpositions_arc[10] = arc_value[8]
            qpos[10] = self.g_jointpositions_arc[11] = arc_value[9]
            qpos[11] = self.g_jointpositions_arc[12] = 0

            qpos[12] = arc_value[10]
            self.g_jointpositions_arc[13] = _reverse_between(
                arc_value[10],
                self.robot_original[10],
                self.robot_fist[10],
                0.0,
                self.robot_original[13],
            )
            qpos[13] = self.g_jointpositions_arc[14] = arc_value[11]
            qpos[14] = self.g_jointpositions_arc[15] = arc_value[12]
            qpos[15] = self.g_jointpositions_arc[16] = 0
            if is_exact_calibration_pose:
                self.smooth_positions_arc = list(self.g_jointpositions_arc)
            else:
                self.g_jointpositions_arc = self._apply_arc_smooth(self.g_jointpositions_arc)

            if self._should_log_thumb_roll_debug():
                self._log_thumb_roll_debug("right", joint_arc, filtered_joint_arc, arc_value)
        else:
            # 手动映射备用
            qpos[20] = joint_arc[4] * 2.2
            qpos[17] = joint_arc[2] * -2.5
            qpos[1] = joint_arc[6] * 0.1 + joint_arc[8] * 0.7
            qpos[9] = joint_arc[10] * 0.1 + joint_arc[12] * 0.7
            qpos[13] = joint_arc[14] * 0.1 + joint_arc[16] * 0.7
            qpos[5] = joint_arc[18] * 0.1 + joint_arc[20] * 0.7

        # ========== 应用平滑滤波 ==========
        self.g_jointpositions = _map_o20_qpos_to_motor(
            self.handcore,
            qpos,
            'right',
            self.effective_robot_original,
            self.effective_robot_fist,
        )
        self._apply_thumb_motor_calibration(qpos)
        self._apply_motor_constraints()
        self.g_jointpositions = self._apply_smooth(self.g_jointpositions)
        # print(self.g_jointpositions[5],self.g_jointpositions[10])
        # for i in range(len(self.g_jointpositions)):
        #     if i % 5 != 0:
        #         self.g_jointpositions[i] = 0

    def _should_log_thumb_roll_debug(self):
        debug_setting = getattr(self.multi_state_mapper, "isdebug", False)
        if isinstance(debug_setting, list):
            enabled = "thumb_rotate" in debug_setting
        else:
            enabled = bool(debug_setting)
        if not enabled:
            return False
        self._debug_roll_counter += 1
        return self._debug_roll_counter % 30 == 1

    def _log_thumb_roll_debug(self, hand, joint_arc, filtered_joint_arc, arc_value):
        config = self.multi_state_mapper.finger_configs.get("thumb_rotate", {})
        joints = list(config.get("joints", []))
        weights = list(config.get("weights", []))
        raw_values = [float(joint_arc[index]) for index in joints if index < len(joint_arc)]
        filtered_values = [
            float(filtered_joint_arc[index])
            for index in joints
            if index < len(filtered_joint_arc)
        ]
        source_values = {}
        for state_name, state in (
            ("open", self.calibrationoriginal),
            ("opose", self.calibrationopose),
            ("fist", self.calibrationfistpose),
        ):
            if state is None:
                continue
            source_values[state_name] = [
                float(state[index]) for index in joints if index < len(state)
            ]

        print(
            "[O20 thumb_cmc_roll debug] "
            f"hand={hand}, joints={joints}, weights={weights}, "
            f"raw={raw_values}, filtered={filtered_values}, "
            f"calibration={source_values}, "
            f"arc_value0={float(arc_value[0])}, "
            f"g_arc0={float(self.g_jointpositions_arc[0])}, "
            f"robot_targets(open/opose/fist)="
            f"({self.robot_original[0]}, {self.robot_opose[0]}, {self.robot_fist[0]})"
        )


    def speed_update(self):
        for i in range(len(self.g_jointpositions)):
            lastpos = self.last_jointpositions[i]
            position_error = int(abs(self.g_jointpositions[i] - lastpos))
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
        self.g_jointpositions = [0] * length
        self.g_jointvelocity = [255] * length
        self.last_jointpositions = [0] * length
        self.last_jointvelocity = [255] * length
        self.g_jointpositions_arc = [0] * length
        self.mujoco_joint_arc_indices = O20_MUJOCO_JOINT_ARC_INDICES
        self.mujoco_joint_arc_signs = O20_MUJOCO_JOINT_ARC_SIGNS
        self.g_jointvelocity_arc = [0] * length
        self.handstate = [0] * length
        self.calibrationoriginal = None
        self.calibrationfistpose = None
        self.calibrationopose = None
        self.glove_version = 'v2'

        self.smooth_enabled = True
        self.smooth_alpha = 0.5
        self.smooth_positions = [0.0] * length
        self.smooth_positions_arc = [0.0] * length
        self.max_step = 20

        self.robot_original = ROBOT_ORIGINAL_LEFT
        self.robot_opose = ROBOT_OPOSE_LEFT
        self.robot_fist = ROBOT_FIST_LEFT
        self.urdf_joint_limits = _resolve_o20_urdf_joint_limits(self.handcore, "left")
        self.mujoco_joint_arc_mirrors = _build_o20_mujoco_joint_arc_mirrors(self.urdf_joint_limits)
        self.mujoco_joint_arc_remaps = _build_o20_mujoco_joint_arc_remaps(
            self.urdf_joint_limits,
            self.robot_original,
            self.robot_opose,
            self.robot_fist,
        )

        finger_configs = _resolve_version_config(FINGER_CONFIGS, self.glove_version)
        self.effective_robot_original, self.effective_robot_opose, self.effective_robot_fist = (
            _build_effective_robot_states(
                finger_configs,
                self.robot_original,
                self.robot_opose,
                self.robot_fist,
            )
        )
        self.multi_state_mapper = O20DynamicWeightMultiStateLinearMapper(
            finger_configs,
            MAPPING_ORDER,
            is_debug=is_debug,
        )

        self.motor_constraints = MOTOR_CONSTRAINTS['left']

    def _apply_motor_constraints(self):
        for i, constraint in enumerate(self.motor_constraints):
            if constraint.get('enabled', False):
                min_val = constraint.get('min', 0)
                max_val = constraint.get('max', 255)
                self.g_jointpositions[i] = int(max(min_val, min(max_val, self.g_jointpositions[i])))

    def _apply_thumb_motor_calibration(self, qpos):
        for motor_idx, calibration in O20_THUMB_MOTOR_CALIBRATION.items():
            robot_idx = calibration["robot_idx"]
            lower = self.effective_robot_original[robot_idx]
            upper = self.effective_robot_fist[robot_idx]
            value = _clamp_to_range(qpos[calibration["qpos_index"]], lower, upper)
            opose = self.effective_robot_opose[calibration["robot_idx"]]
            self.g_jointpositions[motor_idx] = int(round(_piecewise_motor_value(
                value,
                lower,
                opose,
                upper,
                0,
                calibration["opose_motor"],
                255,
            )))

    def _map_thumb_cmc_yaw(self, joint_arc):
        if (self.calibrationoriginal is None or self.calibrationopose is None
                or self.calibrationfistpose is None):
            return None

        return _map_piecewise_linear(
            joint_arc[1],
            self.calibrationoriginal[1],
            self.calibrationopose[1],
            self.calibrationfistpose[1],
            self.effective_robot_original[1],
            self.effective_robot_opose[1],
            self.effective_robot_fist[1],
        )

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

        self.multi_state_mapper.set_state_order(MULTI_SEGMENT_CONFIG['states'])

    def _to_list(self, data):
        """转换为列表"""
        if hasattr(data, 'tolist'):
            return data.tolist()
        elif isinstance(data, np.ndarray):
            # print(111)
            return data.tolist()
        else:
            return list(data)

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

    def _apply_arc_smooth(self, raw_positions):
        if not self.smooth_enabled:
            return raw_positions

        smoothed = []
        for i, raw in enumerate(raw_positions):
            target = self.smooth_alpha * raw + (1 - self.smooth_alpha) * self.smooth_positions_arc[i]

            diff = target - self.smooth_positions_arc[i]
            if abs(diff) > self.max_step:
                target = self.smooth_positions_arc[i] + (self.max_step if diff > 0 else -self.max_step)

            self.smooth_positions_arc[i] = target
            smoothed.append(float(target))

        return smoothed

    def _is_exact_fist_pose(self, joint_arc) -> bool:
        return _matches_calibration_pose(joint_arc, self.calibrationfistpose)

    def _is_exact_calibration_pose(self, joint_arc) -> bool:
        return (
            _matches_calibration_pose(joint_arc, self.calibrationoriginal)
            or _matches_calibration_pose(joint_arc, self.calibrationopose)
            or _matches_calibration_pose(joint_arc, self.calibrationfistpose)
        )

    def joint_update(self, joint_arc):
        """
        右手映射 - 基于标定数据和预期机械手动作的映射器完成
        """
        qpos = np.zeros(25)
        is_exact_fist_pose = self._is_exact_fist_pose(joint_arc)
        is_exact_calibration_pose = self._is_exact_calibration_pose(joint_arc)
        # ========== 使用映射器进行精确映射 ==========
        if self.calibrationoriginal is not None and self.calibrationfistpose is not None and self.calibrationopose is not None:
            if is_exact_calibration_pose and self.smooth_enabled:
                self.multi_state_mapper.reset_filter(joint_arc)
            arc_value = self.multi_state_mapper.map_glove_to_robot(
                joint_arc,
                use_filter=self.smooth_enabled and not is_exact_calibration_pose,
            )
        # ========== 没有标定数据时使用手动映射 ==========
        else:
            arc_value = None

        if arc_value is not None:
            filtered_joint_arc = self.multi_state_mapper.last_filtered_glove
            if filtered_joint_arc is None:
                filtered_joint_arc = joint_arc
            thumb_cmc_yaw = self._map_thumb_cmc_yaw(filtered_joint_arc)
            if thumb_cmc_yaw is not None:
                arc_value[1] = thumb_cmc_yaw
            pinky_roll = _map_config_from_calibration(
                self.multi_state_mapper,
                "pinky_roll",
                filtered_joint_arc,
                self.calibrationoriginal,
                self.calibrationopose,
                self.calibrationfistpose,
                self.robot_original,
                self.robot_opose,
                self.robot_fist,
            )
            if pinky_roll is not None:
                arc_value[13] = pinky_roll
            arc_value = _clamp_robot_arc_values(arc_value, self.urdf_joint_limits)

            qpos[16] = self.g_jointpositions_arc[0] = arc_value[0]
            qpos[17] = self.g_jointpositions_arc[1] = arc_value[1]
            qpos[18] = self.g_jointpositions_arc[2] = arc_value[2]
            qpos[19] = self.g_jointpositions_arc[3] = arc_value[3]

            qpos[0] = self.g_jointpositions_arc[5] = arc_value[4]
            qpos[1] = self.g_jointpositions_arc[6] = arc_value[5]
            qpos[2] = self.g_jointpositions_arc[7] = arc_value[6]
            qpos[3] = self.g_jointpositions_arc[8] = 0

            qpos[4] = self.g_jointpositions_arc[17] = arc_value[13]
            qpos[5] = self.g_jointpositions_arc[18] = arc_value[14]
            qpos[6] = self.g_jointpositions_arc[19] = arc_value[15]
            qpos[7] = self.g_jointpositions_arc[4] = 0

            qpos[8] = self.g_jointpositions_arc[9] = arc_value[7]
            qpos[9] = self.g_jointpositions_arc[10] = arc_value[8]
            qpos[10] = self.g_jointpositions_arc[11] = arc_value[9]
            qpos[11] = self.g_jointpositions_arc[12] = 0

            qpos[12] = self.g_jointpositions_arc[13] = arc_value[10]
            qpos[13] = self.g_jointpositions_arc[14] = arc_value[11]
            qpos[14] = self.g_jointpositions_arc[15] = arc_value[12]
            qpos[15] = self.g_jointpositions_arc[16] = 0
            if is_exact_calibration_pose:
                self.smooth_positions_arc = list(self.g_jointpositions_arc)
            else:
                self.g_jointpositions_arc = self._apply_arc_smooth(self.g_jointpositions_arc)
        else:
            # 手动映射备用
            qpos[20] = joint_arc[4] * 2.2
            qpos[17] = joint_arc[2] * -2.5
            qpos[1] = joint_arc[6] * 0.1 + joint_arc[8] * 0.7
            qpos[9] = joint_arc[10] * 0.1 + joint_arc[12] * 0.7
            qpos[13] = joint_arc[14] * 0.1 + joint_arc[16] * 0.7
            qpos[5] = joint_arc[18] * 0.1 + joint_arc[20] * 0.7

        self.g_jointpositions = _map_o20_qpos_to_motor(
            self.handcore,
            qpos,
            'left',
            self.effective_robot_original,
            self.effective_robot_fist,
        )
        self._apply_thumb_motor_calibration(qpos)

        # print(qpos[4],arc_value[17],self.g_jointpositions[9])
        self._apply_motor_constraints()

    def speed_update(self):
        for i in range(len(self.g_jointpositions)):
            lastpos = self.last_jointpositions[i]
            position_error = int(abs(self.g_jointpositions[i] - lastpos))
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
