import copy
from typing import Dict, List, Sequence

import numpy as np

from linkerhand.filter import MultiChannelKalmanFilter


class SimpleLinearMapper:
    """Direction-preserving calibrated linear mapper for LinkerForce hands."""

    def __init__(self, finger_configs: Dict, mapping_order: Sequence[str], is_debug: bool = False,
                 filter_enabled: bool = False):
        self.finger_configs = copy.deepcopy(finger_configs)
        self.mapping_order = list(mapping_order)
        self.glove_states = {}
        self.robot_states = {}
        self.state_order = []
        self.debug_value = [0.0] * 20
        self.isdebug = is_debug
        self.filter_enabled = filter_enabled
        self.filters = None
        self._filter_channels = 0
        self._filter_initialized = False
        self.last_raw_glove = None
        self.last_filtered_glove = None
        self.raw_history = []
        self.filtered_history = []

    def set_debug(self, enabled=True):
        self.isdebug = enabled

    def add_state(self, state_name: str, glove_angles: Sequence[float], robot_angles: Sequence[float]):
        self.glove_states[state_name] = np.array(glove_angles, dtype=float)
        self.robot_states[state_name] = np.array(robot_angles, dtype=float)
        if state_name not in self.state_order:
            self.state_order.append(state_name)

    def set_state_order(self, state_order: Sequence[str]):
        for state_name in state_order:
            if state_name not in self.glove_states:
                raise ValueError(f"状态 '{state_name}' 未定义")
        self.state_order = list(state_order)

    def get_state_info(self) -> Dict:
        return {
            "states": list(self.glove_states.keys()),
            "state_order": list(self.state_order),
            "has_original": "original" in self.glove_states,
        }

    def map_glove_to_robot(self, glove_current, use_filter=None):
        if len(self.state_order) < 2:
            raise ValueError("请至少设置两个状态")
        if "original" not in self.glove_states:
            raise ValueError("必须包含 'original' 状态作为基准")

        current = np.array(glove_current, dtype=float)
        self.last_raw_glove = current.copy()
        filter_enabled = self.filter_enabled if use_filter is None else use_filter
        if filter_enabled:
            current = self._filter_current(current)
        else:
            self.last_filtered_glove = current.copy()

        robot_angles = self.robot_states["original"].copy()

        for config_name in self.mapping_order:
            config = self.finger_configs[config_name]
            robot_idx = config["robot_idx"]
            angle = self._map_config(current, config)
            robot_angles[robot_idx] = angle
            if robot_idx < len(self.debug_value):
                self.debug_value[robot_idx] = float(angle)

        return robot_angles

    def _filter_current(self, current: np.ndarray) -> np.ndarray:
        if self.filters is None or self._filter_channels != len(current):
            self._filter_channels = len(current)
            self.filters = MultiChannelKalmanFilter(
                num_channels=self._filter_channels,
                process_variance=1e-5,
                measurement_variance=0.0005,
                initial_values=[0.0] * self._filter_channels,
            )
            self._filter_initialized = False

        current_list = current.tolist()
        if not self._filter_initialized:
            self.filters.reset(current_list)
            self._filter_initialized = True

        filtered = np.array(self.filters.update(current_list), dtype=float)
        self.last_filtered_glove = filtered.copy()

        self.raw_history.append(current.copy())
        self.filtered_history.append(filtered.copy())
        max_history = 100
        if len(self.raw_history) > max_history:
            self.raw_history = self.raw_history[-max_history:]
            self.filtered_history = self.filtered_history[-max_history:]

        return filtered

    def reset_filter(self, current=None):
        if current is None:
            self.filters = None
            self._filter_channels = 0
            self._filter_initialized = False
            self.last_filtered_glove = None
            return

        current_array = np.array(current, dtype=float)
        self._filter_channels = len(current_array)
        self.filters = MultiChannelKalmanFilter(
            num_channels=self._filter_channels,
            process_variance=1e-5,
            measurement_variance=0.0005,
            initial_values=current_array.tolist(),
        )
        self.filters.reset(current_array.tolist())
        self._filter_initialized = True
        self.last_filtered_glove = current_array.copy()
        self.raw_history.append(current_array.copy())
        self.filtered_history.append(current_array.copy())
        max_history = 100
        if len(self.raw_history) > max_history:
            self.raw_history = self.raw_history[-max_history:]
            self.filtered_history = self.filtered_history[-max_history:]

    def _map_config(self, current: np.ndarray, config: Dict) -> float:
        states = list(config.get("state_order") or self.state_order)
        for state_name in states:
            if state_name not in self.glove_states:
                raise ValueError(f"状态 '{state_name}' 未定义")

        source_points = [
            self._fused_value(self.glove_states[state_name], config)
            for state_name in states
        ]
        target_points = [
            float(self.robot_states[state_name][config["robot_idx"]])
            for state_name in states
        ]
        value = self._fused_value(current, config)

        if len(states) == 2:
            value = self._apply_scale_factor(value, source_points[0], source_points[1], config)

        if self._should_extend_to_fist(config, states):
            result = self._extend_to_fist(value, source_points, target_points, config)
        else:
            result = self._piecewise_linear(value, source_points, target_points)

        if config.get("reverse_output_direction", False) or config.get("reverse_motion", False):
            lower, upper = self._target_range(config, states)
            result = upper - (self._clamp(result, lower, upper) - lower)

        return result

    def _fused_value(self, data: np.ndarray, config: Dict) -> float:
        joints = config["joints"]
        weights = self._normalize_weights(config["weights"])
        if len(weights) != len(joints):
            raise ValueError("weights 长度必须和 joints 长度一致")
        return float(sum(float(data[joint]) * weight for joint, weight in zip(joints, weights)))

    @staticmethod
    def _normalize_weights(weights: Sequence[float]) -> List[float]:
        values = np.array(weights, dtype=float)
        total = float(values.sum())
        if abs(total) < 1e-12:
            return [0.0] * len(values)
        return (values / total).tolist()

    @classmethod
    def _piecewise_linear(cls, value: float, source_points: List[float], target_points: List[float]) -> float:
        if len(source_points) != len(target_points):
            raise ValueError("source_points 和 target_points 长度必须一致")
        if len(source_points) < 2:
            raise ValueError("至少需要两个映射状态")

        for index in range(len(source_points) - 1):
            source_a = source_points[index]
            source_b = source_points[index + 1]
            target_a = target_points[index]
            target_b = target_points[index + 1]
            if cls._between(value, source_a, source_b):
                return cls._interpolate(value, source_a, source_b, target_a, target_b)

        if abs(value - source_points[0]) <= abs(value - source_points[-1]):
            return target_points[0]
        return target_points[-1]

    @staticmethod
    def _between(value: float, lower: float, upper: float) -> bool:
        return min(lower, upper) <= value <= max(lower, upper)

    @staticmethod
    def _interpolate(value: float, source_a: float, source_b: float,
                     target_a: float, target_b: float) -> float:
        if abs(source_b - source_a) < 1e-12:
            return target_b
        ratio = (value - source_a) / (source_b - source_a)
        ratio = max(0.0, min(1.0, ratio))
        return target_a + ratio * (target_b - target_a)

    def _should_extend_to_fist(self, config: Dict, states: List[str]) -> bool:
        ext_config = config.get("extended_mapping") or {}
        return (
            bool(ext_config.get("enabled", False))
            and len(states) == 2
            and states == ["original", "opose"]
            and "fist" in self.glove_states
            and "fist" in self.robot_states
        )

    def _target_range(self, config: Dict, states: List[str]) -> List[float]:
        range_states = list(config.get("range_states") or states)
        target_values = [
            float(self.robot_states[state_name][config["robot_idx"]])
            for state_name in range_states
            if state_name in self.robot_states
        ]
        if not target_values:
            raise ValueError("range_states 中没有可用的目标状态")
        return [min(target_values), max(target_values)]

    @staticmethod
    def _apply_scale_factor(value: float, source_open: float, source_opose: float,
                            config: Dict) -> float:
        ext_config = config.get("extended_mapping") or {}
        if not ext_config.get("enabled", False):
            return value

        scale_factor = float(ext_config.get("scale_factor", 1.0))
        if abs(scale_factor - 1.0) < 1e-12:
            return value
        if abs(source_opose - source_open) < 1e-12:
            return value

        normalized = (value - source_open) / (source_opose - source_open)
        if abs(normalized - 1.0) < 1e-12:
            return value

        scaled_normalized = normalized * scale_factor
        return source_open + scaled_normalized * (source_opose - source_open)

    def _extend_to_fist(self, value: float, source_points: List[float],
                        target_points: List[float], config: Dict) -> float:
        source_open, source_opose = source_points
        target_open, target_opose = target_points
        source_fist = self._fused_value(self.glove_states["fist"], config)
        target_fist = float(self.robot_states["fist"][config["robot_idx"]])
        exp_factor = float((config.get("extended_mapping") or {}).get("extended_exp_factor", 1.0))

        if abs(source_opose - source_open) < 1e-12:
            normalized = 0.5
        else:
            normalized = (value - source_open) / (source_opose - source_open)

        slope = target_opose - target_open
        if normalized <= 0.0:
            return target_open
        if normalized <= 1.0:
            return target_open + normalized * slope

        if self._at_or_beyond(value, source_opose, source_fist):
            return target_fist

        if abs(source_fist - source_opose) < 1e-12:
            return target_fist

        t = (value - source_opose) / (source_fist - source_opose)
        t = self._clamp(t, 0.0, 1.0)
        progress = t * (1.0 + (exp_factor - 1.0) * t)
        result = target_opose + (target_fist - target_opose) * progress
        return self._clamp(result, min(target_opose, target_fist), max(target_opose, target_fist))

    @staticmethod
    def _at_or_beyond(value: float, start: float, end: float) -> bool:
        if abs(end - start) < 1e-12:
            return True
        if end > start:
            return value >= end
        return value <= end

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return min(upper, max(lower, value))
