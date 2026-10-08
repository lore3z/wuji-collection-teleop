"""Time-constant EMA for Cartesian position and quaternion orientation."""

import math

import numpy as np

from .pose_mapper import normalize_quaternion_xyzw


class PoseEMAFilter:
    def __init__(self, translation_tau_sec, rotation_tau_sec):
        self.translation_tau = self._positive(
            translation_tau_sec, "translation_tau_sec")
        self.rotation_tau = self._positive(
            rotation_tau_sec, "rotation_tau_sec")
        self.position = None
        self.quaternion = None

    @staticmethod
    def _positive(value, name):
        result = float(value)
        if not np.isfinite(result) or result <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
        return result

    @staticmethod
    def _alpha(dt, tau):
        return 1.0 - math.exp(-dt / tau)

    def alphas(self, dt):
        """Return the translation/rotation gains used for a given step."""
        dt = self._positive(dt, "dt")
        return (self._alpha(dt, self.translation_tau),
                self._alpha(dt, self.rotation_tau))

    def reset(self, position, quaternion_xyzw):
        position = np.asarray(position, dtype=float)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("EMA position must contain three finite values")
        self.position = position.copy()
        self.quaternion = normalize_quaternion_xyzw(quaternion_xyzw)

    def step(self, target_position, target_quaternion_xyzw, dt):
        dt = self._positive(dt, "dt")
        target_position = np.asarray(target_position, dtype=float)
        if (target_position.shape != (3,) or
                not np.all(np.isfinite(target_position))):
            raise ValueError("EMA target position must contain three finite values")
        target_quaternion = normalize_quaternion_xyzw(target_quaternion_xyzw)
        if self.position is None:
            self.reset(target_position, target_quaternion)
            return self.position.copy(), self.quaternion.copy()

        position_alpha = self._alpha(dt, self.translation_tau)
        self.position += position_alpha * (target_position - self.position)

        # Shortest-path quaternion interpolation.  Slerp is used away from
        # zero angle; normalized lerp avoids loss of precision nearby.
        current = self.quaternion
        dot = float(np.dot(current, target_quaternion))
        if dot < 0.0:
            target_quaternion = -target_quaternion
            dot = -dot
        rotation_alpha = self._alpha(dt, self.rotation_tau)
        if dot > 0.9995:
            result = current + rotation_alpha * (target_quaternion - current)
        else:
            theta = math.acos(float(np.clip(dot, -1.0, 1.0)))
            sin_theta = math.sin(theta)
            result = (
                math.sin((1.0 - rotation_alpha) * theta) / sin_theta * current +
                math.sin(rotation_alpha * theta) / sin_theta * target_quaternion)
        self.quaternion = normalize_quaternion_xyzw(result)
        return self.position.copy(), self.quaternion.copy()
