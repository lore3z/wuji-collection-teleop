"""Minimal RM-X Cartesian limiter for the first motion rehearsal."""

import numpy as np


class SingleAxisTargetLimiter:
    def __init__(self, initial_position, initial_quaternion_xyzw,
                 max_offset_m=0.005, max_speed_m_s=0.005, max_step_m=None):
        self.initial_position = np.asarray(initial_position, dtype=float)
        self.initial_quaternion = np.asarray(initial_quaternion_xyzw, dtype=float)
        if self.initial_position.shape != (3,) or self.initial_quaternion.shape != (4,):
            raise ValueError("initial pose has invalid shape")
        self.max_offset_m = float(max_offset_m)
        self.max_speed_m_s = float(max_speed_m_s)
        self.max_step_m = (float("inf") if max_step_m is None
                           else float(max_step_m))
        if (self.max_offset_m <= 0.0 or self.max_speed_m_s <= 0.0 or
                self.max_step_m <= 0.0):
            raise ValueError("limits must be positive")
        self.current_x = float(self.initial_position[0])

    def reset(self):
        self.current_x = float(self.initial_position[0])

    def step(self, requested_position, dt):
        requested = np.asarray(requested_position, dtype=float)
        dt = float(dt)
        if requested.shape != (3,) or not np.all(np.isfinite(requested)) or dt <= 0.0:
            raise ValueError("requested position and dt must be valid")
        low = self.initial_position[0] - self.max_offset_m
        high = self.initial_position[0] + self.max_offset_m
        desired_x = float(np.clip(requested[0], low, high))
        max_step = min(self.max_speed_m_s * dt, self.max_step_m)
        self.current_x += float(np.clip(desired_x - self.current_x, -max_step, max_step))
        position = self.initial_position.copy()
        position[0] = self.current_x
        return position, self.initial_quaternion.copy()
