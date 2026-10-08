"""Small 6DoF Cartesian range and rate limiter."""

import numpy as np

from .pose_mapper import (
    normalize_quaternion_xyzw, quaternion_angle_rad,
    quaternion_inverse_xyzw, quaternion_multiply_xyzw,
    quaternion_to_rotation_vector_xyzw, rotation_vector_to_quaternion_xyzw,
)


class CartesianPoseLimiter:
    def __init__(self, initial_position, initial_quaternion_xyzw,
                 max_offset_m, max_linear_speed_m_s,
                 max_angular_speed_rad_s,
                 max_linear_acceleration_m_s2=None):
        self.initial_position = np.asarray(initial_position, dtype=float)
        self.initial_quaternion = normalize_quaternion_xyzw(
            initial_quaternion_xyzw)
        self.max_offset_m = float(max_offset_m)
        self.max_linear_speed = float(max_linear_speed_m_s)
        self.max_angular_speed = float(max_angular_speed_rad_s)
        self.max_linear_acceleration = (
            None if max_linear_acceleration_m_s2 is None else
            float(max_linear_acceleration_m_s2))
        if (self.max_linear_acceleration is not None and
                (not np.isfinite(self.max_linear_acceleration) or
                 self.max_linear_acceleration <= 0.0)):
            raise ValueError("max linear acceleration must be positive")
        self.reset()

    def reset(self):
        self.position = self.initial_position.copy()
        self.quaternion = self.initial_quaternion.copy()
        self.linear_velocity = np.zeros(3)

    def step(self, requested_position, requested_quaternion_xyzw, dt):
        dt = float(dt)
        requested = np.asarray(requested_position, dtype=float)
        if requested.shape != (3,) or not np.all(np.isfinite(requested)) or dt <= 0:
            raise ValueError("invalid Cartesian limiter input")
        low = self.initial_position - self.max_offset_m
        high = self.initial_position + self.max_offset_m
        desired = np.clip(requested, low, high)
        delta = desired - self.position
        distance = float(np.linalg.norm(delta))
        if self.max_linear_acceleration is None:
            max_distance = self.max_linear_speed * dt
            if distance > max_distance:
                delta *= max_distance / distance
            self.position += delta
        elif distance > 0.0:
            # Braking-distance speed target plus a vector acceleration bound.
            desired_speed = min(
                self.max_linear_speed,
                np.sqrt(2.0 * self.max_linear_acceleration * distance))
            desired_velocity = delta * (desired_speed / distance)
            velocity_delta = desired_velocity - self.linear_velocity
            max_velocity_delta = self.max_linear_acceleration * dt
            velocity_delta_norm = float(np.linalg.norm(velocity_delta))
            if velocity_delta_norm > max_velocity_delta:
                velocity_delta *= max_velocity_delta / velocity_delta_norm
            self.linear_velocity += velocity_delta
            speed = float(np.linalg.norm(self.linear_velocity))
            if speed > self.max_linear_speed:
                self.linear_velocity *= self.max_linear_speed / speed
            step = self.linear_velocity * dt
            if (float(np.linalg.norm(step)) >= distance or
                    float(np.dot(step, delta)) <= 0.0):
                step = delta
                self.linear_velocity = np.zeros(3)
            self.position += step
        else:
            self.linear_velocity = np.zeros(3)

        desired_q = normalize_quaternion_xyzw(requested_quaternion_xyzw)
        if float(np.dot(desired_q, self.quaternion)) < 0.0:
            desired_q = -desired_q
        relative = quaternion_multiply_xyzw(
            desired_q, quaternion_inverse_xyzw(self.quaternion))
        angle = quaternion_angle_rad(relative, [0, 0, 0, 1])
        max_angle = self.max_angular_speed * dt
        if angle > max_angle:
            vector = quaternion_to_rotation_vector_xyzw(relative)
            relative = rotation_vector_to_quaternion_xyzw(
                vector * (max_angle / angle))
        self.quaternion = quaternion_multiply_xyzw(relative, self.quaternion)
        return self.position.copy(), self.quaternion.copy()
