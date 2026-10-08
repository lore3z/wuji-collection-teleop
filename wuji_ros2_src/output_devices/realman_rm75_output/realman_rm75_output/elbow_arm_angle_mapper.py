"""Map WUJI elbow direction to a relative RM75 arm-angle target."""

import math

import numpy as np


class InvalidElbowDirection(ValueError):
    pass


def elbow_swivel_angle_rad(wrist_position, elbow_direction):
    """Return signed elbow direction angle around the shoulder-to-wrist axis."""
    wrist = np.asarray(wrist_position, dtype=float)
    direction = np.asarray(elbow_direction, dtype=float)
    if wrist.shape != (3,) or direction.shape != (3,):
        raise InvalidElbowDirection("wrist and elbow direction must be 3-vectors")
    if not np.all(np.isfinite(wrist)) or not np.all(np.isfinite(direction)):
        raise InvalidElbowDirection("wrist and elbow direction must be finite")
    wrist_norm = float(np.linalg.norm(wrist))
    direction_norm = float(np.linalg.norm(direction))
    if wrist_norm < 1e-6 or direction_norm < 1e-6:
        raise InvalidElbowDirection("wrist axis or elbow direction is singular")
    axis = wrist / wrist_norm
    direction = direction - float(direction @ axis) * axis
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm < 1e-6:
        raise InvalidElbowDirection("elbow direction is parallel to wrist axis")
    direction /= direction_norm

    # Right-chest -Y is anatomically down. Project it into the plane normal to
    # the shoulder-wrist axis to form a repeatable zero-angle direction.
    reference = np.array([0.0, -1.0, 0.0])
    basis_x = reference - float(reference @ axis) * axis
    if float(np.linalg.norm(basis_x)) < 1e-6:
        reference = np.array([0.0, 0.0, 1.0])
        basis_x = reference - float(reference @ axis) * axis
    basis_x /= np.linalg.norm(basis_x)
    basis_y = np.cross(axis, basis_x)
    return math.atan2(float(direction @ basis_y), float(direction @ basis_x))


class RelativeArmAngleMapper:
    def __init__(self, rm_initial_angle_deg, scale=0.25,
                 max_step_deg=5.0):
        self.rm_initial_angle_deg = float(rm_initial_angle_deg)
        self.scale = float(scale)
        self.max_step_deg = float(max_step_deg)
        if not all(np.isfinite(v) for v in (
                self.rm_initial_angle_deg, self.scale, self.max_step_deg)):
            raise ValueError("arm-angle mapping parameters must be finite")
        if self.scale <= 0.0 or self.max_step_deg <= 0.0:
            raise ValueError("arm-angle scale and max step must be positive")
        self.previous_direction = None
        self.previous_target_deg = None

    @staticmethod
    def _axis_and_projected_direction(wrist_position, elbow_direction):
        wrist = np.asarray(wrist_position, dtype=float)
        direction = np.asarray(elbow_direction, dtype=float)
        if wrist.shape != (3,) or direction.shape != (3,):
            raise InvalidElbowDirection(
                "wrist and elbow direction must be 3-vectors")
        if not np.all(np.isfinite(wrist)) or not np.all(np.isfinite(direction)):
            raise InvalidElbowDirection(
                "wrist and elbow direction must be finite")
        wrist_norm = float(np.linalg.norm(wrist))
        if wrist_norm < 1e-6:
            raise InvalidElbowDirection("wrist axis is singular")
        axis = wrist / wrist_norm
        projected = direction - float(direction @ axis) * axis
        projected_norm = float(np.linalg.norm(projected))
        if projected_norm < 1e-6:
            raise InvalidElbowDirection(
                "elbow direction is parallel to wrist axis")
        return axis, projected / projected_norm

    def process(self, wrist_position, elbow_direction):
        axis, direction = self._axis_and_projected_direction(
            wrist_position, elbow_direction)
        if self.previous_direction is None:
            self.previous_direction = direction
            self.previous_target_deg = self.rm_initial_angle_deg
            return self.previous_target_deg

        # Parallel-transport the preceding direction into the current
        # shoulder-wrist normal plane. This avoids the absolute-reference basis
        # flip that occurs when a moving wrist axis approaches the reference.
        previous = self.previous_direction - float(
            self.previous_direction @ axis) * axis
        previous_norm = float(np.linalg.norm(previous))
        if previous_norm < 1e-6:
            self.previous_direction = direction
            raise InvalidElbowDirection(
                "previous elbow direction became singular after transport")
        previous /= previous_norm
        delta_rad = math.atan2(
            float(axis @ np.cross(previous, direction)),
            float(np.clip(previous @ direction, -1.0, 1.0)),
        )
        step_deg = self.scale * math.degrees(delta_rad)
        if abs(step_deg) > self.max_step_deg:
            # Treat a discontinuity as a new input basis while holding the RM
            # target. This prevents a rejection latch on every following frame.
            self.previous_direction = direction
            raise InvalidElbowDirection(
                f"arm-angle target step {abs(step_deg):.3f} deg "
                f"exceeds {self.max_step_deg:.3f} deg")
        target = self.previous_target_deg + step_deg
        self.previous_direction = direction
        self.previous_target_deg = target
        return target
