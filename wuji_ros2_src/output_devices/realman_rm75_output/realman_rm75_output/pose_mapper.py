"""ROS-independent relative-pose mapping and safety checks."""

from dataclasses import dataclass
import math
import time
from typing import Optional

import numpy as np


class InvalidPose(ValueError):
    """Input pose is malformed, unsafe, or outside configured limits."""


def normalize_quaternion_xyzw(value: np.ndarray) -> np.ndarray:
    q = np.asarray(value, dtype=float)
    if q.shape != (4,) or not np.all(np.isfinite(q)):
        raise InvalidPose("quaternion must contain four finite values")
    norm = float(np.linalg.norm(q))
    if norm < 1e-9:
        raise InvalidPose("quaternion norm is zero")
    return q / norm


def quaternion_wxyz_to_xyzw(value: np.ndarray) -> np.ndarray:
    """Convert RealMan/API order [w,x,y,z] to internal/ROS [x,y,z,w]."""
    q = np.asarray(value, dtype=float)
    if q.shape != (4,) or not np.all(np.isfinite(q)):
        raise InvalidPose("wxyz quaternion must contain four finite values")
    return normalize_quaternion_xyzw(np.array([q[1], q[2], q[3], q[0]]))


def quaternion_xyzw_to_wxyz(value: np.ndarray) -> np.ndarray:
    """Convert internal/ROS order [x,y,z,w] to RealMan/API [w,x,y,z]."""
    x, y, z, w = normalize_quaternion_xyzw(value)
    return np.array([w, x, y, z])


def continuous_quaternion_xyzw(value: np.ndarray, reference: np.ndarray) -> np.ndarray:
    q = normalize_quaternion_xyzw(value)
    ref = normalize_quaternion_xyzw(reference)
    return -q if float(np.dot(q, ref)) < 0.0 else q


def quaternion_multiply_xyzw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = normalize_quaternion_xyzw(left)
    x2, y2, z2, w2 = normalize_quaternion_xyzw(right)
    return normalize_quaternion_xyzw(np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ]))


def quaternion_inverse_xyzw(value: np.ndarray) -> np.ndarray:
    x, y, z, w = normalize_quaternion_xyzw(value)
    return np.array([-x, -y, -z, w])


def quaternion_to_matrix_xyzw(value: np.ndarray) -> np.ndarray:
    x, y, z, w = normalize_quaternion_xyzw(value)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def matrix_to_quaternion_xyzw(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=float)
    if m.shape != (3, 3) or not np.all(np.isfinite(m)):
        raise InvalidPose("rotation matrix must be finite and 3x3")
    # Stable branch-based conversion.
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = np.array([(m[2, 1]-m[1, 2])/s, (m[0, 2]-m[2, 0])/s,
                      (m[1, 0]-m[0, 1])/s, 0.25*s])
    else:
        i = int(np.argmax(np.diag(m)))
        if i == 0:
            s = math.sqrt(max(0.0, 1.0+m[0, 0]-m[1, 1]-m[2, 2]))*2.0
            q = np.array([0.25*s, (m[0, 1]+m[1, 0])/s,
                          (m[0, 2]+m[2, 0])/s, (m[2, 1]-m[1, 2])/s])
        elif i == 1:
            s = math.sqrt(max(0.0, 1.0+m[1, 1]-m[0, 0]-m[2, 2]))*2.0
            q = np.array([(m[0, 1]+m[1, 0])/s, 0.25*s,
                          (m[1, 2]+m[2, 1])/s, (m[0, 2]-m[2, 0])/s])
        else:
            s = math.sqrt(max(0.0, 1.0+m[2, 2]-m[0, 0]-m[1, 1]))*2.0
            q = np.array([(m[0, 2]+m[2, 0])/s, (m[1, 2]+m[2, 1])/s,
                          0.25*s, (m[1, 0]-m[0, 1])/s])
    return normalize_quaternion_xyzw(q)


def quaternion_angle_rad(left: np.ndarray, right: np.ndarray) -> float:
    dot = abs(float(np.dot(normalize_quaternion_xyzw(left),
                           normalize_quaternion_xyzw(right))))
    return 2.0 * math.acos(float(np.clip(dot, -1.0, 1.0)))


def quaternion_to_rotation_vector_xyzw(value: np.ndarray) -> np.ndarray:
    q = normalize_quaternion_xyzw(value)
    if q[3] < 0.0:
        q = -q
    vector_norm = float(np.linalg.norm(q[:3]))
    if vector_norm < 1e-12:
        return np.zeros(3)
    angle = 2.0 * math.atan2(vector_norm, float(q[3]))
    return q[:3] * (angle / vector_norm)


def rotation_vector_to_quaternion_xyzw(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise InvalidPose("rotation vector must contain three finite values")
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = vector / angle
    return normalize_quaternion_xyzw(
        np.r_[axis * math.sin(angle / 2.0), math.cos(angle / 2.0)])


@dataclass(frozen=True)
class MappingResult:
    pico_position: np.ndarray
    pico_quaternion_xyzw: np.ndarray
    relative_translation: np.ndarray
    relative_rotation_xyzw: np.ndarray
    mapped_relative_rotation_xyzw: np.ndarray
    mapped_translation: np.ndarray
    target_position: np.ndarray
    target_quaternion_xyzw: np.ndarray


class RelativePoseMapper:
    def __init__(self, axis_mapping, position_scale, max_position_jump_m,
                 max_rotation_jump_rad, workspace_min, workspace_max,
                 tracker_timeout_sec, clock=time.monotonic,
                 translation_only=False, freeze_timeout_sec=0.75,
                 rotation_only=False, rotation_scale=1.0):
        self.axis_mapping = np.asarray(axis_mapping, dtype=float).reshape(3, 3)
        if not np.all(np.isfinite(self.axis_mapping)):
            raise ValueError("axis_mapping contains non-finite values")
        if not np.allclose(self.axis_mapping @ self.axis_mapping.T, np.eye(3), atol=1e-6):
            raise ValueError("axis_mapping must be an orthonormal matrix")
        if not math.isclose(float(np.linalg.det(self.axis_mapping)), 1.0,
                            abs_tol=1e-6):
            raise ValueError("axis_mapping must be a proper rotation with determinant +1")
        self.position_scale = float(position_scale)
        self.max_position_jump_m = float(max_position_jump_m)
        self.max_rotation_jump_rad = float(max_rotation_jump_rad)
        self.workspace_min = np.asarray(workspace_min, dtype=float)
        self.workspace_max = np.asarray(workspace_max, dtype=float)
        if np.any(self.workspace_min >= self.workspace_max):
            raise ValueError("workspace_min must be below workspace_max")
        self.tracker_timeout_sec = float(tracker_timeout_sec)
        self.translation_only = bool(translation_only)
        self.rotation_only = bool(rotation_only)
        if self.translation_only and self.rotation_only:
            raise ValueError("translation_only and rotation_only are mutually exclusive")
        self.freeze_timeout_sec = float(freeze_timeout_sec)
        self.rotation_scale = float(rotation_scale)
        scalar_values = (
            self.position_scale, self.max_position_jump_m,
            self.max_rotation_jump_rad, self.tracker_timeout_sec,
            self.freeze_timeout_sec,
            self.rotation_scale,
        )
        if not all(np.isfinite(v) for v in scalar_values):
            raise ValueError("mapping scalar parameters must be finite")
        if self.position_scale <= 0.0:
            raise ValueError("position_scale must be positive")
        if min(self.max_position_jump_m, self.max_rotation_jump_rad,
               self.tracker_timeout_sec) <= 0.0:
            raise ValueError("jump limits and tracker timeout must be positive")
        if self.freeze_timeout_sec <= 0.0:
            raise ValueError("freeze timeout must be positive")
        if self.rotation_scale <= 0.0:
            raise ValueError("rotation scale must be positive")
        if self.workspace_min.shape != (3,) or self.workspace_max.shape != (3,):
            raise ValueError("workspace bounds must each contain three values")
        if not (np.all(np.isfinite(self.workspace_min)) and
                np.all(np.isfinite(self.workspace_max))):
            raise ValueError("workspace bounds must be finite")
        self.clock = clock
        self.rm_initial_position: Optional[np.ndarray] = None
        self.rm_initial_quaternion: Optional[np.ndarray] = None
        self.pico_initial_position: Optional[np.ndarray] = None
        self.pico_initial_quaternion: Optional[np.ndarray] = None
        self.previous_position: Optional[np.ndarray] = None
        self.previous_quaternion: Optional[np.ndarray] = None
        self.last_message_time: Optional[float] = None
        self.last_pose_change_time: Optional[float] = None
        self.last_result: Optional[MappingResult] = None

    def set_rm_initial_pose(self, position, quaternion_xyzw):
        position = np.asarray(position, dtype=float)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise InvalidPose("RM initial position is invalid")
        self.rm_initial_position = position.copy()
        self.rm_initial_quaternion = normalize_quaternion_xyzw(quaternion_xyzw)

    def reset_tracker_reference(self):
        """Forget PICO history so the next valid frame becomes the baseline."""
        self.pico_initial_position = None
        self.pico_initial_quaternion = None
        self.previous_position = None
        self.previous_quaternion = None
        self.last_message_time = None
        self.last_pose_change_time = None
        self.last_result = None

    def process(self, position, quaternion_xyzw, received_at=None) -> MappingResult:
        if self.rm_initial_position is None:
            raise RuntimeError("RM initial pose has not been set")
        now = self.clock() if received_at is None else float(received_at)
        if not np.isfinite(now):
            raise InvalidPose("received_at must be finite")
        position = np.asarray(position, dtype=float)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise InvalidPose("PICO position must contain three finite values")
        quaternion = normalize_quaternion_xyzw(quaternion_xyzw)
        if self.previous_quaternion is not None:
            quaternion = continuous_quaternion_xyzw(quaternion, self.previous_quaternion)
        if self.previous_position is not None:
            position_jump = float(np.linalg.norm(position - self.previous_position))
            rotation_jump = quaternion_angle_rad(quaternion, self.previous_quaternion)
            pose_changed = position_jump > 1e-7 or rotation_jump > 1e-6
            if pose_changed:
                self.last_pose_change_time = now
            elif (self.last_pose_change_time is not None and
                  now - self.last_pose_change_time > self.freeze_timeout_sec):
                raise InvalidPose(
                    f"tracker pose frozen for {now-self.last_pose_change_time:.3f} s")
            if position_jump > self.max_position_jump_m:
                raise InvalidPose(f"position jump {position_jump:.6f} m exceeds limit")
            if not self.translation_only and rotation_jump > self.max_rotation_jump_rad:
                raise InvalidPose(f"rotation jump {rotation_jump:.6f} rad exceeds limit")
        else:
            self.last_pose_change_time = now

        if self.pico_initial_position is None:
            self.pico_initial_position = position.copy()
            self.pico_initial_quaternion = quaternion.copy()

        relative_translation = position - self.pico_initial_position
        relative_quaternion = quaternion_multiply_xyzw(
            quaternion, quaternion_inverse_xyzw(self.pico_initial_quaternion))
        if self.translation_only:
            mapped_relative_quaternion = np.array([0.0, 0.0, 0.0, 1.0])
        else:
            source_rotation = quaternion_to_matrix_xyzw(relative_quaternion)
            mapped_rotation = (
                self.axis_mapping @ source_rotation @ self.axis_mapping.T
            )
            mapped_relative_quaternion = matrix_to_quaternion_xyzw(mapped_rotation)
            mapped_rotation_vector = quaternion_to_rotation_vector_xyzw(
                mapped_relative_quaternion)
            mapped_relative_quaternion = rotation_vector_to_quaternion_xyzw(
                self.rotation_scale * mapped_rotation_vector)
        mapped_translation = self.position_scale * (self.axis_mapping @ relative_translation)
        target_position = (self.rm_initial_position.copy() if self.rotation_only else
                           self.rm_initial_position + mapped_translation)
        if np.any(target_position < self.workspace_min) or np.any(target_position > self.workspace_max):
            raise InvalidPose(f"target position {target_position.tolist()} outside workspace")
        # Relative rotation is expressed in the RM base frame, so left-compose it.
        target_quaternion = quaternion_multiply_xyzw(
            mapped_relative_quaternion, self.rm_initial_quaternion)
        result = MappingResult(position.copy(), quaternion.copy(),
                               relative_translation, relative_quaternion,
                               mapped_relative_quaternion, mapped_translation,
                               target_position, target_quaternion)
        self.previous_position = position.copy()
        self.previous_quaternion = quaternion.copy()
        self.last_message_time = now
        self.last_result = result
        return result

    def tracker_timed_out(self, now=None) -> bool:
        if self.last_message_time is None:
            return True
        current = self.clock() if now is None else float(now)
        return current - self.last_message_time > self.tracker_timeout_sec
