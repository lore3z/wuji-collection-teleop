"""Pure calculations for guided PICO rotational-axis calibration."""

from dataclasses import dataclass
import math
from typing import Dict, Iterable

import numpy as np


ROTATION_PAIRS = (
    ("rm_x_positive", "rm_x_negative", "RM +X rotation (forward axis)"),
    ("rm_y_positive", "rm_y_negative", "RM +Y rotation (left axis)"),
    ("rm_z_positive", "rm_z_negative", "RM +Z rotation (up axis)"),
)


@dataclass(frozen=True)
class RotationCalibrationResult:
    rotation_axis_mapping: np.ndarray
    pair_rotation_vectors_rad: Dict[str, np.ndarray]
    suggested_rotation_scale: float


def normalize_quaternion_xyzw(value: Iterable[float]) -> np.ndarray:
    q = np.asarray(value, dtype=float)
    if q.shape != (4,) or not np.all(np.isfinite(q)):
        raise ValueError("each quaternion must contain four finite values")
    norm = float(np.linalg.norm(q))
    if norm < 1e-9:
        raise ValueError("quaternion norm is zero")
    return q / norm


def quaternion_inverse_xyzw(value: Iterable[float]) -> np.ndarray:
    x, y, z, w = normalize_quaternion_xyzw(value)
    return np.array([-x, -y, -z, w])


def quaternion_multiply_xyzw(left: Iterable[float],
                             right: Iterable[float]) -> np.ndarray:
    x1, y1, z1, w1 = normalize_quaternion_xyzw(left)
    x2, y2, z2, w2 = normalize_quaternion_xyzw(right)
    return normalize_quaternion_xyzw([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ])


def quaternion_to_rotation_vector_xyzw(value: Iterable[float]) -> np.ndarray:
    q = normalize_quaternion_xyzw(value)
    if q[3] < 0.0:
        q = -q
    vector_norm = float(np.linalg.norm(q[:3]))
    if vector_norm < 1e-12:
        return np.zeros(3)
    angle = 2.0 * math.atan2(vector_norm, float(q[3]))
    return q[:3] * (angle / vector_norm)


def relative_rotation_vector(current_xyzw, neutral_xyzw) -> np.ndarray:
    relative = quaternion_multiply_xyzw(
        current_xyzw, quaternion_inverse_xyzw(neutral_xyzw))
    return quaternion_to_rotation_vector_xyzw(relative)


def derive_rotation_axis_mapping(samples: Dict[str, Iterable[float]],
                                 desired_rm_rotation_deg: float = 5.0
                                 ) -> RotationCalibrationResult:
    neutral = normalize_quaternion_xyzw(samples["neutral"])
    rows = []
    axes = []
    pair_vectors = {}
    one_way_angles = []
    for positive, negative, _description in ROTATION_PAIRS:
        positive_vector = relative_rotation_vector(samples[positive], neutral)
        negative_vector = relative_rotation_vector(samples[negative], neutral)
        pair_vector = positive_vector - negative_vector
        magnitude = float(np.linalg.norm(pair_vector))
        if magnitude < math.radians(10.0):
            raise ValueError(
                f"{positive}/{negative} separation is only "
                f"{math.degrees(magnitude):.1f} deg; use clearer 15-25 deg rotations")
        axis = int(np.argmax(np.abs(pair_vector)))
        dominance = abs(pair_vector[axis]) / magnitude
        if dominance < 0.75:
            raise ValueError(
                f"{positive}/{negative} is not a clean single-axis rotation "
                f"(dominance={dominance:.2f})")
        row = np.zeros(3)
        row[axis] = 1.0 if pair_vector[axis] > 0.0 else -1.0
        rows.append(row)
        axes.append(axis)
        pair_vectors[f"{positive}_minus_{negative}"] = pair_vector
        one_way_angles.append(magnitude / 2.0)

    if len(set(axes)) != 3:
        raise ValueError(
            "rotation pairs did not identify three distinct PICO axes; repeat calibration")

    typical_angle = float(np.median(one_way_angles))
    scale = min(1.0, math.radians(desired_rm_rotation_deg) / typical_angle)
    return RotationCalibrationResult(
        rotation_axis_mapping=np.vstack(rows),
        pair_rotation_vectors_rad=pair_vectors,
        suggested_rotation_scale=scale,
    )

