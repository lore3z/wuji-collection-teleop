"""Pure calculations for guided PICO-to-RM75 axis calibration."""

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np


SEMANTIC_PAIRS = (
    ("forward", "backward", "RM +X (forward)"),
    ("left", "right", "RM +Y (left)"),
    ("up", "down", "RM +Z (up)"),
)


@dataclass(frozen=True)
class CalibrationResult:
    axis_mapping: np.ndarray
    pair_displacements_m: Dict[str, np.ndarray]
    suggested_position_scale: float


def _position(value: Iterable[float]) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError("each sample must be a finite XYZ position")
    return result


def derive_axis_mapping(samples: Dict[str, Iterable[float]],
                        desired_rm_motion_m: float = 0.02) -> CalibrationResult:
    """Derive a signed permutation mapping from six held direction samples.

    Rows map PICO translation to RM base +X (forward), +Y (left), +Z (up).
    """
    rows = []
    axes = []
    displacements = {}
    magnitudes = []
    for positive, negative, _description in SEMANTIC_PAIRS:
        delta = _position(samples[positive]) - _position(samples[negative])
        magnitude = float(np.linalg.norm(delta))
        if magnitude < 0.02:
            raise ValueError(
                f"{positive}/{negative} separation is only {magnitude:.3f} m; "
                "use a clearer 5-10 cm movement")
        axis = int(np.argmax(np.abs(delta)))
        dominance = abs(delta[axis]) / magnitude
        if dominance < 0.75:
            raise ValueError(
                f"{positive}/{negative} is not a clean single-axis movement "
                f"(dominance={dominance:.2f})")
        row = np.zeros(3, dtype=float)
        row[axis] = 1.0 if delta[axis] > 0.0 else -1.0
        rows.append(row)
        axes.append(axis)
        displacements[f"{positive}_minus_{negative}"] = delta
        # Half of the positive-to-negative travel approximates one-way travel.
        magnitudes.append(magnitude / 2.0)

    if len(set(axes)) != 3:
        raise ValueError(
            "direction pairs did not identify three distinct PICO axes; repeat calibration")

    typical_motion = float(np.median(magnitudes))
    scale = min(1.0, desired_rm_motion_m / typical_motion)
    return CalibrationResult(
        axis_mapping=np.vstack(rows),
        pair_displacements_m=displacements,
        suggested_position_scale=scale,
    )

