"""FTP-1 canonical right-hand features from MediaPipe-format hand landmarks.

The implementation mirrors the public FTP-1 human-hand geometry definition,
but takes already-decoded ``(T, 21, 3)`` landmarks instead of MANO 6D poses.
Wuji's ``hand_skeleton`` is already expressed in the wrist frame, and we still
subtract wrist to make the calculation invariant to a global translation.
"""

from __future__ import annotations

# Resolve project imports independently of the current working directory.
import sys as _project_sys
from pathlib import Path as _ProjectPath
_project_root = _ProjectPath(__file__).resolve().parents[1]
if str(_project_root) not in _project_sys.path:
    _project_sys.path.insert(0, str(_project_root))


import numpy as np


FTP1_HAND_NAMES = (
    "palm_adduction", "thumb_adduction", "thumb_mcp", "thumb_ip",
    "index_adduction", "index_mcp", "index_pip", "index_dip",
    "middle_adduction", "middle_mcp", "middle_pip", "middle_dip",
    "ring_adduction", "ring_mcp", "ring_pip", "ring_dip",
    "pinky_adduction", "pinky_mcp", "pinky_pip", "pinky_dip", "palm",
)
FTP1_HAND_FAAS_IDX = np.asarray(
    [1, 26, 2, 3, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24, 27],
    dtype=np.int32,
)


def _signed_vector_angle(v1: np.ndarray, v2: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Exact public FTP-1 signed-angle convention (sign * arccos).

    Do not replace this with atan2: at the zero-cross-product degeneracy the
    upstream FTP-1 parser deliberately returns zero through ``sign(0)``.
    Matching that behaviour is necessary before temporal unwrap can remove
    only representation seams.
    """
    cross = np.cross(v1, v2)
    sign = np.sign(np.sum(cross * normal, axis=-1))
    denom = np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1)
    cosine = np.clip(np.sum(v1 * v2, axis=-1) / (denom + 1e-8), -1.0, 1.0)
    return np.arccos(cosine) * sign


def _signed_plane_angle(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray, normal: np.ndarray, *, swap: bool) -> np.ndarray:
    plane_normal = np.cross(v2, v3)
    plane_normal /= np.maximum(np.linalg.norm(plane_normal, axis=-1, keepdims=True), 1e-8)
    projected_v1 = v1 - np.sum(v1 * plane_normal, axis=-1, keepdims=True) * plane_normal
    return _signed_vector_angle(v2, projected_v1, normal) if swap else _signed_vector_angle(projected_v1, v2, normal)


def _fold_axis_angle(angle: np.ndarray) -> np.ndarray:
    """Remove the 180-degree direction ambiguity of a projected bone axis.

    When a curled finger points back towards the wrist, a directed projected
    vector can jump between +pi and -pi even though its MCP ab/adduction barely
    changed.  Adduction describes the *axis* of that bone in the palm plane, so
    directions separated by pi are equivalent.  Folding to [-pi/2, pi/2]
    expresses that physical equivalence without temporal filtering.
    """
    return np.arctan2(np.sin(angle), np.abs(np.cos(angle)))


def _mcp_adduction(
    metacarpal: np.ndarray,
    proximal: np.ndarray,
    palm_normal: np.ndarray,
) -> np.ndarray:
    """Stable MCP adduction from metacarpal and proximal-phalanx axes.

    The old implementation used MCP-to-fingertip.  Once PIP/DIP flexion made
    that long vector project close to zero or point backwards, its sign became
    numerically unstable.  MCP adduction is determined by MCP-to-PIP instead;
    distal flexion must not change it.
    """
    normal = palm_normal / np.maximum(
        np.linalg.norm(palm_normal, axis=-1, keepdims=True), 1e-8
    )
    forward = metacarpal - np.sum(metacarpal * normal, axis=-1, keepdims=True) * normal
    forward /= np.maximum(np.linalg.norm(forward, axis=-1, keepdims=True), 1e-8)
    lateral = np.cross(normal, forward)
    lateral /= np.maximum(np.linalg.norm(lateral, axis=-1, keepdims=True), 1e-8)
    proximal = proximal / np.maximum(
        np.linalg.norm(proximal, axis=-1, keepdims=True), 1e-8
    )
    # Lateral component is invariant to flexion in the forward/normal plane.
    # asin also keeps the physical MCP side-swing in [-pi/2, pi/2].
    return np.arcsin(np.clip(np.sum(proximal * lateral, axis=-1), -1.0, 1.0))


def ftp1_right_hand_joints_from_mediapipe(landmarks: np.ndarray) -> np.ndarray:
    """Return FTP-1 canonical right-hand joint state ``(T, 21)`` in radians.

    ``landmarks`` must follow the 21-point MediaPipe order: wrist; four thumb
    points; then four points for index, middle, ring and pinky.
    """
    kps = np.asarray(landmarks, dtype=np.float64)
    if kps.ndim != 3 or kps.shape[1:] != (21, 3):
        raise ValueError(f"expected (T,21,3) MediaPipe landmarks, got {kps.shape}")
    if not np.all(np.isfinite(kps)):
        raise ValueError("non-finite MediaPipe landmarks")
    kps = kps - kps[:, 0:1, :]

    palm_normal = np.cross(kps[:, 5] - kps[:, 0], kps[:, 13] - kps[:, 0])
    va = _signed_vector_angle
    pa = _signed_plane_angle

    thumb_mcp = va(kps[:, 2] - kps[:, 1], kps[:, 3] - kps[:, 2], kps[:, 2] - kps[:, 5])
    index_mcp = va(kps[:, 5] - kps[:, 0], kps[:, 8] - kps[:, 5], kps[:, 5] - kps[:, 9])
    middle_mcp = va(kps[:, 9] - kps[:, 0], kps[:, 12] - kps[:, 9], kps[:, 9] - kps[:, 13])
    ring_mcp = va(kps[:, 13] - kps[:, 0], kps[:, 16] - kps[:, 13], kps[:, 13] - kps[:, 17])
    pinky_mcp = va(kps[:, 17] - kps[:, 0], kps[:, 20] - kps[:, 17], kps[:, 13] - kps[:, 17])

    # Use only the segment controlled by the corresponding base joint.
    # Distal points are intentionally excluded so curling PIP/DIP cannot flip
    # an MCP adduction value by roughly pi radians.
    thumb_add = _fold_axis_angle(pa(
        kps[:, 2] - kps[:, 1],
        kps[:, 5] - kps[:, 1],
        kps[:, 13] - kps[:, 1],
        palm_normal,
        swap=True,
    ))
    index_add = _mcp_adduction(kps[:, 5] - kps[:, 0], kps[:, 6] - kps[:, 5], palm_normal)
    middle_add = _mcp_adduction(kps[:, 9] - kps[:, 0], kps[:, 10] - kps[:, 9], palm_normal)
    ring_add = _mcp_adduction(kps[:, 13] - kps[:, 0], kps[:, 14] - kps[:, 13], palm_normal)
    pinky_add = _mcp_adduction(kps[:, 17] - kps[:, 0], kps[:, 18] - kps[:, 17], palm_normal)

    thumb_ip = va(kps[:, 3] - kps[:, 2], kps[:, 4] - kps[:, 3], kps[:, 3] - kps[:, 5])
    index_pip = va(kps[:, 6] - kps[:, 5], kps[:, 7] - kps[:, 6], np.cross(kps[:, 6] - kps[:, 5], kps[:, 7] - kps[:, 6]))
    middle_pip = va(kps[:, 10] - kps[:, 9], kps[:, 11] - kps[:, 10], np.cross(kps[:, 10] - kps[:, 9], kps[:, 11] - kps[:, 10]))
    ring_pip = va(kps[:, 14] - kps[:, 13], kps[:, 15] - kps[:, 14], np.cross(kps[:, 14] - kps[:, 13], kps[:, 15] - kps[:, 14]))
    pinky_pip = va(kps[:, 18] - kps[:, 17], kps[:, 19] - kps[:, 18], np.cross(kps[:, 18] - kps[:, 17], kps[:, 19] - kps[:, 18]))
    index_dip = va(kps[:, 7] - kps[:, 6], kps[:, 8] - kps[:, 7], np.cross(kps[:, 7] - kps[:, 6], kps[:, 8] - kps[:, 7]))
    middle_dip = va(kps[:, 11] - kps[:, 10], kps[:, 12] - kps[:, 11], np.cross(kps[:, 11] - kps[:, 10], kps[:, 12] - kps[:, 11]))
    ring_dip = va(kps[:, 15] - kps[:, 14], kps[:, 16] - kps[:, 15], np.cross(kps[:, 15] - kps[:, 14], kps[:, 16] - kps[:, 15]))
    pinky_dip = va(kps[:, 19] - kps[:, 18], kps[:, 20] - kps[:, 19], np.cross(kps[:, 19] - kps[:, 18], kps[:, 20] - kps[:, 19]))
    palm = np.pi / 2.0 - va(kps[:, 2] - kps[:, 1], kps[:, 17] - kps[:, 1], np.cross(kps[:, 2] - kps[:, 1], kps[:, 17] - kps[:, 1]))
    palm_add = _fold_axis_angle(pa(kps[:, 4] - kps[:, 1], kps[:, 9] - kps[:, 1], kps[:, 13] - kps[:, 1], palm_normal, swap=True))

    return np.stack((
        palm_add, thumb_add, thumb_mcp, thumb_ip,
        index_add, index_mcp, index_pip, index_dip,
        middle_add, middle_mcp, middle_pip, middle_dip,
        ring_add, ring_mcp, ring_pip, ring_dip,
        pinky_add, pinky_mcp, pinky_pip, pinky_dip, palm,
    ), axis=-1).astype(np.float32)


def _unwrap_joint_trajectory(values: np.ndarray) -> np.ndarray:
    """Remove only +/-pi representation wraps along time.

    FTP-1's published adduction equations use a directed projected vector.
    That representation is periodic, so a physically continuous movement may
    numerically jump from +pi to -pi.  ``numpy.unwrap`` keeps the published
    geometry while making its time trajectory continuous; it does *not*
    smooth, clamp, or alter real motion.
    """
    if len(values) < 2:
        return values
    output = np.unwrap(values, axis=0)
    # FTP-1's projected-bone adduction equations describe an *axis*, not an
    # arrow: theta and theta+pi are the same physical MCP side-swing.  Deep
    # curl can swap that axis direction, yielding a pi (rather than 2*pi)
    # representation seam.  Unwrap these six axis-valued channels with period
    # pi while retaining the public formula itself.
    axis_channels = (0, 1, 4, 8, 12, 16)
    output[:, axis_channels] = np.unwrap(values[:, axis_channels] * 2.0, axis=0) / 2.0
    return output.astype(np.float32)


def ftp1_right_hand_joints_official_from_mediapipe(landmarks: np.ndarray) -> np.ndarray:
    """Published FTP-1 right-hand equations on MediaPipe landmarks.

    This is the exact right-hand equation structure in FTP-1's
    ``get_hand_joints_mano_single_hand`` adapted from MANO wrist coordinates
    to an already wrist-relative MediaPipe 21-landmark skeleton.  Its output
    is unwrapped only along time to eliminate the +/-pi representation seam.
    Use this as ``right_hand_joints`` for FTP-1; retain the stable and Wuji
    anatomical outputs separately for diagnostics and embodiment-specific
    control.
    """
    kps = np.asarray(landmarks, dtype=np.float64)
    if kps.ndim != 3 or kps.shape[1:] != (21, 3):
        raise ValueError(f"expected (T,21,3) MediaPipe landmarks, got {kps.shape}")
    if not np.all(np.isfinite(kps)):
        raise ValueError("non-finite MediaPipe landmarks")
    kps = kps - kps[:, 0:1, :]
    palm_normal = np.cross(kps[:, 5] - kps[:, 0], kps[:, 13] - kps[:, 0])
    va = _signed_vector_angle
    pa = _signed_plane_angle

    thumb_mcp = va(kps[:, 2] - kps[:, 1], kps[:, 3] - kps[:, 2], kps[:, 2] - kps[:, 5])
    index_mcp = va(kps[:, 5] - kps[:, 0], kps[:, 8] - kps[:, 5], kps[:, 5] - kps[:, 9])
    middle_mcp = va(kps[:, 9] - kps[:, 0], kps[:, 12] - kps[:, 9], kps[:, 9] - kps[:, 13])
    ring_mcp = va(kps[:, 13] - kps[:, 0], kps[:, 16] - kps[:, 13], kps[:, 13] - kps[:, 17])
    pinky_mcp = va(kps[:, 17] - kps[:, 0], kps[:, 20] - kps[:, 17], kps[:, 13] - kps[:, 17])
    thumb_add = pa(kps[:, 4] - kps[:, 2], kps[:, 5] - kps[:, 2], kps[:, 13] - kps[:, 2], palm_normal, swap=True)
    index_add = pa(kps[:, 8] - kps[:, 5], kps[:, 5] - kps[:, 0], kps[:, 9] - kps[:, 0], palm_normal, swap=True)
    middle_add = pa(kps[:, 12] - kps[:, 9], kps[:, 9] - kps[:, 0], kps[:, 13] - kps[:, 0], palm_normal, swap=True)
    ring_add = pa(kps[:, 16] - kps[:, 13], kps[:, 13] - kps[:, 0], kps[:, 17] - kps[:, 0], palm_normal, swap=True)
    pinky_add = pa(kps[:, 20] - kps[:, 17], kps[:, 17] - kps[:, 0], kps[:, 13] - kps[:, 0], palm_normal, swap=True)
    thumb_ip = va(kps[:, 3] - kps[:, 2], kps[:, 4] - kps[:, 3], kps[:, 3] - kps[:, 5])
    index_pip = va(kps[:, 6] - kps[:, 5], kps[:, 7] - kps[:, 6], np.cross(kps[:, 6] - kps[:, 5], kps[:, 7] - kps[:, 6]))
    middle_pip = va(kps[:, 10] - kps[:, 9], kps[:, 11] - kps[:, 10], np.cross(kps[:, 10] - kps[:, 9], kps[:, 11] - kps[:, 10]))
    ring_pip = va(kps[:, 14] - kps[:, 13], kps[:, 15] - kps[:, 14], np.cross(kps[:, 14] - kps[:, 13], kps[:, 15] - kps[:, 14]))
    pinky_pip = va(kps[:, 18] - kps[:, 17], kps[:, 19] - kps[:, 18], np.cross(kps[:, 18] - kps[:, 17], kps[:, 19] - kps[:, 18]))
    index_dip = va(kps[:, 7] - kps[:, 6], kps[:, 8] - kps[:, 7], np.cross(kps[:, 7] - kps[:, 6], kps[:, 8] - kps[:, 7]))
    middle_dip = va(kps[:, 11] - kps[:, 10], kps[:, 12] - kps[:, 11], np.cross(kps[:, 11] - kps[:, 10], kps[:, 12] - kps[:, 11]))
    ring_dip = va(kps[:, 15] - kps[:, 14], kps[:, 16] - kps[:, 15], np.cross(kps[:, 15] - kps[:, 14], kps[:, 16] - kps[:, 15]))
    pinky_dip = va(kps[:, 19] - kps[:, 18], kps[:, 20] - kps[:, 19], np.cross(kps[:, 19] - kps[:, 18], kps[:, 20] - kps[:, 19]))
    palm = np.pi / 2.0 - va(kps[:, 2] - kps[:, 1], kps[:, 17] - kps[:, 1], np.cross(kps[:, 2] - kps[:, 1], kps[:, 17] - kps[:, 1]))
    palm_add = pa(kps[:, 4] - kps[:, 1], kps[:, 9] - kps[:, 1], kps[:, 13] - kps[:, 1], palm_normal, swap=True)
    values = np.stack((
        palm_add, thumb_add, thumb_mcp, thumb_ip,
        index_add, index_mcp, index_pip, index_dip,
        middle_add, middle_mcp, middle_pip, middle_dip,
        ring_add, ring_mcp, ring_pip, ring_dip,
        pinky_add, pinky_mcp, pinky_pip, pinky_dip, palm,
    ), axis=-1).astype(np.float32)
    return _unwrap_joint_trajectory(values)
