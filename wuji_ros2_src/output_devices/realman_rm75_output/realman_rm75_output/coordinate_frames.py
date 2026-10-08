"""Authoritative proper rotations for the RM75 retargeting boundary."""

import numpy as np


# WUJI /right_arm_target_pose uses right-chest axes [forward, up, right].
# acRealman_xr uses Project axes [right, up, back] and maps them to its RM75
# base as [up, back, right].  Composing those two documented conventions gives
# the matrix below.  It must be used for both translation and SO(3) conjugation.
WUJI_RIGHT_CHEST_TO_RM_BASE = np.array([
    [0.0, 1.0, 0.0],
    [-1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
])


def wuji_right_chest_to_rm_base_flat():
    """Return a fresh row-major list suitable for ROS parameters/mappers."""
    return WUJI_RIGHT_CHEST_TO_RM_BASE.reshape(-1).tolist()
