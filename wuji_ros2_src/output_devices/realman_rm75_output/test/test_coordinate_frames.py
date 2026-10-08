import numpy as np

from realman_rm75_output.coordinate_frames import (
    WUJI_RIGHT_CHEST_TO_RM_BASE,
    wuji_right_chest_to_rm_base_flat,
)


def test_wuji_right_chest_mapping_matches_acrealman_composition():
    # chest [forward, up, right] -> Project [right, up, back]
    chest_to_project = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ])
    # acRealman right arm: RM = [Project up, Project back, Project right]
    acrealman_project_to_rm = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ])
    np.testing.assert_allclose(
        WUJI_RIGHT_CHEST_TO_RM_BASE,
        acrealman_project_to_rm @ chest_to_project)


def test_wuji_right_chest_mapping_is_a_proper_rotation():
    matrix = WUJI_RIGHT_CHEST_TO_RM_BASE
    np.testing.assert_allclose(matrix @ matrix.T, np.eye(3))
    assert np.linalg.det(matrix) == 1.0
    assert wuji_right_chest_to_rm_base_flat() == matrix.reshape(-1).tolist()
