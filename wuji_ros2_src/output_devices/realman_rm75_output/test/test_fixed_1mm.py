import numpy as np

from realman_rm75_output.rm75_fixed_1mm_test import fixed_offsets


def test_fixed_offsets_are_one_mm_out_hold_and_home():
    offsets = fixed_offsets()
    assert len(offsets) == 50
    assert offsets[0] == 0.00005
    assert np.max(offsets) == 0.001
    assert offsets[-1] == 0.0
    assert np.max(np.abs(np.diff(offsets))) <= 0.00005 + 1e-12
