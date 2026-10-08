import numpy as np
import pytest

from realman_rm75_output.readonly_preflight import base_x_offsets


def test_base_x_offsets_cover_current_negative_current_positive():
    offsets = base_x_offsets(0.005, 0.0005)
    assert offsets[0] == pytest.approx(0.0)
    assert np.min(offsets) == pytest.approx(-0.005)
    assert np.max(offsets) == pytest.approx(0.005)
    minimum_index = int(np.argmin(offsets))
    assert 0.0 in offsets[minimum_index + 1:]
    assert np.max(np.abs(np.diff(offsets))) <= 0.0005 + 1e-12


@pytest.mark.parametrize("limit,step", [(0, 0.0005), (0.005, 0),
                                         (0.005, 0.006)])
def test_base_x_offsets_reject_invalid_values(limit, step):
    with pytest.raises(ValueError):
        base_x_offsets(limit, step)
