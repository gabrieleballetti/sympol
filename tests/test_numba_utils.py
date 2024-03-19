import pytest
import numpy as np
from sympol._numba_utils import nb_cartesian_product, _cproduct_idx


@pytest.mark.parametrize("disable_numba", [True, False])
def test_nb_cartesian_product(disable_numba):
    a = np.array([1, 2, 3])
    b = np.array([4, 5])

    func = nb_cartesian_product.py_func if disable_numba else nb_cartesian_product

    expected = np.array(
        [
            [1, 4],
            [2, 4],
            [3, 4],
            [1, 5],
            [2, 5],
            [3, 5],
        ]
    )

    assert np.array_equal(func([a, b]), expected)


@pytest.mark.parametrize("disable_numba", [True, False])
def test_cproduct_idx(disable_numba):
    a = np.array([3, 2, 1])

    func = _cproduct_idx.py_func if disable_numba else _cproduct_idx

    expected = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [2, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [2, 1, 0],
        ]
    )

    assert np.array_equal(func(a), expected)
