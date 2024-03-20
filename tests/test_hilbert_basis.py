import pytest

import numpy as np
from sympol import Polytope
from sympol._hilbert_basis import (
    _get_hilbert_basis_hom,
    _check_hilbert_basis_in_polytope,
)


@pytest.mark.parametrize("disable_numba", [True, False])
def test_get_hilbert_basis_hom(disable_numba):
    generators = np.array(
        [
            [1, 0, 0, 0],
            [1, 0, 0, 1],
            [1, 0, 1, -1],
            [1, 0, 1, 0],
            [1, 1, 0, -1],
            [1, 1, 0, 0],
            [1, 1, 1, 2],
            [1, 1, 1, 3],
            [2, 0, 1, 0],
            [2, 1, 1, -1],
            [2, 1, 1, 0],
            [2, 1, 1, 1],
            [2, 1, 1, 2],
            [4, 2, 2, 4],  # redundant
        ],
    )

    inequalities = np.array(
        [
            [4, -3, -3, 1],
            [0, 1, 0, 0],
            [0, 1, 1, 1],
            [0, 0, 1, 0],
            [1, -1, 3, -1],
            [1, 3, -1, -1],
            [1, 0, -1, 0],
            [1, -1, 0, 0],
        ],
        dtype=object,
    )

    hb = _get_hilbert_basis_hom(
        generators=generators,
        inequalities=inequalities,
        stop_at_height=2,
        disable_numba=disable_numba,
    )

    expected_hb = np.array(
        [
            [1, 0, 0, 0],
            [1, 0, 0, 1],
            [1, 0, 1, -1],
            [1, 0, 1, 0],
            [1, 1, 0, -1],
            [1, 1, 0, 0],
            [1, 1, 1, 2],
            [1, 1, 1, 3],
            [2, 1, 1, 1],
        ]
    )

    assert np.array_equal(hb, expected_hb)


@pytest.mark.parametrize("disable_numba", [True, False])
def test_check_hilbert_basis_in_polytope(disable_numba):
    p = Polytope.reeve_simplex(3, 2)

    generators = np.array(p.half_open_parallelotopes_pts[1:, 1:])

    irreducibles = np.empty((0, generators.shape[1]), dtype=generators.dtype)

    ineqs = np.array(p.inequalities)
    cone_ineqs = ineqs[ineqs[:, 0] == 0]
    cone_ineqs = cone_ineqs[:, 1:]
    other_ineqs = ineqs[ineqs[:, 0] != 0]

    assert not _check_hilbert_basis_in_polytope(
        generators=generators,
        irreducibles=irreducibles,
        cone_inequalities=cone_ineqs,
        other_inequalities=other_ineqs,
        disable_numba=disable_numba,
    )

    p = Polytope.reeve_simplex(3, 2) * 2

    generators = np.array(p.half_open_parallelotopes_pts[1:, 1:])

    irreducibles = np.empty((0, generators.shape[1]), dtype=generators.dtype)

    ineqs = np.array(p.inequalities)
    cone_ineqs = ineqs[ineqs[:, 0] == 0]
    cone_ineqs = cone_ineqs[:, 1:]
    other_ineqs = ineqs[ineqs[:, 0] != 0]

    assert _check_hilbert_basis_in_polytope(
        generators=generators,
        irreducibles=irreducibles,
        cone_inequalities=cone_ineqs,
        other_inequalities=other_ineqs,
        disable_numba=disable_numba,
    )
