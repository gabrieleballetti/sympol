import pytest
import numpy as np

from sympol._integer_points_np import find_integer_points
from sympol.polytope import Polytope


@pytest.mark.parametrize("count_only", [True, False])
def test_integer_points(count_only):
    p = Polytope(
        [
            [0, 0, 0],
            [4, 0, 0],
            [0, 2, 0],
            [0, 0, 3],
            [4, 2, 0],
            [4, 0, 3],
            [0, 2, 3],
            [4, 2, 3],
        ]
    )
    (
        interior_points,
        boundary_points,
        saturated_facets,
        n_points,
        n_interior_points,
        forced_stop,
    ) = find_integer_points(
        verts=p.vertices.view(np.ndarray).astype(np.int64),
        ineqs=p.inequalities.view(np.ndarray).astype(np.int64),
        dim=p.dim,
        count_only=count_only,
    )

    assert not forced_stop
    assert n_points == 60
    assert n_interior_points == 6

    if count_only:
        assert interior_points is None
        assert boundary_points is None
        assert saturated_facets is None
    else:
        assert interior_points.shape == (6, 3)
        assert boundary_points.shape == (54, 3)
        assert len(saturated_facets) == 54


def test_integer_points_consistency():
    p = Polytope.random_lattice_polytope(dim=4, n_vertices=20, min=-10, max=10)

    (
        interior_points,
        boundary_points,
        saturated_facets,
        n_points,
        n_interior_points,
        forced_stop,
    ) = find_integer_points(
        verts=p.vertices.view(np.ndarray).astype(np.int64),
        ineqs=p.inequalities.view(np.ndarray).astype(np.int64),
        dim=p.dim,
        count_only=False,
    )

    assert n_points == (interior_points.shape[0]) + boundary_points.shape[0]
    assert len(saturated_facets) == boundary_points.shape[0]


@pytest.mark.parametrize("interior", [True, False])
@pytest.mark.parametrize("count_only", [True, False])
def test_stop_at_max_points(count_only, interior):
    """
    Test that the lattice point enumeration of a (massive) polytope stops
    correctly if the maximum number of requested points is reached.
    """
    p = (
        Polytope(
            vertices=[
                [1, 0, 0],
                [-1, 0, 0],
                [0, 1, 0],
                [0, -1, 0],
                [0, 0, 1],
                [0, 0, -1],
            ]
        )
        * 100
    )

    (
        interior_points,
        boundary_points,
        saturated_facets,
        n_points,
        n_interior_points,
        forced_stop,
    ) = find_integer_points(
        verts=p.vertices.view(np.ndarray).astype(np.int64),
        ineqs=p.inequalities.view(np.ndarray).astype(np.int64),
        dim=p.dim,
        count_only=count_only,
        stop_at=-1 if interior else 10,
        stop_at_interior=10 if interior else -1,
    )

    assert forced_stop

    if interior:
        assert n_interior_points >= 10
        assert n_points >= 10
        if not count_only:
            assert interior_points.shape[0] >= 10
    else:
        assert n_points >= 0
        if not count_only:
            assert interior_points.shape[0] + boundary_points.shape[0] >= 10
