import pytest

from sympol._integer_points import _find_integer_points
from sympol.polytope import Polytope


@pytest.mark.parametrize("disable_numba", [True, False])
@pytest.mark.parametrize("count_only", [True, False])
def test_integer_points(count_only, disable_numba):
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
    ) = _find_integer_points(
        verts=p.vertices,
        ineqs=p.inequalities,
        dim=p.dim,
        count_only=count_only,
        disable_numba=disable_numba,
    )

    assert not forced_stop
    assert n_points == 60
    assert n_interior_points == 6

    if count_only:
        assert interior_points.shape == (0, 3)
        assert boundary_points.shape == (0, 3)
        assert saturated_facets.shape == (0, 6)
    else:
        assert interior_points.shape == (6, 3)
        assert boundary_points.shape == (54, 3)
        assert saturated_facets.shape == (54, 6)


@pytest.mark.parametrize("disable_numba", [True, False])
def test_integer_points_consistency(disable_numba):
    p = Polytope.random_lattice_polytope(dim=3, n_points=6, min=-3, max=3)

    (
        interior_points,
        boundary_points,
        saturated_facets,
        n_points,
        n_interior_points,
        forced_stop,
    ) = _find_integer_points(
        verts=p.vertices,
        ineqs=p.inequalities,
        dim=p.dim,
        count_only=False,
        disable_numba=disable_numba,
    )

    assert n_points == (interior_points.shape[0]) + boundary_points.shape[0]
    assert len(saturated_facets) == boundary_points.shape[0]


@pytest.mark.parametrize("disable_numba", [True, False])
@pytest.mark.parametrize("interior", [True, False])
@pytest.mark.parametrize("count_only", [True, False])
def test_stop_at_max_points(count_only, interior, disable_numba):
    """
    Test that the lattice point enumeration of a (big) polytope stops
    correctly if the maximum number of requested points is reached.
    """
    p = Polytope.cube(2) * 100

    (
        interior_points,
        boundary_points,
        saturated_facets,
        n_points,
        n_interior_points,
        forced_stop,
    ) = _find_integer_points(
        verts=p.vertices,
        ineqs=p.inequalities,
        dim=p.dim,
        count_only=count_only,
        stop_at=-1 if interior else 10,
        stop_at_interior=10 if interior else -1,
        disable_numba=disable_numba,
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


@pytest.mark.parametrize("disable_numba", [True, False])
def test_integer_points_1d(disable_numba):
    p = Polytope([[0], [3]])

    (
        interior_points,
        boundary_points,
        saturated_facets,
        n_points,
        n_interior_points,
        forced_stop,
    ) = _find_integer_points(
        verts=p.vertices,
        ineqs=p.inequalities,
        dim=p.dim,
        count_only=False,
        disable_numba=disable_numba,
    )

    assert n_points == 4
    assert n_interior_points == 2
    assert interior_points.shape == (2, 1)
    assert boundary_points.shape == (2, 1)
    assert saturated_facets.shape == (2, 2)
