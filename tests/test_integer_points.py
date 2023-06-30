import pytest

from sympol.integer_points import _find_integer_points
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
    ) = _find_integer_points(
        verts=p._verts_as_np_array(),
        ineqs=p._ineqs_as_np_array(),
        dim=p.dim,
        count_only=count_only,
    )

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
    ) = _find_integer_points(
        verts=p._verts_as_np_array(),
        ineqs=p._ineqs_as_np_array(),
        dim=p.dim,
        count_only=False,
    )

    assert n_points == (interior_points.shape[0]) + boundary_points.shape[0]
    assert len(saturated_facets) == boundary_points.shape[0]
