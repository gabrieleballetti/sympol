import numpy as np

from sympol.integer_points import _find_integer_points
from sympol.polytope import Polytope


def test_integer_points():
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
    points = _find_integer_points(
        verts=np.array(p.vertices),
        ineqs=np.array(
            [ineq.normal.tolist() + [-ineq.rhs] for ineq in p.linear_inequalities]
        ),
        dim=p.dim,
    )

    assert len(points) == 60
    # check the expected 6 interior points
    assert len([p for p in points if p[1] == frozenset({})]) == 6
    assert points[0] == ((1, 1, 1), frozenset({}))
    assert points[1] == ((1, 1, 2), frozenset({}))
    assert points[2] == ((2, 1, 1), frozenset({}))
    assert points[3] == ((2, 1, 2), frozenset({}))
    assert points[4] == ((3, 1, 1), frozenset({}))
    assert points[5] == ((3, 1, 2), frozenset({}))
