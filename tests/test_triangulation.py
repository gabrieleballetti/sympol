import numpy as np
from sympol._triangulation import _get_upper_or_lower_hull_triangulation


def test_upper_or_lower_hull_triangulation():
    """
    Test that a triangulation of the upper/lower hull of a point list is
    correctly calculated.
    """
    pts = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ]
    )

    triang = _get_upper_or_lower_hull_triangulation(pts, 3)

    assert all(len(simplex) == 4 for simplex in triang)


def test_upper_or_lower_hull_triangulation_simplex():
    """
    Test that a triangulation of the upper/lower hull of the point list
    of the vertices of a simplex is correctly calculated.
    """
    pts = np.array(
        [
            [0, 0],
            [1, 0],
            [0, 1],
        ]
    )

    triang = _get_upper_or_lower_hull_triangulation(pts, 2)

    assert triang == tuple([frozenset({0, 1, 2})])


def test_upper_or_lower_hull_triangulation_low_dim():
    """
    Test that a triangulation of the upper/lower hull of a non full-rank
    point list is correctly calculated.
    """
    pts = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
        ]
    )

    triang = _get_upper_or_lower_hull_triangulation(pts, 2)

    assert all(len(simplex) == 3 for simplex in triang)
