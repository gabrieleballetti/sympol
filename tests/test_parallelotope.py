import pytest
import numpy as np
from sympy import Matrix
from sympol.parallelotope import HalfOpenParallelotope
from sympol.point_list import PointList


def test_smith_normal_form():
    """
    Test calculation of Smith normal form of generator matrix.
    """
    generators = PointList([[1, -1, -1], [1, 1, 0], [1, 0, 1]])
    hop = HalfOpenParallelotope(generators)
    assert hop.m == Matrix([[1, 1, 1], [-1, 1, 0], [-1, 0, 1]])
    assert hop.snf == [1, 1, 3]
    assert hop.det == 3
    assert hop.k == 2


@pytest.mark.parametrize("count_only", [True, False])
@pytest.mark.parametrize("height", [-1, 1])
def test_integer_points(height, count_only):
    """
    Test calculation of number of integer points.
    """
    generators = PointList([[1, -1, -1], [1, 1, 0], [1, 0, 1]])
    hop = HalfOpenParallelotope(generators)
    pts, h = hop.get_integer_points(count_only=count_only, height=height)
    if height == -1:
        if count_only:
            assert h == (1, 1, 1)
            assert pts == tuple()
        else:
            assert h == (1, 1, 1)
            assert set([tuple(p) for p in pts]) == set(
                ((0, 0, 0), (1, 0, 0), (2, 0, 0))
            )
    else:
        if count_only:
            assert h == (0, 1, 0)
            assert pts == tuple()
        else:
            assert h == (0, 1, 0)
            assert set([tuple(p) for p in pts]) == set({(1, 0, 0)})
