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


def test_integer_points():
    """
    Test calculation of number of integer points.
    """
    generators = PointList([[1, -1, -1], [1, 1, 0], [1, 0, 1]])
    hop = HalfOpenParallelotope(generators)
    pts = hop.get_integer_points()
    assert set([tuple(p) for p in pts]) == set(((0, 0, 0), (1, 0, 0), (2, 0, 0)))
