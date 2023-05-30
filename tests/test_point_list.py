from sympy import Rational
from sympol.point import Point
from sympol.point_list import PointList


def test_init():
    """
    Test initialization of a point list
    """
    points = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    point_list = PointList(points)

    assert point_list.shape == (4, 3)
    assert point_list[0] == Point([0, 0, 0])
    assert point_list[1] == Point([1, 0, 0])
    assert point_list[2] == Point([0, 1, 0])
    assert point_list[3] == Point([0, 0, 1])


def test_affine_rank():
    """
    Test calculation of the affine rank of a point list
    """
    points = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    point_list = PointList(points)

    assert point_list.affine_rank() == 3


def test_barycenter():
    """
    Test calculation of the barycenter of a point list
    """
    points = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    point_list = PointList(points)

    assert point_list.barycenter == Point(
        [Rational(1, 4), Rational(1, 4), Rational(1, 4)]
    )
