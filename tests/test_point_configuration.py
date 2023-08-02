from sympy import Rational
from sympol.point import Point
from sympol.point_configuration import PointConfiguration


def test_init():
    """
    Test initialization of a point list
    """
    points = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    point_list = PointConfiguration(points)

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
    point_list = PointConfiguration(points)

    assert point_list.affine_rank == 3


def test_barycenter():
    """
    Test calculation of the barycenter of a point list
    """
    points = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    point_list = PointConfiguration(points)

    assert point_list.barycenter == Point(
        [Rational(1, 4), Rational(1, 4), Rational(1, 4)]
    )


def test_smith_normal_form():
    """
    Test calculation of the affine Smith normal form of a point list with rank
    strictly less than the ambient dimension
    """
    points = [
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [1, 1, 3, 0],
        [0, 0, -2, 0],
    ]
    point_list = PointConfiguration(points)

    assert point_list.snf_diag == [1, 1, 1, 0]


def test_index():
    """
    Test calculation of the index of a point list
    """
    points = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 3]]
    point_list = PointConfiguration(points)
    assert point_list.index == 3

    # this are the vertices of a simplex with non-spanning vertices
    # (since it is not a unimodular simplex), but whose lattice points
    # span the lattice
    points = [
        [0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [5, 5, 5, 5, 8],
        [2, 2, 2, 2, 3],
    ]
    point_list = PointConfiguration(points)
    assert point_list.index == 1
