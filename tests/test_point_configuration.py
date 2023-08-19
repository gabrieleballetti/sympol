import pytest
import numpy as np
from sympy import Rational
from sympol.point import Point
from sympol.point_configuration import PointConfiguration


def test_init():
    """
    Test initialization of a point configuration.
    """
    points = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    pt_cfg = PointConfiguration(points)

    assert pt_cfg.shape == (4, 3)
    assert pt_cfg[0] == Point([0, 0, 0])
    assert pt_cfg[1] == Point([1, 0, 0])
    assert pt_cfg[2] == Point([0, 1, 0])
    assert pt_cfg[3] == Point([0, 0, 1])


def test_point_configuration_from_high_rank_array():
    """
    Test that an exception is raised when trying to initialize a point configuration
    from an array with rank > 2.
    """
    with pytest.raises(ValueError):
        PointConfiguration([[[1, 3], [1, 3]], [[1, 3], [1, 3]]])


def test_empty_init():
    """
    Test initialization of an empty point configuration.
    """
    pt_cfg = PointConfiguration([])

    assert pt_cfg.shape == (0, 0)


def test_get_item():
    """
    Test that __getitem__ is overridden correctly.
    """
    pt_cfg = PointConfiguration([[1, 2], [3, 4]])
    assert pt_cfg[0] == Point([1, 2])
    assert pt_cfg[1] == Point([3, 4])

    assert pt_cfg[0, :] == Point([1, 2])

    assert pt_cfg[0:1] == PointConfiguration([[1, 2]])
    assert pt_cfg[0:2] == PointConfiguration([[1, 2], [3, 4]])

    with pytest.raises(IndexError):
        # For a type which is not int or slice let numpy handle the error
        pt_cfg[0.5]


def test_eq():
    """
    Test that __eq__ is overridden correctly.
    """
    pt_cfg_1 = PointConfiguration([[1, 2], [3, 4]])
    pt_cfg_2 = PointConfiguration([[1, 2], [3, 4]])
    pt_cfg_3 = PointConfiguration([[1, 2], [3, 5]])

    assert pt_cfg_1 == pt_cfg_2
    assert pt_cfg_1 != pt_cfg_3

    with pytest.raises(ValueError):
        # np.array wants a.any()/a.all()
        assert pt_cfg_1 == np.array([[1, 2], [3, 4]])


def test_ne():
    """
    Test that __ne__ is overridden correctly.
    """
    pt_cfg_1 = PointConfiguration([[1, 2], [3, 4]])
    pt_cfg_2 = PointConfiguration([[1, 2], [3, 5]])

    assert pt_cfg_1 != pt_cfg_2


def test_add():
    """
    Test that __add__ is overridden correctly.
    """
    pt_cfg_1 = PointConfiguration([[1, 2], [3, 4]])
    pt_cfg_2 = PointConfiguration([[1, 2], [3, 5]])

    assert pt_cfg_1 + pt_cfg_2 == PointConfiguration([[2, 4], [6, 9]])


def test_sub():
    """
    Test that __sub__ is overridden correctly.
    """
    pt_cfg_1 = PointConfiguration([[1, 2], [3, 4]])
    pt_cfg_2 = PointConfiguration([[1, 2], [3, 5]])

    assert pt_cfg_1 - pt_cfg_2 == PointConfiguration([[0, 0], [0, -1]])


def test_ambient_dimension():
    """
    Test calculation of the ambient dimension of a point configuration.
    """
    pt_cfg = PointConfiguration([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert pt_cfg.ambient_dimension == 3

    with pytest.raises(ValueError):
        PointConfiguration([]).ambient_dimension


def test_affine_rank():
    """
    Test calculation of the affine rank of a point configuration.
    """
    pts = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]

    assert PointConfiguration(pts).affine_rank == 2
    assert PointConfiguration(pts + Point([1, 1, 1])).affine_rank == 2


def test_rank():
    """
    Test calculation of the homogeneous rank of a point configuration.
    """
    pts = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]

    assert PointConfiguration(pts).rank == 2
    assert PointConfiguration(pts + Point([1, 1, 1])).rank == 3


def test_barycenter():
    """
    Test calculation of the barycenter of a point configuration.
    """
    pts = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    pt_cfg = PointConfiguration(pts)

    assert pt_cfg.barycenter == Point([Rational(1, 4), Rational(1, 4), Rational(1, 4)])


def test_smith_normal_form():
    """
    Test calculation of the affine Smith normal form of a point configuration with rank
    strictly less than the ambient dimension
    """
    pts = [
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [1, 1, 3, 0],
        [0, 0, -2, 0],
    ]
    pt_cfg = PointConfiguration(pts)

    assert pt_cfg.snf_diag == [1, 1, 1, 0]


def test_index():
    """
    Test calculation of the index of a point configuration.
    """
    pts = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 3]]
    pt_cfg = PointConfiguration(pts)
    assert pt_cfg.index == 3

    # this are the vertices of a simplex with non-spanning vertices
    # (since it is not a unimodular simplex), but whose lattice points
    # span the lattice
    pts = [
        [0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [5, 5, 5, 5, 8],
        [2, 2, 2, 2, 3],
    ]
    pt_cfg = PointConfiguration(pts)
    assert pt_cfg.index == 1


def test_index_non_full_rank():
    """
    Test that an exception is raised when trying to calculate the index of a non-full
    rank point configuration.
    """
    pts = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    pt_cfg = PointConfiguration(pts)
    with pytest.raises(ValueError):
        pt_cfg.index


def test_can_make_set():
    """
    Test that a point is hashable and can therefore be put in a set.
    """
    test_set = set([PointConfiguration([[1, 3]]), PointConfiguration([[1, 3]])])

    assert len(test_set) == 1


def test_conversion_to_int64():
    """
    Test that a PointConfiguration can be converted to np.int64 type
    """
    pts = PointConfiguration([[1, 3], [2, 4]])
    pts_int64 = pts.view(np.ndarray).astype(np.int64)

    assert pts_int64.dtype == np.int64


def test_triangulation():
    """
    Test calculation of the triangulation of a point configuration.
    This point configuration has two possible triangulations (one with
    one and one with four simplices), check that the former is returned.
    """

    pts = PointConfiguration(
        [
            [0, 0, 0],
            [-1, -1, -1],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    )

    triangulation = pts.triangulation

    assert triangulation == tuple([frozenset({1, 2, 3, 4})])
