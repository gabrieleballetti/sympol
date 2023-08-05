import pytest
from sympy import Matrix
from sympol.parallelotope import HalfOpenParallelotope
from sympol.point_configuration import PointConfiguration


def test_linearly_dependent_generators():
    """
    Test that the HalfOpenParallelotope __init__ method raises an exception
    when the generators are linearly dependent.
    """
    generators = PointConfiguration([[1, 0, 0], [1, 1, 0], [1, 0, 1], [1, 1, 1]])
    with pytest.raises(ValueError):
        HalfOpenParallelotope(generators)


def test_smith_normal_form():
    """
    Test calculation of Smith normal form of generator matrix.
    """
    generators = PointConfiguration([[1, -1, -1], [1, 1, 0], [1, 0, 1]])
    hop = HalfOpenParallelotope(generators)
    assert hop.m == Matrix([[1, 1, 1], [-1, 1, 0], [-1, 0, 1]])
    assert hop.snf == [1, 1, 3]
    assert hop.k == 2


def test_det():
    """
    Test calculation of determinant of generator matrix.
    """
    generators = PointConfiguration([[1, -1, -1], [1, 1, 0], [1, 0, 1]])
    hop = HalfOpenParallelotope(generators)
    assert hop.det == 3


def test_v_d_inv():
    """
    Test calculation of the inversion matrix V*D^-1.
    """
    generators = PointConfiguration([[1, -1, -1], [1, 1, 0], [1, 0, 1]])
    hop = HalfOpenParallelotope(generators)
    assert hop.v_d_inv == Matrix([[0, 0, 1], [0, 0, 1], [0, 0, 1]])


@pytest.mark.parametrize("count_only", [True, False])
@pytest.mark.parametrize("height", [-1, 1])
def test_integer_points(height, count_only):
    """
    Test calculation of number of integer points.
    """
    generators = PointConfiguration([[1, -1, -1], [1, 1, 0], [1, 0, 1]])
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


@pytest.mark.parametrize("count_only", [True, False])
@pytest.mark.parametrize("height", [-1, 1])
def test_integer_points_consistency(height, count_only):
    """
    Check that the two methods (numpy and sympy) give the same results.
    """
    generators = PointConfiguration(
        [
            [1, 1, 1, 1, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 0, 1, 1],
        ]
    )
    hop = HalfOpenParallelotope(generators)
    pts1, h1 = hop.get_integer_points(
        height=height, use_sympy=True, count_only=count_only
    )
    pts2, h2 = hop.get_integer_points(
        height=height, use_sympy=False, count_only=count_only
    )
    assert h1 == h2
    if not count_only:
        assert set([tuple(pt) for pt in pts1]) == set([tuple(pt) for pt in pts2])
