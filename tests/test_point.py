import pytest
import numpy as np
from sympy import Rational
from sympol.point import Point


def test_point_initialization():
    """
    Test initialization of a point, and that __eq__ is overridden.
    """
    point = Point([1, 3])

    assert point == Point([1, 3])
    assert point / 3 == Point([Rational(1, 3), 1])


def test_can_make_set():
    """
    Test that a point is hashable and can therefore be put in a set.
    """
    test_set = set([Point([1, 3]), Point([1, 3])])

    assert len(test_set) == 1


def test_point_from_high_rank_array():
    """
    Test that an exception is raised when trying to initialize a point from an
    array with rank > 1.
    """
    with pytest.raises(ValueError):
        Point([[[1, 3], [1, 3]]])


def test_eq_correctly_overridden():
    """
    Test that __eq__ is overridden correctly.
    """
    assert Point([1, 2]) == Point([1, 2])
    assert Point([1, 2]) != Point([1, 3])

    with pytest.raises(ValueError):
        # np.array wants a.any()/a.all()
        assert Point([1, 2]) == np.array([1, 2])
