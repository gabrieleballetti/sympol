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
