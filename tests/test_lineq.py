from sympy import Point, Rational
from sympol.lineq import LinIneq


def test_init():
    """
    Test initialization of a linear inequality
    """
    normal = Point([1, 2, 3])
    rhs = Rational(4, 5)
    lin_ineq = LinIneq(normal, rhs)

    assert lin_ineq.normal == normal
    assert lin_ineq.rhs == rhs


def test_evaluate():
    """
    Test evaluation of a linear inequality at a point
    """
    normal = Point([1, 2, 3])
    rhs = Rational(4, 5)
    lin_ineq = LinIneq(normal, rhs)

    point1 = Point([0, 0, 0])
    assert lin_ineq.evaluate(point1) == -rhs

    point2 = Point([1, 1, 1])
    assert lin_ineq.evaluate(point2) == normal.dot(point2) - rhs
