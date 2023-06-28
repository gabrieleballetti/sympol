from cdd import NumberTypeable, Fraction
from sympy import Point, Rational
from sympy.abc import x
from sympol.utils import (
    _cdd_fraction_to_simpy_rational,
    _eulerian_number,
    _eulerian_poly,
)


def test_cdd_fraction_to_simpy_rational():
    """
    Test that a cdd fraction is correctly converted to a sympy rational.
    """
    rat = Fraction(10**100, 11**100)
    frac = _cdd_fraction_to_simpy_rational(rat)
    assert frac == Rational(10**100, 11**100)

    n = 10**100
    frac = _cdd_fraction_to_simpy_rational(n)
    assert frac == Rational(10**100)


def test_eulerian_number():
    """
    Test that the Eulerian numbers are correctly calculated.
    """
    assert _eulerian_number(0, 0) == 1

    assert _eulerian_number(1, 0) == 1

    assert _eulerian_number(2, 0) == 1
    assert _eulerian_number(2, 1) == 1

    assert _eulerian_number(3, 0) == 1
    assert _eulerian_number(3, 1) == 4
    assert _eulerian_number(3, 2) == 1

    assert _eulerian_number(4, 0) == 1
    assert _eulerian_number(4, 1) == 11
    assert _eulerian_number(4, 2) == 11
    assert _eulerian_number(4, 3) == 1

    assert _eulerian_number(5, 0) == 1
    assert _eulerian_number(5, 1) == 26
    assert _eulerian_number(5, 2) == 66
    assert _eulerian_number(5, 3) == 26
    assert _eulerian_number(5, 4) == 1


def test_eulerian_polynomial():
    """
    Test that the Eulerian polynomials are correctly calculated. Up to n = 10 they are
    hardcoded so we test them against the explicit formula.
    """
    assert _eulerian_poly(0, x).equals(1)
    for n in range(1, 11):
        assert _eulerian_poly(n, x).equals(
            sum([_eulerian_number(n, k - 1) * x**k for k in range(1, n + 1)])
        )

    assert _eulerian_poly(11, x).equals(
        x**11
        + 2036 * x**10
        + 152637 * x**9
        + 2203488 * x**8
        + 9738114 * x**7
        + 15724248 * x**6
        + 9738114 * x**5
        + 2203488 * x**4
        + 152637 * x**3
        + 2036 * x**2
        + x
    )
