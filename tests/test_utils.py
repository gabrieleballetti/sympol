from cdd import NumberTypeable, Fraction
from sympy import Point, Rational
from sympol.utils import _cdd_fraction_to_simpy_rational


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
