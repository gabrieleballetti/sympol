from cdd import Fraction
from sympy import Rational


def _cdd_fraction_to_simpy_rational(frac):
    """
    Convert a cddlib fraction (or potentially an integer) to a sympy Rational.
    """
    if isinstance(frac, int):
        return Rational(frac)

    if isinstance(frac, Fraction):
        return Rational(frac.numerator, frac.denominator)

    raise TypeError("Expected a cddlib Fraction or an int")
