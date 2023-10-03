from itertools import product
from functools import cache
import numpy as np
from cdd import Fraction
from sympy import binomial, factorial, floor, Integer, Poly, Rational
from sympy.abc import x


def _cdd_fraction_to_simpy_rational(frac):
    """Convert a cddlib fraction (or potentially an integer) to a sympy Rational."""
    if isinstance(frac, int):
        return Rational(frac)

    if isinstance(frac, Fraction):
        return Rational(frac.numerator, frac.denominator)

    raise TypeError("Expected a cddlib Fraction or an int")


def _coefficients(poly: Poly, d: int) -> tuple[Rational]:
    """Return the coefficients of the polynomial poly up to degree d."""
    return tuple(poly.coeff_monomial(x**i) for i in range(d + 1))


def _is_integer(num) -> bool:
    """Return True if num is an integer, False otherwise."""
    return isinstance(num, (int, Integer))


def _np_cartesian_product(*arrays):
    """Return the cartesian product of a list of arrays."""
    prod = np.empty(
        shape=(
            np.prod([a.shape[0] for a in arrays]),
            sum(a.shape[1] for a in arrays),
        )
    )
    for i, arrs in enumerate(product(*arrays)):
        prod[i] = np.concatenate(arrs)

    return prod


def _is_log_concave(iterable):
    """Return True if the iterable is log-concave, False otherwise.

    A sequence a_0, a_1, ..., a_d is log concave if it has no internal zeros and
    (a_i)^2 >= a_{i−1} * a_{i+1} for all i = 1, ..., d − 1.

    Usually it is also required that the sequence is non-negative, but we do not
    check for this here (as we mainly work with positive sequences).

    Log-concavity implies unimodality, but not vice versa.
    """
    first = 0
    while first < len(iterable) and iterable[first] == 0:
        first += 1

    last = len(iterable) - 1
    while last > first and iterable[last] == 0:
        last -= 1

    i = first + 1
    while i < last:
        if iterable[i] == 0:
            return False
        if iterable[i] ** 2 < iterable[i - 1] * iterable[i + 1]:
            return False
        i += 1
    return True


def _is_unimodal(iterable):
    """Return True if the iterable is unimodal, False otherwise."""
    i = 1
    while i < len(iterable) and iterable[i - 1] <= iterable[i]:
        i += 1
    while i < len(iterable) and iterable[i - 1] >= iterable[i]:
        i += 1
    return i == len(iterable)


@cache
def _eulerian_number(n, k):
    """Return the Eulerian number A(n,k)."""
    return sum(
        [(-1) ** i * binomial(n + 1, i) * (k + 1 - i) ** n for i in range(k + 1)]
    )


@cache
def _eulerian_poly(n, x):
    """Calculate Eulerian polynomial A_n(x), first 10 values are given explicitly."""
    if n == 0:
        return Poly(1, x)
    return Poly(sum([_eulerian_number(n, k - 1) * x**k for k in range(1, n + 1)]))


@cache
def _binomial_polynomial(d, k, x):
    """Calculate the binomial polynomial binomial(x + k, d), with x as the variable."""
    poly = Poly(Rational(1, factorial(d)), x, domain="QQ")
    for i in range(d):
        poly *= Poly(x + k - i, x)

    return poly


@cache
def _h_to_gamma(d):
    """Return the transformation matrix G that transforms h to gamma."""
    m = floor(Rational(d, 2))
    g = np.empty((m + 1, m + 1), dtype=object)
    for i in range(m + 1):
        for j in range(m + 1):
            g[i, j] = (-1) ** (i - j) * (
                binomial(d - i - j, i - j) + binomial(d - i - j - 1, i - j - 1)
            )
    return g


@cache
def _gamma_to_h(d):
    """Return the transformation matrix S that transforms gamma to h."""
    m = floor(Rational(d, 2))
    s = np.empty((m + 1, m + 1), dtype=object)
    for i in range(m + 1):
        for j in range(m + 1):
            s[i, j] = binomial(d - 2 * j, i - j)
    return s
