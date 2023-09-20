import numpy as np
from itertools import product
from cdd import Fraction
from sympy import binomial, factorial, Integer, Poly, Rational


def _cdd_fraction_to_simpy_rational(frac):
    """Convert a cddlib fraction (or potentially an integer) to a sympy Rational."""
    if isinstance(frac, int):
        return Rational(frac)

    if isinstance(frac, Fraction):
        return Rational(frac.numerator, frac.denominator)

    raise TypeError("Expected a cddlib Fraction or an int")


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


def _eulerian_number(n, k):
    """Return the Eulerian number A(n,k)."""
    return sum(
        [(-1) ** i * binomial(n + 1, i) * (k + 1 - i) ** n for i in range(k + 1)]
    )


def _eulerian_poly(n, x):
    """Calculate Eulerian polynomial A_n(x), first 10 values are given explicitly."""
    if n == 0:
        return Poly(1, x)
    if n == 1:
        return Poly(x)
    if n == 2:
        return Poly(x**2 + x)
    if n == 3:
        return Poly(x**3 + 4 * x**2 + x)
    if n == 4:
        return Poly(x**4 + 11 * x**3 + 11 * x**2 + x)
    if n == 5:
        return Poly(x**5 + 26 * x**4 + 66 * x**3 + 26 * x**2 + x)
    if n == 6:
        return Poly(
            x**6 + 57 * x**5 + 302 * x**4 + 302 * x**3 + 57 * x**2 + x
        )
    if n == 7:
        return Poly(
            x**7
            + 120 * x**6
            + 1191 * x**5
            + 2416 * x**4
            + 1191 * x**3
            + 120 * x**2
            + x
        )
    if n == 8:
        return Poly(
            x**8
            + 247 * x**7
            + 4293 * x**6
            + 15619 * x**5
            + 15619 * x**4
            + 4293 * x**3
            + 247 * x**2
            + x
        )
    if n == 9:
        return Poly(
            x**9
            + 502 * x**8
            + 14608 * x**7
            + 88234 * x**6
            + 156190 * x**5
            + 88234 * x**4
            + 14608 * x**3
            + 502 * x**2
            + x
        )
    if n == 10:
        return Poly(
            x**10
            + 1013 * x**9
            + 47840 * x**8
            + 455192 * x**7
            + 1310354 * x**6
            + 1310354 * x**5
            + 455192 * x**4
            + 47840 * x**3
            + 1013 * x**2
            + x
        )
    return Poly(sum([_eulerian_number(n, k - 1) * x**k for k in range(1, n + 1)]))


def _binomial_polynomial(d, k, x):
    """Calculate the binomial polynomial binomial(x + k, d), with x as the variable."""
    poly = Poly(Rational(1, factorial(d)), x, domain="QQ")
    for i in range(d):
        poly *= Poly(x + k - i, x)

    return poly
