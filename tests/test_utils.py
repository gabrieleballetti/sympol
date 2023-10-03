import pytest
import numpy as np
from cdd import Fraction
from sympy import Poly, Rational
from sympy.abc import x
from sympol._utils import (
    _cdd_fraction_to_simpy_rational,
    _np_cartesian_product,
    _is_log_concave,
    _is_unimodal,
    _eulerian_number,
    _eulerian_poly,
    _h_to_gamma,
    _gamma_to_h,
    _arrays_equal_up_to_row_permutation,
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

    with pytest.raises(TypeError):
        _cdd_fraction_to_simpy_rational("not a fraction")


def test_np_cartesian_product():
    """
    Test that the cartesian product of two iterables is correctly calculated.
    """
    a = np.array([[0, 1], [2, 3]])
    b = np.array([[4], [5]])

    assert np.array_equal(
        _np_cartesian_product(a, b),
        np.array([[0, 1, 4], [0, 1, 5], [2, 3, 4], [2, 3, 5]]),
    )


def test_is_unimodal():
    """
    Test that a sequence is correctly identified as unimodal or not.
    """
    assert _is_unimodal([1, 2, 3, 4, 5, 5, 5, 4, 3, 2, 1])
    assert _is_unimodal([1, 2, 3, 4, 5, 5, 5, 5, 5, 5, 5])
    assert _is_unimodal([5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0])
    assert _is_unimodal([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    assert not _is_unimodal([1, 2, 3, 4, 5, 4, 5, 4, 3, 2, 1])
    assert not _is_unimodal([2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2])
    assert not _is_unimodal([1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 1])


def test_is_log_concave():
    """
    Test that a sequence is correctly identified as log-concave or not.
    """
    assert _is_log_concave([1, 6, 15, 20, 15, 6, 1])
    assert _is_log_concave([1, 7, 21, 35, 35, 21, 7, 1])
    assert _is_log_concave([5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0])
    assert _is_log_concave([0, 0, 1, 1, 1, 0, 0])
    assert _is_log_concave([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    assert not _is_log_concave([1, 1, 0, 0, 1])
    assert not _is_log_concave([1, 4, 17, 0, 0])
    assert not _is_log_concave([2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2])
    assert not _is_log_concave([1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 1])


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
    assert _eulerian_poly(0, x) == Poly(1, x)
    assert _eulerian_poly(1, x) == Poly(x)
    assert _eulerian_poly(2, x) == Poly(x**2 + x)
    assert _eulerian_poly(3, x) == Poly(x**3 + 4 * x**2 + x)
    assert _eulerian_poly(4, x) == Poly(x**4 + 11 * x**3 + 11 * x**2 + x)

    assert _eulerian_poly(11, x) == Poly(
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


def test_h_to_gamma():
    g = _h_to_gamma(5)

    assert np.array_equal(
        g,
        np.array(
            [
                [1, 0, 0],
                [-5, 1, 0],
                [5, -3, 1],
            ]
        ),
    )


def test_gamma_to_h():
    s = _gamma_to_h(5)

    assert np.array_equal(
        s,
        np.array(
            [
                [1, 0, 0],
                [5, 1, 0],
                [10, 3, 1],
            ]
        ),
    )


def test_arrays_equal_up_to_row_permutation():
    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([[4, 5, 6], [1, 2, 3]])
    c = np.array([[1, 2, 3], [4, 6, 5]])
    d = np.array([[1, 2, 3], [4, 5, 6], [1, 2, 3]])
    e = np.array([[1, 2, 3], [4, 5, 6], [4, 5, 6]])

    assert _arrays_equal_up_to_row_permutation(a, b)
    assert not _arrays_equal_up_to_row_permutation(a, c)
    assert not _arrays_equal_up_to_row_permutation(a, d)
    assert not _arrays_equal_up_to_row_permutation(d, e)
