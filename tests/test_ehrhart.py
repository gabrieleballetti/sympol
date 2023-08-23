from sympy import Integer, Poly
from sympy.abc import x
from sympol.ehrhart import (
    is_valid_h_star_vector,
    ehrhart_to_h_star_polynomial,
    h_star_to_ehrhart_polynomial,
    h_star_vector_of_cartesian_product_from_h_star_vectors,
)
from sympol import Polytope


def test_is_valid_h_star_vector():
    """Test is_valid_h_star_vector function."""

    # ordered so that all the failure cases are covered, in order
    assert not is_valid_h_star_vector((1, 1, 1.0))
    assert not is_valid_h_star_vector((2, 1, 1))
    assert not is_valid_h_star_vector((1, -1, 1))
    assert not is_valid_h_star_vector((1, 1, 2))
    assert not is_valid_h_star_vector((1, 1, 1, 2, 1))
    assert not is_valid_h_star_vector((1, 1, 0, 1, 0))
    assert not is_valid_h_star_vector((1, 2, 1, 1, 2))
    assert not is_valid_h_star_vector((1, 2, 1, 1, 0))

    assert is_valid_h_star_vector((1, 1, 1))
    assert is_valid_h_star_vector((1, 1, 1, 0))
    assert is_valid_h_star_vector((1, 1, Integer(1)))


def test_ehrhart_to_h_star_polynomial():
    """
    Test that the method ehrhart_to_h_star_polynomial works correctly for a
    unit cube.
    """
    p = Polytope.cube(3)

    assert ehrhart_to_h_star_polynomial(
        dim=p.dim, ehrhart_coefficients=p.ehrhart_coefficients
    ) == Poly(x**2 + 4 * x + 1, x)


def test_h_star_to_ehrhart_polynomial():
    """
    Test that the method h_star_to_ehrhart_polynomial works correctly for a
    unit cube.
    """
    p = Polytope.cube(3)

    assert h_star_to_ehrhart_polynomial(
        dim=p.dim, h_star_vector=p.h_star_vector
    ) == Poly((x + 1) ** 3, x, domain="QQ")


def test_h_star_vector_of_cartesian_product_from_h_star_vectors():
    """
    Test that the method h_star_vector_of_cartesian_product_from_h_star_vectors
    """
    h1 = Polytope.cube(2).h_star_vector
    h2 = Polytope.cube(3).h_star_vector

    h = h_star_vector_of_cartesian_product_from_h_star_vectors(h1, h2)

    assert h == (1, 26, 66, 26, 1, 0)  # Polytope.cube(5).h_star_vector
