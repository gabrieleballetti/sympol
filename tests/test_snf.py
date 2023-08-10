from sympy import Matrix
from sympol._snf import smith_normal_form


def test_smith_normal_form():
    """
    Test calculation of Smith normal form of a matrix.
    """
    m = Matrix([[2, 4, 4], [-6, 6, 12], [10, 4, 16]])
    d, u, v = smith_normal_form(m)
    assert d == Matrix([[2, 0, 0], [0, 2, 0], [0, 0, 156]])
    assert d == u * m * v


def test_smith_normal_form_with_zero_row():
    """
    Test calculation of Smith normal form of a matrix with a zero row.
    """
    m = Matrix([[2, 4, 4], [-6, 6, 12], [0, 0, 0]])
    d, u, v = smith_normal_form(m)
    assert d == Matrix([[2, 0, 0], [0, 6, 0], [0, 0, 0]])
    assert d == u * m * v


def test_smith_normal_form_rectangular():
    """
    Test calculation of Smith normal form of a rectangular matrix.
    """
    m = Matrix([[2, 4, 4], [-6, 6, 12]])
    d, u, v = smith_normal_form(m)
    assert d == Matrix([[2, 0, 0], [0, 6, 0]])
    assert d == u * m * v

    d, u, v = smith_normal_form(m.T)
    assert d == Matrix([[2, 0], [0, 6], [0, 0]])
    assert d == u * m.T * v
