from random import shuffle
from sympy import Matrix

from sympol.normal_form import (
    get_normal_form,
    _is_automorphism,
    # _find_canonical_permutation_and_automorphisms,
)
from sympol.point import Point
from sympol.point_list import PointList
from sympol.polytope import Polytope


def test_normal_form():
    """
    Test normal form of a polytope
    """
    verts = [
        [-2, -1, -3],
        [-2, -1, 3],
        [-2, 1, -3],
        [-2, 1, 3],
        [2, -1, -3],
        [2, -1, 3],
        [2, 1, -3],
        [2, 1, 3],
    ]

    polytope_1 = Polytope(vertices=verts)
    normal_form_1 = get_normal_form(polytope_1)

    shuffle(verts)
    uni_map = Matrix(
        [
            [1, 2, 0],
            [-1, -1, 0],
            [-4, 9, 1],
        ]
    )
    polytope_2 = Polytope(vertices=(Matrix(verts) * uni_map))
    normal_form_2 = get_normal_form(polytope_2)

    assert normal_form_1 == normal_form_2


def test_identity_is_automorphisms():
    """
    Test that is_automorphism returns True for a coord permutation
    """

    input_list = PointList(
        [
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ]
    )

    assert _is_automorphism(input_list, input_list)


def test_unimodular_eq_lists_is_automorphism():
    """
    Test that is_automorphism returns True when called with two affinely unimodular
    equivalent lists
    """

    input_list = PointList(
        [
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ]
    )

    matrix_input = Matrix(input_list)
    unim_eq_map = Matrix([[2, 5, 0], [1, 3, 0], [0, 0, -1]])
    output_list = PointList((matrix_input * unim_eq_map).tolist())
    output_list = output_list + Point([2, 6, -1])

    assert _is_automorphism(input_list, output_list)


def test_non_unimodular_eq_lists_is_not_automorphism():
    """
    Test that is_automorphism returns True when called with two affinely unimodular
    equivalent lists
    """

    input_list = PointList(
        [
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ]
    )

    matrix_input = Matrix(input_list)
    unim_eq_map = Matrix([[1, 5, 0], [1, 3, 0], [0, 0, -1]])
    output_list = PointList((matrix_input * unim_eq_map).tolist())
    output_list = output_list + Point([2, 6, -1])

    assert not _is_automorphism(input_list, output_list)
