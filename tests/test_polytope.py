import pytest
from unittest.mock import MagicMock
import numpy as np
from sympy import Matrix, Poly, Rational
from sympy.abc import x

from sympol._half_open_parallelotope import HalfOpenParallelotope
from sympol.point import Point
from sympol.point_configuration import PointConfiguration
from sympol.polytope import Polytope, Simplex
from sympol._utils import _arrays_equal_up_to_row_permutation


def test_init_from_points():
    """
    Test initialization of a polytope from a list of points
    """
    points = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [Rational(1, 4), Rational(1, 4), Rational(1, 4)],
    ]
    polytope = Polytope(points)

    assert polytope._points == PointConfiguration(points)
    assert polytope.ambient_dim == 3

    # check correct dimension initialization
    assert polytope._dim is None
    assert polytope.dim == 3
    assert polytope._dim == 3

    # check correct vertices initialization
    assert polytope._vertices is None
    assert polytope.vertices == PointConfiguration(points[:-1])
    assert polytope._vertices == PointConfiguration(points[:-1])


def test_init_from_vertices():
    """
    Test initialization of a polytope from a list of vertices
    """
    vertices = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]
    polytope = Polytope(vertices=vertices)
    assert polytope._vertices == PointConfiguration(vertices)
    assert polytope.ambient_dim == 3

    # check correct dimension initialization
    assert polytope._dim is None
    assert polytope.dim == 3
    assert polytope._dim == 3

    # check correct vertices initialization
    assert polytope._vertices == PointConfiguration(vertices)
    assert polytope.vertices == PointConfiguration(vertices)

    # check that inequalities are not initialized
    assert polytope._inequalities is None


def test_init_from_inequalities():
    """
    Test initialization of a polytope from inequalities
    """
    ineqs = [
        [0, 1, 0],
        [0, 0, 1],
        [1, -1, 0],
        [1, 0, -1],
    ]

    p = Polytope(inequalities=ineqs)
    assert np.array_equal(p._inequalities, ineqs)
    assert p.ambient_dim == 2

    # check correct dimension initialization
    assert p._dim is None
    assert p.dim == 2
    assert p._dim == 2

    # check that vertices are not initialized
    assert p._vertices is None


def test_init_from_inequalities_and_equalities():
    """
    Test initialization of a polytope from inequalities and equalities
    """
    ineqs = [
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [1, -1, 0, 0],
        [1, 0, -1, 0],
    ]
    eqs = [
        [0, 0, 0, 1],
    ]

    p = Polytope(inequalities=ineqs, equalities=eqs)
    expected_verts = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
        ]
    )

    assert _arrays_equal_up_to_row_permutation(p.vertices, expected_verts)


def test_init_from_inequalities_with_redundancies():
    """
    Test initialization of a polytope from inequalities with redundencies
    """
    ineqs = [
        [0, 1, 0],
        [0, 0, 1],
        [1, -1, 0],
        [1, 0, -1],
        [2, 0, -1],  # redundant
    ]

    p = Polytope(inequalities=ineqs)

    assert _arrays_equal_up_to_row_permutation(p._inequalities, np.array(ineqs))
    assert _arrays_equal_up_to_row_permutation(p.inequalities, np.array(ineqs[:-1]))


def test_init_from_inequalities_equalities_with_redundancies():
    """
    Test initialization of a polytope from inequalities + equalities with redundencies
    """
    ineqs = [
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [1, -1, 0, 0],
        [1, 0, -1, 0],
        [2, 0, -1, 0],  # redundant
    ]

    eqs = [
        [0, 0, 0, 1],
        [0, 0, 0, -1],  # redundant
    ]

    p = Polytope(inequalities=ineqs, equalities=eqs)

    assert _arrays_equal_up_to_row_permutation(p._inequalities, np.array(ineqs))
    assert _arrays_equal_up_to_row_permutation(p._equalities, np.array(eqs))
    assert _arrays_equal_up_to_row_permutation(p.inequalities, np.array(ineqs[:-1]))
    assert _arrays_equal_up_to_row_permutation(p.equalities, np.array(eqs[:-1]))


def test_init_from_inequalities_empty_polytope():
    """
    Test initialization of a polytope from inequalities defining an empty polytope
    """
    ineqs = [[0, 1, 0], [0, 0, 1], [1, -1, 0], [1, 0, -1], [-2, 0, 1]]

    p = Polytope(inequalities=ineqs)

    assert p.is_empty_set
    assert p.n_vertices == 0
    assert p.n_inequalities == 0


def test_init_from_inequalities_unbounded_polytope():
    """
    Test initialization of a polytope from inequalities defining an unbounded polytope
    """
    ineqs = [[0, 1, 0, 0], [0, 0, 1, 0], [1, -1, 0, 0], [1, 0, -1, 0]]

    p = Polytope(inequalities=ineqs)

    assert not p.is_bounded


def test_init_edge_cases():
    """
    Test initialization of a polytope raises exceptions for edge cases
    """
    with pytest.raises(ValueError):
        Polytope()

    with pytest.raises(ValueError):
        Polytope(points=None, vertices=None)

    with pytest.raises(ValueError):
        Polytope(points=[], vertices=[])

    with pytest.raises(ValueError):
        Polytope(equalities=[])

    with pytest.raises(ValueError):
        Polytope(points=[], inequalities=[])


def test_vertices():
    """
    Test that the vertices are correct
    """
    verts = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, Rational(101, 100)]]
    mid = Point([1, 1, 1]) / 4
    points = verts + [mid]
    polytope = Polytope(points)

    assert polytope.vertices == PointConfiguration(verts)


def test_redundant_vertices():
    """
    Test that the redundant vertices are correct
    """
    verts = [[0, 0], [1, 0], [1, 0], [1, 1], [2, 0], [0, 2], [2, 2], [2, 2]]
    p = Polytope(verts)

    assert p.n_vertices == 4


def test_rational_vertices_precision():
    """
    Test that sympy rationals are correctly converted cdd rationals
    """
    verts = [
        [0],
        [Rational(2304652398467913471348138749134013409345716384712935, 2)],
    ]
    polytope = Polytope(verts)

    assert polytope.vertices == PointConfiguration(verts)


def test_inequalities():
    """
    Test that the inequalities of a cube are correctly tranformed from cdd
    format to sympy format.
    """
    p = Polytope.cube(3) * Rational(1, 2)

    expected = np.array(
        [
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [Rational(1, 2), -1, 0, 0],
            [Rational(1, 2), 0, -1, 0],
            [Rational(1, 2), 0, 0, -1],
        ]
    )

    assert _arrays_equal_up_to_row_permutation(p.inequalities, expected)


def test_homogeneous_inequalities():
    """
    Test that the homogenous_inequalities are correctly calculated.
    """
    p = Polytope.cube(3) * Rational(1, 2)

    expected = np.array(
        [
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, -2, 0, 0],
            [1, 0, -2, 0],
            [1, 0, 0, -2],
        ]
    )

    assert _arrays_equal_up_to_row_permutation(p.homogeneous_inequalities, expected)


def test_n_inequalities():
    """
    Test that the number of inequalities is correct
    """
    p = Polytope.cube(3)

    assert p.n_inequalities == 6


def test_n_equalities():
    """
    Test that the number of equalities is correct
    """
    p = Polytope.cube(3)

    assert p.n_equalities == 0


def test_triangulation():
    """
    Test that the triangulation is correct
    """
    p = Polytope.cube(2)

    possible_triangulations = [
        (frozenset({0, 1, 3}), frozenset({0, 2, 3})),
        (frozenset({0, 1, 2}), frozenset({1, 2, 3})),
        (frozenset({0, 2, 3}), frozenset({0, 1, 3})),
        (frozenset({1, 2, 3}), frozenset({0, 1, 2})),
    ]

    assert p.triangulation in possible_triangulations


def test_triangulation_simplex():
    """
    Test that the triangulation is correct for a simplex (as it is not explicitely
    calculated)
    """
    p = Polytope.unimodular_simplex(2)
    p._make_simplex()

    assert p.triangulation == (frozenset({0, 1, 2}),)


def test_half_open_decomposition():
    """
    Test that the half-open decomposition is correct
    """
    p = Polytope.cube(2)

    assert len(p.half_open_decomposition) == 2
    assert len(p.half_open_decomposition[0]) + len(p.half_open_decomposition[1]) == 1


def test_half_open_decomposition_simplex():
    """
    Test that the half-open decomposition is correct for a simplex
    (as it is not explicitely calculated)
    """
    p = Polytope.unimodular_simplex(2)
    p._make_simplex()

    assert p.half_open_decomposition == (frozenset({}),)


def test_induced_boundary_triangulation():
    """
    Test that the triangulation of the boundary of a polytope induced
    by a triangulation of the polytope is correct
    """
    p = Polytope.cube(3)

    bt = p.induced_boundary_triangulation

    assert len(bt) == 12
    assert all(len(s) == 3 for s in bt)


def test_triangulation_lower_dimensional_polytope():
    """
    Test that the triangulation is correct
    """
    p = Polytope([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0]])

    possible_triangulations = [
        (frozenset({0, 1, 3}), frozenset({0, 2, 3})),
        (frozenset({0, 1, 2}), frozenset({1, 2, 3})),
        (frozenset({0, 2, 3}), frozenset({0, 1, 3})),
        (frozenset({1, 2, 3}), frozenset({0, 1, 2})),
    ]

    assert p.triangulation in possible_triangulations


def test_triangulation_one_dimensional_polytope():
    """
    Test that the triangulation is correct
    """
    p = Polytope([[0], [1]])

    assert p.triangulation == (frozenset({0, 1}),)


def test_get_volume_segment():
    """
    Test that the volume is correct
    """
    p = Polytope.cube(1) * 10

    assert p._volume is None
    assert p._normalized_volume is None
    assert p.volume == 10  # This triggers a volume computation
    assert p._volume == 10
    assert p._normalized_volume == 10
    assert p.normalized_volume == 10


def test_get_volume_cube():
    """
    Test that the volume is correct
    """
    polytope = Polytope.cube(3)

    assert polytope._volume is None
    assert polytope._normalized_volume is None
    assert polytope.volume == 1  # This triggers a volume computation
    assert polytope._volume == 1
    assert polytope._normalized_volume == 6
    assert polytope.normalized_volume == 6


def test_get_volume_simplex():
    """
    Test that the volume is correct
    """
    polytope = Polytope.unimodular_simplex(3)

    assert polytope._volume is None
    assert polytope._normalized_volume is None
    assert polytope.volume == Rational(1, 6)  # This triggers a volume computation
    assert polytope._volume == Rational(1, 6)
    assert polytope._normalized_volume == 1
    assert polytope.normalized_volume == 1


def test_volume_computed_only_once():
    """
    Test that the volume is computed only once
    """
    polytope = Polytope.unimodular_simplex(3)

    def mock_calculate_volume():
        polytope._volume = 1
        polytope._normalized_volume = 1

    polytope._calculate_volume = MagicMock(side_effect=mock_calculate_volume)

    assert polytope._volume is None
    assert polytope._normalized_volume is None
    assert polytope.volume == 1  # This triggers a volume computation
    assert polytope._volume == 1
    assert polytope._normalized_volume == 1
    assert polytope.normalized_volume == 1

    # Check that the _calculate_volume method was called only once
    polytope._calculate_volume.assert_called_once()


def test_volume_lower_dimensional_polytope():
    """
    Test that the volume is correct for a lower dimensional polytope
    """
    verts = [[0, 0, 0], [2, 0, 0], [0, 3, 0]]
    p = Polytope(verts)

    assert p.volume == 3
    assert p.normalized_volume == 6


def test_boundary_volume():
    """
    Test that the boundary volume is calculated correctly
    """
    p = Polytope.cube(3)

    assert p.boundary_volume == 6


def test_normalized_boundary_volume():
    """
    Test that the normalized boundary volume is calculated correctly
    """
    p = Polytope.cube(3)
    assert p.normalized_boundary_volume == 12


def test_barycenter():
    """
    Test that the barycenter is correct
    """
    polytope = Polytope.unimodular_simplex(3)

    assert polytope.barycenter == Point(
        [Rational(1, 4), Rational(1, 4), Rational(1, 4)]
    )


def test_translation():
    """
    Test that the translation is correct
    """
    polytope = Polytope.cube(3) + Point([1, 1, 1])

    assert polytope.vertices == PointConfiguration(
        [
            [1, 1, 1],
            [1, 1, 2],
            [1, 2, 1],
            [1, 2, 2],
            [2, 1, 1],
            [2, 1, 2],
            [2, 2, 1],
            [2, 2, 2],
        ]
    )


def test_dilation():
    """
    Test that the dilation is correct
    """
    polytope = Polytope.cube(3) * 2

    assert polytope.vertices == PointConfiguration(
        [
            [0, 0, 0],
            [0, 0, 2],
            [0, 2, 0],
            [0, 2, 2],
            [2, 0, 0],
            [2, 0, 2],
            [2, 2, 0],
            [2, 2, 2],
        ]
    )


def test_translation_and_dilation():
    """
    Test that the translation and dilation are correct
    """
    polytope = Polytope.cube(3) * 2 - Point([1, 1, 1])

    assert polytope.vertices == PointConfiguration(
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


@pytest.mark.parametrize("strict", [True, False])
def test_contains_point(strict):
    """
    Test that the contains method works correctly
    """
    polytope = Polytope.cube(3) * 2 - Point([1, 1, 1])

    assert polytope.contains(Point([0, 0, 0]), strict)
    assert polytope.contains(Point([0, 0, 1]), strict) is (False if strict else True)
    assert polytope.contains(Point([0, 1, 1]), strict) is (False if strict else True)
    assert polytope.contains(Point([1, 1, 1]), strict) is (False if strict else True)
    assert not polytope.contains(Point([0, 0, 2]), strict)


@pytest.mark.parametrize("strict", [True, False])
def test_lower_dim_polytope_contains_point(strict):
    """
    Test that points are correctly contained (or not) in lower dimensional polytopes
    """
    p = Polytope.cube(2) * 2 - Point([1, 1])
    new_verts = [v.tolist() + [0] for v in p.vertices]
    p = Polytope(new_verts)

    assert p.contains(Point([0, 0, 0]), strict)
    assert p.contains(Point([1, 0, 0]), strict) is (False if strict else True)
    assert p.contains(Point([1, 1, 0]), strict) is (False if strict else True)
    assert not p.contains(Point([0, 0, 1]), strict)


@pytest.mark.parametrize("strict", [True, False])
def test_contains_polytope(strict):
    """
    Test that the contains method works correctly for a polytope
    """
    p = Polytope.cube(3) * 3
    p1 = Polytope.cube(3)
    p2 = Polytope.cube(3) + Point([1, 1, 1])

    assert p.contains(p, strict) == (False if strict else True)
    assert p.contains(p1, strict) == (False if strict else True)
    assert p.contains(p2, strict)

    # check that vertices are not used if not available
    pts = PointConfiguration([[a**1, a**2, a**3] for a in range(100)])
    p3 = Polytope(pts)
    assert not p.contains(p3)
    assert p3._vertices is None


def test_lower_dim_polytope_contains_polytope():
    """
    Test that points are correctly contained (or not) in lower dimensional polytopes
    """
    verts = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
    ]
    p = Polytope(vertices=verts)

    assert (p * 2).contains(p)
    assert (p * 2).contains(p * 2)
    assert not (p * 2).contains(p * 3)


def test_contains_wrong_type():
    """
    Test that the contains method raises an exception when called with a wrong type
    """
    p = Polytope.cube(3)

    with pytest.raises(TypeError):
        p.contains(1)


def test_facets():
    """
    Test that the facets of a cube are six and of the right shape
    """
    p = Polytope.cube(3)

    p.inequalities

    assert p.n_facets == 6

    for facet in p.facets:
        assert len(facet) == 4


def test_ridges():
    """
    Test that the ridges of a 3-cube are its edges
    """
    p = Polytope.cube(3)

    assert p.n_ridges == 12

    for ridge in p.ridges:
        assert len(ridge) == 2

    assert frozenset(p.ridges) == frozenset(p.edges)


def test_edges():
    """
    Test that the edges of a cube are twelve
    """
    p = Polytope.cube(3)

    assert p.n_edges == 12

    for edge in p.edges:
        assert len(edge) == 2


def test_faces():
    """
    Test calculation of the faces of a polytope
    """
    p = Polytope.unimodular_simplex(7)

    # test edge cases
    with pytest.raises(ValueError):
        p.faces(-2)
    assert p.faces(-1) == tuple([frozenset()])
    assert p.faces(0) == tuple([frozenset([i]) for i in range(p.n_vertices)])
    assert p.faces(7) == (frozenset(range(p.n_vertices)),)
    assert p.faces(8) == tuple()

    # test length for other dimensions
    assert len(p.faces(6)) == 8
    assert len(p.faces(5)) == 28
    assert len(p.faces(4)) == 56
    assert len(p.faces(3)) == 70
    assert len(p.faces(2)) == 56


def test_faces_remove_temp_faces():
    """
    This is a test to check that the temporary faces are removed correctly
    in the faces algorithm. This helps to achieve 100% code coverage for the
    faces algorithm, which I couldn't achieve with a low dimensional polytope,
    as the facets need to be listed in a particular order for the algorithm
    to remove temporary faces.
    """
    p = Polytope.cross_polytope(5)

    # Reorder the 3-dim faces so that the algorithm will add and then remove a
    # temporary face
    for face in p.faces(3):
        f = face.intersection(p.facets[0])
        if len(f) < 3:
            break
    p._faces[3] = tuple([face] + [f for f in p._faces[3] if f != face])

    assert len(p.faces(2)) == 80


def test_faces_lower_dimensional_polytope():
    """
    Test calculation of the faces of a polytope
    """
    p = Polytope.cube(3)
    p = Polytope([list(v) + [0] for v in p.vertices])

    # test length for other dimensions
    assert len(p.faces(3)) == 1
    assert len(p.faces(2)) == 6
    assert len(p.faces(1)) == 12
    assert len(p.faces(0)) == 8
    assert len(p.faces(-1)) == 1


def test_neighbors():
    """
    Test the neighbors method on a vertex of a cube
    """
    p = Polytope.cube(3)
    neighbors = p.neighbors(0)
    assert len(neighbors) == 3
    assert neighbors == frozenset([1, 2, 4])


@pytest.mark.parametrize("v_repr", [True, False])
def test_vertex_adjacency_matrix(v_repr):
    """
    Test calculation of the vertex adjacency matrix of a polytope
    """
    p = Polytope.cube(3)

    if v_repr:
        expected = np.array(
            [
                [0, 1, 1, 0, 1, 0, 0, 0],
                [1, 0, 0, 1, 0, 1, 0, 0],
                [1, 0, 0, 1, 0, 0, 1, 0],
                [0, 1, 1, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 1, 1, 0],
                [0, 1, 0, 0, 1, 0, 0, 1],
                [0, 0, 1, 0, 1, 0, 0, 1],
                [0, 0, 0, 1, 0, 1, 1, 0],
            ]
        )
    else:
        p = Polytope(inequalities=p.inequalities)
        # different matrix due to vertex order permutation
        expected = np.array(
            [
                [0, 1, 0, 1, 0, 0, 0, 1],
                [1, 0, 1, 0, 0, 0, 1, 0],
                [0, 1, 0, 1, 1, 0, 0, 0],
                [1, 0, 1, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 1, 1, 0],
                [0, 0, 0, 1, 1, 0, 0, 1],
                [0, 1, 0, 0, 1, 0, 0, 1],
                [1, 0, 0, 0, 0, 1, 1, 0],
            ]
        )

    assert np.array_equal(p.vertex_adjacency_matrix, p.vertex_adjacency_matrix.T)
    assert np.array_equal(p.vertex_adjacency_matrix, expected)


@pytest.mark.parametrize("v_repr", [True, False])
def test_facet_adjacency_matrix(v_repr):
    """
    Test calculation of the facet adjacency matrix of a polytope
    """
    p = Polytope.cube(3)
    if not v_repr:
        p = Polytope(inequalities=p.inequalities)

    expected = np.array(
        [
            [0, 1, 1, 0, 1, 1],
            [1, 0, 1, 1, 1, 0],
            [1, 1, 0, 1, 0, 1],
            [0, 1, 1, 0, 1, 1],
            [1, 1, 0, 1, 0, 1],
            [1, 0, 1, 1, 1, 0],
        ]
    )

    assert np.array_equal(p.facet_adjacency_matrix, p.facet_adjacency_matrix.T)
    assert np.array_equal(p.facet_adjacency_matrix, expected)


@pytest.mark.parametrize("v_repr", [True, False])
def test_vertex_facet_matrix(v_repr):
    """
    Test calculation of the vertex facet matrix of a polytope
    """
    p = Polytope.cube(3)
    if v_repr:
        expected = np.array(
            [
                [1, 0, 1, 0, 1, 0, 1, 0],
                [1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 0, 0, 1, 1, 0, 0],
                [0, 1, 0, 1, 0, 1, 0, 1],
                [0, 0, 1, 1, 0, 0, 1, 1],
                [0, 0, 0, 0, 1, 1, 1, 1],
            ]
        )
    else:
        p = Polytope(inequalities=p.inequalities)
        # different matrix due to vertex order permutation
        expected = np.array(
            [
                [1, 1, 1, 1, 0, 0, 0, 0],
                [0, 1, 1, 0, 1, 0, 1, 0],
                [0, 0, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 1, 1, 1, 1],
                [1, 1, 0, 0, 0, 0, 1, 1],
                [1, 0, 0, 1, 0, 1, 0, 1],
            ]
        )
    assert _arrays_equal_up_to_row_permutation(p.vertex_facet_matrix, expected)


def test_vertex_facet_pairing_matrix():
    """
    Test calculation of the vertex facet pairing matrix of a polytope
    """
    vertices = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [-1, 0, 1],
        [0, 1, -1],
        [0, -1, 0],
        [0, 0, -1],
    ]
    polytope = Polytope(vertices)

    expected = np.array(
        [
            [2, 0, 1, 0, 0, 2, 1],
            [1, 0, 0, 0, 1, 2, 2],
            [3, 2, 2, 0, 1, 0, 0],
            [1, 2, 0, 0, 3, 0, 2],
            [3, 1, 2, 0, 0, 1, 0],
            [0, 2, 0, 1, 3, 0, 2],
            [0, 2, 2, 3, 1, 0, 0],
            [0, 1, 2, 3, 0, 1, 0],
            [0, 0, 0, 1, 1, 2, 2],
            [0, 0, 1, 2, 0, 2, 1],
        ]
    )
    assert np.array_equal(polytope.vertex_facet_pairing_matrix, expected)


def test_dual():
    cube = Polytope.cube(3) * 2 - Point([1, 1, 1])
    cp = Polytope.cross_polytope(3)
    assert _arrays_equal_up_to_row_permutation(cube.dual.vertices, cp.vertices)
    assert _arrays_equal_up_to_row_permutation(
        (cube * Rational(1, 2)).dual.vertices, (cp * 2).vertices
    )

    with pytest.raises(ValueError):
        Polytope([[-1, 0], [1, 0]]).dual

    with pytest.raises(ValueError):
        Polytope.cube(3).dual


def test_f_vector():
    """
    Test calculation of the f-vector for an hypercube
    """
    p = Polytope.cube(7)

    expected_f_vector = (1, 128, 448, 672, 560, 280, 84, 14, 1)
    assert p.f_vector == expected_f_vector


def test_vertex_facet_matrix_low_dimensional_polytope():
    """
    Test calculation of the vertex facet matrix of a non-full-dimensional polytope
    """
    verts = [[0, 0, 1], [7, 4, 19], [5, 3, 14], [12, 7, 32]]
    square = Polytope(verts)

    expected_vfm = np.array(
        [[1, 0, 1, 0], [1, 1, 0, 0], [0, 1, 0, 1], [0, 0, 1, 1]],
        dtype=bool,
    )

    expected_vfpm = np.array(
        [
            [0, 1, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 1, 0],
            [1, 1, 0, 0],
        ]
    )
    assert np.array_equal(square.vertex_facet_matrix, expected_vfm)
    assert np.array_equal(square.vertex_facet_pairing_matrix, expected_vfpm)


def test_vertex_facet_pairing_matrix_is_nonnegative():
    """
    Test that the vertex facet pairing matrix is nonnegative
    """
    p = Polytope.random_lattice_polytope(dim=4, n_points=10, min=-5, max=5)
    assert np.all((p.vertex_facet_pairing_matrix) >= 0)


def test_vertex_facet_pairing_matrix_low_dimensional_polytope():
    """
    Test calculation of the vertex facet pairing matrix of a non-full-dimensional
    polytope
    """
    verts = [[0, 0, 1], [1, 0, 6], [0, 1, 5], [1, 1, 10]]
    square = Polytope(verts)

    expected = np.array(
        [
            [0, 0, 1, 1],
            [0, 1, 0, 1],
            [1, 1, 0, 0],
            [1, 0, 1, 0],
        ]
    )
    assert np.array_equal(square.vertex_facet_pairing_matrix, expected)


def test_normal_form():
    """
    Test calculation of the normal form of a polytope
    """
    cube_1 = Polytope.cube(3) * 2 - Point([1, 1, 1])

    unimodular_map = Matrix(
        [
            [4, -1, 0],
            [-7, 2, 0],
            [-2, -1, 1],
        ]
    )

    cube_2 = Polytope(Matrix(cube_1.vertices) * unimodular_map)

    assert cube_1.normal_form == cube_2.normal_form


def test_affine_normal_form():
    """
    Test calculation of the affine normal form of a polytope
    """
    cube_1 = Polytope.random_lattice_polytope(dim=3, n_points=8, min=-2, max=2)

    unimodular_map = Matrix(
        [
            [4, -1, 0],
            [-7, 2, 0],
            [-2, -1, 1],
        ]
    )

    cube_2 = Polytope(vertices=(Matrix(cube_1.vertices) * unimodular_map)) - Point(
        [5, -1, 3]
    )

    assert cube_1.affine_normal_form == cube_2.affine_normal_form


def test_affine_normal_form_idempotent():
    """
    Test that the affine normal form is idempotent
    """
    polytope = Polytope.random_lattice_polytope(dim=4, n_points=10, min=-2, max=2)
    polytope_anf = Polytope(vertices=polytope.affine_normal_form)
    assert polytope.affine_normal_form == polytope_anf.affine_normal_form


def test_affine_normal_lower_dimensional_polytope():
    """
    Test that the affine normal works for lower dimensional polytopes
    """
    p = Polytope.cube(2)
    q = Polytope([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
    assert p.affine_normal_form == q.affine_normal_form


def test_integer_points():
    """
    Test that the integer/interior/boundary points of a polytope are correct
    """
    s = Polytope.unimodular_simplex(3)
    s2 = s * 2
    s3 = s * 3
    s4 = s * 4

    c = Polytope.cube(3)
    c2 = c * 2

    assert _arrays_equal_up_to_row_permutation(s.integer_points, s.vertices)
    assert s.n_interior_points == 0
    assert s.n_boundary_points == 4

    assert s2.n_interior_points == 0
    assert s2.n_integer_points == 10
    assert s2.n_boundary_points == 10

    assert s3.n_integer_points == 20
    assert s3.n_interior_points == 0
    assert s3.n_boundary_points == 20

    assert s4.n_integer_points == 35
    assert s4.n_interior_points == 1
    assert s4.n_boundary_points == 34
    assert s4.interior_points == PointConfiguration([[1, 1, 1]])

    assert _arrays_equal_up_to_row_permutation(c.integer_points, c.vertices)
    assert c.n_interior_points == 0
    assert c.n_boundary_points == 8

    assert c2.n_integer_points == 27
    assert c2.n_interior_points == 1
    assert c2.n_boundary_points == 26
    assert c2.interior_points == PointConfiguration([[1, 1, 1]])


def test_integer_points_consistency():
    """
    Test that the integer/interior/boundary points of a polytope are consistent
    to what is obtained by looking in the parallelotopes over the half-open
    decomposition.
    """
    p = Polytope.cube(3)

    n_pts = 0
    pts = []

    for verts_ids, special_gens_ids in zip(p.triangulation, p.half_open_decomposition):
        hop = HalfOpenParallelotope(
            generators=[Point([1] + list(p.vertices[v_id])) for v_id in verts_ids],
            special_gens_ids=special_gens_ids,
        )

        pts_parallelotope, _ = hop.get_integer_points(height=1)
        n_pts += len(pts_parallelotope)
        pts += [Point(pt[1:]) for pt in pts_parallelotope]

    # compensate for missing vertices
    n_pts += p.dim + 1
    for v_id in p.triangulation[0]:
        pts.append(Point(p.vertices[v_id]))

    assert p.n_integer_points == n_pts
    assert _arrays_equal_up_to_row_permutation(p.integer_points, pts)


def test_integer_points_low_dimensional_polytope():
    """
    Test that an exception is raised when trying to compute the integer points of
    a non full-dimensional polytope
    """
    verts = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    square = Polytope(verts)

    with pytest.raises(ValueError):
        square.integer_points


def test_has_n_interior_points():
    """
    Test that the has_n_interior_points method works correctly and the lattice points
    enumeration stops if n is exceeded
    """
    c = Polytope.cube(3) * 2

    assert not c.has_n_interior_points(0)
    assert c._n_interior_points is None  # enumeration stopped and no value stored

    assert c.has_n_interior_points(1)
    assert c._n_interior_points == 1  # enumeration completed and value stored


def test_boundary_points_facets():
    """
    Test that facets of the boundary points of a polytope are calculated correctly
    """
    p = Polytope.cube(3)
    boundary_points_facets = p.boundary_points_facets

    for pt, f_ids in zip(p.boundary_points, boundary_points_facets):
        for j, ineq in enumerate(p.inequalities):
            if np.dot(ineq[1:], pt) + ineq[0] == 0:
                assert j in f_ids
            else:
                assert j not in f_ids


def test_ehrhart_polynomial():
    """
    Test that the Ehrhart polynomial of a lattice polytope is correct
    """
    with pytest.raises(ValueError):
        p = Polytope.unimodular_simplex(1) * Rational(1, 2)
        p.ehrhart_polynomial

    assert Polytope.unimodular_simplex(1).ehrhart_polynomial == Poly(x + 1, domain="QQ")
    assert Polytope.unimodular_simplex(2).ehrhart_polynomial == Poly(
        x**2 / 2 + 3 * x / 2 + 1, domain="QQ"
    )
    assert Polytope.unimodular_simplex(3).ehrhart_polynomial == Poly(
        x**3 / 6 + x**2 + 11 * x / 6 + 1, domain="QQ"
    )

    for d in range(1, 5):
        assert Polytope.cube(d).ehrhart_polynomial == Poly((x + 1) ** d, domain="QQ")


def test_ehrhart_coefficients():
    """
    Test that the Ehrhart coefficients of a lattice polytope are correct
    """

    assert Polytope.unimodular_simplex(3).ehrhart_coefficients == (
        1,
        Rational(11, 6),
        1,
        Rational(1, 6),
    )


def test_h_star_polynomial():
    """
    Test that the h*-polynomial of a lattice polytope is correct
    """
    with pytest.raises(ValueError):
        p = Polytope.unimodular_simplex(1) * Rational(1, 2)
        p.h_star_polynomial

    for d in range(1, 5):
        assert Polytope.unimodular_simplex(d).h_star_polynomial == Poly(1, x)

    assert Polytope.cube(1).h_star_polynomial == Poly(1, x)
    assert Polytope.cube(2).h_star_polynomial == Poly(x + 1)
    assert Polytope.cube(3).h_star_polynomial == Poly(x**2 + 4 * x + 1)


def test_h_star_vector():
    """
    Test that the h*-vector of a lattice polytope is correct
    """
    assert Polytope.cube(3).h_star_vector == (1, 4, 1, 0)
    assert (Polytope.cube(3) * 2).h_star_vector == (1, 23, 23, 1)


def test_gamma_vector():
    p = Polytope(
        [
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, -1],
            [1, 1, 1],
        ]
    )
    assert p.gamma_vector == (1, -2, 0, 0)

    p = Polytope(
        [
            [-1, 0],
            [0, -1],
            [1, 1],
        ]
    )
    assert p.gamma_vector == (1, -1, 0)


def test_gamma_polynomial():
    p = Polytope(
        [
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, -1],
            [1, 1, 1],
        ]
    )
    assert p.gamma_polynomial == Poly(1 - 2 * x)


def test_degree():
    """
    Test that the degree of a lattice polytope is correct
    """
    assert Polytope.unimodular_simplex(3).degree == 0
    assert (Polytope.unimodular_simplex(3) * 2).degree == 2
    assert (Polytope.unimodular_simplex(3) * 3).degree == 2
    assert (Polytope.unimodular_simplex(3) * 4).degree == 3

    assert (Polytope.cube(3) * 1).degree == 2
    assert (Polytope.cube(3) * 2).degree == 3


def test_half_open_parallelotopes_pts():
    """
    Test that the half-open parallelotopes are calculated correctly
    """
    p = Polytope.cube(3)

    assert len([pt for pt in p.half_open_parallelotopes_pts if pt[0] == 0]) == 1
    assert len([pt for pt in p.half_open_parallelotopes_pts if pt[0] == 1]) == 4
    assert len([pt for pt in p.half_open_parallelotopes_pts if pt[0] == 2]) == 1


def test_avoid_parallelotopes_pts_recomputation():
    """
    Test that the half-open parallelotopes are not recomputed if available.
    Check this by providing a fake half_open_parallelotopes_pts property,
    and make sure that the h*-vector is based on that (and hence wrong).
    """
    p = Polytope.unimodular_simplex(2)

    p._half_open_parallelotopes_pts = [
        Point([0, 0, 0]),
        Point([1, 0, 0]),
        Point([1, 1, 0]),
        Point([2, 0, 0]),
    ]

    assert p.h_star_vector == (1, 2, 1)  # instead of (1, 0, 0)


def test_hilbert_basis():
    """
    Test that the Hilbert basis is calculated correctly
    """
    with pytest.raises(ValueError):
        p = Polytope.unimodular_simplex(1) * Rational(1, 2)
        p.hilbert_basis

    p = Polytope.cube(3)
    assert len(p.hilbert_basis) == p.n_integer_points

    p = Polytope(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 2],
        ]
    )
    assert len(p.hilbert_basis) == p.n_integer_points + 1


def test_add():
    """
    Test that the __add__ method works correctly
    """
    p = Polytope.cube(2) + Point([1, 1])
    expected = PointConfiguration([[1, 1], [1, 2], [2, 1], [2, 2]])
    assert _arrays_equal_up_to_row_permutation(p.vertices, expected)

    p = Polytope.cube(2) + Polytope.cube(2)
    expected = Polytope.cube(2) * 2
    assert _arrays_equal_up_to_row_permutation(p.vertices, expected.vertices)

    with pytest.raises(ValueError):
        Polytope.cube(1) + Polytope.cube(2)

    with pytest.raises(TypeError):
        Polytope.cube(2) + 1


def test_neg():
    """
    Test that the __neg__ method works correctly
    """
    p = -Polytope.cube(2)
    expected = PointConfiguration([[0, 0], [0, -1], [-1, 0], [-1, -1]])
    assert _arrays_equal_up_to_row_permutation(p.vertices, expected)


def test_mult():
    """
    Test that the __mult__ method works correctly
    """
    p = Polytope.cube(2) * 2
    expected = PointConfiguration([[0, 0], [0, 2], [2, 0], [2, 2]])
    assert _arrays_equal_up_to_row_permutation(p.vertices, expected)

    p = Polytope.cube(2) * Rational(1, 2)
    expected = PointConfiguration(
        [
            [0, 0],
            [0, Rational(1, 2)],
            [Rational(1, 2), 0],
            [Rational(1, 2), Rational(1, 2)],
        ]
    )
    assert _arrays_equal_up_to_row_permutation(p.vertices, expected)

    p = Polytope.cube(2) * Polytope([[2], [3]])
    expected = PointConfiguration(
        [
            [0, 0, 2],
            [0, 1, 2],
            [1, 0, 2],
            [1, 1, 2],
            [0, 0, 3],
            [0, 1, 3],
            [1, 0, 3],
            [1, 1, 3],
        ]
    )
    assert _arrays_equal_up_to_row_permutation(p.vertices, expected)

    with pytest.raises(TypeError):
        Polytope.cube(2) * "a not allowed type"


def test_hilbert_basis_empty_polytope():
    """
    Test that the Hilbert basis of the empty polytope is "empty",
    i.e. if it has no lattice points other than the vertices.
    """
    p = Polytope.unimodular_simplex(2)
    assert len(p.hilbert_basis) == 3


@pytest.mark.parametrize("stop_at_height", [-1, 2])
def test_get_hilbert_basis_stopped(stop_at_height):
    """
    Test that the method _get_hilbert_basis stops correctly if requested
    """
    p = Polytope(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 3],
        ]
    )
    hb = p._get_hilbert_basis(stop_at_height=stop_at_height)

    if stop_at_height == -1:
        assert len(hb) == p.n_integer_points + 2
    else:
        assert len(hb) == p.n_integer_points + 1


def test_is_simplicial():
    """
    Test the is_simplicial property
    """
    assert Polytope.unimodular_simplex(3).is_simplicial
    assert not (Polytope.unimodular_simplex(2) * Polytope.cube(1)).is_simplicial
    assert not Polytope.cube(3).is_simplicial


def test_is_simple():
    """
    Test the is_simple property
    """
    assert Polytope.unimodular_simplex(3).is_simple
    assert (Polytope.unimodular_simplex(2) * Polytope.cube(1)).is_simple
    assert Polytope.cube(3).is_simple
    assert not Polytope(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1]]
    ).is_simple


def test_is_lattice_polytope():
    """
    Test the is_lattice_polytope method
    """
    polytope = Polytope.unimodular_simplex(dim=3)
    assert polytope.is_lattice_polytope

    polytope = polytope * Rational(3, 2)
    assert not polytope.is_lattice_polytope


def test_is_lattice_pyramid():
    """
    Test the is_lattice_pyramid method
    """
    p = Polytope.unimodular_simplex(dim=3)

    with pytest.raises(ValueError):
        (p * Rational(1, 2)).is_lattice_pyramid

    assert p.is_lattice_pyramid

    assert not (p * 2).is_lattice_pyramid


def test_is_lattice_pyramid_low_dimensional_polytope():
    """
    Test that the is_lattice_pyramid method works for low dimensional polytopes
    """
    p = Polytope([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    assert p.is_lattice_pyramid

    p = Polytope([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
    assert not p.is_lattice_pyramid


def test_is_hollow():
    """
    Test the is_hollow property
    """
    s = Polytope.unimodular_simplex(3)

    assert s.is_hollow
    assert (s * 2).is_hollow
    assert (s * 3).is_hollow
    assert not (s * 4).is_hollow


def test_has_one_interior_point():
    """
    Test the has_one_interior_point property
    """
    s = Polytope.unimodular_simplex(3)

    assert s.is_hollow
    assert (s * 2).is_hollow
    assert (s * 3).is_hollow
    assert not (s * 4).is_hollow


def test_is_canonical():
    """
    Test the is_canonical property
    """
    s = Polytope.unimodular_simplex(3)

    assert not s.is_canonical
    assert not (s * 2).is_canonical
    assert not (s * 3).is_canonical
    assert not (s * 4).is_canonical
    assert ((s * 4) - Point([1, 1, 1])).is_canonical
    assert not ((s * 5) - Point([1, 1, 1])).is_canonical


def test_is_reflexive():
    """
    Test the is_reflexive property
    """
    assert not Polytope.unimodular_simplex(2).is_reflexive

    s = Polytope.unimodular_simplex(3) * 4 - Point([1, 1, 1])
    c = Polytope.cube(3) * 2 - Point([1, 1, 1])

    assert s.is_reflexive
    assert c.is_reflexive

    pts = [pt for pt in c.integer_points if pt != Point([1, 1, 1])]
    c2 = Polytope(pts)
    assert c2.is_canonical
    assert not c2.is_reflexive


def test_is_gorenstein():
    """
    Test the is_gorenstein property
    """
    assert Polytope.unimodular_simplex(3).is_gorenstein
    assert Polytope.cube(3).is_gorenstein
    assert (Polytope.cube(3) * 2).is_gorenstein
    assert not (Polytope.cube(3) * 3).is_gorenstein


def test_is_ehrhart_positive():
    """
    Test the is_ehrhart_positive property
    """
    s = Polytope.unimodular_simplex(3)

    assert s.is_ehrhart_positive

    p = Polytope(
        [
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [1, 0, 26, 27],
        ]
    )

    assert not p.is_ehrhart_positive


def test_has_log_concave_h_star_vector():
    """
    Test the has_log_concave_h_star_vector property
    """
    p = Polytope(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 15],
            [1, 1, 16],
        ]
    )

    assert p.has_log_concave_h_star_vector

    p = Polytope(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 16],
            [1, 1, 17],
        ]
    )

    assert not p.has_log_concave_h_star_vector


def test_has_unimodal_h_star_vector():
    """
    Test the has_unimodal_h_star_vector property
    """
    s = Polytope.unimodular_simplex(3) * 4

    assert s.has_unimodal_h_star_vector

    p = Polytope([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 2]])

    assert not p.has_unimodal_h_star_vector


def test_is_centrally_symmetric():
    """
    Test that the is_centrally_symmetric property works correctly.
    """
    c = Polytope.cube(2) * 2
    assert not c.is_centrally_symmetric

    c = c - Point([1, 1])
    assert c.is_centrally_symmetric


def test_is_spanning():
    """
    Test the is_spanning property. In the first case only the vertices should be used.
    """
    p = Polytope.unimodular_simplex(3)
    assert p.is_spanning
    assert p._integer_points is None

    p = Polytope([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 2]])
    assert not p.is_spanning
    assert p._integer_points is not None

    p = Polytope([[-1], [1]])
    assert p.is_spanning

    p = Polytope([[-1], [1]])
    p._integer_points = PointConfiguration([[-1], [1]])
    assert not p.is_spanning


def test_is_very_ample():
    """
    Test the is_very_ample property
    """
    p = Polytope.unimodular_simplex(3)
    assert p.is_very_ample

    p = Polytope.cube(3)
    assert p.is_very_ample

    p = Polytope.reeve_simplex(3, 2)
    assert not p.is_very_ample

    p = Polytope([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 2], [0, 0, -1]])
    assert p.is_very_ample

    # very ample, but not idp
    p = Polytope(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 2],
            [1, 1, 3],
            [1, 0, -1],
            [0, 1, -1],
            [0, 0, 1],
        ]
    )
    assert p.is_very_ample

    p = Polytope(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [-1, 1, 1],
            [-2, 1, 1],
            [2, 1, 0],
            [2, 0, -2],
            [1, 0, -2],
        ]
    )
    assert p.is_very_ample


def test_is_idp():
    """
    Test the is_idp property. In the first two cases by computing the Hilbert
    basis, in the last case by noting that the pattice points span a sublattice
    of index > 1.
    """
    p = Polytope([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 2], [0, 0, -1]])
    assert p.is_idp

    p = Polytope([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 3], [0, 0, -1]])
    assert not p.is_idp

    # very ample, but not idp
    p = Polytope(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 2],
            [1, 1, 3],
            [1, 0, -1],
            [0, 1, -1],
            [0, 0, 1],
        ]
    )
    assert not p.is_idp

    p = Polytope([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 2]])
    p._get_hilbert_basis = MagicMock()
    assert not p.is_idp
    assert p._get_hilbert_basis.call_count == 0  # no need to compute the Hilbert basis


def test_is_smooth():
    """
    Test the is_smooth property
    """
    assert Polytope.cube(3).is_smooth
    assert not Polytope([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 2]]).is_smooth
    assert not Polytope(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 3], [0, 0, -1]]
    ).is_smooth


def test_free_sum():
    """
    Test the free_sum method
    """
    assert _arrays_equal_up_to_row_permutation(
        Polytope.cube(1).free_sum(Polytope.cube(1)).vertices,
        PointConfiguration([[0, 0], [0, 1], [1, 0]]),
    )

    with pytest.raises(TypeError):
        assert Polytope.cube(2).free_sum(1)


def test_cayley_sum():
    """
    Test the cayley_sum method
    """
    assert _arrays_equal_up_to_row_permutation(
        Polytope.cube(2).cayley_sum(Polytope.unimodular_simplex(2)).vertices,
        PointConfiguration(
            [
                [0, 0, 0],
                [0, 1, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [0, 1, 1],
            ]
        ),
    )

    with pytest.raises(TypeError):
        assert Polytope.cube(2).cayley_sum(1)

    with pytest.raises(ValueError):
        assert Polytope.cube(2).cayley_sum(Polytope.cube(1))


def test_chisel_vertex():
    """
    Test that the chisel_vertex method works correctly
    """
    with pytest.raises(ValueError):
        assert (Polytope.cube(2) * Rational(1, 2)).chisel_vertex(0, 1)

    with pytest.raises(ValueError):
        assert Polytope.cube(2).chisel_vertex(0, -1)

    with pytest.raises(ValueError):
        assert Polytope.cube(2).chisel_vertex(0, 2)

    c = Polytope.cube(2) * 2
    assert c.chisel_vertex(0, 0) == c
    assert c.chisel_vertex(0, 1).n_vertices == 5


def test_chisel():
    """
    Test that the chisel method works correctly
    """
    with pytest.raises(ValueError):
        assert (Polytope.cube(2) * Rational(1, 2)).chisel(1)

    with pytest.raises(ValueError):
        assert Polytope.cube(2).chisel(-1)

    with pytest.raises(ValueError):
        assert (Polytope.cube(2) * 3).chisel(2)

    assert Polytope.cube(2).chisel(0).vertices == Polytope.cube(2).vertices
    assert (Polytope.cube(2) * 2).chisel(1).n_vertices == 4
    assert (Polytope.cube(2) * 3).chisel(1).n_vertices == 8


def test_intersect_with_affine_subspace():
    p = Polytope.cube(2) * 2 - Point([1, 1])
    pts = PointConfiguration([[1, 0], [0, 1]])
    q = p.intersect_with_affine_subspace(pts)
    assert _arrays_equal_up_to_row_permutation(q.vertices, pts)

    p = Polytope.cube(3) * 2 - Point([1, 1, 1])
    pts = PointConfiguration([[0, 0, 0], [0, 1, 1]])
    q = p.intersect_with_affine_subspace(pts)
    expected = PointConfiguration([[0, 1, 1], [0, -1, -1]])
    assert _arrays_equal_up_to_row_permutation(q.vertices, expected)

    p = Polytope.cube(3) * 2 - Point([1, 1, 1])
    pts = PointConfiguration([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    q = p.intersect_with_affine_subspace(pts)
    expected = PointConfiguration([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])
    assert _arrays_equal_up_to_row_permutation(q.vertices, expected)


def test_unimodular_simplex():
    """
    Test that the unimodular simplex is correctly constructed
    """
    with pytest.raises(ValueError):
        assert Polytope.unimodular_simplex(-1)

    simplex = Polytope.unimodular_simplex(3)

    expected_vertices = PointConfiguration(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    )

    assert simplex.vertices == expected_vertices
    assert simplex.volume == Rational(1, 6)
    assert simplex.normalized_volume == 1


def test_cube():
    """
    Test that the cube is correctly constructed
    """
    with pytest.raises(ValueError):
        assert Polytope.cube(-1)

    cube = Polytope.cube(3)

    expected_vertices = PointConfiguration(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ]
    )

    assert _arrays_equal_up_to_row_permutation(cube.vertices, expected_vertices)
    assert cube.volume == 1
    assert cube.normalized_volume == 6


def test_cross_polytope():
    """
    Test that the cross polytope is correctly constructed
    """
    with pytest.raises(ValueError):
        assert Polytope.cross_polytope(-1)

    cross = Polytope.cross_polytope(3)

    expected_vertices = PointConfiguration(
        [
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
        ]
    )

    assert _arrays_equal_up_to_row_permutation(cross.vertices, expected_vertices)
    assert cross.volume == Rational(4, 3)
    assert cross.normalized_volume == 8


def test_reeve_simplex():
    """
    Test that the reeve simplex is correctly constructed.
    """
    with pytest.raises(ValueError):
        assert Polytope.reeve_simplex(0, 3)

    with pytest.raises(ValueError):
        assert Polytope.reeve_simplex(3, 0)

    s = Polytope.reeve_simplex(3, 3)

    expected_vertices = PointConfiguration(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 2, 3],
        ]
    )

    assert _arrays_equal_up_to_row_permutation(s.vertices, expected_vertices)
    assert isinstance(s, Simplex)

    s = Polytope.reeve_simplex(4, 3)

    expected_vertices = PointConfiguration(
        [
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [1, 2, 3, 0],
            [0, 0, 0, 1],
        ]
    )

    assert _arrays_equal_up_to_row_permutation(s.vertices, expected_vertices)
    assert isinstance(s, Polytope)


def test_gentleman_reeve_polytope():
    """
    Test that the gentleman reeve polytope is correctly constructed.
    """

    with pytest.raises(ValueError):
        assert Polytope.gentleman_reeve_polytope(0, 3)

    with pytest.raises(ValueError):
        assert Polytope.gentleman_reeve_polytope(3, 0)

    p = Polytope.gentleman_reeve_polytope(3, 3)

    expected_vertices = PointConfiguration(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 2, 3],
            [0, -1, -1],
        ]
    )

    assert _arrays_equal_up_to_row_permutation(p.vertices, expected_vertices)

    p = Polytope.gentleman_reeve_polytope(4, 3)

    expected_vertices = PointConfiguration(
        [
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [1, 2, 3, 0],
            [0, -1, -1, 0],
            [0, 0, 0, 1],
        ]
    )

    assert _arrays_equal_up_to_row_permutation(p.vertices, expected_vertices)


def test_higashitample_polytope():
    """
    Test that the higashitample polytope is correctly constructed.
    """
    with pytest.raises(ValueError):
        assert Polytope.higashitample_polytope(0, 1)

    with pytest.raises(ValueError):
        assert Polytope.higashitample_polytope(3, 0)

    p = Polytope.higashitample_polytope(3, 3)
    assert p.is_very_ample
    assert not p.is_idp


def test_set_cdd_polyhedron_from_points():
    """
    Test that _cdd_polyhedron is initialized from a list of points
    """
    points = [
        [0, 0, 0],
        [Rational(1, 7), 0, 0],
        [0, Rational(1, 7), 0],
        [0, 0, Rational(1, 7)],
    ]
    p = Polytope(points)
    p._set_cdd_polyhedron_from_points()
    assert p._cdd_polyhedron is not None


def test_set_ineqs_and_eqs():
    """
    Test that the setter for inequalities and equalities
    """
    verts = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]
    p = Polytope(verts)

    ineqs = np.array([[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, -1, 0], [1, -1, 0, 0]])
    eqs = np.array([[0, 0, 0, 1]])

    assert _arrays_equal_up_to_row_permutation(p.inequalities, ineqs)
    assert _arrays_equal_up_to_row_permutation(p.equalities, eqs)


def test_lower_dimensional_polytope():
    """
    Test that a lower dimensional polytope is correctly constructed
    """
    verts = PointConfiguration(
        [
            [0, 0, 0],
            [1, 1, 0],
        ]
    )

    p = Polytope(verts)
    assert p.n_equalities == 2


def test_simplex_conversion():
    """
    Test that the polytope is correctly converted to a simplex
    """
    simplex = Polytope([[0, 0], [1, 0], [0, 1]])

    # On initialization no checks are done
    assert simplex.__class__ == Polytope

    # Calculating vertices should convert to a simplex
    simplex.vertices
    assert simplex.__class__ == Simplex

    with pytest.raises(ValueError):
        Polytope.cube(2)._make_simplex()


def test_barycentric_coordinates():
    """
    Test that barycentric coordinates are correctly calculated
    """
    simplex = Polytope([[-1, -1], [2, -1], [-1, 2]])
    simplex._make_simplex()

    assert simplex.barycentric_coordinates(Point([0, 0])) == [
        Rational(1, 3),
        Rational(1, 3),
        Rational(1, 3),
    ]
    assert simplex.barycentric_coordinates(Point([-1, 0])) == [
        Rational(2, 3),
        0,
        Rational(1, 3),
    ]


def test_opposite_vertex():
    """
    Test that the opposite vertex is correctly calculated
    """
    s = Simplex.unimodular_simplex(3)

    for facet_id, facet in enumerate(s.facets):
        assert s.opposite_vertex(facet_id) not in facet


def test_weights():
    """
    Test that weights are correctly calculated
    """
    simplex = Polytope([[-1, -1], [1, 0], [0, 1]])
    simplex._make_simplex()

    assert simplex.weights == [1, 1, 1]
