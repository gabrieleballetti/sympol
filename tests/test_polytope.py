import pytest
from unittest.mock import MagicMock
from sympy import Matrix, Poly, Rational
from sympy.abc import x

from sympol.point import Point
from sympol.point_list import PointList
from sympol.polytope import Polytope, Simplex


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

    assert polytope.points == PointList(points)
    assert polytope.ambient_dim == 3

    # check correct dimension initialization
    assert polytope._dim is None
    assert polytope.dim == 3
    assert polytope._dim == 3

    # check correct vertices initialization
    assert polytope._vertices is None
    assert polytope.vertices == PointList(points[:-1])
    assert polytope._vertices == PointList(points[:-1])


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
    assert polytope._points == PointList(vertices)
    assert polytope.points == PointList(vertices)
    assert polytope.ambient_dim == 3

    # check correct dimension initialization
    assert polytope._dim is None
    assert polytope.dim == 3
    assert polytope._dim == 3

    # check correct vertices initialization
    assert polytope._vertices == PointList(vertices)
    assert polytope.vertices == PointList(vertices)


def test_init_from_points_and_vertices():
    """
    Test initialization of a random polytope from a list of vertices
    """
    points = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [Rational(1, 4), Rational(1, 4), Rational(1, 4)],
    ]
    vertices = points[:-1]
    polytope = Polytope(points, vertices=vertices)

    assert polytope.points == PointList(points)
    assert polytope.ambient_dim == 3

    # check correct dimension initialization
    assert polytope._dim is None
    assert polytope.dim == 3
    assert polytope._dim == 3

    # check correct vertices initialization
    assert polytope._vertices == PointList(vertices)


def test_vertices():
    """
    Test that the vertices are correct
    """
    verts = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, Rational(101, 100)]]
    mid = Point([1, 1, 1]) / 4
    points = verts + [mid]
    polytope = Polytope(points)

    assert polytope.vertices == PointList(verts)


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

    assert polytope.vertices == PointList(verts)


def test_triangulation():
    """
    Test that the triangulation is correct
    """
    p = Polytope.cube(2)

    assert p.triangulation == (frozenset({0, 1, 3}), frozenset({0, 2, 3}))


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
    p = Polytope([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])

    assert p.triangulation == (frozenset({0, 1, 3}), frozenset({0, 2, 3}))


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
    polytope = Polytope.cube(3)

    assert polytope.boundary_volume == 6
    assert polytope.normalized_boundary_volume == 12


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

    assert polytope.vertices == PointList(
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

    assert polytope.vertices == PointList(
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

    assert polytope.vertices == PointList(
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
    pts = PointList([[a**1, a**2, a**3] for a in range(100)])
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


def test_linear_inequalities():
    """
    Test that the linear_inequalities of a cube are six and are
    at distance one from the origin
    """
    polytope = Polytope.cube(3) * 2 - Point([1, 1, 1])

    assert len(polytope.linear_inequalities) == 6

    origin = Point([0, 0, 0])
    for lineq in polytope.linear_inequalities:
        assert lineq.evaluate(origin) == 1

    simplex = Polytope.unimodular_simplex(3)
    assert len(simplex.linear_inequalities) == 4
    for lineq in simplex.linear_inequalities:
        assert lineq.evaluate(origin) == 1 or lineq.evaluate(origin) == 0


def test_linear_inequalities_rational_coeffs():
    """
    Test that rational coefficients are correctly passed from cdd to sympy
    """
    p = Polytope.unimodular_simplex(dim=2) * Rational(1, 2)
    assert p.linear_inequalities[0].normal == Point([-1, -1])
    assert p.linear_inequalities[0].rhs == -Rational(1, 2)


def test_facets():
    """
    Test that the facets of a cube are six and are
    at distance one from the origin
    """
    p = Polytope.cube(3)

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
    assert p.faces(-1) == tuple([frozenset()])
    assert p.faces(0) == tuple([frozenset([i]) for i in range(p.n_vertices)])
    assert p.faces(7) == (frozenset(range(p.n_vertices)),)

    # test length for other dimensions
    assert len(p.faces(6)) == 8
    assert len(p.faces(5)) == 28
    assert len(p.faces(4)) == 56
    assert len(p.faces(3)) == 70
    assert len(p.faces(2)) == 56


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


def test_vertex_adjacency_matrix():
    """
    Test calculation of the vertex adjacency matrix of a polytope
    """
    p = Polytope.cube(3)

    expected_matrix = Matrix(
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

    assert p.vertex_adjacency_matrix.is_symmetric()
    assert p.vertex_adjacency_matrix == expected_matrix


def test_f_vector():
    """
    Test calculation of the f-vector for an hypercube
    """
    p = Polytope.cube(7)

    expected_f_vector = (1, 128, 448, 672, 560, 280, 84, 14, 1)
    assert p.f_vector == expected_f_vector


def test_vertex_facet_matrix():
    """
    Test calculation of the vertex facet matrix of a polytope
    """
    polytope = Polytope.unimodular_simplex(dim=2)

    expected_matrix = Matrix(
        [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ]
    )
    assert polytope.vertex_facet_matrix == expected_matrix


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

    expected = Matrix(
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
    assert polytope.vertex_facet_pairing_matrix == expected


def test_vertex_facet_matrix_low_dimensional_polytope():
    """
    Test calculation of the vertex facet matrix of a non-full-dimensional polytope
    """
    verts = [[0, 0, 1], [7, 4, 19], [5, 3, 14], [12, 7, 32]]
    square = Polytope(verts)

    expected_vfm = Matrix(
        [
            [1, 0, 1, 0],
            [1, 1, 0, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 1],
        ]
    )

    expected_vfpm = Matrix(
        [
            [0, 1, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 1, 0],
            [1, 1, 0, 0],
        ]
    )
    assert square.vertex_facet_matrix == expected_vfm
    assert square.vertex_facet_pairing_matrix == expected_vfpm


def test_vertex_facet_pairing_matrix_is_nonnegative():
    """
    Test that the vertex facet pairing matrix is nonnegative
    """
    p = Polytope.random_lattice_polytope(dim=4, n_vertices=10, min=-5, max=5)
    assert min(p.vertex_facet_pairing_matrix) >= 0


def test_vertex_facet_pairing_matrix_low_dimensional_polytope():
    """
    Test calculation of the vertex facet pairing matrix of a non-full-dimensional
    polytope
    TODO: Is this guaranteed to work or should we project to a full-dimensional
    polytope in a lower dimension?
    """
    verts = [[0, 0, 1], [1, 0, 6], [0, 1, 5], [1, 1, 10]]
    square = Polytope(verts)

    expected = Matrix(
        [
            [0, 0, 1, 1],
            [0, 1, 0, 1],
            [1, 1, 0, 0],
            [1, 0, 1, 0],
        ]
    )
    assert square.vertex_facet_pairing_matrix == expected


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
    cube_1 = Polytope.random_lattice_polytope(3, 8, -2, 2)

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
    polytope = Polytope.random_lattice_polytope(dim=4, n_vertices=10, min=-2, max=2)
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

    assert set([pt for pt in s.integer_points]) == set([pt for pt in s.vertices])
    assert s.n_interior_points == 0
    assert s.n_boundary_points == 4

    assert s2.n_integer_points == 10
    assert s2.n_interior_points == 0
    assert s2.n_boundary_points == 10

    assert s3.n_integer_points == 20
    assert s3.n_interior_points == 0
    assert s3.n_boundary_points == 20

    assert s4.n_integer_points == 35
    assert s4.n_interior_points == 1
    assert s4.n_boundary_points == 34
    assert s4.interior_points == PointList([[1, 1, 1]])

    assert set([pt for pt in c.integer_points]) == set([pt for pt in c.vertices])
    assert c.n_interior_points == 0
    assert c.n_boundary_points == 8

    assert c2.n_integer_points == 27
    assert c2.n_interior_points == 1
    assert c2.n_boundary_points == 26
    assert c2.interior_points == PointList([[1, 1, 1]])


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

    for pt, f_ids in zip(p.boundary_points, p.boundary_points_facets):
        for j, lineq in enumerate(p.linear_inequalities):
            if lineq.evaluate(pt) == 0:
                assert j in f_ids
            else:
                assert j not in f_ids


def test_ehrhart_polynomial():
    """
    Test that the Ehrhart polynomial of a lattice polytope is correct
    """
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
    for d in range(1, 5):
        assert Polytope.unimodular_simplex(d).h_star_polynomial == Poly(1, x)

    Polytope.cube(1).h_star_polynomial == Poly(1, x)
    Polytope.cube(2).h_star_polynomial == Poly(x + 1)
    Polytope.cube(3).h_star_polynomial == Poly(x**2 + 4 * x + 1)


def test_h_star_vector():
    """
    Test that the h*-vector of a lattice polytope is correct
    """
    assert Polytope.cube(3).h_star_vector == (1, 4, 1, 0)
    assert (Polytope.cube(3) * 2).h_star_vector == (1, 23, 23, 1)


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


def test_has_unimodal_h_star_vector():
    """
    Test the has_unimodal_h_star_vector property
    """
    s = Polytope.unimodular_simplex(3) * 4

    assert s.has_unimodal_h_star_vector

    p = Polytope([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 2]])

    assert not p.has_unimodal_h_star_vector


def test_is_idp():
    """
    Test the is_idp property
    """
    p = Polytope([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 2], [0, 0, -1]])
    assert p.is_idp

    p = Polytope([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 3], [0, 0, -1]])
    assert not p.is_idp


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
    assert set(Polytope.cube(1).free_sum(Polytope.cube(1)).vertices) == set(
        [Point((0, 0)), Point((0, 1)), Point((1, 0))]
    )

    with pytest.raises(TypeError):
        assert Polytope.cube(2).free_sum(1)


def test_chisel_vertex():
    """
    Test that the chisel_vertex method works correctly
    """
    with pytest.raises(ValueError):
        assert Polytope.cube(2).chisel(-1)

    c = Polytope.cube(2) * 2
    assert c.chisel_vertex(0, 0) == c
    assert c.chisel_vertex(0, 1).n_vertices == 5


def test_chisel():
    """
    Test that the chisel method works correctly
    """
    with pytest.raises(ValueError):
        assert Polytope.cube(2).chisel(-1)

    assert Polytope.cube(2).chisel(0).vertices == Polytope.cube(2).vertices
    assert (Polytope.cube(2) * 2).chisel(1).n_vertices == 4
    assert (Polytope.cube(2) * 3).chisel(1).n_vertices == 8

    with pytest.raises(ValueError):
        assert (Polytope.cube(2) * 3).chisel(2)


def test_unimodular_simplex():
    """
    Test that the unimodular simplex is correctly constructed
    """
    simplex = Polytope.unimodular_simplex(3)

    expected_vertices = PointList(
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
    cube = Polytope.cube(3)

    expected_vertices = PointList(
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

    assert set(cube.vertices) == set(expected_vertices)
    assert cube.volume == 1
    assert cube.normalized_volume == 6


def test_cross_polytope():
    """
    Test that the cross polytope is correctly constructed
    """
    cross = Polytope.cross_polytope(3)

    expected_vertices = PointList(
        [
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
        ]
    )

    assert set(cross.vertices) == set(expected_vertices)
    assert cross.volume == Rational(4, 3)
    assert cross.normalized_volume == 8


def test_get_cdd_polyhedron_from_points():
    """
    Test that _cdd_polyhedron is initialized from a list of points
    """
    points = [
        [0, 0, 0],
        [Rational(1, 7), 0, 0],
        [0, Rational(1, 7), 0],
        [0, 0, Rational(1, 7)],
    ]
    polytope = Polytope(points)
    polytope._get_cdd_polyhedron_from_points()
    assert polytope._cdd_polyhedron is not None


def test_lower_dimensional_polytope():
    """
    Test that the lower dimensional polytope is correctly constructed
    """
    verts = PointList(
        [
            [0, 0, 0],
            [1, 1, 0],
        ]
    )

    p = Polytope(verts)
    eqs = [lin_eq.is_equality for lin_eq in p.linear_inequalities]
    assert len(eqs) > 0


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


def test_weights():
    """
    Test that weights are correctly calculated
    """
    simplex = Polytope([[-1, -1], [1, 0], [0, 1]])
    simplex._make_simplex()

    assert simplex.weights == [1, 1, 1]
