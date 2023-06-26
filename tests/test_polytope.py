from unittest.mock import MagicMock
from sympy import Matrix, Pow, Rational

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


def test_triangulation_lower_dimensional_polytope():
    """
    Test that the triangulation is correct
    """
    p = Polytope([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])

    assert p.triangulation == (frozenset({0, 1, 3}), frozenset({0, 2, 3}))


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


def test_contains_point():
    """
    Test that the contains method works correctly for a point
    """
    polytope = Polytope.cube(3) * 2 - Point([1, 1, 1])

    assert polytope.contains(Point([0, 0, 0]))
    assert polytope.contains(Point([0, 0, 1]))
    assert polytope.contains(Point([0, 1, 1]))
    assert polytope.contains(Point([1, 1, 1]))
    assert not polytope.contains(Point([0, 0, 2]))


def test_lower_dim_polytope_contains_point():
    """
    Test that points are correctly contained (or not) in lower dimensional polytopes
    """
    p = Polytope(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
        ]
    )

    assert p.contains(Point([0, 0, 0]))
    assert not p.contains(Point([0, 0, 1]))


def test_contains_polytope():
    """
    Test that the contains method works correctly for a polytope
    """
    polytope = Polytope.cube(3) * 2 - Point([1, 1, 1])

    assert polytope.contains(polytope)
    assert polytope.contains(Polytope.cube(3))

    # check that vertices are not used if not available
    pts = PointList([[Pow(a, 1), Pow(a, 2), Pow(a, 3)] for a in range(100)])
    polytope_2 = Polytope(pts)
    assert not polytope.contains(polytope_2)
    assert polytope_2._vertices is None


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


def test_is_lattice_polytope():
    """
    Test the is_lattice_polytope method
    """
    polytope = Polytope.unimodular_simplex(dim=3)
    assert polytope.is_lattice_polytope()

    polytope = polytope * Rational(3, 2)
    assert not polytope.is_lattice_polytope()


def test_integer_points():
    """
    Test that the integer points of a polytope are correctly calculated
    """
    s = Polytope.unimodular_simplex(3)
    c = Polytope.cube(3)

    assert s.n_integer_points == s.n_vertices
    assert c.n_integer_points == c.n_vertices


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

    assert cube.vertices == expected_vertices
    assert cube.volume == 1
    assert cube.normalized_volume == 6


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
