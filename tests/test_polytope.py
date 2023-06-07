from unittest.mock import MagicMock
from sympy import Matrix, Rational

from sympol.point import Point
from sympol.point_list import PointList
from sympol.polytope import Polytope


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
    verts = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    mid = Point([1, 1, 1]) / 4
    points = verts + [mid]
    polytope = Polytope(points)

    assert polytope.vertices == PointList(verts)


def test_boundary_triangulation():
    """
    Test that the boundary triangulation has the expected shape
    """
    polytope = Polytope.unimodular_simplex(3)

    # make into a set to ignore order
    expected = {
        frozenset((0, 1, 2)),
        frozenset((0, 1, 3)),
        frozenset((0, 2, 3)),
        frozenset((1, 2, 3)),
    }
    result = {frozenset(simplex_ids) for simplex_ids in polytope.boundary_triangulation}

    assert result == expected


def test_get_volume():
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


def test_facets():
    """
    Test that the facets of a cube are six and are
    at distance one from the origin
    """
    polytope = Polytope.cube(3) * 2 - Point([1, 1, 1])

    assert len(polytope.facets) == 6

    for facet in polytope.facets:
        assert len(facet) == 4


def test_vertex_facet_matrix():
    """
    Test calculation of the vertex facet matrix of a polytope
    """
    polytope = Polytope.unimodular_simplex(dim=2)

    expected_matrix = Matrix([[1, 1, 0], [1, 0, 1], [0, 1, 1]])
    assert polytope.vertex_facet_matrix == expected_matrix


def test_inner_normal_to_facet():
    """
    Test that the inner normal to a facet is correct
    """
    points = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    polytope = Polytope(points)

    facet_ids = [1, 2, 3]
    normal = polytope._inner_normal_to_facet(facet_ids)

    assert normal == Point([-1, -1, -1])


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
            [0, 0, 1, 2, 0, 2, 1],
            [1, 2, 0, 0, 3, 0, 2],
            [0, 2, 0, 1, 3, 0, 2],
            [1, 0, 0, 0, 1, 2, 2],
            [0, 0, 0, 1, 1, 2, 2],
            [3, 2, 2, 0, 1, 0, 0],
            [3, 1, 2, 0, 0, 1, 0],
            [0, 2, 2, 3, 1, 0, 0],
            [0, 1, 2, 3, 0, 1, 0],
        ]
    )
    assert polytope.vertex_facet_pairing_matrix == expected


def test_vertex_facet_pairing_matrix_is_nonnegative():
    """
    Test that the vertex facet pairing matrix is nonnegative
    """
    polytope = Polytope.random_lattice_polytope(dim=4, n_vertices=50, min=-5, max=5)
    assert min(polytope.vertex_facet_pairing_matrix) >= 0


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
    cube_1 = Polytope.cube(3) * 3

    unimodular_map = Matrix(
        [
            [4, -1, 0],
            [-7, 2, 0],
            [-2, -1, 1],
        ]
    )

    cube_2 = Polytope(Matrix(cube_1.vertices) * unimodular_map) - Point([5, -1, 3])

    assert cube_1.affine_normal_form == cube_2.affine_normal_form


def test_affine_normal_form_idempotent():
    """
    Test that the affine normal form is idempotent
    """
    polytope = Polytope.random_lattice_polytope(dim=4, n_vertices=10, min=-2, max=2)
    polytope_anf = Polytope(vertices=polytope.affine_normal_form)
    assert polytope.affine_normal_form == polytope_anf.affine_normal_form


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
