from unittest.mock import MagicMock
from sympy import Rational

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
    points = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    polytope = Polytope(points)

    assert len(polytope.boundary_triangulation) == 4
    assert len(polytope.boundary_triangulation[0]) == 3
    assert len(polytope.boundary_triangulation[0][0]) == 3


def test_get_volume():
    """
    Test that the volume is correct
    """
    points = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    polytope = Polytope(points)

    assert polytope._volume is None
    assert polytope._normalized_volume is None
    assert polytope.volume == 1  # This triggers a volume computation
    assert polytope._volume == 1
    assert polytope._normalized_volume == 6
    assert polytope.normalized_volume == 6


def test_volume_computed_only_once():
    """
    Test that the volume is computed only once
    """
    points = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    polytope = Polytope(points)

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
    points = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    polytope = Polytope(points)

    assert polytope.barycenter == Point(
        [Rational(1, 4), Rational(1, 4), Rational(1, 4)]
    )


def test_hyperplanes():
    """
    Test that the hyperplanes are correct
    """
    points = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    polytope = Polytope(points)

    assert len(polytope.hyperplanes) == 3
    assert polytope.hyperplanes[0].normal_vector == [1, 0, 0]
    assert polytope.hyperplanes[1].normal_vector == [0, 1, 0]
    assert polytope.hyperplanes[2].normal_vector == [0, 0, 1]


def test_inner_normal_to_facet():
    """
    Test that the inner normal to a facet is correct
    """
    points = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    polytope = Polytope(points)

    facet = [Point(v) for v in points[:-1]]
    normal = polytope._inner_normal_to_facet(facet)

    assert normal == Point([0, 0, 1])
