from sympy import Rational
from sympy.geometry.point import Point

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

    assert polytope.points == [Point(p) for p in points]
    assert polytope.ambient_dim == 3

    # check correct dimension initialization
    assert polytope._dim is None
    assert polytope.dim == 3
    assert polytope._dim == 3

    # check correct vertices initialization
    assert polytope._vertices is None
    assert polytope.vertices == [Point(p) for p in points[:-1]]
    assert polytope._vertices == [Point(p) for p in points[:-1]]


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
    assert polytope._points == [Point(p) for p in vertices]
    assert polytope.points == [Point(p) for p in vertices]
    assert polytope.ambient_dim == 3

    # check correct dimension initialization
    assert polytope._dim is None
    assert polytope.dim == 3
    assert polytope._dim == 3

    # check correct vertices initialization
    assert polytope._vertices == [Point(p) for p in vertices]
    assert polytope.vertices == [Point(p) for p in vertices]


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

    assert polytope.points == [Point(p) for p in points]
    assert polytope.ambient_dim == 3

    # check correct dimension initialization
    assert polytope._dim is None
    assert polytope.dim == 3
    assert polytope._dim == 3

    # check correct vertices initialization
    assert polytope._vertices == [Point(p) for p in vertices]


def test_vertices():
    """
    Test that the vertices are correct
    """
    verts = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    mid = Point([1, 1, 1]) / 4
    points = verts + [mid]
    polytope = Polytope(points)

    assert polytope.vertices == [Point(p) for p in verts]


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
