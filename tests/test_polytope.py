import numpy as np
from sympy import Array
from sympy.geometry.point import Point
from poly.polytope import Polytope


def test_random_init():
    """
    Test initialization of a random polytope with integer coordinates
    """
    points = [Point(p) for p in np.random.randint(0, 100, size=(10, 5))]
    polytope = Polytope(points)
    assert polytope.points == points
    assert polytope.ambient_dim == 5
    assert polytope.dim == 5
    assert polytope.is_full_dim


def test_vertices():
    """
    Test that the vertices are correct
    """
    verts = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    mid = Array([1, 1, 1]) / 4
    points = Array(verts + [mid])
    polytope = Polytope(points)
    assert polytope.vertices == Array(verts)


def test_boundary_triangulation():
    """
    Test that the boundary triangulation has the expected shape
    """
    points = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    polytope = Polytope(points)
    assert polytope.boundary_triangulation.shape == (4, 3, 3)


def test_get_volume():
    """
    Test that the volume is correct
    """
    points = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    polytope = Polytope(points)
    assert polytope.get_volume() == 1
    assert polytope.get_normalized_volume() == 6
