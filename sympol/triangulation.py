from sympy import Matrix
from sympol.point_list import PointList
from scipy.spatial import Delaunay


def delaunay_triangulation(points: PointList):
    """
    Return the Delaunay triangulation of the given points.
    It uses scipy.spatial.Delaunay which uses float64 precision.
    """
    points = points.to_numpy()
    delaunay_triangulation = Delaunay(points)
    return tuple([frozenset(s) for s in delaunay_triangulation.simplices])
