from sympy import Matrix
from sympol.point_list import PointList


def get_placing_triangulation(points: PointList):
    """
    Find a triangulation of a point list
    """
    # pick a large initial simplex
