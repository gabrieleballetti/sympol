import numpy as np
from sympy import Rational
import cdd

MAX_INT = np.iinfo(np.int64).max
MIN_INT = np.iinfo(np.int64).min


def _get_upper_or_lower_hull_triangulation(points: np.ndarray, affine_rank: int):
    """Get a triangulation from an upper hull of the PointConfiguration.

    Args:
        points: A point configuration as a numpy array.
        affine_rank: The affine rank of the point configuration. This will be the
            dimension of the simplices in the triangulation.

    Returns:
        A triangulation of the upper hull of the PointConfiguration.
    """

    # simplex case needs to be handled separately
    if points.shape[0] == affine_rank + 1:
        return tuple([frozenset(range(points.shape[0]))])

    while True:
        triangulation = _get_random_upper_or_lower_hull_triangulation(points)
        if all(len(simplex) == affine_rank + 1 for simplex in triangulation):
            return triangulation


def _get_random_upper_or_lower_hull_triangulation(points: np.ndarray):
    # generate random "heights" for the points
    nums = np.random.randint(MIN_INT, MAX_INT, size=points.shape[0], dtype=np.int64)
    dens = np.random.randint(1, 2, size=points.shape[0], dtype=np.int64)
    heights = np.array([Rational(p, q) for p, q in zip(nums, dens)])

    # append the heights to the points
    points = np.hstack((points, heights.reshape(-1, 1)))

    # use cdd to calculate the (d+1)-dimensional polytope and its inequalities
    points = np.hstack((np.ones(shape=(points.shape[0], 1), dtype=int), points))
    cdd_mat = cdd.Matrix(points, number_type="fraction")
    cdd_mat.rep_type = cdd.RepType.GENERATOR
    polyhedron = cdd.Polyhedron(cdd_mat)
    ineqs = polyhedron.get_inequalities()
    eqs = ineqs.lin_set

    # find the upper and lower hulls
    upper = []
    lower = []
    for i, ineq in enumerate(ineqs):
        if i in eqs:
            continue
        if ineq[-1] > 0:
            upper.append(i)
        elif ineq[-1] < 0:
            lower.append(i)

    # pick one of the hulls to use for the triangulation
    hull_to_use = upper if len(upper) <= len(lower) else lower
    facet_vertex_incidence = polyhedron.get_incidence()
    triangulation = tuple(
        [facet_vertex_incidence[facet_id] for facet_id in hull_to_use]
    )

    return triangulation
