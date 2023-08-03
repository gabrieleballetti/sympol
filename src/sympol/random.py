from enum import Enum
import numpy as np

from sympol.polytope import Polytope


class SubpolytopeStrategy(Enum):
    """
    Enum for the different strategies for sampling subpolytopes
    """

    VERTEX_SUBSET = 1
    POINTS_SUBSET = 2
    MINIMIZE_H1 = 3


def sample_polytope_from_normal_distribution(dim, n_points, std_dev=1):
    """
    Return a random polytope as the convex hull of a set of points sampled from a
    normal distribution
    """
    if dim < 1:
        raise ValueError("Dimension must be at least 1")
    if n_points < dim + 1:
        raise ValueError("Number of points must be at least dim + 1")

    d = -1
    while d != dim:
        verts = np.random.default_rng().normal(
            loc=0, scale=std_dev, size=(n_points, dim)
        )
        verts = verts.round().astype(int)
        p = Polytope(verts)
        d = p.dim
    return Polytope(verts)


def random_subpolytope(p, strategy):
    """
    Return a random subpolytope of p
    """
    if strategy == SubpolytopeStrategy.VERTEX_SUBSET:
        n = np.random.randint(p.dim + 1, p.n_vertices)
        verts = np.random.default_rng().choice(p.vertices, n)
        return Polytope(verts)
    elif strategy == SubpolytopeStrategy.POINTS_SUBSET:
        p.integer_points.shape[0]  # trigger enumeration
        n = np.random.randint(p.dim + 1, p.n_integer_points)
        pts = np.random.default_rng().choice(p.integer_points, n)
        return Polytope(pts)

    raise NotImplementedError("Not implemented yet")
