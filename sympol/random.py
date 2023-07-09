import numpy as np

from sympol.polytope import Polytope


def sample_polytope_from_normal_distribution(dim, n_vertices, std_dev=1):
    """ """
    verts = np.random.default_rng().normal(loc=0, scale=std_dev, size=(n_vertices, dim))
    verts = verts.round().astype(int)
    return Polytope(verts)
