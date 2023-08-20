import numpy as np
from sympol import Polytope


def sample_polytope_from_normal_distribution(dim, n_points, std_dev=1):
    """
    Return a random polytope as the convex hull of a set of points sampled from a
    normal distribution.
    """
    if dim < 1:
        raise ValueError("Dimension must be at least 1.")
    if n_points < dim + 1:
        raise ValueError("Number of points must be at least dim + 1.")

    d = -1
    while d != dim:
        verts = np.random.default_rng().normal(
            loc=0, scale=std_dev, size=(n_points, dim)
        )
        verts = verts.round().astype(int)
        p = Polytope(verts)
        d = p.dim
    return Polytope(verts)


if __name__ == "__main__":
    dim = 6
    extra_pts = 0
    while True:
        p = sample_polytope_from_normal_distribution(
            dim, n_points=dim + 1 + extra_pts, std_dev=0.5
        )
        if p.dim < dim:
            continue

        if p.is_lattice_pyramid:
            continue

        print(p.h_star_vector)

        if not p.is_idp:
            continue

        # if not p.has_log_concave_h_star_vector:
        #     print(p.vertices)
