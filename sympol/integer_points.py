import numpy as np


def _find_integer_points(polytope):
    """
    Find all integer points in a polytope. returns a list of pairs (point, facet_ids)
    where facet_ids is a list of the indices of the facets containing the point. Adapted
    from https://github.com/LiamMcAllisterGroup/cytools/blob/main/cytools/polytope.py
    (GPLv3) which in turn is adapted from a SageMath implementation by Volker Braun.
    NOTE: This function is slow and not optimized, but I couldn't find a better
    implementation available in Python. I have plans to implement a faster version
    in Cython, but for now this will do.
    """
    if not polytope.is_full_dim():
        raise ValueError("polytope must be full-dimensional")

    # Find bounding box and sort by decreasing dimension size
    verts = np.array(polytope.vertices)
    box_min = np.array([min(verts[:, i]) for i in range(polytope.dim)])
    box_max = np.array([max(verts[:, i]) for i in range(polytope.dim)])
    box_diff = box_max - box_min
    diameter_index = np.argsort(box_diff)[::-1]
    # Construct the inverse permutation
    orig_dict = {j: i for i, j in enumerate(diameter_index)}
    orig_perm = [orig_dict[i] for i in range(polytope.dim)]
    # Sort box bounds
    box_min = box_min[diameter_index]
    box_max = box_max[diameter_index]
    # Inequalities must also have their coordinates permuted
    ineqs = np.array(
        [ineq.normal.tolist() + [-ineq.rhs] for ineq in polytope.linear_inequalities]
    )
    ineqs[:, :-1] = ineqs[:, diameter_index]
    # Find all lattice points and apply the inverse permutation
    points = []
    facet_ind = []
    p = np.array(box_min)
    while True:
        tmp_v = (
            ineqs[:, 1:-1].dot(p[1:]) + ineqs[:, -1]
            if polytope.dim > 1
            else ineqs[:, -1]
        )
        i_min = box_min[0]
        i_max = box_max[0]
        # Find the lower bound for the allowed region
        while i_min <= i_max:
            if all(i_min * ineqs[i, 0] + tmp_v[i] >= 0 for i in range(len(tmp_v))):
                break
            i_min += 1
        # Find the upper bound for the allowed region
        while i_min <= i_max:
            if all(i_max * ineqs[i, 0] + tmp_v[i] >= 0 for i in range(len(tmp_v))):
                break
            i_max -= 1
        # The points i_min .. i_max are contained in the polytope
        i = i_min
        while i <= i_max:
            p[0] = i
            saturated = frozenset(
                j for j in range(len(tmp_v)) if i * ineqs[j, 0] + tmp_v[j] == 0
            )
            points.append(np.array(p)[orig_perm])
            facet_ind.append(saturated)
            i += 1
        # Increment the other entries in p to move on to next loop
        inc = 1
        if polytope.dim == 1:
            break
        break_loop = False
        while True:
            if p[inc] == box_max[inc]:
                p[inc] = box_min[inc]
                inc += 1
                if inc == polytope.dim:
                    break_loop = True
                    break
            else:
                p[inc] += 1
                break
        if break_loop:
            break
    # The points and saturated inequalities have now been computed.
    points_mat = np.array(points, dtype=int)
    # points_mat = self._inv_transf_matrix.dot(points_mat.T).T

    # Organize the points as explained above.
    points = sorted(
        [(tuple(points_mat[i]), facet_ind[i]) for i in range(len(points))],
        key=(lambda p: (-(len(p[1]) if len(p[1]) > 0 else 1e9),) + tuple(p[0])),
    )

    return points
