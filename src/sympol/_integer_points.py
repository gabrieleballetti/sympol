import numpy as np
import numba as nb


def _find_integer_points(
    verts,
    ineqs,
    dim,
    count_only=False,
    stop_at=-1,
    stop_at_interior=-1,
    disable_numba=False,
):
    """Wrapper for the numba implementation of the integer points enumeration.

    The idea is to allow support to run the wrapped function with pure Python
    to avoid overflows in the numba implementation. Not yet implemented.
    """
    if not disable_numba:
        verts = np.array(verts, dtype=np.int64)
        ineqs = np.array(ineqs, dtype=np.int64)
        func = _find_integer_points_numba
    else:
        verts = np.array(verts)
        ineqs = np.array(ineqs)
        func = _find_integer_points_numba.py_func

    return func(
        verts=verts,
        ineqs=ineqs,
        dim=dim,
        count_only=count_only,
        stop_at=stop_at,
        stop_at_interior=stop_at_interior,
    )


@nb.njit(
    nb.types.Tuple(
        (
            nb.int64[:, :],
            nb.int64[:, :],
            nb.boolean[:, :],
            nb.int64,
            nb.int64,
            nb.boolean,
        )
    )(
        nb.int64[:, :],
        nb.int64[:, :],
        nb.int64,
        nb.boolean,
        nb.int64,
        nb.int64,
    ),
    cache=True,
)
def _find_integer_points_numba(
    verts,
    ineqs,
    dim,
    count_only,
    stop_at,
    stop_at_interior,
):
    """
    Find all integer points in a polytope. returns a list of pairs (point, facet_ids)
    where facet_ids is a list of the indices of the facets containing the point. Adapted
    from the SageMath implementation by Volker Braun.
    NOTE: This function is slow if compared with Barvinok's algorithm but I couldn't
    find a better implementation available in Python.
    """
    dtype = verts.dtype
    # Find bounding box and sort by decreasing dimension size
    box_min = np.array([np.min(verts[:, i]) for i in range(dim)])
    box_max = np.array([np.max(verts[:, i]) for i in range(dim)])
    box_diff = box_max - box_min
    diameter_index = np.argsort(box_diff)[::-1]
    # Construct the inverse permutation
    orig_dict = {j: i for i, j in enumerate(diameter_index)}
    orig_perm = [orig_dict[i] for i in range(dim)]
    # Sort box bounds
    box_min = box_min[diameter_index]
    box_max = box_max[diameter_index]
    # Inequalities must also have their coordinates permuted
    ineqs[:, 1:] = ineqs[:, diameter_index + 1]
    # Find all lattice points and apply the inverse permutation
    num_points = 0
    num_interior_points = 0
    boundary_points = []
    interior_points = []
    # make an empty list of list of integers
    saturated_facets = []
    p = box_min.copy()
    forced_stop = False

    while True:
        if dim > 1:
            tmp_v = np.zeros_like(ineqs[:, 0])
            for i in range(ineqs.shape[0]):
                for j in range(ineqs.shape[1] - 2):
                    tmp_v[i] += ineqs[i, j + 2] * p[j + 1]
            tmp_v += ineqs[:, 0]
        else:
            tmp_v = ineqs[:, 0]
        i_min = box_min[0]
        i_max = box_max[0]
        ii_min = i_min
        ii_max = i_max
        # Find the lower bound for the allowed region
        while i_min <= i_max:
            flag = True
            for i in range(len(tmp_v)):
                if not i_min * ineqs[i, 1] + tmp_v[i] >= 0:
                    flag = False
                    break
            if flag:
                # i_min is in the polytope
                any_flag = False
                for i in range(len(tmp_v)):
                    if i_min * ineqs[i, 1] + tmp_v[i] == 0:
                        any_flag = True
                        break
                if any_flag:
                    # i_min is a boundary point, we need to go further to find interior p
                    ii_min = i_min + 1
                    any_flag_2 = False
                    for i in range(len(tmp_v)):
                        if ii_min * ineqs[i, 1] + tmp_v[i] == 0:
                            any_flag_2 = True
                            break
                    if any_flag_2:
                        # ii_min is also a boundary point, this line will not intersect
                        # the interior of the polytope
                        ii_min = box_max[0] + 1
                    break
            i_min += 1
            ii_min += 1

        # Find the upper bound for the allowed region
        while i_min <= i_max:
            flag = True
            for i in range(len(tmp_v)):
                if not i_max * ineqs[i, 1] + tmp_v[i] >= 0:
                    flag = False
                    break
            if flag:
                # i_max is in the polytope
                any_flag = False
                for i in range(len(tmp_v)):
                    if i_max * ineqs[i, 1] + tmp_v[i] == 0:
                        any_flag = True
                        break
                if any_flag:
                    # i_max is a boundary point, we need to go further to find interior p
                    ii_max = i_max - 1
                    any_flag_2 = False
                    for i in range(len(tmp_v)):
                        if ii_max * ineqs[i, 1] + tmp_v[i] == 0:
                            any_flag_2 = True
                            break
                    if any_flag_2:
                        # ii_max is also a boundary point, this line will not intersect
                        # the interior of the polytope
                        ii_max = box_min[0] - 1
                break
            i_max -= 1
            ii_max -= 1

        # The points i_min .. i_max are contained in the polytope
        num_points += i_max - i_min + 1
        if ii_min <= ii_max:
            num_interior_points += ii_max - ii_min + 1

        if not count_only:
            i = i_min
            while i <= i_max:
                p[0] = i
                if i >= ii_min and i <= ii_max:
                    # this are interior points, so they will not saturate any inequality
                    interior_point = np.empty_like(p)
                    for idx, value in enumerate(orig_perm):
                        interior_point[idx] = p[value]
                    interior_points.append(interior_point)
                else:
                    boundary_point = np.empty_like(p)
                    for idx, value in enumerate(orig_perm):
                        boundary_point[idx] = p[value]
                    boundary_points.append(boundary_point)
                    saturated = [
                        j for j in range(len(tmp_v)) if i * ineqs[j, 1] + tmp_v[j] == 0
                    ]
                    saturated_facets.append(saturated)
                i += 1

        # Break if we have reached the required number of points
        if stop_at > 0 and num_points >= stop_at:
            forced_stop = True
            break
        if stop_at_interior > 0 and num_interior_points >= stop_at_interior:
            forced_stop = True
            break

        # Increment the other entries in p to move on to next loop
        inc = 1
        if dim == 1:
            break
        break_loop = False
        while True:
            if p[inc] == box_max[inc]:
                p[inc] = box_min[inc]
                inc += 1
                if inc == dim:
                    break_loop = True
                    break
            else:
                p[inc] += 1
                break
        if break_loop:
            break

    if count_only:
        interior_points_arr = np.empty((0, dim), dtype=dtype)
        boundary_points_arr = np.empty((0, dim), dtype=dtype)
        saturated_facets_arr = np.empty((0, ineqs.shape[0]), dtype=np.bool_)

        return (
            interior_points_arr,
            boundary_points_arr,
            saturated_facets_arr,
            num_points,
            num_interior_points,
            forced_stop,
        )

    interior_points_arr = np.empty((num_interior_points, dim), dtype=dtype)
    boundary_points_arr = np.empty((num_points - num_interior_points, dim), dtype=dtype)
    saturated_facets_arr = np.zeros(
        (len(saturated_facets), ineqs.shape[0]), dtype=np.bool_
    )

    for i, point in enumerate(interior_points):
        interior_points_arr[i] = point

    for i, point in enumerate(boundary_points):
        boundary_points_arr[i] = point

    for i, facet in enumerate(saturated_facets):
        for j in facet:
            saturated_facets_arr[i, j] = True

    return (
        interior_points_arr,
        boundary_points_arr,
        saturated_facets_arr,
        num_points,
        num_interior_points,
        forced_stop,
    )
