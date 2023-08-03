import numpy as np
cimport numpy as cnp

cnp.import_array()

ctypedef cnp.int64_t DTYPE_t

def _find_integer_points(
        cnp.ndarray[DTYPE_t, ndim=2] verts,
        cnp.ndarray ineqs,
        int dim,
        bint count_only=False,
        int stop_at = -1,
        int stop_at_interior = -1
    ):
    """
    Find all integer points in a polytope. returns a list of pairs (point, facet_ids)
    where facet_ids is a list of the indices of the facets containing the point. Adapted
    from the SageMath implementation by Volker Braun.
    NOTE: This function is slow if compared with Barvinok's algorithm but I couldn't
    find a better implementation available in Python.
    """

    # Find bounding box and sort by decreasing dimension size
    cdef cnp.ndarray[DTYPE_t, ndim=1] box_min = np.array([min(verts[:, i]) for i in range(dim)])
    cdef cnp.ndarray[DTYPE_t, ndim=1] box_max = np.array([max(verts[:, i]) for i in range(dim)])
    cdef cnp.ndarray[DTYPE_t, ndim=1] box_diff = box_max - box_min
    cdef cnp.ndarray[DTYPE_t, ndim=1] diameter_index = np.argsort(box_diff)[::-1]
    # Construct the inverse permutation
    cdef dict orig_dict = {j: i for i, j in enumerate(diameter_index)}
    cdef list orig_perm = [orig_dict[i] for i in range(dim)]
    # Sort box bounds
    box_min = box_min[diameter_index]
    box_max = box_max[diameter_index]
    # Inequalities must also have their coordinates permuted
    ineqs[:, 1:] = ineqs[:, diameter_index + 1]
    # Find all lattice points and apply the inverse permutation
    cdef cnp.int64_t num_points = 0
    cdef cnp.int64_t num_interior_points = 0
    cdef list boundary_points = []
    cdef list interior_points = []
    cdef list saturated_facets = []
    cdef cnp.ndarray[DTYPE_t, ndim=1] p = np.array(box_min)
    cdef cnp.ndarray tmp_v
    cdef DTYPE_t i_min, i_max, i
    cdef int inc
    cdef bint forced_stop = False
    while True:
        tmp_v = ineqs[:, 2:].dot(p[1:]) + ineqs[:, 0] if dim > 1 else ineqs[:, 0]
        i_min = box_min[0]
        i_max = box_max[0]
        ii_min = i_min
        ii_max = i_max
        # Find the lower bound for the allowed region
        while i_min <= i_max:
            if all(i_min * ineqs[i, 1] + tmp_v[i] >= 0 for i in range(len(tmp_v))):
                # i_min is in the polytope
                if any(i_min * ineqs[i, 1] + tmp_v[i] == 0 for i in range(len(tmp_v))):
                    # i_min is a boundary point, we need to go further to find interior p
                    ii_min = i_min + 1
                    if any(ii_min * ineqs[i, 1] + tmp_v[i] == 0 for i in range(len(tmp_v))):
                        # ii_min is also a boundary point, this line will not intersect
                        # the interior of the polytope
                        ii_min = box_max[0] + 1
                break
            i_min += 1
            ii_min += 1

        # Find the upper bound for the allowed region
        while i_min <= i_max:
            if all(i_max * ineqs[i, 1] + tmp_v[i] >= 0 for i in range(len(tmp_v))):
                # i_max is in the polytope
                if any(i_max * ineqs[i, 1] + tmp_v[i] == 0 for i in range(len(tmp_v))):
                    # i_max is a boundary point, we need to go further to find interior p
                    ii_max = i_max - 1
                    if any(ii_max * ineqs[i, 1] + tmp_v[i] == 0 for i in range(len(tmp_v))):
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
                    interior_points.append(np.array(p)[orig_perm])
                else:
                    boundary_points.append(np.array(p)[orig_perm])
                    saturated = frozenset(
                        j for j in range(len(tmp_v)) if i * ineqs[j, 1] + tmp_v[j] == 0
                    )
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
        return None, None, None, num_points, num_interior_points, forced_stop

    # The points and saturated inequalities have now been computed.
    cdef cnp.ndarray[DTYPE_t, ndim=2] interior_points_arr = np.array(interior_points, dtype=np.int64, ndmin=2)
    cdef cnp.ndarray[DTYPE_t, ndim=2] boundary_points_arr = np.array(boundary_points, dtype=np.int64, ndmin=2)

    return interior_points_arr, boundary_points_arr, saturated_facets, num_points, num_interior_points, forced_stop
