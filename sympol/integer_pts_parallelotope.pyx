import copy
import itertools
import numpy as np
cimport numpy as cnp

cnp.import_array()

ctypedef cnp.int64_t DTYPE_t

cpdef tuple get_parallelotope_points(
        cnp.ndarray[DTYPE_t, ndim=1] snf,
        DTYPE_t det,
        cnp.ndarray[DTYPE_t, ndim=2] VDinv,
        cnp.ndarray[DTYPE_t, ndim=2] R,
        cnp.ndarray[DTYPE_t, ndim=2] A,
        cnp.ndarray[DTYPE_t, ndim=1] b,
        bint check_inequalities=False,
    ):
    """
    Get all the integer points in the half-open parallelotope generated by
    the columns of R. This is a modification of the SageMath implementation
    at sage/src/sage/geometry/integral_points.pxi (GPLv2+).
    Description of the parameters:
    snf: the diagonal Smith normal form of the matrix R
    det: the determinant of R
    VDinv: the inverse of the matrix VD, where D is the diagonal matrix
        in the Smith normal form of R
    R: the matrix whose columns generate the parallelotope
    A: a vector of coefficients for a linear inequality Ax <= b
    b: the right-hand side of the linear inequality Ax <= b
    Returns the tuple of the integer points in the parallelotope satisfying Ax <= b are returned.
    """
    cdef int i, j
    cdef int dim = VDinv.shape[0]
    cdef int ambient_dim = R.shape[0]
    cdef DTYPE_t s = 0
    cdef list gens = []
    cdef cnp.ndarray[DTYPE_t, ndim=1] gen = np.zeros(ambient_dim, dtype=np.int64)
    cdef cnp.ndarray[DTYPE_t, ndim=1] q_times_d = np.zeros(dim, dtype=np.int64)
    for base in itertools.product(*[range(i) for i in snf]):
        for i in range(dim):
            s = 0
            for j in range(dim):
                s += VDinv[i, j] * base[j]
            q_times_d[i] = s % det
        for i in range(ambient_dim):
            s = 0
            for j in range(dim):
                s += R[i, j] * q_times_d[j]
            gen[i] = s / det
        if check_inequalities:
            s = 0
            for i in range(ambient_dim):
                s += A[i] * gen[i]
            if s > b:
                continue
        gens.append(copy.copy(gen))
    if not check_inequalities:
        assert(len(gens) == det)
    return tuple(gens)