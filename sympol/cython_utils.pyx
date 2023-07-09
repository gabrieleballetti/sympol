import numpy as np
cimport numpy as cnp

cnp.import_array()

ctypedef cnp.int64_t DTYPE_t

def _sum_of_points(
        cnp.ndarray[DTYPE_t, ndim=2] P,
        cnp.ndarray[DTYPE_t, ndim=2] Q
    ):
    """
    Given two sets of points P and Q, return the set of points
    obtained by summing each point in P with each point in Q.
    (also remove duplicates)
    """
    cdef int i, j, k, n, m, d
    cdef cnp.ndarray[DTYPE_t, ndim=2] S
    n = P.shape[0]
    m = Q.shape[0]
    d = P.shape[1]
    S = np.zeros((n*m, d), dtype=np.int64)
    k = 0
    for i in range(n):
        for j in range(m):
            for k in range(d):
                S[i*m+j, k] = P[i, k] + Q[j, k]
    return np.unique(S, axis=0)





