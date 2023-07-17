import copy
import itertools
import numpy as np
from sympy import diag, Matrix

from sympol.integer_pts_parallelotope_np import (
    get_parallelotope_points_np,
)
from sympol.point import Point
from sympol.snf import smith_normal_form


class HalfOpenParallelotope:
    """
    Class for representing a half-open parallelotope and counting its
    number of integer points.
    """

    def __init__(self, generators, special_gens_ids=None, check_rank=True):
        """
        Initializes a half-open parallelotope with the given generators.
        """
        if special_gens_ids is None:
            special_gens_ids = []

        self.t = sum(
            [generators[i] for i in special_gens_ids], Point([0 for _ in generators[0]])
        )

        self.m = Matrix([g - self.t for g in generators]).T

        # check linear independence of generators
        if check_rank and self.m.rank() != self.m.shape[1]:
            raise ValueError("Generators must be linearly independent.")

        self._snf = None
        self._v_d_inv = None
        self._det = None

    @property
    def snf(self):
        """
        Returns the Smith normal form of the generator matrix.
        """
        if self._snf is None:
            self._calculate_smith_normal_form()

        return self._snf

    @property
    def det(self):
        """
        Returns the determinant of the generator matrix.
        """
        if self._det is None:
            self._calculate_smith_normal_form()

        return self._det

    @property
    def v_d_inv(self):
        if self._v_d_inv is None:
            self._calculate_smith_normal_form()

        return self._v_d_inv

    def _calculate_smith_normal_form(self):
        """
        Calculates the Smith normal form of the generator matrix, i.e. the
        diagonal matrix D such that D = U * M * V, where R is the matrix having
        the generators as columns and U and V are integer unimodular matrices.
        """
        d, u, v = smith_normal_form(self.m)

        self._snf = d.diagonal().flat()
        self._det = 1
        for i in self._snf:
            self._det *= i

        d_inv = diag(*[self._det // e_i for e_i in self._snf])

        # Apply a % self._det to each entry of v * d_inv to keep the
        # entries in the range [0, self._det)
        self._v_d_inv = v * d_inv % self._det

    def get_integer_points(self, height=-1):
        """
        Returns the number of integer points in the half-open parallelotope.
        """
        try:
            pts = get_parallelotope_points_np(
                np.array(self.snf, dtype=np.int64),
                self.det,
                np.array(self.v_d_inv, dtype=np.int64),
                np.array(self.m, dtype=np.int64),
                height,
            )
        except OverflowError as e:
            # This can be uset to allow to fallback in the pure Python implementation if
            # there is a failure due to overflow. But in practice this will be too slow
            # to be useful in most of the cases. Can still be used for debugging by
            # commenting out the following line.
            raise OverflowError(e)
            pts = get_parallelotope_points(
                self.snf,
                self.det,
                self.v_d_inv,
                self.m,
                height=height,
            )
        return pts


def get_parallelotope_points(snf, det, VDinv, R, height=-1):
    """
    Get all the integer points in the half-open parallelotope generated by
    the columns of R. This is a modification of the SageMath implementation
    at sage/src/sage/geometry/integral_points.pxi (GPLv2+).

    snf: the diagonal Smith normal form of the matrix R
    det: the determinant of R
    VDinv: the inverse of the matrix VD, where D is the diagonal matrix
        in the Smith normal form of R
    R: the matrix whose columns generate the parallelotope
    height: if given, only return points with the given "height" (= first coordinate)
    """
    dim = VDinv.shape[0]
    ambient_dim = R.shape[0]
    s = 0
    gens = []
    gen = np.zeros(ambient_dim, dtype=np.int64)
    q_times_d = np.zeros(dim, dtype=np.int64)
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
            if i == 0 and height >= 0 and gen[0] != height:
                break
        else:
            # only executed if the loop did not break
            gens.append(copy.copy(gen))
    return tuple(gens)
