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

        translated_gens = []
        for i, gen in enumerate(generators):
            if i in special_gens_ids:
                translated_gens.append((-gen).tolist())
            else:
                translated_gens.append(gen.tolist())

        self.m = Matrix(translated_gens).T

        # check linear independence of generators
        if check_rank and self.m.rank() != self.m.shape[1]:
            raise ValueError("Generators must be linearly independent.")

        self._snf = None
        self._v_d_inv = None
        self._det = None
        self._k = None

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

    @property
    def k(self):
        """
        Return the value of k, i.e. the index of the first non-one entry in the
        (diagonal) of the Smith normal form of the generator matrix. This is used
        to speed up the computation of the integer points enumeration.
        """
        if self._k is None:
            self._k = 0
            for i, d_i in enumerate(self.snf):
                if d_i != 1:
                    self._k = i
                    break

        return self._k

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

    def get_integer_points(self, height=-1, count_only=False, use_sympy=False):
        """
        Returns the number of integer points in the half-open parallelotope.
        """
        if use_sympy:
            pts, h = get_parallelotope_points_simpy(
                snf=self.snf,
                det=self.det,
                VDinv=self.v_d_inv,
                k=self.k,
                R=self.m,
                t=self.t,
                height=height,
                count_only=count_only,
            )
        else:
            pts, h = get_parallelotope_points_np(
                snf=np.array(self.snf, dtype=np.int64),
                det=self.det,
                VDinv=np.array(self.v_d_inv, dtype=np.int64),
                k=self.k,
                R=np.array(self.m, dtype=np.int64),
                t=np.array(self.t, dtype=np.int64),
                height=height,
                count_only=count_only,
            )
        return pts, tuple(h)


def get_parallelotope_points_simpy(
    snf,
    det,
    VDinv,
    k,
    R,
    t,
    height=-1,
    count_only=False,
):
    """
    See get_parallelotope_points_np for the documentation of the parameters.
    """
    dim = VDinv.shape[0]
    ambient_dim = R.shape[0]
    s = 0
    gens = []
    gen = np.zeros(ambient_dim, dtype=np.int64)
    q_times_d = np.zeros(dim, dtype=np.int64)
    h = np.zeros(ambient_dim, dtype=np.int64)
    for base in itertools.product(*[range(i) for i in snf]):
        for i in range(dim):
            s = 0
            for j in range(k, dim):
                s += VDinv[i, j] * base[j]
            q_times_d[i] = s % det
        for i in range(ambient_dim):
            s = 0
            for j in range(dim):
                s += R[i, j] * q_times_d[j]
            gen[i] = s // det + t[i]
            if i == 0:
                if height >= 0 and gen[0] != height:
                    break
                h[gen[0]] += 1
                if count_only:
                    break
        else:
            # only executed if the loop did not break
            gens.append(copy.copy(gen))
    return tuple(gens), h
