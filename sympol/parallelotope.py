import numpy as np
from sympy import diag, Matrix
from sympy.matrices.normalforms import hermite_normal_form, smith_normal_form
from sympol.integer_pts_parallelotope import get_parallelotope_points


class HalfOpenParallelotope:
    """
    Class for representing a half-open parallelotope and counting its
    number of integer points.
    """

    def __init__(self, generators, check_rank=True):
        """
        Initializes a half-open parallelotope with the given generators.
        """
        self.m = Matrix(generators).T

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
        # column style HNF
        h = hermite_normal_form(self.m)
        v = self.m.inv() * h

        # row style HNF
        # d = hermite_normal_form(h.T).T
        # u = d * h.inv() # not needed
        d = smith_normal_form(h.T).T

        self._snf = d.diagonal().flat()
        self._det = 1
        for i in self._snf:
            self._det *= i

        d_inv = diag(*[self._det // e_i for e_i in self._snf])

        self._v_d_inv = v * d_inv

    def get_integer_points(self):
        """
        Returns the number of integer points in the half-open parallelotope.
        """
        pts = get_parallelotope_points(
            np.array(self.snf, dtype=np.int64),
            self.det,
            np.array(self.v_d_inv, dtype=np.int64),
            np.array(self.m, dtype=np.int64),
            np.zeros(shape=[0, 0], dtype=np.int64),
            np.zeros(shape=[0], dtype=np.int64),
            check_inequalities=False,
        )
        return pts
