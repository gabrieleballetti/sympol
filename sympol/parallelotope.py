from sympy import diag, Matrix
from sympy.matrices.normalforms import hermite_normal_form


class HalfOpenParallelotope:
    """
    Class for representing a half-open parallelotope and counting its
    number of integer points.
    """

    def __init__(self, generators, check_rank=True):
        """
        Initializes a half-open parallelotope with the given generators.
        """
        self.generators = generators

        # check linear independence of generators
        if check_rank and self.generators.rank != self.generators.shape[0]:
            raise ValueError("Generators must be linearly independent.")

        self._m = None
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

    @property
    def m(self):
        if self._m is None:
            self._calculate_smith_normal_form()

        return self._m

    def _calculate_smith_normal_form(self):
        """
        Calculates the Smith normal form of the generator matrix, i.e. the
        diagonal matrix D such that D = U * M * V, where R is the matrix having
        the generators as columns and U and V are integer unimodular matrices.
        """
        self._m = Matrix(self.generators).T

        # column style HNF
        h = hermite_normal_form(self._m)
        v = self._m.inv() * h

        # row style HNF
        d = hermite_normal_form(h.T).T
        # u = d * h.inv() # not needed

        self._snf = d.diagonal().flat()
        self._det = 1
        for i in self._snf:
            self._det *= i

        d_inv = diag(*[self._det // e_i for e_i in self._snf])

        self._v_d_inv = v * d_inv

    def find_integer_points(self):
        """
        Returns the number of integer points in the half-open parallelotope.
        """
        # TODO
        pass
