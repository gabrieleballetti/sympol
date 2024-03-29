import numpy as np
import numba as nb
from sympy import diag, Matrix

from sympol.point import Point
from sympol._snf import smith_normal_form
from sympol._numba_utils import nb_cartesian_product


class HalfOpenParallelotope:
    """Class for representing a half-open parallelotope."""

    def __init__(self, generators, special_gens_ids=None, check_rank=True):
        """Initialize a half-open parallelotope with the given generators."""
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
        """Return the Smith normal form of the generator matrix."""
        if self._snf is None:
            self._calculate_smith_normal_form()

        return self._snf

    @property
    def det(self):
        """Return the determinant of the generator matrix."""
        if self._det is None:
            self._calculate_smith_normal_form()

        return self._det

    @property
    def v_d_inv(self):
        """Return the matrix VDinv, where D is the SNF."""
        if self._v_d_inv is None:
            self._calculate_smith_normal_form()

        return self._v_d_inv

    @property
    def k(self):
        """Return the index of the first non-one entry in the (diagonal of the) SNF."""
        if self._k is None:
            self._k = 0
            for i, d_i in enumerate(self.snf):
                if d_i != 1:
                    self._k = i
                    break

        return self._k

    def _calculate_smith_normal_form(self):
        """Calculates the Smith normal form (SNF) of the generator matrix.

        The SNF is the diagonal matrix D such that D = U * M * V, where R is the matrix
        having the generators as columns and U and V are integer unimodular matrices.
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

    def get_integer_points(
        self, height=-1, count_only=False, count=True, disable_numba=False
    ):
        """Returns the integer points in the half-open parallelotope."""
        if not disable_numba:
            snf = np.array(self.snf, dtype=np.int64)
            det = np.int64(self.det)
            VDinv = np.array(self.v_d_inv, dtype=np.int64)
            R = np.array(self.m, dtype=np.int64)
            t = np.array(self.t, dtype=np.int64)
            func = _get_parallelotope_points
        else:
            snf = np.array(self.snf, dtype=object)
            VDinv = np.array(self.v_d_inv, dtype=object)
            det = self.det
            R = np.array(self.m, dtype=object)
            t = np.array(self.t, dtype=object)
            func = _get_parallelotope_points.py_func

        pts, h = func(
            snf=snf,
            det=det,
            VDinv=VDinv,
            k=self.k,
            R=R,
            t=t,
            height=height,
            count_only=count_only,
            count=count,
        )

        pts = tuple([Point(p) for p in pts])

        return pts, tuple(h)


@nb.njit(
    nb.types.Tuple((nb.int64[:, :], nb.int64[:]))(
        nb.int64[:],
        nb.int64,
        nb.int64[:, :],
        nb.int64,
        nb.int64[:, :],
        nb.int64[:],
        nb.int64,
        nb.boolean,
        nb.boolean,
    ),
    cache=True,
)
def _get_parallelotope_points(
    snf,
    det,
    VDinv,
    k,
    R,
    t,
    height=-1,
    count_only=False,
    count=True,
):
    """Get the integer points in the half-open parallelotope generated by R.

    Args:
        snf: the diagonal Smith normal form of the matrix R
        det: the determinant of R
        VDinv: the inverse of the matrix VD, where D is the diagonal matrix
            in the Smith normal form of R
        k: the index of the first non-one entry in the diagonal of the Smith normal
            form
        R: the matrix whose columns generate the parallelotope
        t: the translation vector of the parallelotope
        height: if given, only get points with the given height (i.e. first coord)
        count_only: if true, count the points at each height instead of getting them
        count: if false, avoid counting at all

    Returns:
        A tuple of the integer points in the half-open parallelotope and the number
        of points at each height.
    """
    dtype = R.dtype
    dim = VDinv.shape[0]
    ambient_dim = R.shape[0]
    s = 0
    gen = np.zeros(ambient_dim, dtype=dtype)
    q_times_d = np.zeros(dim, dtype=dtype)
    h = np.zeros(ambient_dim, dtype=dtype)
    arange_list = nb.typed.List([np.arange(i, dtype=np.int32) for i in snf])
    cart = nb_cartesian_product(arange_list)
    gens = np.empty((cart.shape[0], ambient_dim), dtype=dtype)
    last = 0
    for b in range(cart.shape[0]):
        base = cart[b]
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
                if count:
                    h[gen[0]] += 1
                if count_only:
                    break
        else:
            # only executed if the loop did not break
            gens[last] = gen
            last += 1
    gens = gens[:last]
    return gens, h
