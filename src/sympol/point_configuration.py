"""Module for the PointConfiguration class."""

import numpy as np
from numpy.typing import ArrayLike
from sympy import Abs, Matrix, prod, ZZ, Rational
from sympy.matrices.normalforms import smith_normal_form

from sympol.point import Point


class PointConfiguration(np.ndarray):
    """PointConfiguration class based on a numpy array with sympy rational entries.

    The PointConfiguration inherit from numpy.ndarray. It is a two dimensional array
    whose entries are converted to sympy Rational upon initialization. This object can
    be thought as a list of Point objects in the same ambient space.

    Example usage:

    .. code-block:: python

        from sympol import PointConfiguration

        PointConfiguration([[1, 0, 0], [0, 1, 0]]) / 2
        # PointConfiguration([[1/2, 0, 0],
        #                     [0, 1/2, 0]], dtype=object)

    """

    def __new__(cls, data: ArrayLike, shape=None, **kwargs):
        """Create a new PointConfiguration object."""

        _arr = np.array(data)
        if _arr.size == 0:
            _arr = _arr.reshape(0, 0)
        if _arr.ndim != 2:
            raise ValueError("Point configuration must be a rank 2 array.")
        if _arr.size > 0:
            make_sympy_rational = np.vectorize(lambda x: Rational(x))
            _arr = make_sympy_rational(_arr)
        return _arr.view(cls)

    def __init__(self, data: ArrayLike, shape=None, **kwargs):
        """Initialize a PointConfiguration object."""

        self._ambient_dimension = None
        self._rank = None
        self._affine_rank = None
        self._barycenter = None
        self._snf_diag = None
        self._index = None

    def __getitem__(self, index):
        """
        Overload the getitem method to return a Point
        """
        if isinstance(index, int):
            return Point(super().__getitem__(index))
        elif isinstance(index, slice):
            return PointConfiguration(super().__getitem__(index))
        else:
            return super().__getitem__(index)

    def __eq__(self, other):
        """
        Override the == operator for the PointConfiguration class.
        """
        if isinstance(other, PointConfiguration):
            return np.array_equal(self, other)
        else:
            return super().__eq__(other)

    def __ne__(self, other):
        """
        Override the != operator for the PointConfiguration class.
        """
        return not self.__eq__(other)

    def __hash__(self):
        """
        Override the hash method for the PointConfiguration class.
        """
        return tuple(map(tuple, self)).__hash__()

    @property
    def ambient_dimension(self):
        """
        Get the ambient dimension of the point list
        """
        if self.shape[0] == 0:
            raise ValueError("Point list is empty")

        if self._ambient_dimension is None:
            self._ambient_dimension = self[0].ambient_dimension

        return self._ambient_dimension

    @property
    def rank(self):
        """
        Get the rank of the point list (avoid the variable name "rank" to avoid
        conflict with the rank method of the Array class)
        """
        if self._rank is None:
            self._rank = Matrix(self).rank()

        return self._rank

    @property
    def affine_rank(self):
        """
        Get the affine rank of the point list
        """
        if self._affine_rank is None:
            translated_points = [p - self[0] for p in self]
            self._affine_rank = Matrix(translated_points).rank()

        return self._affine_rank

    @property
    def barycenter(self):
        """
        Get the barycenter of the point list
        """
        if self._barycenter is None:
            self._barycenter = (
                Point([sum(coords) for coords in zip(*self)]) / self.shape[0]
            )

        return self._barycenter

    @property
    def snf_diag(self):
        """
        Get the (diagonal entries of the) affine Smith Normal Form of the point list.
        """
        if self._snf_diag is None:
            m = Matrix(self[1:] - self[0])
            self._snf_diag = smith_normal_form(m, domain=ZZ).diagonal().flat()

        return self._snf_diag

    @property
    def index(self):
        """
        Get the index of the sublattice spanned by the point list in the integer
        lattice.
        """
        if self._index is None:
            if self.affine_rank < self.ambient_dimension:
                raise ValueError("Point list is not full rank")
            self._index = Abs(prod(self.snf_diag))

        return self._index
