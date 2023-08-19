"""Module for the PointConfiguration class."""

import numpy as np
from numpy.typing import ArrayLike
from sympy import Abs, Matrix, prod, ZZ, Rational
from sympy.matrices.normalforms import smith_normal_form

from sympol.point import Point
from sympol._triangulation import _get_upper_or_lower_hull_triangulation


class PointConfiguration(np.ndarray):
    """PointConfiguration class based on a numpy array with sympy rational entries.

    The PointConfiguration inherit from numpy.ndarray. It is a two dimensional array
    whose entries are converted to sympy Rational upon initialization. This object
    represents a list of Point objects in the same ambient space.

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

        self._triangulation = None

    def __getitem__(self, index):
        """Override __getitem__ to return Point objects for integers values."""
        if isinstance(index, int):
            return Point(super().__getitem__(index))
        elif isinstance(index, tuple) and len(index) > 0 and isinstance(index[0], int):
            # Also Point for indices like [0, ...]
            return Point(super().__getitem__(index[0])).__getitem__(index[1:])
        else:
            return super().__getitem__(index)

    def __eq__(self, other):
        """Define __eq__ between two PointConfiguration objects."""
        if isinstance(other, PointConfiguration):
            return np.array_equal(self, other)
        else:
            return super().__eq__(other)

    def __ne__(self, other):
        """Define __ne__ between two PointConfiguration objects."""
        return not self.__eq__(other)

    def __hash__(self):
        """Define __hash__ for PointConfiguration objects."""
        return tuple(map(tuple, self)).__hash__()

    @property
    def ambient_dimension(self) -> int:
        """Get the ambient dimension of the point.

        Returns:
            The ambient dimension of the point.

        Raises:
            ValueError: If the point configuration is empty.
        """
        if self.shape[0] == 0:
            raise ValueError("Point configuration is empty")

        if self._ambient_dimension is None:
            self._ambient_dimension = self[0].ambient_dimension

        return self._ambient_dimension

    @property
    def rank(self) -> int:
        """Get the rank of the point configuration.

        Note that the rank of a point configuration is the rank of the matrix defined
        by the points, which is the size of the largest square submatrix with
        non-zero determinant. The rank might not be translation-invariant, see
        :attr:`affine_rank` for the translation-invariant version.

        Returns:
            The rank of the point list.
        """
        if self._rank is None:
            self._rank = Matrix(self).rank()

        return self._rank

    @property
    def affine_rank(self) -> int:
        """Get the affine rank of the point list.

        Note that the affine rank of a point configuration is the rank of the matrix
        defined by the points after translating one to the origin. Use
        :attr:`rank` for the rank of the matrix defined by the points in the
        PointConfiguration without translation.

        Returns:
            The affine rank of the point list.
        """
        if self._affine_rank is None:
            translated_points = [p - self[0] for p in self]
            self._affine_rank = Matrix(translated_points).rank()

        return self._affine_rank

    @property
    def barycenter(self) -> Point:
        """Get the barycenter of the point list.

        The barycenter (or center of mass) of the PointConfiguration is the average of
        its points.

        Returns:
            The barycenter of the point list.
        """
        if self._barycenter is None:
            self._barycenter = (
                Point([sum(coords) for coords in zip(*self)]) / self.shape[0]
            )

        return self._barycenter

    @property
    def snf_diag(self) -> list:
        """Get the affine Smith Normal Form of the PointConfiguration.

        The affine Smith Normal Form of a PointConfiguration is the diagonal of the
        matrix defined by the points in the PointConfiguration after translating one of
        them to the origin.

        Returns:
            The diagonal entries of the affine Smith Normal Form of the
            PointConfiguration.
        """
        if self._snf_diag is None:
            m = Matrix(self[1:] - self[0])
            self._snf_diag = smith_normal_form(m, domain=ZZ).diagonal().flat()

        return self._snf_diag

    @property
    def index(self):
        """Get the index of the sublattice spanned by the PointConfiguration.

        The index of a sublattice is the index of the sublattice L in the ambient
        integer lattice ZZ^d. It equals the cardinality of the quotient group ZZ^d / L.

        Returns:
            The index of the sublattice spanned by the PointConfiguration.

        Raises:
            ValueError: If the PointConfiguration is not full rank.
        """
        if self._index is None:
            if self.affine_rank < self.ambient_dimension:
                raise ValueError("Point list is not full rank")
            self._index = Abs(prod(self.snf_diag))

        return self._index

    @property
    def triangulation(self):
        """Get a triangulation of on the PointConfiguration.

        Returns:
            A triangulation of the PointConfiguration.
        """
        if self._triangulation is None:
            self._triangulation = _get_upper_or_lower_hull_triangulation(
                points=self, affine_rank=self.affine_rank
            )

        return self._triangulation
