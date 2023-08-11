"""Module for the Point class."""

import numpy as np
from numpy.typing import ArrayLike
from sympy import Rational


class Point(np.ndarray):
    """Point class based on a numpy array with sympy rational entries.

    The Point class inherit from numpy.ndarray. It is a one dimensional array whose
    entries are converted to sympy Rational upon initialization.

    Example usage:

    .. code-block:: python

        from sympol import Point

        Point([1, 2, 3]) / 2
        # Point([1/2, 1, 3/2], dtype=object)

    """

    def __new__(cls, data: ArrayLike, **kwargs):
        """Create a new Point object."""
        make_sympy_rational = np.vectorize(lambda x: Rational(x))
        return make_sympy_rational(np.array(data)).view(cls)

    def __init__(self, data: np.typing.ArrayLike, **kwargs):
        """Initialize a Point object."""
        if self.ndim != 1:
            raise ValueError("Point must be a rank 1 array.")

    def __eq__(self, other):
        """Define __eq__ between two Point classes."""
        if isinstance(other, Point):
            return np.array_equal(self, other)
        else:
            return super().__eq__(other)

    def __ne__(self, other: "Point"):
        """Define __ne__ between two Point classes."""
        return not self.__eq__(other)

    def __hash__(self):
        """Define __hash__ for Point class."""
        return tuple(self).__hash__()

    @property
    def ambient_dimension(self) -> int:
        """Get the ambient dimension of the point.

        Returns:
            The ambient dimension of the point.

        """
        return self.shape[0]
