"""Module for the Point class."""

import numpy as np
from sympy import Rational


class Point(np.ndarray):
    """Point class based on a numpy array with sympy rational entries.

    Example usage:
    >>> from sympol.point import Point
    >>> a = Point([1, 2, 3])
    >>> a / 2
    Point([1/2, 1, 3/2])

    """

    def __new__(cls, data):
        """Create a new Point object.

        Args:
            data (list): A list of values to be converted to sympy Rational objects.

        Returns:
            A new Point object.

        """
        make_sympy_rational = np.vectorize(lambda x: Rational(x))
        return make_sympy_rational(np.array(data)).view(cls)

    def __init__(self, data, **kwargs):
        """Initialize a Point object.

        Args:
            data (list): A list of values to be converted to sympy Rational objects.

        Returns:
            None.

        Raises:
            ValueError: If the Point is not a rank 1 array.

        """
        if self.ndim != 1:
            raise ValueError("Point must be a rank 1 array.")

    def __eq__(self, other):
        """Override the == operator for the Point class.

        Args:
            other (Point): Another Point object to compare with.

        Returns:
            bool: True if the two Point objects are equal, False otherwise.

        """
        if isinstance(other, Point):
            return np.array_equal(self, other)
        else:
            return super().__eq__(other)

    def __ne__(self, other):
        """Override the != operator for the Point class.

        Args:
            other (Point): Another Point object to compare with.

        Returns:
            bool: True if the two Point objects are not equal, False otherwise.

        """
        return not self.__eq__(other)

    def __hash__(self):
        """Override the hash function for the Point class.

        Returns:
            int: The hash value of the Point object.

        """
        return tuple(self).__hash__()

    @property
    def ambient_dimension(self):
        """Get the ambient dimension of the Point.

        Returns:
            int: The ambient dimension of the Point.

        """
        return self.shape[0]
