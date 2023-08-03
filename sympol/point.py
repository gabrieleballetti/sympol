import numpy as np
from sympy import Rational


class Point(np.ndarray):
    """
    Point class based an numpy array with sympy rational entries.
    """

    def __new__(cls, data):
        make_sympy_rational = np.vectorize(lambda x: Rational(x))
        return make_sympy_rational(np.array(data)).view(cls)

    def __init__(self, data, **kwargs):
        """
        Initialize a point
        """
        if self.ndim != 1:
            raise ValueError("Point must be a rank 1 array.")

    def __eq__(self, other):
        """
        Override the == operator for the Point class.
        """
        if isinstance(other, Point):
            return np.array_equal(self, other)
        else:
            return super().__eq__(other)

    def __ne__(self, other):
        """
        Override the != operator for the Point class.
        """
        return not self.__eq__(other)

    def __hash__(self):
        """
        Override the hash method for the Point class.
        """
        return tuple(self).__hash__()

    @property
    def ambient_dimension(self):
        """
        Get the ambient dimension of the point.
        """
        return self.shape[0]
