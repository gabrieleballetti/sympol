from sympy import Array


class Point(Array):
    """
    Point class based on sympy Array
    """

    def __init__(self, iterable, shape=None, **kwargs):
        """
        Initialize a point
        """
        if self.rank() > 1:
            raise ValueError("Point must be a rank 1 array at most")

    def dot(self, other):
        """
        Define the dot product between two points
        """
        return sum([x * y for x, y in zip(self, other)])

    @property
    def ambient_dimension(self):
        """
        Get the ambient dimension of the point
        """
        return self.shape[0]
