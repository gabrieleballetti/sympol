from sympy import Array, Matrix
from sympol.point import Point


class PointList(Array):
    """
    Point list class based on sympy Array
    """

    def __init__(self, iterable, shape=None, **kwargs):
        """
        Initialize a point list
        """

        if self.rank() > 2:
            raise ValueError("Point list must be a rank 2 array at most")

        self._barycenter = None

    def __getitem__(self, index):
        """
        Overload the getitem method to return a Point
        """
        return Point(super().__getitem__(index))

    def affine_rank(self):
        """
        Get the affine rank of the point list
        """
        translated_points = [p - self[0] for p in self]
        return Matrix(translated_points).rank()

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
