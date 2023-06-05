from sympy import Array, Matrix, NDimArray
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

        self._ambient_dimension = None
        self._hom_rank = None
        self._affine_rank = None
        self._barycenter = None

    def __getitem__(self, index):
        """
        Overload the getitem method to return a Point
        """
        return Point(super().__getitem__(index))

    def __add__(self, other):
        """
        Overload the + operator to add allow translation by a vector
        """
        if isinstance(other, NDimArray) and self.shape[1] == other.shape[0]:
            return PointList([p + other for p in self])
        return super().__add__(other)

    def __sub__(self, other):
        """
        Overload the - operator to add allow translation by a vector
        """
        if isinstance(other, NDimArray) and self.shape[1] == other.shape[0]:
            return PointList([p - other for p in self])
        return super().__sub__(other)

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
    def hom_rank(self):
        """
        Get the rank of the point list (avoid the variable name "rank" to avoid
        conflict with the rank method of the Array class)
        """
        if self._hom_rank is None:
            self._hom_rank = Matrix(self).rank()

        return self._hom_rank

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
