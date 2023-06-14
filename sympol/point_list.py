from sympy import Abs, Array, Matrix, NDimArray, ZZ
from sympy.polys.matrices import DomainMatrix
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
            # Attempt to convert to a rank 2 array
            if all([i == 1 for i in self.shape[2:]]):
                self = self.reshape(*self.shape[:2])
            else:
                raise ValueError("Point list must be a rank 2 array at most")

        self._ambient_dimension = None
        self._hom_rank = None
        self._affine_rank = None
        self._barycenter = None
        self._basis = None
        self._index = None

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

    def __mul__(self, other):
        """
        Overload the * operator to allow scaling by a scalar
        """
        if isinstance(other, (int, float)):
            return PointList([p * other for p in self])
        return super().__mul__(other)

    def tolist(self):
        """
        Convert the point list to a list of numbers
        """
        return [p.tolist() for p in self]

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

    @property
    def basis(self):
        """
        Get the basis of the sublattice spanned by the point list. Use the LLL
        algorithm to compute the basis (which is probably an overkill). Assume
        that the point list is full rank.
        """
        if self._basis is None:
            if self.affine_rank < self.ambient_dimension:
                raise ValueError("Point list is not full rank")
            m = Matrix(self - self[0])
            m = DomainMatrix.from_Matrix(m, ZZ)
            basis, _ = m.transpose().lll_transform()
            basis = basis.transpose().to_list()

            # the last d rows of the LLL basis are the basis of the sublattice
            basis = basis[-self.ambient_dimension :]
            self._basis = PointList(basis)

        return self._basis

    @property
    def index(self):
        """
        Get the index of the sublattice spanned by the point list in the integer
        lattice.
        """
        if self._index is None:
            if self.affine_rank < self.ambient_dimension:
                raise ValueError("Point list is not full rank")
            self._index = Abs(Matrix(self.basis).det())

        return self._index
