from sympy import Point, Rational


class LinIneq:
    """
    Linear inequality class
    """

    def __init__(
        self,
        normal: Point,
        rhs: Rational,
    ):
        """
        Initialize a linear inequality
        """
        self._normal = normal
        self._rhs = rhs

    @property
    def normal(self):
        """
        Get the normal vector of the linear inequality
        """
        return self._normal

    @property
    def rhs(self):
        """
        Get the right hand side of the linear inequality
        """
        return self._rhs

    def evaluate(self, point: Point):
        """
        Evaluate the linear inequality at a point
        """
        return self._normal.dot(point) - self._rhs
