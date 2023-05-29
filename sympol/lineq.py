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
