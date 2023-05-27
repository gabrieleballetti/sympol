from scipy.spatial import ConvexHull
from sympy import Abs, factorial, Matrix, Rational
from sympy.geometry.point import Point


class Polytope:
    """
    Polytope class
    """

    def __init__(
        self,
        points: list = None,
        dim: int = None,
        vertices: list = None,
    ):
        """
        Initialize a polytope from a list of points or vertices
        """
        if points is None and vertices is None:
            raise ValueError("Either points or vertices must be given")

        if points is not None and len(points) == 0:
            raise ValueError("Points cannot be empty")

        if vertices is not None and len(vertices) == 0:
            raise ValueError("Vertices cannot be empty")

        if vertices is not None:
            self._vertices = [Point(v) for v in vertices]
        else:
            self._vertices = None

        if points is not None:
            self._points = [Point(p) for p in points]
        else:
            self._points = [Point(v) for v in vertices]

        self._ambient_dim = self._points[0].ambient_dimension

        # Need affine_rank to be calculated (unless given by user)
        self._dim = dim
        self._scipy_conv_hull = None

        # Need scipy_conv_hull to be calculated (unless given by user)
        self._boundary_triangulation = None

        # Need volume to be calculated
        self._volume = None
        self._normalized_volume = None

        # TODO: implement
        self._hyperplanes = None
        self._edges = None

    @property
    def points(self):
        """
        Get the defining points of the polytope
        """
        return self._points

    @property
    def ambient_dim(self):
        """
        Get the ambient dimension of the polytope
        """
        return self._ambient_dim

    @property
    def dim(self):
        """
        Get the dimension of the polytope
        """
        if self._dim is None:
            self._dim = Point.affine_rank(*self._points)

            # TODO: this should be removed!
            # Add support for non-full dimensional polytopes
            if self._dim < self._ambient_dim:
                raise NotImplementedError(
                    "Non-full dimensional polytopes are not supported yet"
                )

        return self._dim

    @property
    def is_full_dim(self):
        """
        Check if the polytope is full dimensional
        """
        return self.dim == self.ambient_dim

    @property
    def scipy_conv_hull(self):
        """
        Get the scipy convex hull of the polytope
        """
        if self._scipy_conv_hull is None:
            # TODO: add support for non-full dimensional polytopes
            if not self.is_full_dim:
                raise NotImplementedError(
                    "Non-full dimensional polytopes are not supported yet"
                )
            self._scipy_conv_hull = ConvexHull(self.points)

        return self._scipy_conv_hull

    @property
    def vertices(self):
        """
        Get the vertices of the polytope
        """
        if self._vertices is None:
            self._vertices = [
                self.points[i] for i in self.scipy_conv_hull.vertices.tolist()
            ]

        return self._vertices

    @property
    def boundary_triangulation(self):
        """
        Get the boundary triangulation of the polytope
        """
        if self._boundary_triangulation is None:
            self._boundary_triangulation = [
                [self.points[j] for j in simplex_ids]
                for simplex_ids in self.scipy_conv_hull.simplices.tolist()
            ]
        return self._boundary_triangulation

    @property
    def volume(self):
        """
        Get the normalized volume of the polytope
        """
        if self._volume is None:
            self._calculate_volume()
        return self._volume

    @property
    def normalized_volume(self):
        """
        Get the normalized volume of the polytope
        """
        if self._normalized_volume is None:
            self.calculate_volume()
        return self._normalized_volume

    def _calculate_volume(self):
        """
        Calculate the volume of the polytope, sets both _volume and _normalized_volume
        """
        volume = Rational(0)

        for simplex in self.boundary_triangulation:
            if self.vertices[0] in simplex:
                continue
            translated_simplex = [vertex - self.vertices[0] for vertex in simplex]
            volume += Abs(Matrix(translated_simplex).det())

        self._volume = volume
        self._normalized_volume = volume * factorial(self.dim)

    @property
    def hyperplanes(self):
        """
        Get the inequalities of the polytope
        TODO: implement
        """
        if self._hyperplanes is not None:
            return self._hyperplanes

        return None

    @property
    def edges(self):
        """
        Get the edges of the polytope
        TODO: implement
        """

        if self._edges is not None:
            return self._edges

        return None
