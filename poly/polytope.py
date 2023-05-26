from scipy.spatial import ConvexHull
from sympy import Abs, Array, factorial, Matrix, Rational
from sympy.geometry.point import Point


class Polytope:
    """
    Polytope class
    """

    def __init__(self, points: Array):
        self.points = [Point(p) for p in points]

        if len(self.points) == 0:
            raise ValueError("Polytope must have at least one point")

        self.ambient_dim = self.points[0].ambient_dimension
        self.dim = Point.affine_rank(*self.points)

        self.is_full_dim = self.dim == self.ambient_dim

        if self.dim < self.ambient_dim:
            # decide what to do if the polytope is not full dimensional
            raise ValueError("Polytope is not full dimensional")

        self._conv_hull = ConvexHull(self.points)
        self._ch_simplices = self._conv_hull.simplices.tolist()
        self._ch_vertices = self._conv_hull.vertices.tolist()

        self.vertices = Array([self.points[i] for i in self._ch_vertices])
        self.boundary_triangulation = Array(
            [
                [self.points[j] for j in simplex_ids]
                for simplex_ids in self._ch_simplices
            ]
        )

        # Available on demand
        self.volume = None
        self.normalized_volume = None
        self.hyperplanes = None

    def get_volume(self):
        """
        Get the symbolic normalized volume of the polytope
        """
        if self.volume is not None:
            return self.volume

        self.calculate_volume()
        return self.volume

    def get_normalized_volume(self):
        """
        Get the symbolic normalized volume of the polytope
        """
        if self.normalized_volume is not None:
            return self.normalized_volume

        self.calculate_volume()
        return self.normalized_volume

    def calculate_volume(self):
        """
        Calculate the volume of the polytope
        """
        volume = Rational(0)

        for simplex in self.boundary_triangulation:
            if self.vertices[0] in simplex:
                continue
            translated_simplex = [vertex - self.vertices[0] for vertex in simplex]
            volume += Abs(Matrix(translated_simplex).det())

        self.volume = volume
        self.normalized_volume = volume * factorial(self.dim)

    def get_hyperplanes(self):
        """
        Get the symbolic inequalities of the polytope
        TODO: implement
        """
        if self.hyperplanes is not None:
            return self.hyperplanes

        return None
