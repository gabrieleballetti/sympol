import numpy as np
from scipy.spatial import ConvexHull
from sympy import Abs, factorial, lcm, Number, Matrix, Rational
from sympy.matrices import zeros

from sympol.isomorphism import get_normal_form
from sympol.point import Point
from sympol.point_list import PointList
from sympol.lineq import LinIneq


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
            vertices = PointList(vertices)
        else:
            vertices = None

        if points is not None:
            self._points = PointList(points)
        else:
            self._points = PointList(vertices)

        self._ambient_dim = self._points[0].ambient_dimension

        self._dim = dim
        self._scipy_conv_hull = None
        self._vertices = vertices
        self._boundary_triangulation = None
        self._volume = None
        self._normalized_volume = None

        self._linear_inequalities = None
        self._facets = None
        self._n_vertices = None
        self._n_facets = None
        self._vertex_facet_matrix = None
        self._vertex_facet_pairing_matrix = None
        self._normal_form = None
        self._affine_normal_form = None

        # TODO: implement
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
            self._dim = self.points.affine_rank

            # TODO: this should be removed!
            # Add support for non-full dimensional polytopes
            if self._dim < self._ambient_dim:
                raise NotImplementedError(
                    "Non-full dimensional polytopes are not supported yet"
                )

        return self._dim

    @property
    def scipy_conv_hull(self):
        """
        Get the scipy convex hull of the polytope
        """
        if self._scipy_conv_hull is None:
            # TODO: add support for non-full dimensional polytopes
            if not self.is_full_dim():
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
            self._vertices = PointList(
                [self.points[i] for i in self.scipy_conv_hull.vertices.tolist()]
            )

        return self._vertices

    @property
    def boundary_triangulation(self):
        """
        Get the boundary triangulation of the polytope
        """
        if self._boundary_triangulation is None:
            self._boundary_triangulation = []
            for simplex_ids in self.scipy_conv_hull.simplices.tolist():
                simplex_vs = PointList([self.points[i] for i in simplex_ids])
                if simplex_vs.affine_rank == self.dim - 1:
                    self._boundary_triangulation.append(simplex_ids)

        return self._boundary_triangulation

    @property
    def barycenter(self):
        """
        Get the barycenter of the polytope
        """
        return self.vertices.barycenter

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

    @property
    def linear_inequalities(self):
        """
        Get the defining inequalities of the polytope
        """
        if self._linear_inequalities is None:
            self._calculate_facets_and_lin_eqs()

        return self._linear_inequalities

    @property
    def facets(self):
        """
        Get the facets of the polytope
        """
        if self._facets is None:
            self._calculate_facets_and_lin_eqs()

        return self._facets

    @property
    def n_vertices(self):
        """
        Get the number of vertices of the polytope
        """
        if self._n_vertices is None:
            self._n_vertices = self.vertices.shape[0]

        return self._n_vertices

    @property
    def n_facets(self):
        """
        Get the number of facets of the polytope
        """
        if self._n_facets is None:
            self._n_facets = len(self.facets)

        return self._n_facets

    @property
    def vertex_facet_matrix(self):
        """
        Get the vertex facet matrix of the polytope
        """
        if self._vertex_facet_matrix is None:
            self._vertex_facet_matrix = zeros(len(self.facets), self.vertices.shape[0])

            for i, facet in enumerate(self.facets):
                for vertex_id in facet:
                    self._vertex_facet_matrix[i, vertex_id] = 1

        return self._vertex_facet_matrix

    @property
    def vertex_facet_pairing_matrix(self):
        """
        Get the vertex facet matrix of the polytope
        """
        if self._vertex_facet_pairing_matrix is None:
            self._vertex_facet_pairing_matrix = zeros(
                len(self.facets), self.vertices.shape[0]
            )

            for i, lineq in enumerate(self.linear_inequalities):
                for j, vertex in enumerate(self.vertices):
                    self._vertex_facet_pairing_matrix[i, j] = lineq.evaluate(vertex)

        return self._vertex_facet_pairing_matrix

    def _calculate_facets_and_lin_eqs(self):
        """
        Calculate the facets and the linear_inequalities of the polytope by merging
        scipy_conv_hull.simplices into facets.
        This sets _facets and _linear_inequalities.
        """
        self._linear_inequalities = []
        facets_dict = dict()

        for simplex_ids in self.boundary_triangulation:
            normal = self._inner_normal_to_facet(simplex_ids)
            # make sure the normal has integer coefficients
            normal = normal * lcm([frac.q for frac in normal])
            if normal in [lineq._normal for lineq in self._linear_inequalities]:
                facets_dict[normal].update(simplex_ids)
                continue
            lineq = LinIneq(normal, normal.dot(self.points[simplex_ids[0]]))
            self._linear_inequalities.append(lineq)
            facets_dict[normal] = set(simplex_ids)

        self._facets = [list(ids) for ids in facets_dict.values()]

    def _inner_normal_to_facet(self, simplex_ids):
        """
        Calculate the inner normal to a codimension 1 boundary simplex
        """
        translated_simplex = (
            PointList([self.points[i] for i in simplex_ids[1:]])
            - self.points[simplex_ids[0]]
        )
        if translated_simplex.hom_rank != self.ambient_dim - 1:
            # TODO: this is here to fix a bug, should be removed
            raise ValueError(
                "The simplex {} is not a boundary simplex".format(simplex_ids)
            )
        matrix = Matrix(translated_simplex)
        normal = matrix.nullspace()[0].transpose()  # guaranteed to exist and be 1-dim
        normal = Point(normal.tolist()[0])  # convert to Point

        # make sure it points inwards
        if normal.dot(self.barycenter - self.points[simplex_ids[0]]) < 0:
            normal = -normal

        return normal

    def _calculate_volume(self):
        """
        Calculate the volume of the polytope, sets both _volume and _normalized_volume
        """
        volume = Rational(0)

        for simplex_ids in self.boundary_triangulation:
            if 0 in simplex_ids:
                continue
            translated_simplex = [
                self.vertices[id] - self.vertices[0] for id in simplex_ids
            ]
            volume += Abs(Matrix(translated_simplex).det())

        self._normalized_volume = volume
        self._volume = volume / factorial(self.dim)

    @property
    def edges(self):
        """
        Get the edges of the polytope
        TODO: implement
        """

        if self._edges is not None:
            return self._edges

        return None

    @property
    def normal_form(self):
        """
        Return a polytope in normal form
        """
        if self._normal_form is None:
            self._normal_form = get_normal_form(polytope=self)

        return self._normal_form

    @property
    def affine_normal_form(self):
        """
        Return a polytope in affine normal form
        """
        if self._affine_normal_form is None:
            self._affine_normal_form = get_normal_form(polytope=self, affine=True)

        return self._affine_normal_form

    # Helper functions
    def is_full_dim(self):
        """
        Check if the polytope is full dimensional
        """
        return self.dim == self.ambient_dim

    def is_lattice_polytope(self):
        """
        Check if the polytope is a lattice polytope
        """
        return all([all([i.is_integer for i in v]) for v in self.vertices])

    # Polytope operations

    def __add__(self, other):
        """
        Return the the sum of self and other:
            - the translation of self by other if other is a Point
            - the Minkowski sum of self and other if other is a Polytope (TODO)
        """
        if isinstance(other, Point):
            verts = self.vertices + other
            return Polytope(vertices=verts)

        if isinstance(other, Polytope):
            raise NotImplementedError("Minkowski sum not implemented yet")

        raise TypeError(
            "A polytope can only be added to a Point (translation)"
            " or another polytope (Minkowski sum)"
        )

    def __sub__(self, other):
        """
        Return the the difference of self and other:
            - the translation of self by -other if other is a Point
            - the Minkowski difference of self and other if other is a Polytope (TODO)
        """
        return self + (-other)

    def __neg__(self):
        """
        Return the negation of self
        """
        return self * (-1)

    def __mul__(self, other):
        """
        Return the product of self and other:
            - the dilation of self by other if other is a scalar
            - the cartesian product of self and other if other is a Polytope
        """
        if isinstance(other, Number) or isinstance(other, int):
            verts = self.vertices * other
            return Polytope(vertices=verts)

        if isinstance(other, Polytope):
            verts = []
            for v1 in self.vertices:
                for v2 in other.vertices:
                    verts.append(v1.tolist() + v2.tolist())
            return Polytope(vertices=verts)

        raise TypeError(
            "A polytope can only be multiplied with a scalar (dilation)"
            " or another polytope (cartesian product)"
        )

    # Polytope constructors

    @classmethod
    def unimodular_simplex(cls, dim):
        """
        Return a unimodular simplex in the given dimension
        """

        # check if dim is an integer > 0
        if not isinstance(dim, int) or dim < 1:
            raise ValueError("Dimension must be a positive integer")

        verts = [[0] * dim]
        for i in range(dim):
            vert = [0] * dim
            vert[i] = 1
            verts.append(vert)

        simplex = cls(vertices=verts)

        return simplex

    @classmethod
    def cube(cls, dim):
        """
        Return a unit cube in the given dimension
        """

        # check if dim is an integer > 0
        if not isinstance(dim, int) or dim < 1:
            raise ValueError("Dimension must be a positive integer")

        segment_verts = [[0], [1]]
        cube = cls(vertices=segment_verts)

        for _ in range(dim - 1):
            cube = cube * cls(vertices=segment_verts)

        return cube

    @classmethod
    def random_lattice_polytope(cls, dim, n_vertices, min=0, max=1):
        """
        Return a random lattice polytope in the given dimension from a given number of
        point with integer coordinates between min and max (included, default 0 and 1)
        """
        pts = np.random.randint(min, max + 1, size=(n_vertices, dim))
        return cls(points=pts)
