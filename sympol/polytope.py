import numpy as np
import cdd
from sympy import Abs, factorial, gcd, lcm, Number, Matrix, Rational
from sympy.matrices import zeros

from sympol.isomorphism import get_normal_form
from sympol.point import Point
from sympol.point_list import PointList
from sympol.lineq import LinIneq
from sympol.triangulation import get_placing_triangulation
from sympol.utils import _cdd_fraction_to_simpy_rational


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

        self._cdd_polyhedron = None

        self._vertices = vertices

        self._triangulation = None
        self._volume = None
        self._normalized_volume = None

        self._linear_inequalities = None
        self._facets = None
        self._n_vertices = None
        self._n_facets = None
        self._n_edges = None

        self._cdd_vertex_adjacency = None
        self._cdd_vertex_facet_incidence = None
        self._vertex_adjacency_matrix = None
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
    def cdd_polyhedron(self):
        """
        Get the cdd polyhedron of the polytope
        """
        if self._cdd_polyhedron is None:
            if self._vertices is not None or self._points is not None:
                self._get_cdd_polyhedron_from_points()
            elif self._linear_inequalities is not None:
                self._get_cdd_polyhedron_from_inequalities()
            else:
                raise ValueError("No points or inequalities given")

        return self._cdd_polyhedron

    @property
    def cdd_vertex_facet_incidence(self):
        """
        Get the cdd vertex facet incidence output
        """
        if self._cdd_vertex_facet_incidence is None:
            self._cdd_vertex_facet_incidence = self.cdd_polyhedron.get_incidence()

        return self._cdd_vertex_facet_incidence

    @property
    def cdd_vertex_adjacency(self):
        """
        Get the cdd vertex adjacency output
        """
        if self._cdd_vertex_adjacency is None:
            self._cdd_vertex_adjacency = self.cdd_polyhedron.get_input_adjacency()

        return self._cdd_vertex_adjacency

    @property
    def vertices(self):
        """
        Return the vertices of the polytope or calculate them if they are not
        already calculated
        """
        if self._vertices is None:
            self._get_vertices()

        return self._vertices

    def _get_vertices(self):
        """
        Calculate the vertices of the polytope
        """
        mat_gens = self.cdd_polyhedron.get_generators()
        mat_gens.canonicalize()  # remove redundant points
        self._vertices = PointList([p[1:] for p in mat_gens])

    @property
    def barycenter(self):
        """
        Get the barycenter of the polytope
        """
        return self.vertices.barycenter

    @property
    def triangulation(self):
        """
        Get the triangulation of the polytope
        """
        if self._triangulation is None:
            self._triangulation = get_placing_triangulation(self.vertices)

        return self._triangulation

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
            self._get_linear_inequalities()

        return self._linear_inequalities

    @property
    def facets(self):
        """
        Get the facets of the polytope
        """
        if self._facets is None:
            self._facets = self.cdd_vertex_facet_incidence

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
            self._n_facets = len(self.linear_inequalities)

        return self._n_facets

    @property
    def n_edges(self):
        """
        Get the number of edges of the polytope
        """
        if self._n_edges is None:
            self._n_edges = len(self.edges)

        return self._n_edges

    @property
    def vertex_adjacency_matrix(self):
        """
        Get the vertex adjacency matrix of the polytope:
            a_ij = 1 if vertex i adjacent to vertex j,
            a_ij = 0 otherwise
        """
        if self._vertex_adjacency_matrix is None:
            self._vertex_adjacency_matrix = zeros(self.n_vertices, self.n_vertices)
            for i, ads in enumerate(self.cdd_vertex_adjacency):
                for j in ads:
                    self._vertex_adjacency_matrix[i, j] = 1

        return self._vertex_adjacency_matrix

    @property
    def vertex_facet_matrix(self):
        """
        Get the vertex facet incidence matrix of the polytope:
            m_ij = 1 if vertex j is in facet i,
            m_ij = 0 otherwise
        """
        if self._vertex_facet_matrix is None:
            self._vertex_facet_matrix = zeros(self.n_facets, self.n_vertices)

            for i, facet in enumerate(self.facets):
                for j in facet:
                    self._vertex_facet_matrix[i, j] = 1

        return self._vertex_facet_matrix

    @property
    def vertex_facet_pairing_matrix(self):
        """
        Get the vertex facet pairing matrix of the polytope:
            m_ij = <F_j, v_i> (distance of vertex j to facet i)
        """
        if self._vertex_facet_pairing_matrix is None:
            self._vertex_facet_pairing_matrix = zeros(self.n_facets, self.n_vertices)
            for i, lineq in enumerate(self.linear_inequalities):
                for j, vertex in enumerate(self.vertices):
                    self._vertex_facet_pairing_matrix[i, j] = lineq.evaluate(vertex)

        return self._vertex_facet_pairing_matrix

    # property setters

    def _get_cdd_polyhedron_from_points(self):
        """
        Get the cdd polyhedron from the v-representation of the polytope
        """
        points_to_use = self._vertices if self._vertices is not None else self.points
        formatted_points = [[1] + [c for c in p] for p in points_to_use]
        mat = cdd.Matrix(formatted_points, number_type="fraction")
        mat.rep_type = cdd.RepType.GENERATOR
        self._cdd_polyhedron = cdd.Polyhedron(mat)

    def _get_cdd_polyhedron_from_inequalities(self):
        """
        Get the cdd polyhedron from the h-representation of the polytope
        """
        # TODO
        pass

    def _get_linear_inequalities(self):
        """
        Get the linear inequalities of the polytope from cdd_polyhedron
        """
        mat_ineq = self.cdd_polyhedron.get_inequalities()
        # TODO: (this might be worth not doing for certain applications)
        mat_ineq.canonicalize()  # remove redundant inequalities

        self._linear_inequalities = []

        for ineq in mat_ineq:
            # convert cdd rational to sympy rational
            ineq = [_cdd_fraction_to_simpy_rational(coeff) for coeff in ineq]

            # make the normal integer and primitive
            lcm_ineq = lcm([rat_coeff.q for rat_coeff in ineq[1:]])
            ineq = [rat_coeff * lcm_ineq for rat_coeff in ineq]

            gcd_ineq = gcd([int_coeff for int_coeff in ineq[1:]])
            ineq = [int_coeff / gcd_ineq for int_coeff in ineq]
            self._linear_inequalities.append(LinIneq(Point(ineq[1:]), -ineq[0]))

    # def _calculate_volume(self):
    #     """
    #     Calculate the volume of the polytope, sets both _volume and _normalized_volume
    #     """
    #     volume = Rational(0)

    #     for simplex_ids in self.boundary_triangulation:
    #         if 0 in simplex_ids:
    #             continue
    #         translated_simplex = [
    #             self.vertices[id] - self.vertices[0] for id in simplex_ids
    #         ]
    #         volume += Abs(Matrix(translated_simplex).det())

    #     self._normalized_volume = volume
    #     self._volume = volume / factorial(self.dim)

    @property
    def edges(self):
        """
        Get the edges of the polytope
        """

        if self._edges is None:
            edges = []
            for i, ads in enumerate(self.cdd_vertex_adjacency):
                for j in ads:
                    if i < j:
                        edges.append((i, j))
            self._edges = tuple(edges)

        return self._edges

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

    # Polytope relations

    def contains(self, other):
        """
        Check if the polytope contains a point or another polytope
        """
        if isinstance(other, Point):
            for lineq in self.linear_inequalities:
                if lineq.evaluate(other) < 0:
                    return False
            return True

        if isinstance(other, Polytope):
            pts = other._vertices if other._vertices is not None else other.points
            for p in pts:
                if not self.contains(p):
                    return False
            return True

        raise TypeError("contains() only accepts a Point or a Polytope as argument")

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
