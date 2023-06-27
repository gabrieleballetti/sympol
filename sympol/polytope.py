import numpy as np
import cdd
from scipy.spatial import Delaunay


from sympy import Abs, factorial, gcd, lcm, Number, Matrix, Rational
from sympy.abc import x
from sympy.matrices import zeros
from sympy.matrices.normalforms import hermite_normal_form
from sympy.polys.polyfuncs import interpolate

from sympol.integer_points import _find_integer_points
from sympol.isomorphism import get_normal_form
from sympol.point import Point
from sympol.point_list import PointList
from sympol.lineq import LinIneq
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

        self._linear_inequalities = None

        self._vertices = vertices
        self._facets = None
        self._ridges = None
        self._edges = None

        self._n_vertices = None
        self._n_edges = None
        self._n_ridges = None
        self._n_facets = None

        self._faces = None
        self._f_vector = None

        self._cdd_polyhedron = None
        self._cdd_inequalities = None
        self._cdd_equality_ids = None
        self._cdd_vertex_adjacency = None
        self._cdd_facet_adjacency = None
        self._cdd_vertex_facet_incidence = None

        self._vertex_adjacency_matrix = None
        self._vertex_facet_matrix = None
        self._vertex_facet_pairing_matrix = None

        self._triangulation = None
        self._volume = None
        self._normalized_volume = None

        self._integer_points_raw = None
        self._integer_points = None
        self._interior_points = None
        self._boundary_points = None
        self._n_integer_points = None
        self._n_interior_points = None
        self._n_boundary_points = None

        self._ehrhart_polynomial = None

        self._full_dim_projection = None
        self._normal_form = None
        self._affine_normal_form = None

        # Simplex specific attributes
        self._weights = None

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

            # check if it is a simplex, but only if vertices are
            # already calculated
            if self._vertices is not None and self.is_simplex():
                self._make_simplex()

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
    def cdd_inequalities(self):
        """
        Get the cdd inequalities of the polytope
        """
        if self._cdd_inequalities is None:
            self._cdd_inequalities = self.cdd_polyhedron.get_inequalities()

            # check if we have redunancies
            r1, r2 = self._cdd_inequalities.copy().canonicalize()
            assert r1 == (frozenset({}))
            assert r2 == frozenset({})

        return self._cdd_inequalities

    @property
    def cdd_equality_ids(self):
        """
        Get the indices of the cdd inequalities that are actually equalities
        """
        if self._cdd_equality_ids is None:
            self._cdd_equality_ids = self.cdd_inequalities.lin_set

        return self._cdd_equality_ids

    @property
    def cdd_vertex_facet_incidence(self):
        """
        Get the cdd vertex facet incidence output
        """
        if self._cdd_vertex_facet_incidence is None:
            # make sure vertices and inequalities are calculated (and simplified)
            _ = self.vertices
            _ = self.cdd_inequalities
            self._cdd_vertex_facet_incidence = self.cdd_polyhedron.get_incidence()

        return self._cdd_vertex_facet_incidence

    @property
    def cdd_vertex_adjacency(self):
        """
        Get the cdd vertex adjacency output
        """
        if self._cdd_vertex_adjacency is None:
            # make sure vertices are calculated (and simplified)
            _ = self.vertices
            self._cdd_vertex_adjacency = self.cdd_polyhedron.get_input_adjacency()

        return self._cdd_vertex_adjacency

    @property
    def cdd_facet_adjacency(self):
        """
        Get the cdd facet adjacency
        """
        if self._cdd_facet_adjacency is None:
            # make sure inequalities are calculated (and simplified)
            _ = self.cdd_inequalities
            self._cdd_facet_adjacency = self.cdd_polyhedron.get_adjacency()

        return self._cdd_facet_adjacency

    @property
    def vertices(self):
        """
        Return the vertices of the polytope or calculate them if they are not
        already calculated
        """
        if self._vertices is None:
            mat_gens = self.cdd_polyhedron.get_generators()

            # remove redundant generators and update the cdd polyhedron
            mat_gens.canonicalize()
            self._cdd_polyhedron = cdd.Polyhedron(mat_gens)

            self._vertices = PointList([p[1:] for p in mat_gens])

            # check if the polytope is a simplex
            if self.is_simplex():
                self._make_simplex()

        return self._vertices

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
        Get the facets of the polytope (dim(P)-1 dimensional faces)
        """
        if self._facets is None:
            self._facets = tuple(
                [
                    f
                    for i, f in enumerate(self.cdd_vertex_facet_incidence)
                    if i not in self.cdd_equality_ids
                ]
            )

        return self._facets

    @property
    def ridges(self):
        """
        Get the ridges of the polytope (d-2 dimensional faces)
        """
        if self._ridges is None:
            self._ridges = []
            for i, ads in enumerate(self.cdd_facet_adjacency):
                for j in ads:
                    if i < j:
                        self._ridges.append(self.facets[i].intersection(self.facets[j]))
            self._ridges = tuple(self._ridges)

        return self._ridges

    @property
    def edges(self):
        """
        Get the edges of the polytope (1 dimensional faces)
        """

        if self._edges is None:
            self._edges = []
            for i, ads in enumerate(self.cdd_vertex_adjacency):
                for j in ads:
                    if i < j:
                        self._edges.append(frozenset((i, j)))
            # sort the edges and make a tuple
            self._edges = tuple(self._edges)

        return self._edges

    @property
    def all_faces(self):
        """
        Get all the faces of the polytope. If only the faces of a certain dimension
        are needed, use the faces(dim) method instead to avoid unnecessary computations.
        """
        if self._faces is None:
            for d in range(-1, self.dim + 1):
                # trigger the calculation of the faces
                self.faces(d)

        return self._faces

    def faces(self, dim):
        """
        Get the faces of the polytope of a given dimension. Faces of dimension and
        codimension lower ore equal to two are found from the cdd polyhedron. Other
        faces are found from higher dimensional faces, via intersection with facets.
        TODO: This could be done more efficiently especially as low dimensional faces
        need all the higher dimensional faces to be calculated first.
        """
        if dim < -1 or dim > self.dim:
            raise ValueError(
                "The dimension of the face should be between -1 and the dimension of"
                " the polytope"
            )

        if self._faces is None:
            self._faces = {i: None for i in range(-1, self.dim + 1)}

        if self._faces[dim] is None:
            if dim == -1:
                self._faces[dim] = tuple([frozenset()])
            elif dim == 0:
                self._faces[dim] = tuple(
                    [frozenset([i]) for i in range(self.n_vertices)]
                )
            elif dim == 1:
                self._faces[dim] = self.edges
            elif dim == self.dim - 2:
                self._faces[dim] = self.ridges
            elif dim == self.dim - 1:
                self._faces[dim] = self.facets
            elif dim == self.dim:
                self._faces[dim] = (frozenset(range(self.n_vertices)),)
            else:
                new_faces = []
                for facet in self.facets:
                    for face in self.faces(dim + 1):
                        if face.issubset(facet):
                            continue
                        f = face.intersection(facet)
                        add_f = True
                        for f2 in new_faces:
                            if f.issubset(f2):
                                add_f = False
                                break
                            if f2.issubset(f):
                                new_faces.remove(f2)
                        if add_f:
                            new_faces.append(f)
                self._faces[dim] = tuple(new_faces)

        return self._faces[dim]

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
    def n_ridges(self):
        """
        Get the number of ridges of the polytope
        """
        if self._n_ridges is None:
            self._n_ridges = len(self.ridges)

        return self._n_ridges

    @property
    def n_edges(self):
        """
        Get the number of edges of the polytope
        """
        if self._n_edges is None:
            self._n_edges = len(self.edges)

        return self._n_edges

    @property
    def f_vector(self):
        """
        Get the f-vector of the polytope
        """
        if self._f_vector is None:
            self._f_vector = tuple(
                [len(self.faces(dim)) for dim in range(-1, self.dim + 1)]
            )

        return self._f_vector

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
            for i, lineq in enumerate(
                [h for h in self.linear_inequalities if not h.is_equality]
            ):
                for j, vertex in enumerate(self.vertices):
                    self._vertex_facet_pairing_matrix[i, j] = lineq.evaluate(vertex)

        return self._vertex_facet_pairing_matrix

    @property
    def barycenter(self):
        """
        Get the barycenter of the polytope
        """
        return self.vertices.barycenter

    @property
    def triangulation(self):
        """
        Get the triangulation of the polytope (uses scipy.spatial.Delaunay)
        NOTE: scipy.spatial.Delaunay uses Qhull, which is float based!
        """
        if self._triangulation is None:
            # if the polytope is not full-dimensional, we need to project it
            # to a full-dimensional subspace
            if self.is_full_dim():
                delaunay_triangulation = Delaunay(np.array(self.vertices))
                self._triangulation = tuple(
                    [
                        frozenset([int(i) for i in s])
                        for s in delaunay_triangulation.simplices
                    ]
                )
            else:
                self._triangulation = self.full_dim_projection.triangulation

        return self._triangulation

    @property
    def volume(self):
        """
        Get the normalized volume of the polytope
        """
        if self._volume is None:
            if self.is_full_dim():
                self._calculate_volume()
            else:
                self._volume = self.full_dim_projection.volume
        return self._volume

    @property
    def normalized_volume(self):
        """
        Get the normalized volume of the polytope
        """
        if self._normalized_volume is None:
            if self.is_full_dim():
                self.calculate_volume()
            else:
                self._normalized_volume = self.full_dim_projection.normalized_volume
        return self._normalized_volume

    @property
    def full_dim_projection(self):
        """
        An affine unimodular copy of the polytope in a lower dimensional space
        """
        if self._full_dim_projection is None:
            m = Matrix(self.vertices - self.vertices[0])
            hnf = hermite_normal_form(m)
            self._full_dim_projection = Polytope(hnf)

        return self._full_dim_projection

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

    @property
    def integer_points_raw(self):
        """
        Get the raw output of the integer points function
        """
        if self._integer_points_raw is None:
            self._integer_points_raw = _find_integer_points(polytope=self)

        return self._integer_points_raw

    @property
    def integer_points(self):
        """
        Get the integer points of the polytope
        """
        if self._integer_points is None:
            self._integer_points = PointList([pt[0] for pt in self.integer_points_raw])

        return self._integer_points

    @property
    def interior_points(self):
        """
        Get the interior integer points of the polytope
        """
        if self._interior_points is None:
            int_pts = []
            for pt in self.integer_points_raw:
                if pt[1] != frozenset({}):
                    break
                else:
                    int_pts.append(pt[0])
            self._interior_points = PointList(int_pts)

        return self._interior_points

    @property
    def boundary_points(self):
        """
        Get the boundary integer points of the polytope
        """
        if self._boundary_points is None:
            int_pts = []
            for pt in self.integer_points_raw:
                if pt[1] == frozenset({}):
                    continue
                int_pts.append(pt[0])
            self._boundary_points = PointList(int_pts)

        return self._boundary_points

    @property
    def n_integer_points(self):
        """
        Get the number of integer points of the polytope
        """
        if self._n_integer_points is None:
            self._n_integer_points = self.integer_points.shape[0]

        return self._n_integer_points

    @property
    def n_interior_points(self):
        """
        Get the number of interior integer points of the polytope
        """
        if self._n_interior_points is None:
            self._n_interior_points = self.interior_points.shape[0]

        return self._n_interior_points

    @property
    def n_boundary_points(self):
        """
        Get the number of boundary integer points of the polytope
        """
        if self._n_boundary_points is None:
            self._n_boundary_points = self.boundary_points.shape[0]

        return self._n_boundary_points

    @property
    def ehrhart_polynomial(self):
        """
        Get the Ehrhart polynomial of the polytope
        """
        if self._ehrhart_polynomial is None:
            if not self.is_lattice_polytope:
                raise ValueError(
                    "Ehrhart polynomial is only defined for lattice polytopes"
                )

            data = {
                0: 1,
                1: self.n_integer_points,
                -1: (-1) ** self.dim * self.n_interior_points,
            }

            for k in range(2, (self.dim + 1) // 2 + 1):
                dilation = self * k
                data[k] = dilation.n_integer_points
                data[-k] = (-1) ** (self.dim) * dilation.n_interior_points
            self._ehrhart_polynomial = interpolate(data, x)

        return self._ehrhart_polynomial

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

        self._linear_inequalities = []

        for i, ineq in enumerate(self.cdd_inequalities):
            # convert cdd rational to sympy rational
            ineq = [_cdd_fraction_to_simpy_rational(coeff) for coeff in ineq]

            # make the normal integer and primitive
            lcm_ineq = lcm([rat_coeff.q for rat_coeff in ineq[1:]])
            ineq = [rat_coeff * lcm_ineq for rat_coeff in ineq]

            gcd_ineq = gcd([int_coeff for int_coeff in ineq[1:]])
            ineq = [int_coeff / gcd_ineq for int_coeff in ineq]
            self._linear_inequalities.append(
                LinIneq(
                    normal=Point(ineq[1:]),
                    rhs=-ineq[0],
                    is_equality=i in self.cdd_equality_ids,
                )
            )

    def _calculate_volume(self):
        """
        Calculate the volume of the polytope, sets both _volume and _normalized_volume
        """
        volume = Rational(0)

        for simplex_ids in self.triangulation:
            verts = list(simplex_ids)
            translated_simplex = [
                self.vertices[id] - self.vertices[verts[0]] for id in verts[1:]
            ]
            volume += Abs(Matrix(translated_simplex).det())

        self._normalized_volume = volume
        self._volume = volume / factorial(self.dim)

    # Helper functions

    def is_full_dim(self):
        """
        Check if the polytope is full dimensional
        """
        return self.dim == self.ambient_dim

    def is_simplex(self):
        """
        Check if the polytope is a simplex
        """
        return self.n_vertices == self.dim + 1

    def _make_simplex(self):
        """
        Make the polytope a simplex
        """
        if not self.is_simplex():
            raise ValueError("Polytope is not a simplex")
        self.__class__ = Simplex

    def is_lattice_polytope(self):
        """
        Check if the polytope is a lattice polytope
        """
        return all([all([i.is_integer for i in v]) for v in self.vertices])

    def neighbors(self, vertex_id):
        """
        Get the neighbors of a vertex
        """
        return self.cdd_vertex_adjacency[vertex_id]

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
                if lineq.is_equality and lineq.evaluate(other) != 0:
                    return False
                elif lineq.evaluate(other) < 0:
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

        simplex = Simplex(vertices=verts)

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


class Simplex(Polytope):
    """
    Simplex class
    """

    @property
    def triangulation(self):
        """
        Triangulation of the simplex
        """
        if self._triangulation is None:
            self._triangulation = tuple([frozenset({i for i in range(self.dim + 1)})])

        return self._triangulation

    def barycentric_coordinates(self, point: Point):
        """
        Return the barycentric coordinates of a point in the simplex, or the
        barycentric coordinates of the origin if no point is given
        """
        m = Matrix([[1] + v.tolist() for v in self.vertices]).T
        b = m.LUsolve(Matrix([1] + point.tolist()))

        return b.flat()

    @property
    def weights(self):
        """
        Return the weights of the simplex, i.e. the barycentric coordinates of the
        origin given as integers
        """
        if self._weights is None:
            b = self.barycentric_coordinates(Point([0] * self.dim))

            lcm_b = lcm([frac.q for frac in b])
            b = [frac * lcm_b for frac in b]
            gcd_b = gcd(b)
            self._weights = [i / gcd_b for i in b]

        return self._weights
