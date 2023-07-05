import numpy as np
import cdd

from scipy.spatial import Delaunay


from sympy import Abs, factorial, gcd, lcm, Number, Matrix, Poly, Rational
from sympy.abc import x
from sympy.matrices import zeros
from sympy.matrices.normalforms import hermite_normal_form
from sympy.polys.polyfuncs import interpolate
from sympol.integer_points import _find_integer_points
from sympol.isomorphism import get_normal_form
from sympol.point import Point
from sympol.point_list import PointList
from sympol.lineq import LinIneq
from sympol.utils import _cdd_fraction_to_simpy_rational, _eulerian_poly, is_unimodal


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
        self._induced_boundary_triangulation = None
        self._volume = None
        self._normalized_volume = None
        self._boundary_volume = None
        self._normalized_boundary_volume = None

        self._integer_points = None
        self._interior_points = None
        self._boundary_points = None
        self._boundary_points_facets = None
        self._n_integer_points = None
        self._n_interior_points = None
        self._n_boundary_points = None

        self._ehrhart_polynomial = None
        self._ehrhart_coefficients = None
        self._h_star_polynomial = None
        self._h_star_vector = None

        self._full_dim_projection = None
        self._normal_form = None
        self._affine_normal_form = None

        self._is_simplicial = None
        self._is_simple = None
        self._is_lattice_polytope = None
        self._is_hollow = None
        self._has_one_interior_point = None
        self._is_canonical = None
        self._is_reflexive = None
        self._is_ehrhart_positive = None
        self._has_unimodal_h_star_vector = None
        self._is_idp = None
        self._is_smooth = None

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
        NOTE: scipy.spatial.Delaunay uses Qhull, which is float based, use with care
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
    def induced_boundary_triangulation(self):
        """
        Get the triangulation of the boundary of the polytope induced by the
        triangulation of the polytope
        """
        if self._induced_boundary_triangulation is None:
            self._induced_boundary_triangulation = tuple(
                [
                    ss
                    for s in self.triangulation
                    for f in self.facets
                    if len(ss := f.intersection(s)) == self.dim
                ]
            )

        return self._induced_boundary_triangulation

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
                self._calculate_volume()
            else:
                self._normalized_volume = self.full_dim_projection.normalized_volume
        return self._normalized_volume

    @property
    def boundary_volume(self):
        """
        Get the normalized volume of the boundary of the polytope
        """
        if self._boundary_volume is None:
            self._calculate_boundary_volume()
        return self._boundary_volume

    @property
    def normalized_boundary_volume(self):
        """
        Get the normalized volume of the boundary of the polytope
        """
        if self._normalized_boundary_volume is None:
            self._calculate_boundary_volume()
        return self._normalized_boundary_volume

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
    def integer_points(self):
        """
        Get the integer points of the polytope
        """
        if self._integer_points is None:
            _integer_points = [pt for pt in self.boundary_points]
            if self.n_interior_points > 0:
                _integer_points += [pt for pt in self.interior_points]
            self._integer_points = PointList(_integer_points)
        return self._integer_points

    @property
    def interior_points(self):
        """
        Get the interior integer points of the polytope
        """
        if self._interior_points is None:
            self._get_integer_points()

        return self._interior_points

    @property
    def boundary_points(self):
        """
        Get the boundary integer points of the polytope
        """
        if self._boundary_points is None:
            self._get_integer_points()

        return self._boundary_points

    @property
    def boundary_points_facets(self):
        """
        Get the ids facets of the polytope that contain boundary integer points
        """
        if self._boundary_points_facets is None:
            self._get_integer_points()

        return self._boundary_points_facets

    @property
    def n_integer_points(self):
        """
        Get the number of integer points of the polytope
        """
        if self._n_integer_points is None:
            self._get_integer_points(count_only=True)

        return self._n_integer_points

    @property
    def n_interior_points(self):
        """
        Get the number of interior integer points of the polytope
        """
        if self._n_interior_points is None:
            self._get_integer_points(count_only=True)

        return self._n_interior_points

    @property
    def n_boundary_points(self):
        """
        Get the number of boundary integer points of the polytope
        """
        if self._n_boundary_points is None:
            self._n_boundary_points = self.n_integer_points - self.n_interior_points

        return self._n_boundary_points

    @property
    def ehrhart_polynomial(self):
        """
        Get the Ehrhart polynomial of the polytope
        """
        if self._ehrhart_polynomial is None:
            if not self.is_lattice_polytope:
                raise ValueError(
                    "Ehrhart polynomial is only implemented for lattice polytopes"
                )

            self._ehrhart_polynomial = Poly(1, x, domain="QQ")

            if self.dim == 0:
                return self._ehrhart_polynomial

            # manually add the coefficient c_d
            self._ehrhart_polynomial += self.volume * x**self.dim

            if self.dim == 1:
                return self._ehrhart_polynomial

            # manually add the coefficient c_{d-1}
            # (it is generally faster than determining it through interpolation)
            self._ehrhart_polynomial += (
                Rational(1, 2) * self.boundary_volume * x ** (self.dim - 1)
            )

            if self.dim == 2:
                return self._ehrhart_polynomial

            values_needed = self.dim - 2
            values = {0: 0}
            while values_needed > 0:
                k = (self.dim - values_needed) // 2
                dilation = self * k if k > 1 else self

                # there's no additional computation to count also interior points
                # so we can add them both at the same time
                values[k] = dilation.n_integer_points - self._ehrhart_polynomial.eval(k)
                values[-k] = (-1) ** (
                    self.dim
                ) * dilation.n_interior_points - self._ehrhart_polynomial.eval(-k)
                values_needed -= 2

            self._ehrhart_polynomial += interpolate(values, x)

        return self._ehrhart_polynomial

    @property
    def ehrhart_coefficients(self):
        """
        Get the Ehrhart coefficients of the polytope
        """
        if self._ehrhart_coefficients is None:
            self._ehrhart_coefficients = tuple(self.ehrhart_polynomial.coeffs()[::-1])

        return self._ehrhart_coefficients

    @property
    def h_star_polynomial(self):
        """
        Get the h*-polynomial of the polytope
        """
        if self._h_star_polynomial is None:
            if not self.is_lattice_polytope:
                raise ValueError(
                    "h*-polynomial is only implemented for lattice polytopes"
                )

            self._h_star_polynomial = sum(
                [
                    self.ehrhart_coefficients[i]
                    * _eulerian_poly(i, x)
                    * (1 - x) ** (self.dim - i)
                    for i in range(self.dim + 1)
                ]
            ).simplify()

        return self._h_star_polynomial

    @property
    def h_star_vector(self):
        """
        Get the h*-vector of the polytope
        """
        if self._h_star_vector is None:
            self._h_star_vector = tuple(
                [1]
                + [
                    self.h_star_polynomial.coeff_monomial(x**d)
                    for d in range(1, self.dim + 1)
                ]
            )

        return self._h_star_vector

    @property
    def is_simplicial(self):
        """
        Check if the polytope is simplicial, i.e. if all its facets are simplices.
        """
        if self._is_simplicial is None:
            self._is_simplicial = all([len(f) == self.dim for f in self.facets])

        return self._is_simplicial

    @property
    def is_simple(self):
        """
        Check if the polytope is simple, i.e. if each vertex is contained in exactly
        d edges, where d is the dimension of the polytope.
        """
        if self._is_simple is None:
            self._is_simple = all(
                [
                    len(self.neighbors(v_id)) == self.dim
                    for v_id in range(self.n_vertices)
                ]
            )

        return self._is_simple

    @property
    def is_lattice_polytope(self):
        """
        Check if the polytope is a lattice polytope
        """
        if self._is_lattice_polytope is None:
            self._is_lattice_polytope = all(
                [all([i.is_integer for i in v]) for v in self.vertices]
            )

        return self._is_lattice_polytope

    @property
    def is_hollow(self):
        """
        Check if the polytope is hollow, i.e. if it has no interior points
        """

        if self._is_hollow is None:
            self._is_hollow = self.has_n_interior_points(0)

        return self._is_hollow

    @property
    def has_one_interior_point(self):
        """
        Check if the polytope has exactly one interior point
        """
        if self._has_one_interior_point is None:
            self._has_one_interior_point = self.has_n_interior_points(1)

        return self._has_one_interior_point

    @property
    def is_canonical(self):
        """
        Check if the polytope is canonical Fano, i.e. if it is a lattice polytope with
        the origin as unique interior point
        """
        if self._is_canonical is None:
            self._is_canonical = (
                self.is_lattice_polytope
                and self.has_one_interior_point
                and self.contains(self._origin(), strict=True)
            )

        return self._is_canonical

    @property
    def is_reflexive(self):
        """
        Check if the polytope is reflexive
        """
        if self._is_reflexive is None:
            self._is_reflexive = self.is_canonical and all(
                le.evaluate(self._origin()) == 1
                for le in self.linear_inequalities
                if not le.is_equality
            )

        return self._is_reflexive

    @property
    def is_ehrhart_positive(self):
        """
        Check if the polytope is Ehrhart positive, i.e. if its Ehrhart polynomial has
        only positive coefficients
        """
        if self._is_ehrhart_positive is None:
            self._is_ehrhart_positive = all([i >= 0 for i in self.ehrhart_coefficients])

        return self._is_ehrhart_positive

    @property
    def has_unimodal_h_star_vector(self):
        """
        Check if the polytope has a unimodal h* vector
        """
        if self._has_unimodal_h_star_vector is None:
            self._has_unimodal_h_star_vector = is_unimodal(self.h_star_vector)

        return self._has_unimodal_h_star_vector

    @property
    def is_idp(self):
        """
        Check if the polytope P has the Integer Decomposition Property (IDP), i.e. if
        if for every >= 2 and for every lattice point in kP there exist v1, . . . , vk
        lattice points in P such that u = v1 + · · · + vk. A polytope P with this
        property is also called integrally closed.
        """
        if self._is_idp is None:
            self._is_idp = True
            pts = self.integer_points
            for d in range(2, self.dim):
                n_pts = self.ehrhart_polynomial.eval(d)
                gen_pts = set(p1 + p2 for p1 in pts for p2 in self.integer_points)
                if len(gen_pts) != n_pts:
                    self._is_idp = False
                    break

        return self._is_idp

    @property
    def is_smooth(self):
        """
        Check if the polytope is smooth, i.e. if it is simple and the tangent cone at
        each vertex is unimodular.
        """
        if self._is_smooth is None:
            if not self.is_simple:
                self._is_smooth = False
                return False

            for v_id in range(self.n_vertices):
                v = self.vertices[v_id]
                mat = Matrix([self.vertices[u_id] - v for u_id in self.neighbors(v_id)])
                if not Abs(mat.det()) == 1:
                    self._is_smooth = False
                    return False
            self._is_smooth = True

        return self._is_smooth

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
            ineq = [rat_coeff * Abs(lcm_ineq) for rat_coeff in ineq]

            gcd_ineq = gcd([int_coeff for int_coeff in ineq[1:]])
            ineq = [int_coeff / Abs(gcd_ineq) for int_coeff in ineq]
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

        if self.ambient_dim == 1:
            volume += Abs(self.vertices[1][0] - self.vertices[0][0])
            self._normalized_volume = volume
            self._volume = volume
            return

        for simplex_ids in self.triangulation:
            verts = list(simplex_ids)
            translated_simplex = [
                self.vertices[id] - self.vertices[verts[0]] for id in verts[1:]
            ]
            volume += Abs(Matrix(translated_simplex).det())

        self._normalized_volume = volume
        self._volume = volume / factorial(self.dim)

    def _calculate_boundary_volume(self):
        """
        Calculate the volume of the boundary of the polytope, i.e. the sum of the
        volumes of the facets of the polytopes, wrt their affinely spanned lattice.
        Sets both _boundary_volume and _normalized_boundary_volume
        """
        boundary_norm_volume = Rational(0)

        # for facet in self.facets:
        #     facet_polytope = Polytope([self.vertices[i] for i in facet])
        #     boundary_norm_volume += facet_polytope.normalized_volume

        for s in self.induced_boundary_triangulation:
            simplex = Simplex(vertices=[self.vertices[i] for i in s])
            boundary_norm_volume += simplex.normalized_volume

        self._normalized_boundary_volume = boundary_norm_volume
        self._boundary_volume = boundary_norm_volume / factorial(self.dim - 1)

    def _get_integer_points(self, count_only=False, stop_at_interior=-1):
        """
        Get the integer points, or optionally just the count, and populate the
        correct properties
        """
        if not self.is_full_dim():
            raise ValueError("polytope must be full-dimensional")
        (
            _interior_points,
            _boundary_points,
            _boundary_points_facets,
            _n_integer_points,
            _n_interior_points,
            forced_stop,
        ) = _find_integer_points(
            verts=self._verts_as_np_array(),
            ineqs=self._ineqs_as_np_array(),
            dim=self.dim,
            count_only=count_only,
            stop_at_interior=stop_at_interior,
        )

        if forced_stop:
            # enumeration has been interrupted, do not populate any property
            return

        if not count_only:
            self._interior_points = PointList(_interior_points)
            self._boundary_points = PointList(_boundary_points)
        self._boundary_points_facets = _boundary_points_facets
        self._n_integer_points = _n_integer_points
        self._n_interior_points = _n_interior_points

    # Helper functions

    def _origin(self):
        """
        Return the origin of the ambient space
        """
        return Point([0 for _ in range(self.ambient_dim)])

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

    def neighbors(self, vertex_id):
        """
        Get the neighbors of a vertex
        """
        return self.cdd_vertex_adjacency[vertex_id]

    def _verts_as_np_array(self):
        """
        Return the vertices of the polytope as a numpy array
        """
        return np.array(self.vertices, dtype=np.int64)

    def _ineqs_as_np_array(self):
        """
        Return the linear inequalities of the polytope as a numpy array
        """
        return np.array(
            [ineq.normal.tolist() + [-ineq.rhs] for ineq in self.linear_inequalities],
            dtype=np.int64,
        )

    def has_n_interior_points(self, n):
        """
        Check if the polytope has *exactly* n interior points
        """
        if self._n_interior_points is None:
            self._get_integer_points(count_only=True, stop_at_interior=n + 1)

        # use the "private" _n_interior_points as _get_integer_points might
        # have been interrupted in case more than n interior points were found
        # (in that case _n_interior_points would be None and we do not need to
        # know the exact number)
        return self._n_interior_points == n

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

    def free_sum(self, other):
        """
        Return the free sum of self and other.
        """
        if not isinstance(other, Polytope):
            raise TypeError("The free sum is only defined for polytopes")

        verts = [v.tolist() + [0] * other.ambient_dim for v in self.vertices]
        verts += [[0] * self.ambient_dim + v.tolist() for v in other.vertices]

        # make set to quickly remove duplicates (can only be origin)
        verts = set(map(tuple, verts))

        return Polytope(vertices=verts)

    def chisel_vertex(self, vertex_id, dist):
        """
        Return a new polytope obtained by "chiseling" a vertex at a given lattice
        distance along its neighboring edges.
        """
        if dist < 0:
            raise ValueError("dist must be positive")
        if dist == 0:
            return self

        new_verts = [u for u_id, u in enumerate(self.vertices) if u_id != vertex_id]
        v = self.vertices[vertex_id]
        recompute = False
        neighbors = self.neighbors(vertex_id)
        if len(neighbors) > self.dim:
            # if P is not simple we might need to recompute the vertices
            recompute = True
        for u_id in neighbors:
            u = self.vertices[u_id]
            vs_dir = u - v
            vs_gcd = gcd(vs_dir)
            if dist >= vs_gcd:
                raise ValueError("Trying to chisel too greedily and too deep.")
            w = vs_dir / vs_gcd
            new_verts.append(v + w * dist)

        return Polytope(new_verts) if recompute else Polytope(vertices=new_verts)

    def chisel(self, dist):
        """
        Return a new polytope obtained by "chiseling" all the vertices at a given
        lattice distance along their neighboring edges.
        """
        # NOTE: code repetition with the previous method is intentional for
        # performance reasons
        if dist < 0:
            raise ValueError("dist must be positive")
        if dist == 0:
            return self

        new_verts = set()
        recompute = False
        for v_id in range(self.n_vertices):
            v = self.vertices[v_id]
            neighbors = self.neighbors(v_id)
            if len(neighbors) > self.dim:
                # if P is not simple we might need to recompute the vertices
                recompute = True
            for u_id in neighbors:
                u = self.vertices[u_id]
                vs_dir = u - v
                vs_gcd = gcd(vs_dir)
                if dist > vs_gcd // 2:
                    raise ValueError("Trying to chisel too greedily and too deep.")
                w = vs_dir / vs_gcd
                new_verts.add(v + w * dist)

        return Polytope(new_verts) if recompute else Polytope(vertices=new_verts)

    # Polytope relations

    def contains(self, other, strict=False):
        """
        Check if the polytope contains a point or another polytope, optionally
        only checking the ralative interior
        """
        if isinstance(other, Point):
            for lineq in self.linear_inequalities:
                if lineq.is_equality:
                    if lineq.evaluate(other) != 0:
                        return False
                else:
                    if not strict and lineq.evaluate(other) < 0:
                        return False
                    if strict and lineq.evaluate(other) <= 0:
                        return False
            return True

        if isinstance(other, Polytope):
            pts = other._vertices if other._vertices is not None else other.points
            for p in pts:
                if not self.contains(p, strict=strict):
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

        simplex = Simplex(verts)

        return simplex

    @classmethod
    def cube(cls, dim):
        """
        Return a unit hypercube in the given dimension
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
    def cross_polytope(cls, dim):
        """
        Return a cross polytope centered in 0 in the given dimension
        """

        # check if dim is an integer > 0
        if not isinstance(dim, int) or dim < 1:
            raise ValueError("Dimension must be a positive integer")

        segment_verts = [[-1], [1]]
        cross_polytope = cls(vertices=segment_verts)

        for _ in range(dim - 1):
            cross_polytope = cross_polytope.free_sum(cls(vertices=segment_verts))

        return cross_polytope

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
            b = self.barycentric_coordinates(self._origin())

            lcm_b = lcm([frac.q for frac in b])
            b = [frac * lcm_b for frac in b]
            gcd_b = gcd(b)
            self._weights = [i / gcd_b for i in b]

        return self._weights
