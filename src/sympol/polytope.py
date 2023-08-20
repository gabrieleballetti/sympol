"""Module for the Polytope class."""

from enum import Enum
import numpy as np
import cdd

from sympy import Abs, factorial, gcd, lcm, Number, Matrix, Poly, Rational
from sympy.abc import x
from sympy.matrices.normalforms import hermite_normal_form

from sympol._hilbert_basis_np import get_hilbert_basis_np
from sympol._integer_points_np import find_integer_points
from sympol._isomorphism import get_normal_form
from sympol._half_open_parallelotope import HalfOpenParallelotope
from sympol.point import Point
from sympol.point_configuration import PointConfiguration
from sympol._utils import (
    _binomial_polynomial,
    _cdd_fraction_to_simpy_rational,
    _eulerian_poly,
    _np_cartesian_product,
    _is_log_concave,
    _is_unimodal,
)


class _PolytopeRepresentation(Enum):
    """An enum for representing the different ways a polytope can be defined."""

    UNKNOWN = 0
    V_REPRESENTATION = 1
    H_REPRESENTATION = 2


class Polytope:
    """A class for representing the convex hull of a finite set of points.

    A polytope can be either defined by a list of points (V-representation) or by
    a list of inequalities or hyperplanes (H-representation). The lists can be
    redundant, an a irreduntant representation is calculated automatically if needed.

    Example usage:

    .. code-block:: python

        from sympol import Polytope

        p = Polytope([[-1, -1], [-1, 1], [1, -1], [1, 1], [0, 0]])
        p.vertices
        # PointConfiguration([[-1, -1],
        #             [-1, 1],
        #             [1, -1],
        #             [1, 1]], dtype=object)

        p.volume
        # 4

        p.f_vector
        # (1, 4, 4, 1)
    """

    def __init__(
        self,
        points: list = None,
        vertices: list = None,
        inequalities: list = None,
        equalities: list = None,
    ):
        """Initialize a Polytope object."""

        self._repr = _PolytopeRepresentation.UNKNOWN

        if points is not None or vertices is not None:
            self._repr = _PolytopeRepresentation.V_REPRESENTATION
            if inequalities is not None or equalities is not None:
                raise ValueError(
                    "Cannot initialize a polytope with both points and inequalities."
                )
        elif inequalities is not None or equalities is not None:
            self._repr = _PolytopeRepresentation.H_REPRESENTATION
        else:
            raise ValueError(
                "A polytope needs to be initialized with either points"
                " (V-representation) or inequalities (H-representation)."
            )

        self._points = None
        self._vertices = None
        self._inequalities = None
        self._equalities = None

        if self._repr == _PolytopeRepresentation.V_REPRESENTATION:
            if points is not None and vertices is not None:
                raise ValueError(
                    "Cannot initialize a polytope with both points and vertices."
                )
            elif vertices is not None:
                self._vertices = PointConfiguration(vertices)
                self._points = self._vertices
            elif points is not None:
                self._points = PointConfiguration(points)

        elif self._repr == _PolytopeRepresentation.H_REPRESENTATION:
            if inequalities is None:
                raise ValueError("Cannot initialize a polytope with only equalities.")

            if inequalities is not None:
                self._inequalities = np.array(inequalities)
            if equalities is not None:
                self._equalities = np.array(equalities)

        self._ambient_dim = None
        self._dim = None

        self._is_empty_set = None
        self._is_bounded = None

        self._homogeneous_inequalities = None
        self._n_inequalities = None
        self._n_equalities = None

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

        self._vertex_adjacency_matrix = None
        self._facet_adjacency_matrix = None
        self._vertex_facet_matrix = None

        self._vertex_facet_pairing_matrix = None

        self._triangulation = None
        self._half_open_decomposition = None
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
        self._degree = None

        self._half_open_parallelotopes_pts = None
        self._hilbert_basis = None

        self._full_dim_projection = None
        self._normal_form = None
        self._affine_normal_form = None

        self._is_simplicial = None
        self._is_simple = None
        self._is_lattice_polytope = None
        self._is_lattice_pyramid = None
        self._is_hollow = None
        self._has_one_interior_point = None
        self._is_canonical = None
        self._is_reflexive = None
        self._is_gorenstein = None
        self._is_ehrhart_positive = None
        self._has_log_concave_h_star_vector = None
        self._has_unimodal_h_star_vector = None
        self._is_idp = None
        self._is_smooth = None

        # Simplex specific attributes
        self._weights = None

    @property
    def points(self) -> PointConfiguration:
        """Get the (possibly redundant) defining points of the polytope.

        Returns:
            The defining points of the polytope.
        """

        if self._points is None:
            self._points = self.vertices

        return self._points

    @property
    def ambient_dim(self) -> int:
        """Get the ambient dimension of the polytope.

        Returns:
            The ambient dimension of the polytope.
        """
        if self._ambient_dim is None:
            if self._repr == _PolytopeRepresentation.V_REPRESENTATION:
                self._ambient_dim = self.points.shape[1]
            elif self._repr == _PolytopeRepresentation.H_REPRESENTATION:
                self._ambient_dim = self.inequalities.shape[1] - 1

        return self._ambient_dim

    @property
    def dim(self) -> int:
        """Get the dimension of the polytope.

        Returns:
            The dimension of the polytope.
        """
        if self._dim is None:
            if self._repr == _PolytopeRepresentation.V_REPRESENTATION:
                self._dim = self.points.affine_rank
            elif self._repr == _PolytopeRepresentation.H_REPRESENTATION:
                self._dim = self.ambient_dim - self.n_equalities

        return self._dim

    @property
    def is_empty_set(self) -> bool:
        """Check if the polytope is the empty set.

        Returns:
            True if the polytope is the empty set, False otherwise.
        """
        if self._is_empty_set is None:
            self._is_empty_set = self.points.shape[0] == 0

        return self._is_empty_set

    @property
    def is_bounded(self) -> bool:
        """Check if the polytope is bounded.

        Returns:
            True if the polytope is bounded, False otherwise.
        """
        if self._is_bounded is None:
            # NOTE: this check could be done just by looking at the inequalities
            # in case of H-representation
            _ = self.vertices  # init the vertices to check if the polytope is bounded

        return self._is_bounded

    @property
    def cdd_polyhedron(self) -> cdd.Polyhedron:
        """Get the cdd polyhedron of the polytope.

        The Polyhedron object from the pycddlib library, which is used to switch between
        V-representation and H-representation.

        Returns:
            The cdd polyhedron of the polytope.
        """
        if self._cdd_polyhedron is None:
            if self._repr == _PolytopeRepresentation.V_REPRESENTATION:
                self._set_cdd_polyhedron_from_points()
            elif self._repr == _PolytopeRepresentation.H_REPRESENTATION:
                self._set_cdd_polyhedron_from_inequalities()

        return self._cdd_polyhedron

    @property
    def vertices(self) -> PointConfiguration:
        """Get the irredundant list of vertices of the polytope.

        Returns:
            The vertices of the polytope as a PointConfiguration object.
        """
        if self._vertices is None:
            mat_gens = self.cdd_polyhedron.get_generators()

            if len(mat_gens.lin_set) > 0:
                # unbounded polyhedron case
                self._is_bounded = False
            else:
                self._is_bounded = True

            if mat_gens.row_size > 0:
                # remove redundant generators and update the cdd polyhedron
                mat_gens.canonicalize()
                self._cdd_polyhedron = cdd.Polyhedron(mat_gens)

            self._vertices = PointConfiguration([p[1:] for p in mat_gens])

            # check if the polytope is a simplex
            if self.is_simplex():
                self._make_simplex()

        return self._vertices

    @property
    def inequalities(self) -> np.ndarray:
        """Get the irredundant list of defining inequalities of the polytope.

        The inequalities are an irredundant array [-b A] such that the polytope
        satisfies Ax >= b. Together with the equalities [-b' A'] they define the
        polytope as {x | Ax >= b, A'x = b'}.

        Returns:
            The defining inequalities of the polytope as a numpy array.
        """
        if self._inequalities is None:
            self._set_ineqs_and_eqs()
        elif self._cdd_polyhedron is None:
            # If the inequalities are already set, but the cdd polyhedron is not,
            # then they might have redundancies. Calculate the cdd polyhedron
            # to remove them.
            self._set_cdd_polyhedron_from_inequalities()

        return self._inequalities

    @property
    def equalities(self) -> np.ndarray:
        """Get the irredundant list of defining equalities of the polytope.

        The equalities are an irredundant array [-b' A'] such that the polytope
        satisfies A'x = b'. Together with the inequalities [-b A] they define the
        polytope as {x | Ax >= b, A'x = b'}.

        There are always as many equalities as the ambient dimension of the polytope
        minus the dimension of the polytope.

        Returns:
            The defining equalities of the polytope as a numpy array.
        """
        if self._equalities is None:
            self._set_ineqs_and_eqs()

        return self._equalities

    @property
    def homogeneous_inequalities(self) -> np.ndarray:
        """Get the defining homogeneous inequalities of the polytope.

        This is basically a copy of the ``inequalities`` property, where each line
        (inequality) is multiplied by the denominator of the right hand side b, making
        this array integer. This can be thought of as the defining homogeneous
        inequalities {x | Ax >= 0} of the cone positively spanned by {1} x P.

        Returns:
            The defining homogeneous inequalities of the polytope as a numpy array.
        """
        if self._homogeneous_inequalities is None:
            self._homogeneous_inequalities = np.empty_like(self.inequalities)
            for i, ineq in enumerate(self.inequalities):
                self._homogeneous_inequalities[i] = ineq * ineq[0].q

        return self._homogeneous_inequalities

    @property
    def n_inequalities(self) -> int:
        """Get the number of defining inequalities of the polytope.

        Returns:
            The number of defining inequalities of the polytope.
        """
        return self.inequalities.shape[0]

    @property
    def n_equalities(self) -> int:
        """Get the number of defining equalities of the polytope.

        Returns:
            The number of defining equalities of the polytope.
        """
        return self.equalities.shape[0]

    @property
    def facets(self) -> tuple:
        """Get the facets of the polytope.

        The facets of the polytope are the (d-1)-dimensional faces of the polytope,
        where d is the dimension of the polytope.

        Returns:
            The facets of the polytope as a tuple of frozensets of vertex ids.
        """
        if self._facets is None:
            self._facets = tuple(
                [frozenset(np.where(row)[0]) for row in self.vertex_facet_matrix]
            )

        return self._facets

    @property
    def ridges(self) -> tuple:
        """Get the ridges of the polytope.

        The ridges of the polytope are the (d-2)-dimensional faces of the polytope,
        where d is the dimension of the polytope.

        Returns:
            The ridges of the polytope as a tuple of frozensets of vertex ids.
        """
        if self._ridges is None:
            self._ridges = []
            i, j = np.where(np.triu(self.facet_adjacency_matrix, k=1))
            self._ridges = tuple(
                self.facets[i].intersection(self.facets[j]) for i, j in zip(i, j)
            )

        return self._ridges

    @property
    def edges(self) -> tuple:
        """Get the edges of the polytope.

        The edges of the polytope are the 1-dimensional faces of the polytope.

        Returns:
            The edges of the polytope as a tuple of frozensets of vertex ids.
        """
        if self._edges is None:
            i, j = np.where(np.triu(self.vertex_adjacency_matrix, k=1))
            self._edges = tuple(frozenset({i, j}) for i, j in zip(i, j))

        return self._edges

    def faces(self, dim) -> tuple:
        """Get the faces of the polytope of a given dimension.

        Faces of dimension and codimension lower or equal to two are deduced from the
        cdd polyhedron. Other faces are found from higher dimensional faces, via
        intersection with facets.

        Args:
            dim: The dimension of the faces to be returned.

        Returns:
            The faces of the polytope of dimension dim as a tuple of frozensets of
            vertex ids.
        """
        # TODO: This could be done more efficiently especially as low dimensional faces
        # need all the higher dimensional faces to be calculated first. Can also be
        # compiled with cython.
        if dim < -1:
            raise ValueError(
                "The dimension of the face must be greater than or equal to -1"
            )

        if dim > self.dim:
            return tuple()

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
                        remove_faces = []
                        if face.issubset(facet):
                            continue
                        f = face.intersection(facet)
                        add_f = True
                        for f2 in new_faces:
                            if f.issubset(f2):
                                add_f = False
                                break
                            if f2.issubset(f):
                                remove_faces.append(f2)
                        if add_f:
                            new_faces.append(f)
                        new_faces = [f for f in new_faces if f not in remove_faces]
                self._faces[dim] = tuple(new_faces)

        return self._faces[dim]

    @property
    def n_vertices(self) -> int:
        """Get the number of vertices of the polytope.

        Returns:
            The number of vertices of the polytope.
        """
        if self._n_vertices is None:
            self._n_vertices = self.vertices.shape[0]

        return self._n_vertices

    @property
    def n_facets(self) -> int:
        """Get the number of facets of the polytope.

        Returns:
            The number of facets of the polytope.
        """
        return self.n_inequalities

    @property
    def n_ridges(self) -> int:
        """Get the number of ridges of the polytope.

        Returns:
            The number of ridges of the polytope.
        """
        if self._n_ridges is None:
            self._n_ridges = len(self.ridges)

        return self._n_ridges

    @property
    def n_edges(self) -> int:
        """Get the number of edges of the polytope.

        Returns:
            The number of edges of the polytope.
        """
        if self._n_edges is None:
            self._n_edges = len(self.edges)

        return self._n_edges

    @property
    def f_vector(self) -> tuple:
        """Get the f-vector of the polytope.

        The f-vector of a polytope is a vector of the number of faces of each dimension.
        It should be thought as indexed by the dimension of the faces, where the first
        and last entries are 1 as they represent the empty set (contained in all faces)
        and the polytope itself (containing all faces), respectively.

        Returns:
            The f-vector of the polytope.
        """
        if self._f_vector is None:
            self._f_vector = tuple(
                [len(self.faces(dim)) for dim in range(-1, self.dim + 1)]
            )

        return self._f_vector

    @property
    def vertex_adjacency_matrix(self) -> np.ndarray:
        """Get the vertex adjacency matrix of the polytope.

        The vertex adjacency matrix A of the polytope is a matrix of size n x n, where n
        is the number of vertices of the polytope and
        * a_ij = 1 if vertex i adjacent to vertex j,
        * a_ij = 0 otherwise.

        Returns:
            The vertex adjacency matrix of the polytope.
        """
        if self._vertex_adjacency_matrix is None:
            if self._repr == _PolytopeRepresentation.V_REPRESENTATION:
                get_vertex_adjacency = self.cdd_polyhedron.get_input_adjacency
            elif self._repr == _PolytopeRepresentation.H_REPRESENTATION:
                get_vertex_adjacency = self.cdd_polyhedron.get_adjacency
            self._vertex_adjacency_matrix = np.zeros(
                shape=(self.n_vertices, self.n_vertices), dtype=bool
            )
            for i, ads in enumerate(get_vertex_adjacency()):
                for j in ads:
                    self._vertex_adjacency_matrix[i, j] = 1

        return self._vertex_adjacency_matrix

    @property
    def facet_adjacency_matrix(self) -> np.ndarray:
        """Get the facet adjacency matrix of the polytope.

        The facet adjacency matrix A of the polytope is a matrix of size n x n, where n
        is the number of facets of the polytope and
        * a_ij = 1 if facet i adjacent to facet j,
        * a_ij = 0 otherwise.

        Returns:
            The facet adjacency matrix of the polytope.
        """
        if self._facet_adjacency_matrix is None:
            if self._repr == _PolytopeRepresentation.V_REPRESENTATION:
                get_facet_adjacency = self.cdd_polyhedron.get_adjacency
            elif self._repr == _PolytopeRepresentation.H_REPRESENTATION:
                get_facet_adjacency = self.cdd_polyhedron.get_input_adjacency
            self._facet_adjacency_matrix = np.zeros(
                shape=(self.n_facets, self.n_facets), dtype=bool
            )
            for i, ads in enumerate(get_facet_adjacency()):
                for j in ads:
                    self._facet_adjacency_matrix[i, j] = 1

        return self._facet_adjacency_matrix

    @property
    def vertex_facet_matrix(self) -> np.ndarray:
        """Get the vertex facet incidence matrix of the polytope.

        The vertex facet incidence matrix M of the polytope is a matrix of size
        n_facets x n_vertices, where
        * m_ij = 1 if vertex j is in facet i,
        * m_ij = 0 otherwise.

        Returns:
            The vertex facet incidence matrix of the polytope.
        """
        if self._vertex_facet_matrix is None:
            if self._repr == _PolytopeRepresentation.V_REPRESENTATION:
                get_incidence = self.cdd_polyhedron.get_incidence
            elif self._repr == _PolytopeRepresentation.H_REPRESENTATION:
                get_incidence = self.cdd_polyhedron.get_input_incidence
            self._vertex_facet_matrix = np.zeros(
                shape=(self.n_facets, self.n_vertices), dtype=bool
            )
            _cdd_vertex_facet_incidence = list(get_incidence())
            # remove all occurrences of frozenset({1,...,n}) from the list as
            # this correspond to equalities
            _cdd_vertex_facet_incidence = [
                f for f in _cdd_vertex_facet_incidence if len(f) < self.n_vertices
            ]
            for i, facet in enumerate(_cdd_vertex_facet_incidence):
                for j in facet:
                    self._vertex_facet_matrix[i, j] = True

        return self._vertex_facet_matrix

    @property
    def vertex_facet_pairing_matrix(self) -> np.ndarray:
        """Get the vertex facet pairing matrix of the polytope.

        The vertex facet pairing matrix M of the polytope is a matrix of size
        n_facets x n_vertices, with m_ij = <F_i, v_j> (distance of vertex j to facet i).

        Returns:
            The vertex facet pairing matrix of the polytope.
        """
        if self._vertex_facet_pairing_matrix is None:
            self._vertex_facet_pairing_matrix = (
                self.inequalities[:, 1:] @ self.vertices.view(np.ndarray).T
                + self.inequalities[:, :1]
            )

        return self._vertex_facet_pairing_matrix

    @property
    def barycenter(self) -> Point:
        """Get the barycenter of the polytope.

        The barycenter (or center of mass) of the Polytope is the average of
        its vertices.

        Returns:
            The barycenter of the polytope.
        """
        return self.vertices.barycenter

    @property
    def triangulation(self) -> tuple[frozenset[int]]:
        """Get a triangulation of the polytope in simplices.

        It uses a custom symbolic implementation through upper or lower hull of the
        vertices in a codimension one space, with random heights.

        Returns:
            A triangulation of the polytope in simplices.
        """
        if self._triangulation is None:
            self._triangulation = self.vertices.triangulation

        return self._triangulation

    @property
    def half_open_decomposition(self) -> tuple:
        """Get the half open decomposition of the polytope.

        The half open decomposition of the polytope is given as tuple of vertices ids,
        indicating - for each simplex in the triangulation - which facets (opposite to
        the vertices whose ids are given) are missing from such simplex. A simplex with
        some faces removed is said half-open simplex. The simplices in the triangulation
        of the polytope - once made into half-open simplices - have the property that
        their union is the polytope itself, but they do not have pairwise intersections.

        They are ordered according to the respective simplices in the ``triangulation``
        property.

        Returns:
            The half open decomposition of the polytope.
        """
        if self._half_open_decomposition is None:
            self._half_open_decomposition = []
            for i, simplex_ids in enumerate(self.triangulation):
                verts = [self.vertices[i] for i in simplex_ids]
                simplex = Simplex(vertices=verts)
                special_gens_ids = []
                if i == 0:
                    # Define a reference point in the first simplex of the
                    # triangulation, make sure its coordinates do not satisfy
                    # any rational linear relations. This ensures that any of the
                    # inequalities below will not evaluate to zero at this point.
                    # TODO: it is probably more efficient to take a random point
                    # and check if it strictly satisfies the inequalities, starting over
                    # if it does not.
                    weights = [2 ** Rational(1, i + 2) for i in range(len(verts))]
                    ref_pt = self._origin()
                    for v, w in zip(verts, weights):
                        ref_pt += v * w
                    ref_pt /= sum(weights)
                else:
                    for facet_id, ineq in enumerate(simplex.inequalities):
                        # find the vertex that is not in the facet
                        v_id = simplex.opposite_vertex(facet_id)
                        if np.dot(ineq[1:], ref_pt) + ineq[0] < 0:
                            special_gens_ids.append(v_id)
                self._half_open_decomposition.append(frozenset(special_gens_ids))
            self._half_open_decomposition = tuple(self._half_open_decomposition)

        return self._half_open_decomposition

    @property
    def induced_boundary_triangulation(self) -> tuple:
        """Get the triangulation of the boundary of the polytope.

        Get the triangulation of the boundary of the polytope induced by the
        triangulation of the polytope.

        Returns:
            The triangulation of the boundary of the polytope.
        """
        if self._induced_boundary_triangulation is None:
            self._induced_boundary_triangulation = tuple(
                [
                    ss
                    for s in self.triangulation
                    for f in self.facets
                    if len(ss := f.intersection(s)) == self.dim
                    and PointConfiguration([self.vertices[i] for i in ss]).affine_rank
                    == self.dim - 1
                ]
            )

        return self._induced_boundary_triangulation

    @property
    def volume(self) -> Rational:
        """Get the euclidean volume of the polytope.

        Returns:
            The euclidean volume of the polytope as a sympy Rational.
        """
        if self._volume is None:
            if self.is_full_dim():
                self._calculate_volume()
            else:
                self._volume = self.full_dim_projection.volume
        return self._volume

    @property
    def normalized_volume(self) -> Rational:
        """Get the normalized volume of the polytope.

        The normalized volume of a polytope is the euclidean volume of the polytope
        multiplied by d!, where d is the dimension of the polytope. In particular, the
        normalized volume of a unit hypercube is 1.

        The normalized volume of a lattice polytope is always an integer.

        Returns:
            The normalized volume of the polytope as a sympy Rational.
        """
        if self._normalized_volume is None:
            if self.is_full_dim():
                self._calculate_volume()
            else:
                self._normalized_volume = self.full_dim_projection.normalized_volume
        return self._normalized_volume

    @property
    def boundary_volume(self) -> Rational:
        """Get the boundary volume of the polytope.

        The boundary volume is calculated as the sum of the volumes of the facets of the
        polytope, where each for each facet, the volume is calculated as the euclidean
        volume with respect to the affine lattice obtained as the intersection of the
        original full dimensional lattice with the hyperplane supporting the facet.

        Returns:
            The boundary volume of the polytope as a sympy Rational.
        """
        if self._boundary_volume is None:
            self._calculate_boundary_volume()
        return self._boundary_volume

    @property
    def normalized_boundary_volume(self) -> Rational:
        """Get the normalized boundary volume of the polytope.

        The normalized boundary volume of a polytope is the boundary volume of the
        polytope multiplied by d!, where d is the dimension of the polytope.

        Returns:
            The normalized boundary volume of the polytope as a sympy Rational.
        """
        if self._normalized_boundary_volume is None:
            self._calculate_boundary_volume()
        return self._normalized_boundary_volume

    @property
    def full_dim_projection(self) -> "Polytope":
        """Get a full dimensional projection of the polytope.

        Use Hermite Normal Form to find a full dimensional projection of the polytope,
        which is a polytope which is image of a lattice preserving unimodular affine
        map from the affine span of the polytope, to a d-dimensional space, where d is
        the dimension of the polytope.

        Returns:
            A full dimensional projection of the polytope.
        """
        if self._full_dim_projection is None:
            m = Matrix(self.vertices - self.vertices[0])
            hnf = hermite_normal_form(m)
            self._full_dim_projection = Polytope(hnf)

        return self._full_dim_projection

    @property
    def normal_form(self) -> PointConfiguration:
        """Return the vertices of the polytope in normal form.

        The normal form of a polytope is a "canonical" representative of the equivalence
        class of polytopes under unimodular integer transformations. It only depends on
        the equivalence class of the polytope, and not on the actual instance of the
        polytope.

        To also consider affine transformations, i.e. to include translations in the
        equivalence class, use ``affine_normal_form``.

        Returns:
            The vertices of the polytope in normal form, as a PointConfiguration object.
        """
        if self._normal_form is None:
            self._normal_form = get_normal_form(polytope=self)

        return self._normal_form

    @property
    def affine_normal_form(self) -> PointConfiguration:
        """Return the vertices of the polytope in affine normal form.

        The affine normal form of a polytope is a "canonical" representative of the
        equivalence class of polytopes under affine unimodular transformations. It only
        depends on the equivalence class of the polytope, and not on the actual instance
        of the polytope.

        This can be used to check if two polytopes are equivalent under affine lattice
        preserving unimodular transformations.

        Returns:
            The vertices of the polytope in affine normal form, as a PointConfiguration
            object.
        """
        if self._affine_normal_form is None:
            self._affine_normal_form = get_normal_form(polytope=self, affine=True)

        return self._affine_normal_form

    @property
    def integer_points(self) -> PointConfiguration:
        """Get the integer points of the polytope.

        Returns:
            The integer points of the polytope as a PointConfiguration object.
        """
        if self._integer_points is None:
            _integer_points = [pt for pt in self.boundary_points]
            if self.n_interior_points > 0:
                _integer_points += [pt for pt in self.interior_points]
            self._integer_points = PointConfiguration(_integer_points)
        return self._integer_points

    @property
    def interior_points(self) -> PointConfiguration:
        """Get the interior integer points of the polytope.

        Returns:
            The interior integer points of the polytope as a PointConfiguration object.
        """
        if self._interior_points is None:
            self._get_integer_points()

        return self._interior_points

    @property
    def boundary_points(self) -> PointConfiguration:
        """Get the boundary integer points of the polytope.

        Returns:
            The boundary integer points of the polytope as a PointConfiguration object.
        """
        if self._boundary_points is None:
            self._get_integer_points()

        return self._boundary_points

    @property
    def boundary_points_facets(self) -> list:
        """Get the ids facets of the polytope that contain boundary integer points.

        This is a list of list of frozensets of facet ids. The i-th set in the list
        contains the ids of the facets that contain the i-th boundary integer point.

        Returns:
            The ids facets of the polytope that contain boundary integer points as
            a list of list of frozensets of facet ids.
        """
        if self._boundary_points_facets is None:
            self._get_integer_points()

        return self._boundary_points_facets

    @property
    def n_integer_points(self) -> int:
        """Get the number of integer points of the polytope.

        Returns:
            The number of integer points of the polytope.
        """
        if self._n_integer_points is None:
            self._get_integer_points(count_only=True)

        return self._n_integer_points

    @property
    def n_interior_points(self) -> int:
        """Get the number of interior integer points of the polytope.

        Returns:
            The number of interior integer points of the polytope.
        """
        if self._n_interior_points is None:
            self._get_integer_points(count_only=True)

        return self._n_interior_points

    @property
    def n_boundary_points(self) -> int:
        """Get the number of boundary integer points of the polytope.

        Returns:
            The number of boundary integer points of the polytope.
        """
        if self._n_boundary_points is None:
            self._n_boundary_points = self.n_integer_points - self.n_interior_points

        return self._n_boundary_points

    @property
    def ehrhart_polynomial(self) -> Poly:
        """Get the Ehrhart polynomial of the polytope.

        The Ehrhart polynomial of a polytope is a polynomial in one variable with
        integer coefficients, which encodes the number of integer points in the dilates
        of the polytope as a function of the dilation factor. `More details
        <https://en.wikipedia.org/wiki/Ehrhart_polynomial>`__.

        Returns:
            The Ehrhart polynomial of the polytope as a sympy Poly object.
        """
        if self._ehrhart_polynomial is None:
            if not self.is_lattice_polytope:
                raise ValueError(
                    "Ehrhart polynomial is only implemented for lattice polytopes"
                )

            self._ehrhart_polynomial = sum(
                [
                    h_i * _binomial_polynomial(self.dim, self.dim - i, x)
                    for i, h_i in enumerate(self.h_star_vector)
                    if h_i != 0
                ]
            )

        return self._ehrhart_polynomial

    @property
    def ehrhart_coefficients(self) -> tuple:
        """Get the coefficients of the Ehrhart polynomial of the polytope.

        Returns:
            A list of the coefficients of the Ehrhart polynomial of the polytope
            as a tuple of sympy Rational objects.
        """
        if self._ehrhart_coefficients is None:
            self._ehrhart_coefficients = tuple(self.ehrhart_polynomial.coeffs()[::-1])

        return self._ehrhart_coefficients

    @property
    def h_star_polynomial(self) -> Poly:
        """Get the h*-polynomial of the polytope.

        The h*-polynomial of a polytope is a polynomial in one variable encoding
        information about the number of integer points in the dilates of the polytope.
        `More details
        <https://en.wikipedia.org/wiki/Ehrhart_polynomial#Ehrhart_series>`__.

        Returns:
            The h*-polynomial of the polytope as a sympy Poly object.
        """
        if self._h_star_polynomial is None:
            if not self.is_lattice_polytope:
                raise ValueError(
                    "h*-polynomial is only implemented for lattice polytopes"
                )

            self._h_star_polynomial = Poly(
                sum([h_i * x**i for i, h_i in enumerate(self.h_star_vector)]), x
            )

        return self._h_star_polynomial

    @property
    def h_star_vector(self) -> tuple:
        """Get the h*-vector of the polytope.

        The h*-vector of a polytope is the vector of coefficients of the h*-polynomial
        of the polytope. It can be shown that these coefficients are non-negative
        integers.

        Returns:
            The h*-vector of the polytope as a tuple of integers.
        """
        if self._h_star_vector is None:
            self._h_star_vector = [0 for _ in range(self.ambient_dim + 1)]

            if self._half_open_parallelotopes_pts is not None:
                # if available, use the half-open parallelotopes points to calculate
                # the h*-vector
                for pt in self._half_open_parallelotopes_pts:
                    self._h_star_vector[pt[0]] += 1
            else:
                for verts_ids, special_gens_ids in zip(
                    self.triangulation, self.half_open_decomposition
                ):
                    hop = HalfOpenParallelotope(
                        generators=[
                            Point([1] + list(self.vertices[v_id])) for v_id in verts_ids
                        ],
                        special_gens_ids=special_gens_ids,
                    )

                    _, delta_h = [pt for pt in hop.get_integer_points(count_only=True)]

                    for i, h_i in enumerate(delta_h):
                        self._h_star_vector[i] += h_i

            self._h_star_vector = tuple(self._h_star_vector)

        return self._h_star_vector

    @property
    def degree(self) -> int:
        """Get the degree of the h*-polynomial of the polytope.

        Returns:
            The degree of the h*-polynomial of the polytope.
        """
        if self._degree is None:
            self._degree = self.h_star_polynomial.degree()

        return self._degree

    @property
    def half_open_parallelotopes_pts(self) -> PointConfiguration:
        """Return the integer points in the half-open parallelotopes of the polytope.

        Return all the points in the half-open parallelotopes of the triangulation
        of the polytope. This can be seen as a multi-graded version of the h*-vector.

        Note that while their number at each height is invariative under unimodular
        transformations, the actual points are not as they depend on the chosen
        half-open decomposition of the polytope into simplices.

        Returns:
            The integer points in the half-open parallelotopes of the polytope as a
            tuple of Point objects.
        """
        if self._half_open_parallelotopes_pts is None:
            self._half_open_parallelotopes_pts = []
            for verts_ids, special_gens_ids in zip(
                self.triangulation, self.half_open_decomposition
            ):
                hop = HalfOpenParallelotope(
                    generators=[
                        Point([1] + list(self.vertices[v_id])) for v_id in verts_ids
                    ],
                    special_gens_ids=special_gens_ids,
                )
                pts, _ = hop.get_integer_points()
                self._half_open_parallelotopes_pts += [Point(pt) for pt in pts]
            self._half_open_parallelotopes_pts = PointConfiguration(
                self._half_open_parallelotopes_pts
            )

        return self._half_open_parallelotopes_pts

    @property
    def hilbert_basis(self) -> tuple:
        """Get the Hilbert basis of the polytope.

        The Hilbert basis is a minimal generating set for the semigroup of the integer
        points in the cone positively spanned by {1} x P.

        Returns:
            The Hilbert basis of the polytope as a tuple of Point objects.
        """
        if self._hilbert_basis is None:
            self._hilbert_basis = self._get_hilbert_basis()

        return self._hilbert_basis

    @property
    def is_simplicial(self) -> bool:
        """Check if the polytope is simplicial.

        A polytope is simplicial if all its facets are simplices.

        Returns:
            True if the polytope is simplicial, False otherwise.
        """
        if self._is_simplicial is None:
            self._is_simplicial = all([len(f) == self.dim for f in self.facets])

        return self._is_simplicial

    @property
    def is_simple(self) -> bool:
        """Check if the polytope is simple.

        A polytope is simple if each vertex is contained in exactly d edges, where d is
        the dimension of the polytope.

        Returns:
            True if the polytope is simple, False otherwise.
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
    def is_lattice_polytope(self) -> bool:
        """Check if the polytope is a lattice polytope.

        A polytope is a lattice polytope if all its vertices have integer coordinates.

        Returns:
            True if the polytope is a lattice polytope, False otherwise.
        """
        if self._is_lattice_polytope is None:
            self._is_lattice_polytope = np.all(
                np.vectorize(lambda x: x.is_Integer)(self.vertices.view(np.ndarray))
            )

        return self._is_lattice_polytope

    @property
    def is_lattice_pyramid(self) -> bool:
        """Check if the polytope is a lattice pyramid.

        A polytope is a lattice pyramid if it is a lattice polytope if all of its
        vertices but one are on a facet, from which the last vertex is at distance 1.

        Returns:
            True if the polytope is a lattice pyramid, False otherwise.

        Raises:
            ValueError: If the polytope is not a lattice polytope.
        """
        if self._is_lattice_pyramid is None:
            if not self.is_lattice_polytope:
                raise ValueError("Only lattice polytopes can be being lattice pyramids")

            self._is_lattice_pyramid = False
            for ineq in self.inequalities:
                sum_dists = 0
                for vert in self.vertices:
                    sum_dists += np.dot(ineq[1:], vert) + ineq[0]
                    if sum_dists > 1:
                        break
                if sum_dists == 1:
                    # the polytope is a lattice pyramid wrt the current ineq
                    self._is_lattice_pyramid = True
                    break

        return self._is_lattice_pyramid

    @property
    def is_hollow(self) -> bool:
        """Check if the polytope is hollow, i.e. if it has no interior points.

        Returns:
            True if the polytope is hollow, False otherwise.
        """
        if self._is_hollow is None:
            self._is_hollow = self.has_n_interior_points(0)

        return self._is_hollow

    @property
    def has_one_interior_point(self) -> bool:
        """Check if the polytope has exactly one interior point.

        Returns:
            True if the polytope has exactly one interior point, False otherwise.
        """
        if self._has_one_interior_point is None:
            self._has_one_interior_point = self.has_n_interior_points(1)

        return self._has_one_interior_point

    @property
    def is_canonical(self) -> bool:
        """Check if the polytope is a canonical Fano polytope.

        A polytope is canonical Fano if it is a lattice polytope with
        the origin as unique interior point.

        Returns:
            True if the polytope is a canonical Fano polytope, False otherwise.
        """
        if self._is_canonical is None:
            self._is_canonical = (
                self.is_lattice_polytope
                and self.has_one_interior_point
                and self.contains(self._origin(), strict=True)
            )

        return self._is_canonical

    @property
    def is_reflexive(self) -> bool:
        """Check if the polytope is reflexive.

        A polytope is reflexive if it is a lattice polytope with the origin as unique
        interior point and all its facets at distance 1 from the origin. Equivalently,
        a lattice polytope is reflexive if its dual is a lattice polytope.

        Returns:
            True if the polytope is reflexive, False otherwise.
        """
        if self._is_reflexive is None:
            if not self.is_canonical:
                self._is_reflexive = False
                return False

            # check that all the facets are at distance 1 from the origin
            for ineqs in self.inequalities:
                if ineqs[0] != 1:
                    self._is_reflexive = False
                    return False
            self._is_reflexive = True

        return self._is_reflexive

    @property
    def is_gorenstein(self) -> bool:
        """Check if the polytope is Gorenstein.

        A polytope is Gorenstein if its h*-vector is symmetric, equivalently if it
        is reflexive up to a translation and an integer dilation.

        Returns:
            True if the polytope is Gorenstein, False otherwise.
        """
        if self._is_gorenstein is None:
            # if h* is not available, it would be faster to check if it has an integer
            # dilation that is reflexive (up to translation), but this is not
            # implemented yet

            # check that the h*-vector is symmetric
            hsv = self.h_star_vector[: self.degree + 1]
            self._is_gorenstein = hsv == hsv[::-1]

        return self._is_gorenstein

    @property
    def is_ehrhart_positive(self) -> bool:
        """Check if the polytope is Ehrhart positive.

        A polytope is Ehrhart positive if its Ehrhart polynomial has non-negative
        coefficients.

        Returns:
            True if the polytope is Ehrhart positive, False otherwise.
        """
        if self._is_ehrhart_positive is None:
            self._is_ehrhart_positive = all([i >= 0 for i in self.ehrhart_coefficients])

        return self._is_ehrhart_positive

    @property
    def has_log_concave_h_star_vector(self) -> bool:
        """Check if the polytope has a log-concave h*-vector.

        A sequence a_0, a_1, ..., a_d is log concave if (a_i)^2 >= a_{i−1} * a_{i+1}
        holds for all i = 1, ..., d − 1 and it has no internal zeros.

        Returns:
            True if the polytope has a log-concave h*-vector, False otherwise.
        """
        if self._has_log_concave_h_star_vector is None:
            self._has_log_concave_h_star_vector = _is_log_concave(self.h_star_vector)

        return self._has_log_concave_h_star_vector

    @property
    def has_unimodal_h_star_vector(self) -> bool:
        """Check if the polytope has a unimodal h*-vector.

        A unimodal vector is a vector whose entries are non-negative integers and
        which is unimodal, i.e. there is an index i such that the first i entries
        are increasing and the last i entries are decreasing.

        Returns:
            True if the polytope has a unimodal h*-vector, False otherwise.
        """
        if self._has_unimodal_h_star_vector is None:
            self._has_unimodal_h_star_vector = _is_unimodal(self.h_star_vector)

        return self._has_unimodal_h_star_vector

    @property
    def is_idp(self) -> bool:
        """Check if the polytope P has the Integer Decomposition Property (IDP).

        A polytope is IDP for every k >= 2 and for every lattice point u in kP there
        exist k lattice points v1, . . . , vk in P such that u = v1 + · · · + vk.
        A polytope P with this property is also called integrally closed.

        Returns:
            True if the polytope is IDP, False otherwise.
        """
        if self._is_idp is None:
            # a quicker check is to check that the lattice points of the polytope
            # span the whole lattice (note that the half-open parallelotopes points
            # need to be calculated anyway for the hilbert basis). This is a necessary
            # condition for IDP-ness, but not sufficient.
            index = PointConfiguration(
                [v for v in self.vertices]
                + [pt[1:] for pt in self.half_open_parallelotopes_pts if pt[0] == 1]
            ).index
            if index > 1:
                self._is_idp = False
            else:
                hilbert_basis = self._get_hilbert_basis(stop_at_height=2)

                # check if the last element of the hilbert basis is at height >= 2
                if hilbert_basis[-1][0] >= 2:
                    self._is_idp = False
                else:
                    self._is_idp = True

        return self._is_idp

    @property
    def is_smooth(self) -> bool:
        """Check if the polytope is smooth.

        A lattice polytope is is smooth if it is simple and the tangent cone at
        each vertex is unimodular.

        Returns:
            True if the polytope is smooth, False otherwise.
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

    def _set_cdd_polyhedron_from_points(self) -> None:
        """Set cdd_polyhedron from the v-representation of the polytope."""
        points_to_use = self._vertices if self._vertices is not None else self._points
        formatted_points = [[1] + [c for c in p] for p in points_to_use]
        mat = cdd.Matrix(formatted_points, number_type="fraction")
        mat.rep_type = cdd.RepType.GENERATOR
        self._cdd_polyhedron = cdd.Polyhedron(mat)

    def _set_cdd_polyhedron_from_inequalities(self):
        """Set the cdd polyhedron from the h-representation of the polytope."""
        ineqs = np.array(self._inequalities)
        mat_ineqs = cdd.Matrix(ineqs, number_type="fraction")
        if self._equalities is not None:
            eqs = np.array(self.equalities)
            mat_eqs = cdd.Matrix(eqs, number_type="fraction")
            mat_ineqs.extend(mat_eqs, linear=True)

        mat_ineqs.rep_type = cdd.RepType.INEQUALITY
        self._cdd_polyhedron = cdd.Polyhedron(mat_ineqs)

        # remove redundancies and update the inequalities and equalities
        mat_ineqs = self.cdd_polyhedron.get_inequalities()
        mat_ineqs.canonicalize()
        self._cdd_polyhedron = cdd.Polyhedron(mat_ineqs)
        self._set_ineqs_and_eqs()

    def _set_ineqs_and_eqs(self) -> None:
        """Set the defining inequalities and the equalities of the polytope."""
        cdd_ineqs = self.cdd_polyhedron.get_inequalities()
        eq_ids = cdd_ineqs.lin_set

        self._inequalities = np.empty(shape=(0, cdd_ineqs.col_size), dtype=object)
        self._equalities = np.empty(shape=(0, cdd_ineqs.col_size), dtype=object)

        for i, ineq in enumerate(cdd_ineqs):
            # convert cdd rational to sympy rational
            ineq = [_cdd_fraction_to_simpy_rational(coeff) for coeff in ineq]

            # make the normal integer and primitive
            lcm_ineq = lcm([rat_coeff.q for rat_coeff in ineq[1:]])
            ineq = [rat_coeff * Abs(lcm_ineq) for rat_coeff in ineq]

            gcd_ineq = gcd([int_coeff for int_coeff in ineq[1:]])
            ineq = [int_coeff / Abs(gcd_ineq) for int_coeff in ineq]
            if i in eq_ids:
                self._equalities = np.vstack([self._equalities, ineq])
            else:
                self._inequalities = np.vstack([self._inequalities, ineq])

    def _calculate_volume(self) -> None:
        """Set both _volume and _normalized_volume."""
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

    def _calculate_boundary_volume(self) -> None:
        """Set both _boundary_volume and _normalized_boundary_volume."""
        boundary_norm_volume = Rational(0)

        for s in self.induced_boundary_triangulation:
            simplex = Simplex(vertices=[self.vertices[i] for i in s])
            boundary_norm_volume += simplex.normalized_volume

        self._normalized_boundary_volume = boundary_norm_volume
        self._boundary_volume = boundary_norm_volume / factorial(self.dim - 1)

    def _get_integer_points(self, count_only=False, stop_at_interior=-1) -> None:
        """Set all the integer points properties of the polytope."""
        if not self.is_full_dim():
            raise ValueError("polytope must be full-dimensional")
        (
            _interior_points,
            _boundary_points,
            _boundary_points_facets,
            _n_integer_points,
            _n_interior_points,
            forced_stop,
        ) = find_integer_points(
            verts=self.vertices.view(np.ndarray).astype(np.int64),
            ineqs=self.homogeneous_inequalities.view(np.ndarray).astype(np.int64),
            dim=self.dim,
            count_only=count_only,
            stop_at_interior=stop_at_interior,
        )

        if forced_stop:
            # enumeration has been interrupted, do not populate any property
            return

        if not count_only:
            self._interior_points = PointConfiguration(_interior_points)
            self._boundary_points = PointConfiguration(_boundary_points)
        self._boundary_points_facets = _boundary_points_facets
        self._n_integer_points = _n_integer_points
        self._n_interior_points = _n_interior_points

    # Helper functions

    def _origin(self) -> Point:
        """Return the origin of the ambient space."""
        return Point([0 for _ in range(self.ambient_dim)])

    def is_full_dim(self) -> bool:
        """Check if the polytope is full dimensional.

        Returns:
            True if the polytope is full dimensional, False otherwise.
        """
        return self.dim == self.ambient_dim

    def is_simplex(self) -> bool:
        """Check if the polytope is a simplex.

        Returns:
            True if the polytope is a simplex, False otherwise.
        """
        if self._repr == _PolytopeRepresentation.V_REPRESENTATION:
            return self.n_vertices == self.dim + 1
        elif self._repr == _PolytopeRepresentation.H_REPRESENTATION:
            return self.n_facets == self.dim + 1

    def _make_simplex(self) -> None:
        """Specialize the Polytope class to a Simplex class.

        This is only possible if the polytope is a simplex.

        Raises:
            ValueError: If the polytope is not a simplex.
        """
        if not self.is_simplex():
            raise ValueError("Polytope is not a simplex")
        self.__class__ = Simplex

    def neighbors(self, vertex_id) -> frozenset[int]:
        """Get the neighbors of a vertex.

        The neighbors of a vertex are the vertices that share an edge with it.

        Args:
            vertex_id: The id of the vertex.

        Returns:
            The ids of the neighbors of the vertex.
        """
        row = self.vertex_adjacency_matrix[vertex_id]
        return frozenset(np.where(row)[0].tolist())

    def has_n_interior_points(self, n) -> bool:
        """Check if the polytope has *exactly* n interior points.

        This might result in a quicker calculation than comparing against
        ``n_interior_points``, as the calculation can be interrupted as soon as
        more than n interior points are found.

        Args:
            n: The number of interior points to check for.

        Returns:
            True if the polytope has exactly n interior points, False otherwise.
        """
        if self._n_interior_points is None:
            self._get_integer_points(count_only=True, stop_at_interior=n + 1)

        # use the "private" _n_interior_points as _get_integer_points might
        # have been interrupted in case more than n interior points were found
        # (in that case _n_interior_points would be None and we do not need to
        # know the exact number)
        return self._n_interior_points == n

    def _ehrhart_to_h_star_polynomial(self) -> Poly:
        """Get the h*-polynomial from the h*-vector."""
        return sum(
            [
                self.ehrhart_coefficients[i]
                * _eulerian_poly(i, x)
                * (1 - x) ** (self.dim - i)
                for i in range(self.dim + 1)
            ]
        ).simplify()

    def _get_hilbert_basis(self, stop_at_height=-1) -> tuple:
        """Get the Hilbert basis of the polytope.

        See the ``hilbert_base`` property for more details.

        Args:
            stop_at_height: If > 0, stop the enumeration of the Hilbert basis when
                a point at a heigh greater then or equal to given height is found.

        Returns:
            The Hilbert basis of the polytope as a tuple of Point objects.
        """
        if not self.is_lattice_polytope:
            raise ValueError("Hilbert basis is only implemented for lattice polytopes")

        # a (possibly redundant) set of generators for the semigroup is given by
        # the half-open parallelotopes points, except the origin (assumed to
        # be the first point), plus the missing vertices of the first simplex
        # in the triangulation of the polytope
        generators = np.array(self.half_open_parallelotopes_pts[1:], dtype=np.int64)
        for v_id in self.triangulation[0]:
            generators = np.vstack(
                (
                    generators,
                    np.array([1] + list(self.vertices[v_id]), dtype=np.int64),
                )
            )

        hilbert_basis = tuple(
            get_hilbert_basis_np(
                generators=generators,
                inequalities=self.homogeneous_inequalities.view(np.ndarray).astype(
                    np.int64
                ),
                stop_at_height=stop_at_height,
            )
        )

        return hilbert_basis

    # Polytope operations

    def __add__(self, other) -> "Polytope":
        """Return the the sum of self and other.

        This is:
        * the translation of self by other if other is a Point,
        * the Minkowski sum of self and other if other is a Polytope. (TODO)

        Args:
            other: The other polytope.

        Returns:
            The sum of self and other.
        """
        if isinstance(other, Point):
            verts = self.vertices + other
            return Polytope(vertices=verts)

        if isinstance(other, Polytope):
            raise NotImplementedError("Minkowski sum not implemented yet")

        raise TypeError(
            "A polytope can only be added to a Point (translation) "
            "or another polytope (Minkowski sum)."
        )

    def __sub__(self, other) -> "Polytope":
        """Return the the difference of self and other.

        This is:
        * the translation of self by -other if other is a Point,
        * the Minkowski difference of self and -other if other is a Polytope. (TODO)
        """
        return self + (-other)

    def __neg__(self) -> "Polytope":
        """Return the "flipped" polytope P * (-1)."""
        return self * (-1)

    def __mul__(self, other) -> "Polytope":
        """Return the product of self and other.

        This is:
        * the dilation of self by other if other is a scalar,
        * the cartesian product of self and other if other is a Polytope.

        Args:
            other: The scalar or the other polytope.

        Returns:
            The product of self and other.

        Raises:
            TypeError: If other is not a scalar or a polytope.
        """
        if isinstance(other, Number) or isinstance(other, int):
            verts = self.vertices * other
            return Polytope(vertices=verts)

        if isinstance(other, Polytope):
            verts = _np_cartesian_product(self.vertices, other.vertices)
            return Polytope(vertices=verts)

        raise TypeError(
            "A polytope can only be multiplied with a scalar (dilation)"
            " or another polytope (cartesian product)"
        )

    def free_sum(self, other) -> "Polytope":
        """Return the free sum of the polytope with another polytope.

        The free sum of two polytopes P and Q is the polytope obtained by taking the
        convex hull of the union of the vertices of P and Q, where Q and P are on
        orthogonal spaces in a dim(P) + dim(Q) dimensional space.

        Args:
            other: The other polytope.

        Returns:
            The free sum of the polytope with another polytope.
        """
        if not isinstance(other, Polytope):
            raise TypeError("The free sum is only defined for polytopes")

        verts = [v.tolist() + [0] * other.ambient_dim for v in self.vertices]
        verts += [[0] * self.ambient_dim + v.tolist() for v in other.vertices]

        # make set to quickly remove duplicates (can only be origin)
        verts = list(set(map(tuple, verts)))

        return Polytope(vertices=verts)

    def chisel_vertex(self, vertex_id, dist) -> "Polytope":
        """Return a polytope with a "chiseled" vertex at a given distance.

        Return a new polytope obtained by cutting a vertex at a given lattice
        distance along its neighboring edges.

        Args:
            vertex_id: The id of the vertex to chisel.
            dist: The distance to chisel the vertex at.

        Returns:
            A new polytope obtained by chiseling the vertex at the given distance.

        Raises:
            ValueError: If the polytope is not a lattice polytope, if the distance
                is negative or if the distance is too large.
        """
        if not self.is_lattice_polytope:
            raise ValueError(
                "The chisel operation is only defined for lattice polytopes."
            )

        if dist < 0:
            raise ValueError("The argument dist must be positive.")
        if dist == 0:
            return self

        new_verts = [u for u_id, u in enumerate(self.vertices) if u_id != vertex_id]
        v = self.vertices[vertex_id]
        neighbors = self.neighbors(vertex_id)
        for u_id in neighbors:
            u = self.vertices[u_id]
            vs_dir = u - v
            vs_gcd = gcd(vs_dir)
            if dist >= vs_gcd:
                raise ValueError("Trying to chisel too greedily and too deep.")
            w = vs_dir / vs_gcd
            new_verts.append(v + w * dist)

        return Polytope(new_verts)

    def chisel(self, dist) -> "Polytope":
        """Return a polytope with all vertices "chieseled" at a given distance.

        This is done by applying the chisel operation to all vertices, see
        ``chisel_vertex`` for details.

        Args:
            dist: The distance to chisel the vertices at.

        Returns:
            A new polytope obtained by chiseling all vertices at the given distance.

        Raises:
            ValueError: If the polytope is not a lattice polytope, if the distance
                is negative or if the distance is too large.
        """
        if not self.is_lattice_polytope:
            raise ValueError(
                "The chisel operation is only defined for lattice polytopes."
            )

        if dist < 0:
            raise ValueError("dist must be positive")
        if dist == 0:
            return self

        new_verts = []
        for v_id in range(self.n_vertices):
            v = self.vertices[v_id]
            neighbors = self.neighbors(v_id)
            for u_id in neighbors:
                u = self.vertices[u_id]
                vs_dir = u - v
                vs_gcd = gcd(vs_dir)
                if dist > vs_gcd // 2:
                    raise ValueError("Trying to chisel too greedily and too deep.")
                w = vs_dir / vs_gcd
                pt = v + w * dist
                if pt not in new_verts:
                    new_verts.append(pt)

        return Polytope(new_verts)

    # Polytope relations

    def contains(self, other, strict=False) -> bool:
        """Check if the polytope contains a point or another polytope.

        Args:
            other: The point or polytope to check for containment.
            strict: If True, check for strict containment.

        Returns:
            True if the polytope contains the point or polytope, False otherwise.

        Raises:
            TypeError: If the argument is neither a point nor a polytope.
        """
        if isinstance(other, Point):
            if np.any(np.dot(self.equalities[:, 1:], other) != -self.equalities[:, 0]):
                return False
            if strict:
                if np.any(
                    np.dot(self.inequalities[:, 1:], other) <= -self.inequalities[:, 0]
                ):
                    return False
            else:
                if np.any(
                    np.dot(self.inequalities[:, 1:], other) < -self.inequalities[:, 0]
                ):
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
    def unimodular_simplex(cls, dim) -> "Simplex":
        """Construct a unimodular simplex in the given dimension.

        Args:
            dim: The dimension of the simplex.

        Returns:
            A unimodular simplex in the given dimension.

        Raises:
            ValueError: If the dimension is not a positive integer.
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
    def cube(cls, dim) -> "Polytope":
        """Return a unit hypercube in the given dimension.

        Args:
            dim: The dimension of the hypercube.

        Returns:
            A unit hypercube in the given dimension.
        """
        # check if dim is an integer > 0
        if not isinstance(dim, int) or dim < 1:
            raise ValueError("Dimension must be a positive integer")

        segment_verts = np.array([[0], [1]])
        cube = cls(vertices=_np_cartesian_product(*[segment_verts] * dim))

        return cube

    @classmethod
    def cross_polytope(cls, dim):
        """Return a cross polytope in the given dimension.

        Args:
            dim: The dimension of the cross polytope.

        Returns:
            A cross polytope in the given dimension.

        Raises:
            ValueError: If the dimension is not a positive integer.
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
    def random_lattice_polytope(cls, dim, n_points, min=0, max=1):
        """Return a random lattice polytope in the given dimension.

        Generate n_points random lattice points in dimension dim and calculate their
        convex hull. Consequently, the actual number of vertices and the dimension
        might be lower than the given values.

        Args:
            dim: The dimension of the polytope.
            n_points: The number of points to generate the polytope.
            min: The minimum coordinate of the vertices.
            max: The maximum coordinate of the vertices.

        Returns:
            A random lattice polytope in the given dimension.
        """
        pts = np.random.randint(min, max + 1, size=(n_points, dim))
        return cls(points=pts)


class Simplex(Polytope):
    """A class representing a simplex, as a specialization of the Polytope class.

    A simplex is a polytope with exactly dim + 1 vertices, where dim is the dimension
    of the ambient space. This class overrides some of the methods of the Polytope with
    simpler implementations, and adds some simplex-specific methods.
    """

    @property
    def half_open_decomposition(self) -> tuple:
        """Get the trivial half-open decomposition of the simplex.

        Returns:
            The trivial half-open decomposition of the simplex as a tuple with one
            empty frozenset.
        """
        if self._half_open_decomposition is None:
            self._half_open_decomposition = tuple([frozenset({})])

        return self._half_open_decomposition

    # Simplex-specific properties and methods

    def opposite_vertex(self, facet_id) -> int:
        """Get the opposite vertex of a given facet.

        Args:
            facet_id: The id of the facet.

        Returns:
            The id of the opposite vertex.
        """
        return np.where(self.vertex_facet_matrix[facet_id] == 0)[0][0]

    @property
    def weights(self) -> list:
        """Return the weights of the simplex.

        The weights of a simplex are the barycentric coordinates of the
        origin. They are uniquely determined up to a scalar, they are given as
        integers.

        Returns:
            The weights of the simplex as a list of integers.
        """
        if self._weights is None:
            b = self.barycentric_coordinates(self._origin())

            lcm_b = lcm([frac.q for frac in b])
            b = [frac * lcm_b for frac in b]
            gcd_b = gcd(b)
            self._weights = [i / gcd_b for i in b]

        return self._weights

    def barycentric_coordinates(self, point: Point) -> list:
        """Return the barycentric coordinates of a point in the simplex.

        Args:
            point: The point in the simplex.

        Returns:
            The barycentric coordinates of the point as a list of sympy Rational
            objects.
        """
        m = Matrix([[1] + v.tolist() for v in self.vertices]).T
        b = m.LUsolve(Matrix([1] + point.tolist()))

        return b.flat()
