from enum import Enum
from igraph import Graph

from sympy import Abs, Matrix
from sympy.matrices.normalforms import hermite_normal_form

from sympol.point_list import PointList


class VertexType(Enum):
    """
    Enum for vertex type in the vertex-facet pairing graph.
    """

    VERTEX = 0
    FACET = 1
    LATTICE_DISTANCE = 2


def get_normal_form(polytope, affine=False):
    """
    Find the normal form of a lattice polytope by:
        1. associate to each polytope a graph encoding the vertex-facet relations,
        2. put the graph in canonical form with the igraph Bliss algorithm,
        3. find the automorphism group of the graph with the igraph vf2 algorithm,
        4. find the lexicographically smallest hermite form among all the vertex
           permutations induced by the automorphisms.
    :param polytope: polytope
    :return: normal form of the polytope
    """
    if not polytope.is_full_dim():
        raise NotImplementedError(
            "Non-full dimensional polytopes are not supported yet"
        )

    graph = _get_vertex_facet_pairing_graph(polytope)

    perm = graph.canonical_permutation(color=[v["color"] for v in graph.vs])
    graph = graph.permute_vertices(perm)

    aut_group = graph.get_automorphisms_vf2(color=[v["color"] for v in graph.vs])

    normal_form = None
    min_compare_tuple = None
    for aut in aut_group:
        temp_graph = graph.permute_vertices(aut)
        permuted_verts = [
            v["point"] for v in temp_graph.vs if v["type"] == VertexType.VERTEX
        ]
        if affine:
            for v in permuted_verts:
                candidate_normal_form = hermite_normal_form(
                    Matrix(PointList(permuted_verts) - v)
                )
                compare_tuple = tuple(candidate_normal_form.flat())
                if normal_form is None or compare_tuple < min_compare_tuple:
                    normal_form = candidate_normal_form
                    min_compare_tuple = compare_tuple
        else:
            candidate_normal_form = hermite_normal_form(Matrix(permuted_verts))
            compare_tuple = tuple(candidate_normal_form.flat())
            if normal_form is None or compare_tuple < min_compare_tuple:
                normal_form = candidate_normal_form
                min_compare_tuple = compare_tuple

    return PointList(normal_form)


def _get_vertex_facet_pairing_graph(polytope):
    """
    Return a graph encoding the vertex-facet relations of the input polytope. Color each
    vertex of the graph with the maximum of the correponging row/column of the vertex-
    facet pairing matrix. (multiply by -1 for the vertices associated to the facets to
    avoid matching between vertices and facets)
    :param polytope: polytope
    :return: graph
    """
    vfpm = polytope.vertex_facet_pairing_matrix

    graph = Graph()
    graph.add_vertices(
        polytope.n_facets,
        attributes={
            "type": [VertexType.FACET for _ in range(polytope.n_facets)],
            "color": [-max(vfpm.row(i).flat()) for i in range(polytope.n_facets)],
        },
    )
    graph.add_vertices(
        polytope.n_vertices,
        attributes={
            "type": [VertexType.VERTEX for _ in range(polytope.n_vertices)],
            "color": [max(vfpm.col(j).flat()) for j in range(polytope.n_vertices)],
            "point": [v for v in polytope.vertices],
        },
    )

    for i in range(polytope.n_facets):
        for j in range(polytope.n_vertices):
            if vfpm[i, j] > 0:
                graph.add_edge(i, j + polytope.n_facets)

    return graph


# NOTE: This is not used in the current implementation
# def _is_automorphism(input_list: PointList, output_list: PointList):
#     """
#     Check if the mapping between two lists of points can be extended to an
#     affine unimodular map
#     :param input_list: list of input points
#     :param output_list: list of output points
#     :return: True if the mapping can be extended to an affine unimodular map
#     """

#     if len(input_list) != len(output_list):
#         raise ValueError("Input and output must have the same length")

#     if len(input_list) == 0:
#         raise ValueError("Input and output must be non-empty")

#     shifted_input_list = input_list - input_list.barycenter
#     shifted_output_list = output_list - output_list.barycenter

#     ambient_dim = len(input_list[0])
#     rank_input = shifted_input_list.hom_rank
#     rank_output = shifted_output_list.hom_rank

#     if rank_input < ambient_dim or rank_output < ambient_dim:
#         raise ValueError("Input and output must span the ambient space")

#     # Extract a basis from the input
#     rank = 1
#     matrix_input = Matrix([shifted_input_list[0]])
#     matrix_output = Matrix([shifted_output_list[0]])

#     for i in range(1, input_list.shape[0]):
#         if rank == ambient_dim:
#             break

#         temp_matrix_input = Matrix(
#             matrix_input.tolist() + [shifted_input_list[i].tolist()]
#         )

#         if temp_matrix_input.rank() == rank + 1:
#             matrix_input = temp_matrix_input
#             matrix_output = Matrix(
#                 matrix_output.tolist() + [shifted_output_list[i].tolist()]
#             )
#             rank += 1

#     # In order for the map to be a lattice map we need that the following matrix has
#     # integer, is unimodular, and maps the input to the output
#     map_candidate = matrix_input.inv() * matrix_output

#     if not all([entry.is_integer for entry in map_candidate]):
#         return False
#     if not Matrix(shifted_input_list) * map_candidate == Matrix(shifted_output_list):
#         return False
#     if not Abs(map_candidate.det()) == 1:
#         return False

#     return True
