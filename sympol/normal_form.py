from enum import Enum
from igraph import Graph

from sympy import Abs, Matrix
from sympy.matrices.normalforms import hermite_normal_form

from sympol.point_list import PointList


class VertexType(Enum):
    """
    Enum for vertex type
    """

    VERTEX = 0
    FACET = 1
    LATTICE_DISTANCE = 2


def get_normal_form(polytope):
    """
    Find the normal form a lattice polytope by finding a canonical
    labeling of the bipartite graph of the vertex-facet pairing matrix. Uses the igraph
    Bliss algorithm. Also returns the automorphism group of the graph with the vf2
    algorithm.
    :param polytope: polytope
    :return: canonical permutation
    """
    graph = _get_vertex_facet_pairing_graph(polytope)

    # Find the canonical permutation of the bipartite graph
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
        candidate_normal_form = hermite_normal_form(Matrix(permuted_verts))

        compare_tuple = tuple(candidate_normal_form.flat())
        if normal_form is None or compare_tuple < min_compare_tuple:
            normal_form = candidate_normal_form
            min_compare_tuple = compare_tuple

    return PointList(normal_form)


def _get_vertex_facet_pairing_graph(polytope):
    """
    Get the colored tri-partite graph of the vertex-facet pairing matrix
    :param polytope: polytope
    :return: graph
    """
    vfpm = polytope.vertex_facet_pairing_matrix

    graph = Graph()
    n_facets = vfpm.shape[0]
    n_verts = vfpm.shape[1]
    graph.add_vertices(
        n_facets,
        attributes={
            "type": [VertexType.FACET for _ in range(n_facets)],
            "color": [-1 for _ in range(n_facets)],
        },
    )
    graph.add_vertices(
        n_verts,
        attributes={
            "type": [VertexType.VERTEX for _ in range(n_verts)],
            "color": [0 for _ in range(n_verts)],
            "point": [v for v in polytope.vertices],
        },
    )

    for i in range(n_facets):
        for j in range(n_verts):
            if vfpm[i, j] > 0:
                facet_id = i
                vert_id = j + n_facets
                new_vertex = graph.add_vertex(
                    type=VertexType.LATTICE_DISTANCE,
                    color=vfpm[i, j],
                )
                graph.add_edge(vert_id, new_vertex)
                graph.add_edge(new_vertex, facet_id)

    return graph


def _is_automorphism(input_list: PointList, output_list: PointList):
    """
    Check if the mapping between two lists of points can be extended to an
    affine unimodular map
    :param input_list: list of input points
    :param output_list: list of output points
    :return: True if the mapping can be extended to an affine unimodular map
    """

    if len(input_list) != len(output_list):
        raise ValueError("Input and output must have the same length")

    if len(input_list) == 0:
        raise ValueError("Input and output must be non-empty")

    shifted_input_list = input_list - input_list.barycenter
    shifted_output_list = output_list - output_list.barycenter

    ambient_dim = len(input_list[0])
    rank_input = shifted_input_list.hom_rank
    rank_output = shifted_output_list.hom_rank

    if rank_input < ambient_dim or rank_output < ambient_dim:
        raise ValueError("Input and output must span the ambient space")

    # Extract a basis from the input
    rank = 1
    matrix_input = Matrix([shifted_input_list[0]])
    matrix_output = Matrix([shifted_output_list[0]])

    for i in range(1, input_list.shape[0]):
        if rank == ambient_dim:
            break

        temp_matrix_input = Matrix(
            matrix_input.tolist() + [shifted_input_list[i].tolist()]
        )

        if temp_matrix_input.rank() == rank + 1:
            matrix_input = temp_matrix_input
            matrix_output = Matrix(
                matrix_output.tolist() + [shifted_output_list[i].tolist()]
            )
            rank += 1

    # In order for the map to be a lattice map we need that the following matrix has
    # integer, is unimodular, and maps the input to the output
    map_candidate = matrix_input.inv() * matrix_output

    if not all([entry.is_integer for entry in map_candidate]):
        return False
    if not Matrix(shifted_input_list) * map_candidate == Matrix(shifted_output_list):
        return False
    if not Abs(map_candidate.det()) == 1:
        return False

    return True
