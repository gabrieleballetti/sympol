import numpy as np
from igraph import Graph

from sympy import Abs, Matrix
from sympy.combinatorics import Permutation
from sympy.combinatorics.generators import symmetric
from sympy.matrices.normalforms import hermite_normal_form

from sympol.point import Point
from sympol.point_list import PointList


def get_normal_form(polytope):
    """
    Find the normal form a lattice polytope by finding a canonical
    labeling of the bipartite graph of the vertex-facet pairing matrix. Uses the igraph
    Bliss algorithm. Also returns the automorphism group of the graph with the vf2
    algorithm.
    :param polytope: polytope
    :return: canonical permutation
    """
    vfm = polytope.vertex_facet_matrix.tolist()
    vfpm = polytope.vertex_facet_pairing_matrix.tolist()

    bipartite_graph = Graph.Incidence(vfm)

    verts_ids = np.array([v.index for v in bipartite_graph.vs if v["type"]])
    facets_ids = np.array([v.index for v in bipartite_graph.vs if not v["type"]])

    # Add the vertices coords to the vertices of the graph
    for i, v in zip(verts_ids, polytope.vertices):
        bipartite_graph.vs[i]["point"] = v

    color = [0 if v["type"] else -1 for v in bipartite_graph.vs]
    perm = bipartite_graph.canonical_permutation(color=color)

    bipartite_graph = bipartite_graph.permute_vertices(perm)

    color = [0 if v["type"] else -1 for v in bipartite_graph.vs]
    aut_group = bipartite_graph.get_automorphisms_vf2(color=color)

    normal_form = None
    min_compare_tuple = None
    for aut in aut_group:
        temp_bg = bipartite_graph.permute_vertices(aut)
        permuted_verts = [v["point"] for v in temp_bg.vs if v["type"]]
        candidate_normal_form = hermite_normal_form(Matrix(permuted_verts))

        compare_tuple = tuple(candidate_normal_form.flat())
        if normal_form is None or compare_tuple < min_compare_tuple:
            normal_form = candidate_normal_form
            min_compare_tuple = compare_tuple

    return PointList(normal_form)


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
