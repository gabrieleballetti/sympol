import sys

sys.path.insert(0, ".")

import json
from sympol.polytope import Polytope


def get_random_polytope():
    return Polytope.random_lattice_polytope(3, 10, -5, 5)


def get_polytope_data(p):
    data = {
        "dim": p.dim,
        "vertices": [[int(c) for c in v] for v in p.vertices],
        "f_vector": p.f_vector,
        "vertex_adjacency_matrix": [
            [int(c) for c in r] for r in p.vertex_adjacency_matrix.tolist()
        ],
        "vertex_facet_pairing_matrix": [
            [int(c) for c in r] for r in p.vertex_facet_pairing_matrix.tolist()
        ],
        "normalized_volume": int(p.normalized_volume),
        "normalized_boundary_volume": int(p.normalized_boundary_volume),
        "n_integer_points": p.n_integer_points,
        "n_interior_points": p.n_interior_points,
        "h_star_vector": [int(c) for c in p.h_star_vector],
        "affine_normal_form": [[int(c) for c in v] for v in p.affine_normal_form],
        "is_simplicial": p.is_simplicial,
        "is_simple": p.is_simple,
        "is_lattice_polytope": p.is_lattice_polytope,
        "is_hollow": p.is_hollow,
        "has_one_interior_point": p.has_one_interior_point,
        "is_canonical": p.is_canonical,
        "is_reflexive": p.is_reflexive,
        "is_ehrhart_positive": p.is_ehrhart_positive,
        "has_unimodal_h_star_vector": p.has_unimodal_h_star_vector,
        "is_idp": p.is_idp,
        "is_smooth": p.is_smooth,
    }
    return data


p = get_random_polytope()
data = get_polytope_data(p)

with open("polytope_data.json", "w") as f:
    json.dump(data, f)
