import sys

sys.path.insert(0, ".")

import pathlib
import hashlib
import json
import numpy as np
from sympol.polytope import Polytope
from sympol.random import (
    sample_polytope_from_normal_distribution,
    random_subpolytope,
    SubpolytopeStrategy,
)


def get_random_polytope():
    return Polytope.random_lattice_polytope(3, 10, -5, 5)


def get_polytope_data(p):
    data = {
        "dim": p.dim,
        "vertices": [[int(c) for c in v] for v in p.vertices],
        "f_vector": None,
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
        "affine_normal_form": None,  # [[int(c) for c in v] for v in p.affine_normal_form],
        "is_simplicial": p.is_simplicial,
        "is_simple": p.is_simple,
        "is_hollow": p.is_hollow,
        "has_one_interior_point": p.has_one_interior_point,
        "is_canonical": p.is_canonical,
        "is_reflexive": p.is_reflexive,
        "is_gorenstein": p.is_gorenstein,
        "is_ehrhart_positive": p.is_ehrhart_positive,
        "has_unimodal_h_star_vector": p.has_unimodal_h_star_vector,
        "is_idp": p.is_idp,
        "is_smooth": p.is_smooth,
    }
    return data


def get_polytope_hash(p):
    p_anf = ([[int(c) for c in v] for v in p.vertices],)
    p_bytes = str(p_anf).encode("utf-8")
    p_hash = hashlib.sha256(p_bytes).hexdigest()
    return p_hash


def save_polytope_data(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f)


import time

dim = 6

for i in range(1000):
    p = None

    while p is None or (p.dim == dim and p.n_integer_points > dim + 1):
        if p is None:
            n_points = int(abs(round(np.random.normal(0, 10)))) + dim + 1
            p = sample_polytope_from_normal_distribution(dim, n_points, 1)
        else:
            p = random_subpolytope(p, SubpolytopeStrategy.POINTS_SUBSET)

        if p.dim < dim:
            break

        data = get_polytope_data(p)
        polytope_hash = get_polytope_hash(p)

        filename = pathlib.Path(f"data/{p.dim}/{polytope_hash}.json")
        filename.parent.mkdir(parents=True, exist_ok=True)
        save_polytope_data(data, filename)

        print(f"{p.h_star_vector} - {polytope_hash}")
        if p.is_smooth:
            print("Smooth")
        if p.is_smooth and not p.is_idp:
            print(p.vertices)
            raise Exception("Found one")
        if p.is_idp and not p.has_unimodal_h_star_vector:
            print(p.vertices)
            raise Exception("Found one")

    print("\n")
