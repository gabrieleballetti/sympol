""" Test with random edge polytopes. """
import numpy as np
from sympol import Polytope
from igraph import Graph


def random_edge_polytope(n_vertices, edge_prob):
    graph = Graph.Erdos_Renyi(n_vertices, edge_prob)
    edges = graph.get_edgelist()
    verts = np.zeros((len(edges), n_vertices), dtype=int)
    for i, (u, v) in enumerate(edges):
        verts[i, u] = 1
        verts[i, v] = 1
    return Polytope(vertices=verts[:, 1:])


if __name__ == "__main__":
    dim = 13
    while True:
        p = random_edge_polytope(n_vertices=dim, edge_prob=0.5)
        if p.dim < dim - 1:
            continue
        print(p.h_star_vector)
