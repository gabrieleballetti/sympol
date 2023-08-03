from sympy import divisors

from sympol.polytope import Polytope


def classify_simplices(dim, max_volume):
    """
    Classify all unimodular classes of simplices in a given dimension, up to a certain
    volume. This is done by iterating through all the possible Hermite Normal Forms.
    """
    all_simplices = {}

    for volume in range(1, max_volume + 1):
        vol_simplices = set()
        for hnf in _all_hnfs(volume, dim):
            vol_simplices.add(Polytope(vertices=hnf).affine_normal_form)
        all_simplices[volume] = vol_simplices
        print(f"{volume} - {len(vol_simplices)}")

    return all_simplices


def _all_hnfs(n, dim, row=0):
    """
    Return a generator for the Hermite Normal Forms in a given dimension, up to a given
    volume
    """
    if row == dim - 1:
        return [[[0] * dim, [n] + [0] * (dim - 1)]]
    return (
        hnf + [left + [div] + [0] * row]
        for div in divisors(n)
        for left in _all_left_values(div, dim - row - 1)
        for hnf in _all_hnfs(n // div, dim, row + 1)
    )


def _all_left_values(k, size):
    """
    Return a generator for all list with k entries and values in [0, k-1]
    """
    if size == 0:
        return [[]]
    return (left + [i] for left in _all_left_values(k, size - 1) for i in range(k))
