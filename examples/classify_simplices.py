"""Example: classify all simplices up to a certain volume.

This script contains code to enumerate all unimodular classes of simplices in a given
dimension, up to a certain volume. This is done by iterating through all the possible
Hermite Normal Forms.
"""

from sympy import divisors
from sympol import Polytope, PointConfiguration


def classify_simplices(dim: int, max_volume: int) -> dict[int, set[PointConfiguration]]:
    """Collect all of simplices in fixed dimension, up to a certain volume.

    This is done by iterating through all the possible Hermite Normal Forms.

    Args:
        dim: Dimension of the simplices.
        max_volume: Maximum volume of the simplices.

    Returns:
        A dictionary with the volume as key and the set of the vertices of one
        representative for each equivalence class as value.
    """
    all_simplices = {}

    for volume in range(1, max_volume + 1):
        vol_simplices = set()
        for hnf in _all_hnfs(volume, dim):
            vol_simplices.add(Polytope(vertices=hnf).affine_normal_form)
        all_simplices[volume] = vol_simplices
        print(f"Volume {volume:3d} - {len(vol_simplices):3d} classes")

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


if __name__ == "__main__":
    classify_simplices(dim=3, max_volume=10)
