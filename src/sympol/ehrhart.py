"""Module defining additional functions for Ehrhart Theory related properties."""

from sympy import floor


def is_valid_h_star_vector(h: tuple[int]) -> bool:
    """Check if the given integer vector violates any known h*-vector inequalities.

    This method checks that the h*-vector satisfies a list of know inequalities.
    This does not guarantee that the resulting vector is an h*-vector.

    The list of checked properties/inequalities are:
    * h_i is an integer for all i,
    * h_0 = 1,
    * h_i >= 0 for all i,
    * h_1 >= h_d,
    * h_2 + ... + h_i >= h_{d-i+1} + ... + h_{d-1} for all i in [2, floor(d/2)],
    * h_0 + ... + h_i <= h_{s-i} + ... + h_s for all i in [0, floor(s/2)],
    * if s = d, then h_1 <= h_i for all i up to d-1,
    * if s < d, then h_0 + h_1 <= h_{i-d+s} + ... + h_i for all i in [1, d-1].

    Returns:
        True if the vector satisfies all known inequalities, False otherwise.
    """
    if any(not isinstance(h_i, int) for h_i in h):
        return False

    if any(h_i < 0 for h_i in h):
        return False

    if h[0] != 1:
        return False

    # d and s are shorthand for dimension and degree respectively
    d = len(h) - 1
    for s in range(d, 0, -1):
        if h[s] != 0:
            break

    if h[1] < h[d]:
        return False

    for i in range(2, floor(d / 2) + 1):
        if not sum(h[k] for k in range(2, i + 1)) >= sum(
            h[k] for k in range(d - i + 1, d)
        ):
            # does not satisfy Stanley's dim inequality
            return False

    for i in range(0, floor(s / 2) + 1):
        if not sum(h[k] for k in range(0, i + 1)) <= sum(
            h[k] for k in range(s - i, s + 1)
        ):
            # does not satisfy Stanley's dim inequality
            return False

    if s == d:
        for i in range(1, d - 1):
            if not h[1] <= h[i]:
                return False
    else:
        for i in range(1, d):
            if not h[0] + h[1] <= sum(h[k] for k in range(i - d + s, i + 1)):
                return False  # pragma: no cover
                # could not find a counterexample for this case

    return True
