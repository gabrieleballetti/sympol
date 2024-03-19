import numpy as np
import numba as nb


@nb.njit(
    nb.int32[:, :](nb.int32[:]),
    cache=True,
)
def _cproduct_idx(sizes: np.ndarray):
    """Generates ids tuples for a cartesian product"""
    assert len(sizes) >= 2
    tuples_count = np.prod(sizes)
    tuples = np.zeros((tuples_count, len(sizes)), dtype=np.int32)
    tuple_idx = 0
    # stores the current combination
    current_tuple = np.zeros(len(sizes))
    while tuple_idx < tuples_count:
        tuples[tuple_idx] = current_tuple
        current_tuple[0] += 1
        if current_tuple[0] == sizes[0]:
            current_tuple[0] = 0
            current_tuple[1] += 1
            for i in range(1, len(sizes) - 1):
                if current_tuple[i] == sizes[i]:
                    current_tuple[i + 1] += 1
                    current_tuple[i] = 0
                else:
                    break
        tuple_idx += 1
    return tuples


@nb.njit(cache=True)
def nb_cartesian_product(arrays):
    sizes = [len(a) for a in arrays]
    sizes = np.array(sizes, dtype=np.int32)
    tuples_count = np.prod(sizes)
    array_ids = _cproduct_idx(sizes)
    tuples = np.zeros((tuples_count, len(sizes)))
    for i in range(len(arrays)):
        tuples[:, i] = arrays[i][array_ids[:, i]]
    return tuples
