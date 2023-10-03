import numpy as np
from sympol import Polytope
import pathlib
from itertools import islice
from sympol._utils import _arrays_equal_up_to_row_permutation

DATA_FOLDER = ".data/4p_reflexive"


def _get_n_lines_iterator(filename, n):
    with open(filename) as fp:
        for line in fp:
            yield [line] + list(islice(fp, n - 1))

    # None of the elements in B were found too many times, and the lists are
    # the same length, they are a permutation
    return True


if __name__ == "__main__":
    # read the data from a file, one line at a time
    dim = 4
    n_verts = 5
    tot = 0
    while True:
        count = 0
        filename = pathlib.Path(DATA_FOLDER) / f"v{n_verts:02d}"
        if n_verts > 36:
            break
        if not filename.exists():
            n_verts += 1
            continue
        print(f"Checking v{n_verts:02d}...")

        if n_verts % 2 != 0:
            n_verts += 1
            continue

        for lines in _get_n_lines_iterator(filename, dim + 1):
            if len(lines) < dim + 1:
                break
            # remove the newline characters
            lines = [line.strip() for line in lines]
            lines = lines[1:]
            lines = [list(map(int, line.split())) for line in lines]
            vertices = np.array(lines).T

            if _arrays_equal_up_to_row_permutation(vertices, -vertices):
                p = Polytope(vertices=vertices)
                h = p.h_star_vector
                g = p.gamma_vector

                gamma_positive = all(g_i >= 0 for g_i in g)
                if not gamma_positive:
                    print(h)
                    print(g)

            # print(p.h_star_vector)
            count += 1
        # print(f"v{n_verts:02d} - {count}")
        n_verts += 1
        tot += count
    print(tot)
