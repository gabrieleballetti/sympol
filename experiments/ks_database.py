import numpy as np
from sympol import Polytope
import pathlib
from itertools import islice

DATA_FOLDER = ".data/4p_reflexive"


def _get_n_lines_iterator(filename):
    with open(filename) as fp:
        for line in fp:
            try:
                n, m = line.split()[:2]
                n, m = int(n), int(m)
            except:  # noqa
                break
            n_lines = 4 if n == 4 else n
            yield list(islice(fp, n_lines))

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

        for lines in _get_n_lines_iterator(filename):
            # remove the newline characters
            lines = [line.strip() for line in lines]
            lines = [list(map(int, line.split())) for line in lines]
            vertices = np.array(lines)
            if vertices.shape[0] == dim:
                vertices = vertices.T

            p = Polytope(vertices=vertices)
            # do something

        n_verts += 1
