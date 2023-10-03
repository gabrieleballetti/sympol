from sympy import Abs
from sympol import Polytope
from sympol._utils import _is_log_concave
import pathlib

DATA_FOLDER = ".data/"


def check_log_concavity(ep):
    values = [Abs(ep(i)) for i in range(-1, -12, -1)]
    return _is_log_concave(values)


if __name__ == "__main__":
    # read the data from a file, one line at a time
    d = 6
    i = 12
    polytopes_type = "polytopes"
    while True:
        filename = pathlib.Path(DATA_FOLDER) / f"{d}-{polytopes_type}" / f"v{i}.txt"
        if not filename.exists():
            break
        print(f"Checking {d}-{polytopes_type} of volume {i}...")
        i += 1
        with open(filename, "r") as f:
            for line in f:
                verts = eval(line)
                p = Polytope(vertices=verts)

                if p.is_lattice_pyramid:
                    continue
