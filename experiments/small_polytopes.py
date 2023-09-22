from sympy import Abs
from sympol import Polytope
from sympol._utils import _is_log_concave
import pathlib

DATA_FOLDER = "experiments/data/"


def check_log_concavity(ep):
    values = [Abs(ep(i)) for i in range(-1, -12, -1)]
    return _is_log_concave(values)


if __name__ == "__main__":
    # read the data from a file, one line at a time
    d = 3
    i = 16
    polytopes_type = "polytopes"
    while True:
        filename = pathlib.Path(DATA_FOLDER) / f"{d}-{polytopes_type}" / f"v{i}.txt"
        if not filename.exists():
            break
        print(f"Checking {d}-{polytopes_type} of volume {i}...")
        with open(filename, "r") as f:
            for line in f:
                verts = eval(line)
                p = Polytope(vertices=verts)

                if p.is_idp:
                    if not p.is_very_ample:
                        print("IDP but not very ample")
                        print(p.vertices)
                    if not p.is_spanning:
                        print("IDP but not spanning")
                        print(p.vertices)
                elif p.is_very_ample:
                    if not p.is_spanning:
                        print("Very ample but not spanning")
                        print(p.vertices)
                    h = p.h_star_vector
                    print(h)
        i += 1
