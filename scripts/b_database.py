""" Script to iterate through the B. database of "small" polytopes."""

from sympol import Polytope
import pathlib

DATA_FOLDER = ".data/"


def _b_database_iterator(dim, vol, ptype="polytopes"):
    """Iterate through the database of polytopes of dimension `dim` and volume
    `vol`."""
    filename = pathlib.Path(DATA_FOLDER) / f"{dim}-{ptype}" / f"v{vol}.txt"
    if not filename.exists():
        return
    print(f"Checking {dim}-{ptype} of volume {vol}...")
    vol += 1
    with open(filename, "r") as f:
        for line in f:
            verts = eval(line)
            p = Polytope(vertices=verts)
            yield p


if __name__ == "__main__":
    for dim in range(2, 7):
        for vol in range(1, 10):
            for p in _b_database_iterator(dim, vol):
                print(p.h_star_vector)
