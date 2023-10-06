from sympy import Abs
from sympol import Polytope

if __name__ == "__main__":
    verts = [
        [0, 0, 0, 0, 0],
        [10, 0, 0, 0, 0],
        [0, 20, 0, 0, 0],
        [0, 0, 2, 0, 0],
        [0, 0, 0, 2, 0],
        [0, 0, 0, 0, 2],
    ]
    p = Polytope(verts)

    ep = p.ehrhart_polynomial
    print(ep)
    seq = [Abs(ep(-i)) for i in range(1, 12)]
    print(seq)
    print(seq[1] ** 2 - seq[0] * seq[2])
    print(seq[2] ** 2 - seq[1] * seq[3])
