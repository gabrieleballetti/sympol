import time
from sympol import Polytope
import random
import numpy as np

# create 100 random polytopes, calculate their h*-vectors and check how long it takes


start = time.time()
for i in range(10):
    # set a new seed for each polytope
    random.seed(i)
    np.random.seed(i)

    p = Polytope.random_lattice_polytope(4, 12, min=0, max=9)
    # print(p.vertices)
    print(p.h_star_vector)

end = time.time()
print(end - start)

# cython
# 8.487079620361328 20.2

# python
# 31.13806462287903

# numba
# 12.00000
