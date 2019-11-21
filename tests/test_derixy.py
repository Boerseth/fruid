import sys

import numpy as np

sys.path.append("..")
from fruid.mask import Mask
from fruid.derixy import SpatialDerivative


def test_require_that_first_derivatives_work():
    N = 3
    mask = Mask(N * N, [i for i in range(N * N) if i != 2])
    assert len(mask._mask) == N * N - 1
    D_rho = SpatialDerivative(N, mask, edge_bc="DIRICHLET", interior_bc="NEUMANN")
    rho = mask.apply_on_dense(np.array([4 + (i // N) + (i % N) for i in range(N ** 2)]))
    print(rho)
    D_x = D_rho.x.toarray()
    print(D_x)
    drhodx = D_x.dot(rho)
    print(drhodx)


test_require_that_first_derivatives_work()
