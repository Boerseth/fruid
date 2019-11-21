import copy

import numpy as np

from fruid.mask import Mask
from fruid.triple import TripleVector
from fruid.derixy import SpatialDerivative
from fruid.derit import (
    RK4,
    TimeDerivativeU,
    TimeDerivativeV,
    TimeDerivativeRho,
    TimeDerivativeTriple,
)


# test_require_that_triple_vector_class_works_as_expected()
# test_require_that_anti_mask_works_as_expected()
# test_require_that_mask_works_on_sparse()


def solver_example(N, h, N_t, mu, kappa, x0, p, mask):
    """
    Builds the function `f` for the PDE
        ∂x/∂t = f(x)
    and numerically integrates with time step `h` by Runge-Kutta 4.
    """
    D_u = SpatialDerivative(N, mask, edge_bc="NEUMANN", interior_bc="DIRICHLET")
    D_rho = SpatialDerivative(N, mask, edge_bc="DIRICHLET", interior_bc="NEUMANN")
    dudt = TimeDerivativeU(N, mu, kappa, p, mask, D_u, D_rho)
    dvdt = TimeDerivativeV(N, mu, kappa, p, mask, D_u, D_rho)
    drdt = TimeDerivativeRho(N, mu, kappa, p, mask, D_u, D_rho)

    f = TimeDerivativeTriple(dudt, dvdt, drdt)

    states = [x0]
    for n_t in range(N_t):
        states.append(RK4(h, states[-1], f))
    return states


if __name__ != "__main__":
    N = 50
    N_t = 10
    h = 0.1
    mask = Mask(N * N, list([i for i in range(N * N)]))

    mu = 1.0
    zeta = 1.0
    kappa = zeta + (mu / 3)
    p = mask.apply_mask(np.zeros(N * N))  # TODO: Actually define it from function

    u0 = mask.apply_mask(np.zeros(N * N))
    v0 = mask.apply_mask(np.zeros(N * N))
    rho0 = copy.deepcopy(p)

    x0 = TripleVector(u0, v0, rho0)

    solution = solver_example(N, h, N_t, mu, kappa, x0, p, mask)
