import copy

import numpy as np
import scipy.sparse as sp
import pytest

from fruid.mask import Mask
from fruid.triple import TripleVector
from fruid.derixy import SpatialDerivative
from fruid.derit import *

def RK4(h, x, f):
    k1 = h * f(x)
    k2 = h * f(x + 0.5 * k1)
    k3 = h * f(x + 0.5 * k2)
    k4 = h * f(x + k3)
    return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6


def test_require_that_first_derivatives_work():
    pass


def test_require_that_triple_vector_class_works_as_expected():
    u1 = np.array([1.0, 1.0, 1.0])
    v1 = np.array([1.0, -1.0, 2.0])
    rho1 = np.array([1.0, 2.0, 3.0])
    x1 = TripleVector(u1, v1, rho1)
    x2 = TripleVector(u1, v1, rho1)
    assert type(x1 - x2) == TripleVector
    assert type(x1 + x2) == TripleVector
    assert type(x1 + (-2) * x2) == TripleVector
    assert type(x1 + x2 * (-5)) == TripleVector
    assert type(x1 + x2 / (-5)) == TripleVector
    with pytest.raises(IllegalMultiplicationException):
        assert x1 * x2
    with pytest.raises(IllegalDivisionException):
        assert x1 / x2
    assert type(RK4(1.0, x1, lambda x: -x)) == TripleVector


def test_require_that_anti_mask_works_as_expected():
    with pytest.raises(InvalidMaskInputException):
        mask_object = Mask(1, [1, 0])
    mask_list = [2, 5, 7, 8]
    M = 10
    mask_object = Mask(M, mask_list)
    assert mask_object._anti_mask == [0, 1, 3, 4, 6, 9]
    with pytest.raises(ObjectNotMaskableException):
        mask_object.apply_mask("not valid maskable")
    with pytest.raises(NotMaskableTensorDegreeException):
        mask_object.apply_mask(np.array([[[1.0]]]))


def test_require_that_mask_works_on_sparse():
    M = 5
    mask = Mask(M, [1, 3, 4])
    data = [i for i in range(M ** 2)]
    i = [i // M for i in range(M ** 2)]
    j = [i % M for i in range(M ** 2)]
    elements = mask.apply_mask(list(zip(i, j, data)))
    new_i = [el[0] for el in elements]
    new_j = [el[1] for el in elements]
    new_data = [el[2] for el in elements]
    new_M = len(mask._mask)
    coo = sp.coo_matrix((new_data, (new_i, new_j)), shape=(new_M, new_M))
    array = coo.toarray()
    expected_array = np.array([[6, 8, 9], [16, 18, 19], [21, 23, 24]])
    assert np.allclose(array, expected_array)


test_require_that_triple_vector_class_works_as_expected()
test_require_that_anti_mask_works_as_expected()
test_require_that_mask_works_on_sparse()


class TimeDerivative(object):
    def __init__(self, N, mu, kappa, p, D_u, D_rho):
        #  type: (Int, Float, Float, Ndarray, SpatialDerivative, SpatialDerivative) -> None
        self.N = N
        self.mu = mu
        self.kappa = kappa
        self.p = p
        self.D_u = D_u
        self.D_rho = D_rho


class TimeDerivativeU(TimeDerivative):
    """ 
    An instance `dudt` is essentially a function,
        (R^N², R^N², R^N²)  ╶─>  R^N²
               (u, v, rho)  ├─>  ∂u/∂t
    """

    def __init__(self, *args):
        super(TimeDerivativeU, self).__init__(*args)

    def __call__(self, triple_vector):
        # TODO: Write the actual formula
        return -triple_vector.u


class TimeDerivativeV(TimeDerivative):
    """ 
    An instance `dvdt` is essentially a function,
        (R^N², R^N², R^N²)  ╶─>  R^N²
               (u, v, rho)  ├─>  ∂v/∂t
    """

    def __init__(self, *args):
        super(TimeDerivativeV, self).__init__(*args)

    def __call__(self, triple_vector):
        # TODO: Write the actual formula
        return -triple_vector.v


class TimeDerivativeRho(TimeDerivative):
    """ 
    An instance `drdt` is essentially a function,
        (R^N², R^N², R^N²)  ╶─>  R^N²
               (u, v, rho)  ├─>  ∂rho/∂t
    """

    def __init__(self, *args):
        super(TimeDerivativeRho, self).__init__(*args)

    def __call__(self, triple_vector):
        # TODO: Write the actual formula
        return -triple_vector.rho


class TimeDerivativeTriple:
    """ 
    An instance `dxdt` is essentially a function,
        (R^N², R^N², R^N²)  ╶─>  (R^N², R^N², R^N²)
               (u, v, rho)  ├─>  (∂u/∂t, ∂v/∂t, ∂rho/∂t)
    """

    def __init__(self, dudt, dvdt, drdt):
        self.dudt = dudt
        self.dvdt = dvdt
        self.drdt = drdt

    def __call__(self, triple_vector):
        return TripleVector(
            self.dudt(triple_vector),
            self.dvdt(triple_vector),
            self.drdt(triple_vector),
        )


def solver_example(N, h, N_t, mu, kappa, x0, p, mask):
    """
    Builds the function `f` for the PDE
        ∂x/∂t = f(x)
    and numerically integrates with time step `h` by Runge-Kutta 4.
    """
    D_u = SpatialDerivative(N, mask, edge_bc="NEUMANN", interior_bc="DIRICHLET")
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
