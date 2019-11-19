import copy

import numpy as np
import scipy.sparse as sp
import pytest


def RK4(h, x, f):
    k1 = h * f(x)
    k2 = h * f(x + 0.5 * k1)
    k3 = h * f(x + 0.5 * k2)
    k4 = h * f(x + k3)
    return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6


class IllegalMultiplicationException(Exception):
    pass


class IllegalDivisionException(Exception):
    pass


class TripleVector:
    def __init__(self, u, v, rho):
        self.u = u
        self.v = v
        self.rho = rho

    def split(self,):
        return (self.u, self.v, self.rho)

    def __neg__(self,):
        return TripleVector(-self.u, -self.v, -self.rho)

    def __add__(self, other):
        return TripleVector(self.u + other.u, self.v + other.v, self.rho + other.rho,)

    def __sub__(self, other):
        return TripleVector(self.u - other.u, self.v - other.v, self.rho - other.rho,)

    def __mul__(self, other):
        if not type(other) in [int, float]:
            raise IllegalMultiplicationException()
        return TripleVector(self.u * other, self.v * other, self.rho * other,)

    def __truediv__(self, other):
        if not type(other) in [int, float]:
            raise IllegalDivisionException()
        return TripleVector(
            self.u.__truediv__(other),
            self.v.__truediv__(other),
            self.rho.__truediv__(other),
        )

    def __rmul__(self, other):
        return self * other

    def __str__(self,):
        return "<u: {}, v: {}, rho: {}>".format(self.u, self.v, self.rho)

    def __repr__(self,):
        return "<TripleVector({}, {}, {})>".format(
            repr(self.u), repr(self.v), repr(self.rho)
        )


def is_sorted(l, compare):
    for l_1, l_2 in zip(l[:-1], l[1:]):
        if not compare(l_1, l_2):
            return False
    return True


class InvalidMaskInputException(Exception):
    pass


class NotMaskableTensorDegreeException(Exception):
    pass


class ObjectNotMaskableException(Exception):
    pass


class Mask:
    def __init__(self, M, mask):
        if not is_sorted(mask, lambda l_1, l_2: l_1 < l_2):
            raise InvalidMaskInputException()
        self._M = M
        self._mask = mask
        self._anti_mask = self._make_anti_mask()

    def _make_anti_mask(self,):
        anti_mask = list([m for m in range(self._M)])
        for m in self._mask[::-1]:
            del anti_mask[m]
        return anti_mask

    def apply_mask(self, x):
        if type(x) == np.ndarray:
            self._apply_mask_on_numpy_array(x)
        elif type(x) == list:
            return self._apply_mask_on_sparse_matrix(x)
        else:
            raise ObjectNotMaskableException()

    def _apply_mask_on_numpy_array(self, np_array):
        degree = len(np_array.shape)
        if degree == 1:
            return self._apply_mask_on_vector(np_array)
        elif degree == 2:
            return self._apply_mask_on_matrix(np_array)
        else:
            raise NotMaskableTensorDegreeException()

    def _apply_mask_on_vector(self, v):
        return v[self._mask]

    def _apply_mask_on_matrix(self, A):
        return A[self._mask, :][:, self._mask]

    def _apply_mask_on_sparse_matrix_inefficiently(self, elements):
        return [
            (i, j, value)
            for i, j, value in elements
            if i in self._mask and j in self._mask
        ]
        # This is very computationally expensive, O(N^4)
        # Better, O(N^2 log(N)):

    def _apply_mask_on_sparse_matrix(self, elements):
        elements = self._filter_axis(elements, axis=0)
        elements = self._filter_axis(elements, axis=1)
        return elements

    def _filter_axis(self, elements, axis):
        elements.sort(key=lambda el: el[axis])
        anti_mask = copy.deepcopy(self._anti_mask)
        i = 0
        shift = 0
        while i < len(elements) and len(anti_mask) > 0:
            if elements[i][axis] == anti_mask[0]:
                del elements[i]
            elif elements[i][axis] > anti_mask[0]:
                del anti_mask[0]
                shift = shift + 1
            elif elements[i][axis] < anti_mask[0]:
                new_element = (
                    elements[i][0] - (shift if axis == 0 else 0),
                    elements[i][1] - (shift if axis == 1 else 0),
                    elements[i][2]
                )
                elements[i] = new_element
                i += 1
        #
        # Shift pass through:
        while i < len(elements):
            new_element = (
                elements[i][0] - (shift if axis == 0 else 0),
                elements[i][1] - (shift if axis == 1 else 0),
                elements[i][2]
            )
            elements[i] = new_element
            i += 1

        return elements


class SpatialDerivative:
    """
    The choice in naming for the derivative matrices is such that, when naming
    instances as e.g. `D_z`, the corresponding derivatives become `D_z.xy`.
    This gets the programmatic notation very close to mathematical (or at least
    my own).
    """

    def __init__(self, N, mask, edge_bc="DIRICHLET", interior_bc="NEUMANN"):
        self.N = N
        self.mask = mask
        self.edge_bc = edge_bc
        self.interior_bc = interior_bc
        self.x = self.make_D_x
        self.y = self.make_D_y
        self.xy = self.make_D_xy
        self.xx = self.make_D_xx
        self.yy = self.make_D_yy

    """
    TODO: Write out derivative expressions
    Note on which sparse matrices to use:
        - `sp.coo_matrix((data, (i,j)), shape=(N^2, N^2))` when building
        - `sp.csr_matrix()` when doing vector products, by `coo.tocsr()`
    """
    def make_D_x(self,):
        pass

    def make_D_y(self,):
        pass

    def make_D_xy(self,):
        pass

    def make_D_xx(self,):
        pass

    def make_D_yy(self,):
        pass


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
    mask = Mask(M, [0, 1, 3, 4])
    data = [i for i in range(M**2)]
    i = [i//M for i in range(M**2)]
    j = [i%M for i in range(M**2)]
    elements = mask.apply_mask(list(zip(i, j, data)))
    new_i = [el[0] for el in elements]
    new_j = [el[1] for el in elements]
    new_data = [el[2] for el in elements]
    new_M = len(mask._mask)
    coo = sp.coo_matrix((new_data, (new_i, new_j)), shape=(new_M,new_M))
    array = coo.toarray()
    #assert all(a == 0 for a in array[2,:])
    #assert all(a == 0 for a in array[:,2])
    print(array)


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
