import copy

import numpy as np

N = 50
h = 0.1
mask = Mask([0.0] * (N * N))


def apply_mask(x):
    return np.ma.masked_array(x, mask)


mu = 1.0
zeta = 1.0
kappa = zeta + (mu / 3)
p = 0  # TODO

u_0 = mask.apply_mask(np.zeros(N * N))
v_0 = mask.apply_mask(np.zeros(N * N))
rho_0 = mask.apply_mask(copy.deepcopy(p))  # TODO


# I think the following structure makes sense:


class IllegalMultiplicationException(Exception):
    pass


class TripleVector:
    def __init__(self, u, v, rho):
        self.u = u
        self.v = v
        self.rho = rho

    def __add__(self, other):
        return TripleVector(self.u + other.u, self.v + other.v, self.rho + other.rho,)

    def __sub__(self, other):
        return TripleVector(self.u - other.u, self.v - other.v, self.rho - other.rho,)

    def __mul__(self, other):
        if not type(other) in [int, float]:
            raise IllegalMultiplicationException()
        return TripleVector(self.u * other, self.v * other, self.rho * other,)

    def __rmul__(self, other):
        return self * other


class Mask:
    def __init__(self, mask):
        self._mask = mask

    def apply_mask(self, x):
        """
        This function may not need all the asserts.
        Move to other functions? Remove?
        """
        dimensions = len(x.shape)
        if dimensions == 1:
            assert len(x) >= len(mask)
            return self._apply_mask_on_vector(x)
        elif dimensions == 2:
            assert len(x) == len(x[0])
            assert len(x) >= len(mask)
            return self._apply_mask_on_matrix(x)
        elif dimensions == 3:
            assert len(x) == len(x[0])
            assert len(x) == len(x[0][0])
            assert len(x) >= len(mask)
            return self._apply_mask_on_tensor(x)

    def _apply_mask_on_vector(self, v):
        return v[self._mask]

    def _apply_mask_on_matrix(self, A):
        # TODO: will not work for sparse matrices
        return A[self._mask, :][:, self._mask]

    def _apply_mask_on_tensor(self, T):
        # TODO: will not work for sparse tensors
        return T[self._mask, :, :][:, self._mask, :][:, :, self._mask]


class TimeDerivative(object):
    def __init__(
        self, N, mu, kappa, p, mask, edge_bc="DIRICHLET", interior_bc="NEUMANN"
    ):
        self.N = N
        self.mu = mu
        self.kappa = kappa
        self.p = p
        self.mask = mask
        self.edge_bc = edge_bc
        self.interior_bc = interior_bc
        self.D_x = self.make_D_x
        self.D_y = self.make_D_y
        self.D_xy = self.make_D_xy
        self.D_xx = self.make_D_xx
        self.D_yy = self.make_D_yy

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


class TimeDerivativeU(TimeDerivative):
    def __init__(self, *args):
        super(TimeDerivativeU, self).__init__(*args)

    def __call__(self, triple_vector):
        """ 
        With this, the class becomes a mathematical function,
        D: (R^N, R^N, R^N)  -->  R^N
                 u, v, rho  |->  ∂u/∂t(u,v,rho) 
        """
        # TODO
        return triple_vector.u


class TimeDerivativeV(TimeDerivative):
    def __init__(self, *args):
        super(TimeDerivativeV, self).__init__(*args)

    def __call__(self, triple_vector):
        # TODO
        return triple_vector.v


class TimeDerivativeRho(TimeDerivative):
    def __init__(self, *args):
        super(TimeDerivativeRho, self).__init__(*args)

    def __call__(self, triple_vector):
        # TODO
        return triple_vector.rho


class TimeDerivativeTriple:
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
