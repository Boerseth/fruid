from .triple import TripleVector


def RK4(h, x, f):
    k1 = h * f(x)
    k2 = h * f(x + 0.5 * k1)
    k3 = h * f(x + 0.5 * k2)
    k4 = h * f(x + k3)
    return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6


class _TimeDerivative(object):
    def __init__(self, N, mu, kappa, p, D_u, D_rho):
        #  type: (Int, Float, Float, Ndarray, SpatialDerivative, SpatialDerivative) -> None
        self.N = N
        self.mu = mu
        self.kappa = kappa
        self.p = p
        self.D_u = D_u
        self.D_rho = D_rho


class TimeDerivativeU(_TimeDerivative):
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


class TimeDerivativeV(_TimeDerivative):
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


class TimeDerivativeRho(_TimeDerivative):
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
