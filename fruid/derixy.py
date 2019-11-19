


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
