import scipy.sparse as sp


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
        self.shape = (len(self.mask._mask), len(self.mask._mask))
        self.edge_bc = edge_bc
        self.interior_bc = interior_bc
        self.x = self.make_D_x()
        self.y = self.make_D_y()
        self.xy = self.make_D_xy()
        self.xx = self.make_D_xx()
        self.yy = self.make_D_yy()

    """
    TODO: Write out derivative expressions
    Note on which sparse matrices to use:
        - `sp.coo_matrix((data, (i,j)), shape=(N^2, N^2))` when building
        - `sp.csr_matrix()` when doing vector products, by `coo.tocsr()`
    """

    def make_D_x(self,):
        # construct
        data = [1, 1, 1, 1, 1]
        i = [0, 1, 5, 3, 4]
        j = [0, 1, 5, 3, 4]
        return self.mask_and_build(i, j, data)

    def make_D_y(self,):
        pass

    def make_D_xy(self,):
        pass

    def make_D_xx(self,):
        pass

    def make_D_yy(self,):
        pass

    def mask_and_build(self, i, j, data):
        i_m, j_m, data_m = self.mask.apply_on_sparse(i, j, data)
        coo = sp.coo_matrix((data_m, (i_m, j_m)), shape=self.shape)
        csr = sp.csr_matrix(coo)
        return csr
