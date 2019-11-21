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
