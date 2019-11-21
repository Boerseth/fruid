import sys

import numpy as np
import pytest

sys.path.append("..")
from fruid.triple import (
    TripleVector,
    IllegalMultiplicationException,
    IllegalDivisionException,
)
from fruid.derit import RK4


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
