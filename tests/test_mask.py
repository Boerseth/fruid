import sys

import numpy as np
from scipy.sparse import coo_matrix
import pytest

sys.path.append("..")
from fruid.mask import (
    Mask,
    InvalidMaskInputException,
    NotMaskableTensorDegreeException,
)


def test_require_that_anti_mask_works_as_expected():
    with pytest.raises(InvalidMaskInputException):
        mask_object = Mask(1, [1, 0])
    mask_list = [2, 5, 7, 8]
    M = 10
    mask_object = Mask(M, mask_list)
    assert mask_object._anti_mask == [0, 1, 3, 4, 6, 9]
    with pytest.raises(NotMaskableTensorDegreeException):
        mask_object.apply_on_dense(np.array([[[1.0]]]))


def test_require_that_mask_works_on_sparse():
    M = 5
    mask = Mask(M, [1, 3, 4])
    data = [i for i in range(M ** 2)]
    i = [i // M for i in range(M ** 2)]
    j = [i % M for i in range(M ** 2)]

    new_i, new_j, new_data = mask.apply_on_sparse(i, j, data)
    new_M = len(mask._mask)
    coo = coo_matrix((new_data, (new_i, new_j)), shape=(new_M, new_M))
    array = coo.toarray()

    expected_array = np.array([[6, 8, 9], [16, 18, 19], [21, 23, 24]])
    assert np.allclose(array, expected_array)
