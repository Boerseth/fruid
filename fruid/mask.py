import numpy as np
import copy

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
        while i < len(elements):
            if len(anti_mask) > 0 and elements[i][axis] == anti_mask[0]:
                del elements[i]
            elif len(anti_mask) > 0 and elements[i][axis] > anti_mask[0]:
                del anti_mask[0]
                shift += 1
            elif len(anti_mask) == 0 or elements[i][axis] < anti_mask[0]:
                elements[i] = (
                    elements[i][0] - (shift if axis == 0 else 0),
                    elements[i][1] - (shift if axis == 1 else 0),
                    elements[i][2],
                )
                i += 1
        return elements


