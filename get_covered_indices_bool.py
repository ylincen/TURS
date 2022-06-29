from numba import jit, njit
import numpy as np


@njit
def get_covered_indices_bool(unique_membership, membership):
    select_rows = np.ones(len(membership), dtype="bool")
    for row_i in range(len(membership) - 1):  # if row_i is a subset of row_j, or the reverse, exclude the bigger one
        if not unique_membership[row_i]:
            continue
        for row_j in range(row_i + 1, len(membership)):
            if not unique_membership[row_j]:
                continue
            if np.all(membership[row_j][membership[row_i]]):
                select_rows[row_j] = False  # exclude row_j
            elif np.all(membership[row_i][membership[row_j]]):
                select_rows[row_i] = False  # exclude row_i
            else:
                pass

    covered_indices_bool = (np.sum(membership[unique_membership & select_rows], axis=0) >= 1)
    return covered_indices_bool