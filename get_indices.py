# cython: profile=True

import cython
from constant import *
import numpy as np


def get_indices(feature, feature_type, cut, original_indices):
    feature = feature[original_indices]
    if feature_type == NUMERIC:
        left_indices = original_indices[feature <= cut]
        right_indices = original_indices[feature > cut]
        return [left_indices, right_indices]
    else:
        within_indices = original_indices[np.isin(feature, cut)]
        return within_indices