import itertools
import numpy as np
from constant import *
from numba import njit
from numba import jit, float64, int32, boolean
# Input
# feature type: categorical as 1, numerical as 0
# features: matrix, each column corresponds to one variable
# categorical features are encoded by integer: 0, 1, 2, ...
def generate_candidate_cuts(features, max_num_bin, max_batch_size=5, remove_start_end=True):
    candidate_cuts = [] # for calculating the code length of model
    candidate_cuts_for_search = []
    dim_iter_counter = -1

    for feature, feature_type, max_num_bin_this_dim in zip(features.values, features.types, max_num_bin):
        dim_iter_counter += 1
        if feature_type == CATEGORICAL:
            unique_feature = np.unique(feature)
            candidate_cut_this_dimension = []
            if max_batch_size < len(unique_feature):
                for i in range(max_batch_size):
                    candidate_cut_this_dimension.extend(list(itertools.combinations(unique_feature, r=i+1)))
            else:
                for i in range(len(unique_feature)-1):
                    candidate_cut_this_dimension.extend(list(itertools.combinations(unique_feature, r=i+1)))
            candidate_cuts.append(candidate_cut_this_dimension)
            candidate_cuts_for_search.append(candidate_cut_this_dimension)
        else:
            sort_feature = np.sort(np.unique(feature))
            candidate_cut_this_dimension = (sort_feature[0:(len(sort_feature)-1)] + sort_feature[1:len(sort_feature)])/2

            # to set the bins for each numeric dimension
            if max_num_bin_this_dim > 1:

                select_indices = np.linspace(0, len(candidate_cut_this_dimension)-1, max_num_bin_this_dim+1,
                                             endpoint=True, dtype=int)
                if remove_start_end:
                    select_indices = select_indices[1:(len(select_indices)-1)] # remote the start and end point
                candidate_cuts.append(candidate_cut_this_dimension)
                candidate_cuts_for_search.append(candidate_cut_this_dimension[select_indices])
            else:
                candidate_cuts.append(candidate_cut_this_dimension)
                candidate_cuts_for_search.append(candidate_cut_this_dimension)
    return [candidate_cuts, candidate_cuts_for_search]


# Get the indices when the cut is fixed, during the process of rule growing
# Input:
# original_indice: numpy array
# feature_type: 0 -> NUMERIC, 1 -> CATEGORICAL
# Output:
# left_indices and right_indices, both are python list
def get_indices(feature, feature_type, cut, original_indices):
    feature = feature[original_indices]
    if feature_type == NUMERIC:
        left_indices = original_indices[feature <= cut]
        right_indices = original_indices[feature > cut]
        return [left_indices, right_indices]
    else:
        within_indices = original_indices[np.isin(feature, cut)]
        return within_indices

@njit
def calc_probs(target, num_class, smoothed=False):
    counts = np.bincount(target, minlength=num_class)
    if smoothed:
        counts = counts + np.ones(num_class, dtype='int64')
    if np.sum(counts) == 0:
        return counts / 1.0
    else:
        return counts / np.sum(counts)


def y_to_int(y):
    y_train_new = []
    labels = list(np.unique(y))
    for v in y:
        for i, l in enumerate(labels):
            if v == l:
                y_train_new.append(i)
    return np.array(y_train_new)
