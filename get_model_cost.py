# This script collects methods that cache and calculate the code length of model
import numpy as np


def get_cache_model_cost(features, candidate_cuts):
    cached_model_cost = {}
    for feature_type, feature_name, candidate_cut in \
            zip(features.types, features.names, candidate_cuts):
        if len(candidate_cut) == 0:
            model_cost = 0
        else:
            if feature_type == 0:
                model_cost = np.log2(len(candidate_cut)) + 1  # 1 is for "left" and "right"
            else:
                model_cost = np.log2(len(candidate_cut))

        cached_model_cost[feature_name] = model_cost
    return cached_model_cost
