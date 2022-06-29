# This script defines the Subgroup class, assuming that the target is categorical;
import numpy as np
import math
from constant import *

class Subgroup:
    def __init__(self, rule, cuts, cuts_options, features_names, indices, regret, model_cost,
                 indices_list, cl_model_list, encoded_feature, total_cl_per_data=np.nan, indices_for_score=[], indices_for_score_list=[]):
        self.rule = rule
        self.cuts = cuts
        self.cuts_options = cuts_options
        self.features_names = features_names
        self.indices = indices
        self.regret = regret
        self.model_cost = model_cost
        self.indices_list = indices_list  # the list of all best_indices, for the whole process of rule growing
        self.cl_model_list = cl_model_list  # the list of best CL of model, for the whole process of rule growing
        # NOTE: total_cl_per_data DOES NOT INCLUDE the instances covered by previous rules in the rule set, as
        #       this is used for the local heuristics in the search.
        self.total_cl_per_data = total_cl_per_data

        if len(indices_for_score) == 0:
            self.indices_for_score = indices
            self.indices_for_score_list = indices_list
        else:
            self.indices_for_score = indices_for_score
            self.indices_for_score_list = indices_for_score_list
        self.encoded_feature = encoded_feature


    def get_model_cost(self, cached_model_cost, nocl=True):
        if nocl:
            return 0
        # cl_num_vars = np.log2(len(self.encoded_feature)) # uniform prior on how many variables to encode
        # num_vars = np.sum(self.encoded_feature) > 0
        # cl_which_vars = np.log2(math.comb(len(self.encoded_feature), num_vars))
        #
        # encoded_feature_left = np.zeros(len(self.encoded_feature), dtype=bool)
        # encoded_feature_right = np.zeros(len(self.encoded_feature), dtype=bool)
        # encoded_feature_within = np.zeros(len(self.encoded_feature), dtype=bool)
        # cl_model = cl_num_vars + cl_which_vars
        #
        # for name, option in zip(self.features_names, self.cuts_options):
        #     if self.encoded_feature[name] == 0:
        #         continue
        #
        #     if option == WITHIN_CUT and ~encoded_feature_within[name]:
        #         encoded_feature_within[name] = True
        #         cl_model += cached_model_cost[name]
        #     elif option == LEFT_CUT and ~encoded_feature_left[name]:
        #         encoded_feature_left[name] = True
        #         cl_model += cached_model_cost[name] + 1 # one is for left or right
        #     elif option == RIGHT_CUT and ~encoded_feature_right[name]:
        #         encoded_feature_right[name] = True
        #         cl_model += cached_model_cost[name] + 1
        #     else:
        #         continue
        # # for one numeric variable with two cuts, we do not need to encode left & right, and we can switch the order,
        # # so we save 3 bits in total
        # cl_model = cl_model - 3 * np.sum(np.bitwise_and(encoded_feature_right, encoded_feature_left))
        #
        # # for edge case, say, numeric variable with only one value.
        # if cl_model < 0:
        #     cl_model = 0
        # return cl_model


class GivenRule:
    def __init__(self, rule, cuts, cuts_options, features_names, indices, regret, model_cost,
                 indices_list, cl_model_list, best_gain_incl_list, best_gain_excl_list, best_regret_list):
        self.rule = rule
        self.cuts = cuts
        self.cuts_options = cuts_options
        self.features_names = features_names
        self.indices = indices
        self.regret = regret
        self.model_cost = model_cost
        self.indices_list = indices_list  # the list of all best_indices, for the whole process of rule growing
        self.cl_model_list = cl_model_list  # the list of best CL of model, for the whole process of rule growing

        self.best_gain_incl_list = best_gain_incl_list
        self.best_gain_excl_list = best_gain_excl_list
        self.best_regret_list = best_regret_list

class Features:
    def __init__(self, names, values, types):
        self.names = names
        self.values = values
        self.types = types


class Parameters:
    def __init__(self, max_depth, max_levels_per_batch, max_num_subgroups, max_iter, max_num_bin, num_class):
        self.max_depth = max_depth
        self.max_levels_per_batch = max_levels_per_batch
        self.max_num_subgroups = max_num_subgroups
        self.max_iter = max_iter
        self.max_num_bin = max_num_bin
        self.num_class = num_class

