# cython: profile=True

import math
from utils import *
import nml_regret
import sys
from tree_cl import *
from get_covered_indices_bool import *
import cython

def get_mdl_local_for_rule_set(target, candidate_cut, cut, feature_type, indices,
                               covered_indices_here, uncovered_indices_here,
                               model_cost_so_far, num_class, len_rule, previous_mean_cl,
                               repeated_feature_name_and_cut_option=False):

    cl_model = get_model_cost(candidate_cut=candidate_cut, cut=cut, feature_type=feature_type,
                              model_cost_so_far=model_cost_so_far, len_rule=len_rule,
                              repeated_feature_name_and_cut_option=repeated_feature_name_and_cut_option)
    probs = calc_probs(target[indices], num_class)
    cl_data = -np.sum(np.log2(probs[target[uncovered_indices_here]]))


    reg = nml_regret.regret(len(uncovered_indices_here), num_class)
    total_cl = reg + cl_data + cl_model

    mdl_gain = previous_mean_cl * (len(uncovered_indices_here)) - total_cl
    return [mdl_gain, total_cl, cl_data, reg, cl_model]


def get_mdl_score_given_rule(subgroup_set, target, indices, cl_model,
                             default_MDL_score, membership, num_class, mode,
                             x_train):
    if len(indices) == 0:
        sys.exit("error get_mdl_score: len(indices) is 0!")
    total_neg_log_emp_probs, total_regret, cl_data_this_rule, else_rule_gain, \
        cl_data_excluding_tree, regret_excluding_tree, tree_cl_data, tree_regret = \
        get_emp_neg_log_probs(indices=indices, target=target,
                              membership=membership,
                              subgroup_set=subgroup_set, num_class=num_class,
                              mode=mode, x_train=x_train)
    total_cl_model = cl_model
    for subgroup in subgroup_set:
        total_cl_model = total_cl_model + subgroup.model_cost
    absolute_gain = default_MDL_score - (total_cl_model + total_regret + total_neg_log_emp_probs)
    absolute_gain_excluding_tree = default_MDL_score - (total_cl_model + regret_excluding_tree + cl_data_excluding_tree)

    normalized_gain = (absolute_gain_excluding_tree) / np.sum(
        np.bitwise_or.reduce(membership, axis=0))  # no-else cumulative normalized gain

    return [absolute_gain, normalized_gain, total_regret, cl_model, total_neg_log_emp_probs,
            total_cl_model, total_regret, absolute_gain_excluding_tree, tree_cl_data, tree_regret]


# get the mdl score for the local coverage (rule description, regret, minus_log_likelihood)
def get_mdl_score_local(
        target, candidate_cut, cut, feature_type, indices,
        model_cost_so_far, num_class, len_rule, previous_mean_cl, cached_model_cost, feature_name,
        repeated_feature_name_and_cut_option=False):
    cl_model = get_model_cost(candidate_cut=candidate_cut, cut=cut, feature_type=feature_type,
                              model_cost_so_far=model_cost_so_far, len_rule=len_rule,
                              repeated_feature_name_and_cut_option=repeated_feature_name_and_cut_option)

    probs = calc_probs(target[indices], num_class)
    cl_data = -np.sum(np.log2(probs[target[indices]]))
    reg = nml_regret.regret(len(indices), num_class)
    total_cl = reg + cl_data + cl_model

    mdl_gain = previous_mean_cl * len(indices) - total_cl
    return [mdl_gain, total_cl, cl_data, reg, cl_model]

# get the MDL_score for the whole subgroup_set
# Input:
# subgroup_set: subgroup_set so far
# candidate_cuts: needed for calculating the code length needed to encode the rule
# indices: indices of data points covered by the rule being checked now
# memberships: membership of each data point, a list of lists
def get_mdl_score(subgroup_set, target, candidate_cut, cut, feature_type, indices,
                  default_MDL_score, membership, model_cost_so_far, num_class, mode, len_rule,
                  x_train, repeated_feature_name_and_cut_option=False):
    if len(indices) == 0:
        sys.exit("error get_mdl_score: len(indices) is 0!")
    total_neg_log_emp_probs, total_regret, cl_data_this_rule, else_rule_gain, \
        cl_data_excluding_tree, regret_excluding_tree, tree_cl_data, tree_regret = \
        get_emp_neg_log_probs(indices=indices, target=target,
                              membership=membership,
                              subgroup_set=subgroup_set, num_class=num_class,
                              mode=mode, x_train=x_train)
    cl_model = get_model_cost(candidate_cut=candidate_cut, cut=cut, feature_type=feature_type,
                              model_cost_so_far=model_cost_so_far, len_rule=len_rule,
                              repeated_feature_name_and_cut_option=repeated_feature_name_and_cut_option)

    total_cl_model = cl_model
    for subgroup in subgroup_set:
        # total_cl_model = total_cl_model + subgroup.model_cost - math.log(len(subgroup_set) + 1, 2)
        total_cl_model = total_cl_model + subgroup.model_cost
    total_cl_model = total_cl_model - math.log(len(subgroup_set) + 1, 2) # since the order of rules does not matter!

    absolute_gain = default_MDL_score - (total_cl_model + total_regret + total_neg_log_emp_probs)
    absolute_gain_excluding_tree = default_MDL_score - (total_cl_model + regret_excluding_tree + cl_data_excluding_tree)

    normalized_gain = (absolute_gain - else_rule_gain) / np.sum(np.bitwise_or.reduce(membership, axis=0)) # no-else cumulative normalized gain

    return [absolute_gain, normalized_gain, total_regret, cl_model, total_neg_log_emp_probs,
            total_cl_model, total_regret, absolute_gain_excluding_tree, tree_cl_data, tree_regret]


def get_emp_neg_log_probs(indices, target, membership, subgroup_set, num_class, mode, x_train):

    # get unique membership labels
    # membership[len(subgroup_set), indices] = 1
    indices_bool = np.zeros(len(target), dtype=bool)
    indices_bool[indices] = True
    membership = np.vstack((membership[:len(membership)-1], indices_bool))

    unique_membership_list, corresponding_indices = np.unique(membership, axis=1, return_inverse=True)
    # get probs and cl_data
    cl_data = 0
    regret = 0
    cl_data_excluding_tree = 0
    regret_excluding_tree = 0
    tree_cl_data, tree_regret = 0, 0
    cl_data_this_rule = 0
    for i, unique_membership in enumerate(unique_membership_list.T):
        update_indiecs = np.where(corresponding_indices == i)[0] # indices of this "cover group"

        if np.sum(unique_membership) == 0:
            if mode == CLASSIFICATION:
                overlapping_probs = calc_probs(target[update_indiecs], num_class)
                covered_indices_bool = np.zeros(len(target), dtype=bool)
                covered_indices_bool[update_indiecs] = True

                tree_cl_data, tree_regret = get_tree_cl(x_train.T[covered_indices_bool], target[covered_indices_bool],
                                                        num_class)
                cl_data += tree_cl_data
                regret += tree_regret

                else_rule_prob_default = 0
                else_rule_cl_default = 0
            else:
                sys.exit("error mode in get cl data!")
        else:
            # check whether some rule cover other rule
            if len(membership) == 1: # the case when there is only 1 rule in total
                covered_indices_bool = membership[0]
            else:
                covered_indices_bool = get_covered_indices_bool(unique_membership, membership)
            overlapping_probs = calc_probs(target[covered_indices_bool], num_class)

            cl_data += -np.sum(np.log2(overlapping_probs[target[update_indiecs]]))

        cl_data_excluding_tree += -np.sum(np.log2(overlapping_probs[target[update_indiecs]]))

        if unique_membership[len(unique_membership)-1] == 1:
            cl_data_this_rule += -np.sum(np.log2(overlapping_probs[target[update_indiecs]]))

        else_rule_gain = 0 # initialize else rule gain in case no data points are in the else rule
        if np.sum(unique_membership) == 0:
            else_rule_gain = else_rule_cl_default - (-np.sum(np.log2(overlapping_probs[target[update_indiecs]])))

    for s in subgroup_set:
        regret_plus = nml_regret.regret(len(s.indices), num_class)
        regret += regret_plus
        regret_excluding_tree += regret_plus

    regret_this_rule = nml_regret.regret(len(indices), num_class)
    regret += regret_this_rule
    regret_excluding_tree += regret_this_rule

    return [cl_data, regret, cl_data_this_rule, else_rule_gain, cl_data_excluding_tree, regret_excluding_tree,
            tree_cl_data, tree_regret]


def get_model_cost(candidate_cut, cut, feature_type, model_cost_so_far, len_rule,
                   repeated_feature_name_and_cut_option, no_cl_model=True):
    if no_cl_model:
        return 0

    if repeated_feature_name_and_cut_option:
        sys.exit("feature not used anymore!")
        model_cost = model_cost_so_far
    else:
        if feature_type == 0:
            model_cost = model_cost_so_far + math.log(len(candidate_cut), 2) + 1 # 1 is for "left" and "right"
        else:
            num_levels_chosen = len(cut)
            cl_num_level = math.log(len(candidate_cut), 2)
            cl_which_levels = math.log(math.comb(len(candidate_cut), num_levels_chosen), 2)
            model_cost = model_cost_so_far + cl_num_level + cl_which_levels

        if model_cost < 0:
            print("debug")
    return model_cost


def get_default_MDL_score(target, default_probs):
    n = len(target)
    k = len(default_probs)

    regret = nml_regret.regret(n, k)
    cl_model = 0

    neg_log_emp_probs = -np.sum(np.log2(default_probs[target]))
    score = neg_log_emp_probs + cl_model + regret

    return score

