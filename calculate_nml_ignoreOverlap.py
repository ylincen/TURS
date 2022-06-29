from calculate_NML import *


def get_mdl_score_given_rule_for_list(subgroup_set, target, indices, cl_model,
                                      default_MDL_score, membership, num_class, mode,
                                      x_train, covered_indices):
    if len(indices) == 0:
        sys.exit("error get_mdl_score: len(indices) is 0!")

    total_cl = 0
    for s in subgroup_set:
        total_cl += s.total_cl_per_data * len(s.indices_for_score)

    probs = calc_probs(target[indices], num_class)
    cl_this_rule = -np.sum(np.log2(probs[target[indices]])) + nml_regret.regret(len(indices), num_class)

    if len(membership) > 0:
        uncovered_indices_for_tree = (np.sum(membership, axis=0) == 0)
    else:
        uncovered_indices_for_tree = np.ones(len(target), dtype=bool)
    uncovered_indices_for_tree[indices] = False

    if np.any(uncovered_indices_for_tree):
        tree_cl_data, tree_regret = get_tree_cl(x_train.T[uncovered_indices_for_tree],
                                                target[uncovered_indices_for_tree],
                                                num_class)

        probs_else = calc_probs(target[uncovered_indices_for_tree], num_class)
        else_rule_total_cl = -np.sum(np.log2(probs_else[target[uncovered_indices_for_tree]])) + \
                             nml_regret.regret(np.sum(uncovered_indices_for_tree), num_class)
    else:
        tree_cl_data, tree_regret, else_rule_total_cl = 0, 0, 0

    total_cl_excl_tree = total_cl + cl_this_rule + else_rule_total_cl
    total_cl = total_cl + cl_this_rule + tree_cl_data + tree_regret

    absolute_gain = default_MDL_score - total_cl
    absolute_gain_excluding_tree = default_MDL_score - total_cl_excl_tree
    normalized_gain = absolute_gain_excluding_tree / len(indices)

    return [absolute_gain, normalized_gain, absolute_gain_excluding_tree, tree_cl_data, tree_regret]

