from GrowRuleIgnoreOverlap import *
from GrowRuleFuther import *
from joblib import Parallel, delayed
from calculate_NML import *
from get_model_cost import *

def further_grow_for_ruleset(subgroup_set, new_subgroup_init,
                             best_absolute_gain_init, exclude_score_equal_include_score_init,
                             previous_best_absolute_gain, target, features,
                             parameters, candidate_cuts_for_search, covered_indices,
                             nbeam, memberships, memberships_cumulative,
                             mode, default_MDL_score, cached_model_cost):
    if len(subgroup_set) == 0:
        new_subgroup, best_absolute_gain, exclude_score_equal_include_score = \
            new_subgroup_init, best_absolute_gain_init, exclude_score_equal_include_score_init
    elif len(new_subgroup_init.rule) == 0:
        new_subgroup, best_absolute_gain, exclude_score_equal_include_score = \
            new_subgroup_init, previous_best_absolute_gain, exclude_score_equal_include_score_init
    else:
        new_subgroup, best_absolute_gain, exclude_score_equal_include_score = \
            GrowRuleNew.grow_further_onlist(new_subgroup_init, target, subgroup_set, features,
                                            parameters, candidate_cuts_for_search,
                                            covered_indices, nbeam,
                                            memberships, memberships_cumulative, mode, default_MDL_score,
                                            cached_model_cost=cached_model_cost,
                                            use_surrogate_score=True)
        if len(new_subgroup.rule) == 0:  # no new condition is added to the rule_base
            new_subgroup, best_absolute_gain, exclude_score_equal_include_score = \
                new_subgroup_init, best_absolute_gain, exclude_score_equal_include_score

    return [new_subgroup, best_absolute_gain, exclude_score_equal_include_score]

def find_rule_set_initby_list_paralle(target, features, parameters, mode, num_cores, nbeam, num_initial_rule_list,
                                      early_stop=True, early_num=20):
    subgroup_set = []  # an array of subgroup objects
    memberships = np.zeros((1, len(target)), dtype='bool')
    memberships_cumulative = copy.deepcopy(memberships[0])
    memberships_for_list = np.zeros((1, len(target)), dtype='bool')

    default_probs = calc_probs(target, parameters.num_class)  # return a numpy array corresponding to 0, 1, 2, ...
    default_MDL_score = get_default_MDL_score(target, default_probs)
    candidate_cuts, candidate_cuts_for_search = generate_candidate_cuts(features=features,
                                                                        max_num_bin=parameters.max_num_bin,
                                                                        max_batch_size=parameters.max_levels_per_batch)
    previous_best_absolute_gain = 0
    stop_adding_new_rule = False
    covered_indices = None

    # cached model cost
    cached_model_cost = get_cache_model_cost(features, candidate_cuts_for_search)

    # below are used to determine the number of rules in the rule set.
    absolute_gain_list = []
    covered_indices_list = []
    for i in range(parameters.max_num_subgroups):
        # best_absolute_gain is the gain excluding the tree (non-surrogate score)
        new_subgroup_init_s, best_absolute_gain_init_s, exclude_score_equal_include_score_init_s = \
            GrowIndividualRule.grow_by_mdl_gain_for_rule_list(target, subgroup_set, features, parameters,
                                                              candidate_cuts_for_search, covered_indices, nbeam,
                                                              memberships_for_list, memberships_cumulative,
                                                              mode, default_MDL_score,
                                                              num_best_rules_return=num_initial_rule_list,
                                                              cached_model_cost=cached_model_cost,
                                                              use_surrogate_score=True)
        best_best_absolute_gain = -np.inf
        if num_cores == 1:
            results = [further_grow_for_ruleset(subgroup_set, new_subgroup_init,
                                                  best_absolute_gain_init, exclude_score_equal_include_score_init,
                                                  previous_best_absolute_gain, target, features,
                                                  parameters, candidate_cuts_for_search, covered_indices,
                                                  nbeam, memberships, memberships_cumulative,
                                                  mode, default_MDL_score, cached_model_cost=cached_model_cost)
                for new_subgroup_init, best_absolute_gain_init,
                    exclude_score_equal_include_score_init in
                zip(new_subgroup_init_s, best_absolute_gain_init_s,
                    exclude_score_equal_include_score_init_s)]
        else:
            results = Parallel(n_jobs=num_cores)(delayed(further_grow_for_ruleset)(subgroup_set, new_subgroup_init,
                                 best_absolute_gain_init, exclude_score_equal_include_score_init,
                                 previous_best_absolute_gain, target, features,
                                 parameters, candidate_cuts_for_search, covered_indices,
                                 nbeam, memberships, memberships_cumulative,
                                 mode, default_MDL_score, cached_model_cost=cached_model_cost)
                                                 for new_subgroup_init, best_absolute_gain_init,
                                                     exclude_score_equal_include_score_init in
                                                 zip(new_subgroup_init_s, best_absolute_gain_init_s,
                                                     exclude_score_equal_include_score_init_s))

        for j in range(len(results)):
            if results[j][1] > best_best_absolute_gain:
                best_best_absolute_gain = results[j][1]
                best_new_subgroup = results[j][0]
                best_exclude_score_equal_include_score = results[j][2]

        new_subgroup = best_new_subgroup
        best_absolute_gain = best_best_absolute_gain
        exclude_score_equal_include_score = best_exclude_score_equal_include_score

        early_stop_flag = False
        if early_stop and len(absolute_gain_list) > early_num:
            dif = np.diff(absolute_gain_list)
            if all(dif[(len(dif) - early_num):] < 0):
                early_stop_flag = True
                print("early stop with ", early_num, " consecutive absolute gain decrease!")
            else:
                early_stop_flag = False

        if len(new_subgroup.rule) > 0 and i < parameters.max_num_subgroups - 1 and not stop_adding_new_rule and \
                best_absolute_gain > 0 and not early_stop_flag:
            if exclude_score_equal_include_score:
                stop_adding_new_rule = True

            absolute_gain_list.append(best_absolute_gain)
            memberships[-1][new_subgroup.indices] = 1

            covered_indices2 = np.where(np.bitwise_or(memberships[-1], memberships_cumulative))[0]
            if len(subgroup_set) > 0:
                if len(covered_indices) > 0:
                    update_indices = np.setdiff1d(new_subgroup.indices, covered_indices2)
                else:
                    update_indices = new_subgroup.indices
            else:
                update_indices = new_subgroup.indices
            memberships_for_list[-1][update_indices] = 1

            subgroup_set.append(new_subgroup)
            previous_best_absolute_gain = best_absolute_gain

            memberships_cumulative = np.bitwise_or(memberships[-1], memberships_cumulative)

            memberships = np.vstack((memberships, np.zeros((1, len(target)), dtype=bool)))
            memberships_for_list = np.vstack((memberships_for_list, np.zeros((1, len(target)), dtype=bool)))
            covered_indices = np.where(memberships_cumulative)[0]
            covered_indices_list.append(covered_indices)

        elif len(new_subgroup.rule) > 0 and i == parameters.max_num_subgroups - 1:
            sys.exit("needs more iterations (parameters.max_num_subgroups)! ")

        else:
            best_num_rules = np.argmax(absolute_gain_list)

            subgroup_set = subgroup_set[:best_num_rules + 1]
            memberships = memberships[:best_num_rules + 1]
            return [subgroup_set, memberships, default_MDL_score - absolute_gain_list[best_num_rules],
                    absolute_gain_list[best_num_rules]]
