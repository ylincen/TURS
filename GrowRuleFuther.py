# cython: profile=True

from Subgroup import *
from calculate_NML import *
import copy


class GrowRuleNew:
    @staticmethod
    def grow_further_onlist(rule_base, target, subgroup_set, features, parameters, candidate_cuts_for_search,
                            covered_indices, nbeam,
                            memberships, memberships_cumulative, mode, default_MDL_score, cached_model_cost,
                            use_surrogate_score=True):
        if len(subgroup_set) == 0:
            sys.exit("grow_further_onlist must have subgroup_set with length larger than 0!")

        original_indices = np.arange(start=0, stop=len(target), step=1)

        # initialize things
        exclude_score_equal_include_score = False
        jarcard_dist_threshold = 0.95
        covered_indices_bool = np.zeros(len(target), dtype=bool)
        covered_indices_bool[covered_indices] = True
        if len(covered_indices) == 0:
            uncovered_indices = rule_base.indices
            uncovered_indices_bool = np.zeros(len(target), dtype=bool)
            uncovered_indices_bool[rule_base.indices] = True
        else:
            uncovered_indices_bool = np.zeros(len(target), dtype=bool)
            uncovered_indices_bool[rule_base.indices] = True
            uncovered_indices_bool[covered_indices] = False

        probs = calc_probs(target[rule_base.indices], parameters.num_class)
        cl_data = -np.sum(np.log2(probs[target[uncovered_indices_bool]]))
        reg = nml_regret.regret(np.sum(uncovered_indices_bool), parameters.num_class)
        total_cl_per_data = (cl_data + reg + rule_base.model_cost) / np.sum(uncovered_indices_bool)
        rule_base.total_cl_per_data = total_cl_per_data

        # initialize the beam search
        rule_beam_list = [] # rule beams for every rule length.
        rule_beam = [] # a set of rules
        for i in range(nbeam):
            rule_beam.append(copy.deepcopy(rule_base))
        rule_beam_list.append(rule_beam)
        # start searching
        for d in range(parameters.max_depth):
            res = {} # dictionary to store the results for each candidate cut point
            for rule_base in rule_beam:
                for feature, feature_type, feature_name in zip(features.values, features.types, features.names):
                    for cut in candidate_cuts_for_search[feature_name]:
                        if feature_type == NUMERIC:  # numeric features
                            both_side_indices = get_indices(feature=feature, feature_type=feature_type, cut=cut,
                                                            original_indices=rule_base.indices)
                            for cut_option in [LEFT_CUT, RIGHT_CUT]:
                                indices = both_side_indices[cut_option]
                                if len(indices) == len(rule_base.indices):
                                    continue
                                if covered_indices is None:
                                    uncovered_indices_here = indices
                                    covered_indices_here = np.array([],dtype=bool)
                                else:
                                    indices_bool = np.zeros(len(target), dtype=bool)
                                    indices_bool[indices] = True
                                    uncovered_indices_here = copy.deepcopy(indices_bool)
                                    uncovered_indices_here[covered_indices_bool] = False
                                    uncovered_indices_here = np.where(uncovered_indices_here)[0]

                                    covered_indices_here = np.bitwise_and(covered_indices_bool, indices_bool)
                                if np.sum(uncovered_indices_here) == 0:
                                    continue

                                mdl_gain, total_cl, cl_data, reg, cl_model = get_mdl_local_for_rule_set(
                                    target, candidate_cuts_for_search[feature_name], cut,
                                    feature_type, indices, covered_indices_here, uncovered_indices_here,
                                    model_cost_so_far=rule_base.model_cost, num_class=parameters.num_class,
                                    len_rule=len(rule_base.rule), previous_mean_cl=rule_base.total_cl_per_data,
                                    repeated_feature_name_and_cut_option=False)

                                if cut_option == LEFT_CUT:
                                    direction = "<="
                                else:
                                    direction = ">"
                                res[(rule_base, feature_name, cut, cut_option)] = \
                                    {"mdl_gain": mdl_gain,
                                     "total_cl": total_cl,
                                     "indices": indices,
                                     "indices_for_score": indices,
                                     "feature_name": feature_name,
                                     "cut": cut,
                                     "cut_option": cut_option,
                                     "feature_type": feature_type,
                                     "cl_model": cl_model,
                                     "regret": reg,
                                     "total_cl_per_data": total_cl/(sum(uncovered_indices_here)),
                                     "grow_step": 'X' + str(feature_name) + direction + str(cut)}
                        else: # for categorical features
                            indices = get_indices(feature=feature, feature_type=feature_type, cut=cut,
                                                  original_indices=rule_base.indices)
                            if len(indices) == len(rule_base.indices):
                                continue
                            if covered_indices is None:
                                uncovered_indices_here = indices
                                covered_indices_here = np.array([], dtype=bool)
                            else:
                                indices_bool = np.zeros(len(target), dtype=bool)
                                indices_bool[indices] = True
                                uncovered_indices_here = copy.deepcopy(indices_bool)
                                uncovered_indices_here[covered_indices_bool] = False
                                uncovered_indices_here = np.where(uncovered_indices_here)[0]
                                covered_indices_here = np.bitwise_and(covered_indices_bool, indices_bool)
                            if len(uncovered_indices_here) == 0:
                                continue

                            cut_option = WITHIN_CUT

                            mdl_gain, total_cl, cl_data, reg, cl_model = get_mdl_local_for_rule_set(
                                target, candidate_cuts_for_search[feature_name], cut,
                                feature_type, indices, covered_indices_here, uncovered_indices_here,
                                model_cost_so_far=rule_base.model_cost, num_class=parameters.num_class,
                                len_rule=len(rule_base.rule), previous_mean_cl=rule_base.total_cl_per_data,
                                repeated_feature_name_and_cut_option=False)

                            res[(rule_base, feature_name, cut, cut_option)] = \
                                {"mdl_gain": mdl_gain,
                                 "total_cl": total_cl,
                                 "indices": indices,
                                 "indices_for_score": indices,
                                 "feature_name": feature_name,
                                 "cut": cut,
                                 "cut_option": cut_option,
                                 "feature_type": feature_type,
                                 "cl_model": cl_model,
                                 "regret": reg,
                                 "total_cl_per_data": total_cl/(len(uncovered_indices_here)),
                                 "grow_step": 'X' + str(feature_name) + "in" + str(cut)}

            # search for the best nbeam results
            res_values = list(res.values())
            res_keys = list(res.keys())

            mdl_gain_list = []
            for v in res.values():
                mdl_gain_list.append(v["mdl_gain"])
            if len(mdl_gain_list) == 0:
                break

            if np.max(mdl_gain_list) < 0: # stop growing when we have negative mdl_gain
                break
            else:
                mdl_gain_sorted_index = np.arange(len(mdl_gain_list))[np.argsort(-np.array(mdl_gain_list))]
                best_mdl_gain_index = []
                for kk, ind in enumerate(mdl_gain_sorted_index):
                    if len(best_mdl_gain_index) >= nbeam:
                        break

                    if kk == 0:
                        best_mdl_gain_index.append(ind)

                    flag_skip = False # indicate whether to skip this (kk, ind) because one rule in the best_mdl_gain_index
                                      # has a very similar coverage;
                    # check the indices similarity
                    for ll, best_ind in enumerate(best_mdl_gain_index):
                        jarcard_dist = \
                            len(np.intersect1d(res_values[best_ind]["indices_for_score"],
                                               res_values[ind]["indices_for_score"])) / \
                            len(np.union1d(res_values[best_ind]["indices_for_score"],
                                           res_values[ind]["indices_for_score"]))
                        if jarcard_dist > jarcard_dist_threshold:
                            flag_skip = True
                            break
                    if flag_skip:
                        pass
                    else:
                        best_mdl_gain_index.append(ind)

                best_mdl_gain_index = np.array(best_mdl_gain_index)
            # update the beam
            rule_beam = []
            for i in best_mdl_gain_index:
                values_here = res_values[i]
                keys_here = res_keys[i]

                rule_base_here = copy.deepcopy(keys_here[0])

                rule_here = rule_base_here.rule
                rule_here.append(values_here["grow_step"])

                cuts_here = rule_base_here.cuts
                cuts_here.append(values_here["cut"])

                cuts_option_here = rule_base_here.cuts_options
                cuts_option_here.append(values_here["cut_option"])

                features_names_here = rule_base_here.features_names
                features_names_here.append(values_here["feature_name"])

                indices_here = values_here["indices"]
                reg_here = values_here["regret"]
                model_cost_here = values_here["cl_model"]

                cl_model_list_here = rule_base_here.cl_model_list
                cl_model_list_here.append(values_here["cl_model"])

                indice_list_here = rule_base_here.indices # deprecated!!!

                encoded_feature_here = rule_base_here.encoded_feature
                encoded_feature_here[values_here["feature_name"]] += 1

                rule_beam.append(Subgroup(rule=rule_here, cuts=cuts_here, cuts_options=cuts_option_here,
                                          features_names=features_names_here, indices=indices_here, regret=reg_here,
                                          model_cost=model_cost_here,
                                          cl_model_list=cl_model_list_here, indices_list=indice_list_here,
                                          total_cl_per_data=values_here["total_cl_per_data"],
                                          encoded_feature=encoded_feature_here)) # udpate
            rule_beam_list.append(rule_beam)
        if len(rule_beam_list) == 1: # i.e., no new rule is found
            absolute_gain, normalized_gain, regret, model_cost, total_emp_neg_log_probs, \
            total_model_cost, total_regret, absolute_gain_excluding_tree, \
            tree_cl_data, tree_regret \
                = get_mdl_score_given_rule(subgroup_set, target, rule_base.indices, rule_base.model_cost,
                                           default_MDL_score, memberships, parameters.num_class, mode,
                                           x_train=features.values)
            return [Subgroup(rule=[], cuts=[], cuts_options=[],
                             features_names=[], indices=[],
                             regret=[], model_cost=[],
                             cl_model_list=[], indices_list=[], total_cl_per_data=total_cl_per_data,
                             encoded_feature=[]),
                    absolute_gain_excluding_tree, (absolute_gain == absolute_gain_excluding_tree)]

        best_rule_list = []
        incl_gain_list = []
        excl_gain_list = []
        norm_gain_list = []
        exclude_score_equal_include_score_list = []
        for rule_beam in rule_beam_list:
            best_gain_excl_tree = -np.inf
            best_gain_incl_tree = -np.inf
            for r in rule_beam:
                indices_for_score = r.indices
                if len(indices_for_score) == 0:
                    continue

                r.model_cost = r.get_model_cost(cached_model_cost)
                absolute_gain, normalized_gain, regret, model_cost, total_emp_neg_log_probs, \
                total_model_cost, total_regret, absolute_gain_excluding_tree, \
                tree_cl_data, tree_regret \
                    = get_mdl_score_given_rule(subgroup_set, target, indices_for_score, r.model_cost,
                                               default_MDL_score, memberships, parameters.num_class, mode,
                                               x_train=features.values)

                if use_surrogate_score:
                    if absolute_gain > best_gain_incl_tree:
                        best_gain_incl_tree = absolute_gain
                        best_r = r
                        best_gain_excl_tree = absolute_gain_excluding_tree
                        norm_gain_excl_tree = absolute_gain_excluding_tree / len(indices_for_score)
                        exclude_score_equal_include_score = (best_gain_excl_tree == best_gain_incl_tree)
                else:
                    if absolute_gain_excluding_tree > best_gain_excl_tree:
                        best_gain_incl_tree = absolute_gain
                        best_r = r
                        best_gain_excl_tree = absolute_gain_excluding_tree
                        norm_gain_excl_tree = absolute_gain_excluding_tree / len(indices_for_score)
                        exclude_score_equal_include_score = (best_gain_excl_tree == best_gain_incl_tree)
            best_rule_list.append(best_r)
            incl_gain_list.append(best_gain_incl_tree)
            excl_gain_list.append(best_gain_excl_tree)
            norm_gain_list.append(norm_gain_excl_tree)
            exclude_score_equal_include_score_list.append(exclude_score_equal_include_score)

        if use_surrogate_score:
            best_index_incl_score = np.argmax(incl_gain_list)
        else:
            best_index_incl_score = np.argmax(excl_gain_list)

        return [best_rule_list[best_index_incl_score], excl_gain_list[best_index_incl_score],
                exclude_score_equal_include_score_list[best_index_incl_score]]
