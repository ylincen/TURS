# cython: profile=True
from Subgroup import *
import copy
from calculate_nml_ignoreOverlap import *
from calculate_NML import *
import nml_regret
from utils import *

class GrowIndividualRule:
    @staticmethod
    def grow_by_mdl_gain_for_rule_list(target, subgroup_set, features, parameters,
                                       candidate_cuts_for_search, covered_indices, nbeam,
                                       memberships, memberships_cumulative,
                                       mode, default_MDL_score, num_best_rules_return, cached_model_cost,
                                       use_surrogate_score=True):
        # initialize things
        original_indices = np.arange(start=0, stop=len(target), step=1)
        jarcard_dist_threshold = 0.95

        # calculate the default probs and MDL scores for uncovered data
        uncovered_indices = original_indices[~memberships_cumulative]
        probs = calc_probs(target[original_indices], parameters.num_class)

        if len(uncovered_indices) == 0:
            return [[Subgroup(rule=[], cuts=[], cuts_options=[],
                             features_names=[], indices=[],
                             regret=[], model_cost=[],
                             cl_model_list=[], indices_list=[], encoded_feature=np.zeros(len(features.names),dtype=int),
                             total_cl_per_data=np.nan), 0, False]] # only rule=[] matters here, as "len(rule)==0" is a flag.
        cl_data = -np.sum(np.log2(probs[target[uncovered_indices]]))
        reg = nml_regret.regret(len(uncovered_indices), parameters.num_class)
        total_cl_per_data = (cl_data + reg) / len(uncovered_indices)

        # initialize the beam search
        rule_beam_list = [] # rule beams for every rule length.
        rule_beam = [] # a set of rules
        for i in range(nbeam):
            rule_beam.append(Subgroup(rule=[], cuts=[], cuts_options=[],
                                      features_names=[], indices=original_indices, regret=reg, model_cost=0,
                                      cl_model_list=[], indices_list=[], total_cl_per_data=total_cl_per_data,
                                      encoded_feature=np.zeros(len(features.names),dtype=int)))
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
                                    indices_for_score = indices
                                else:
                                    indices_for_score = indices[~memberships_cumulative[indices]]
                                if len(indices_for_score) == 0:
                                    continue

                                mdl_gain, total_cl, cl_data, reg, cl_model = get_mdl_score_local(
                                    target, candidate_cuts_for_search[feature_name], cut,
                                    feature_type, indices_for_score,
                                    model_cost_so_far=rule_base.model_cost, num_class=parameters.num_class,
                                    len_rule=len(rule_base.rule), previous_mean_cl=rule_base.total_cl_per_data,
                                    repeated_feature_name_and_cut_option=False, cached_model_cost=cached_model_cost,
                                    feature_name=feature_name)
                                if cut_option == LEFT_CUT:
                                    direction = "<="
                                else:
                                    direction = ">"
                                res[(rule_base, feature_name, cut, cut_option)] = \
                                    {"mdl_gain": mdl_gain,
                                     "total_cl": total_cl,
                                     "indices": indices,
                                     "indices_for_score": indices_for_score,
                                     "feature_name": feature_name,
                                     "cut": cut,
                                     "cut_option": cut_option,
                                     "feature_type": feature_type,
                                     "cl_model": cl_model,
                                     "regret": reg,
                                     "total_cl_per_data": total_cl / len(indices_for_score),
                                     "grow_step": 'X' + str(feature_name) + direction + str(cut)}
                        else: # for categorical features
                            indices = get_indices(feature=feature, feature_type=feature_type, cut=cut,
                                                  original_indices=rule_base.indices)
                            if len(indices) == len(rule_base.indices):
                                continue
                            if covered_indices is None:
                                indices_for_score = indices
                            else:
                                indices_for_score = indices[~memberships_cumulative[indices]]
                            if len(indices_for_score) == 0:
                                continue

                            cut_option = WITHIN_CUT

                            mdl_gain, total_cl, cl_data, reg, cl_model = get_mdl_score_local(
                                target, candidate_cuts_for_search[feature_name], cut,
                                feature_type, indices_for_score,
                                model_cost_so_far=rule_base.model_cost, num_class=parameters.num_class,
                                len_rule=len(rule_base.rule), previous_mean_cl=rule_base.total_cl_per_data,
                                repeated_feature_name_and_cut_option=False,
                                cached_model_cost=cached_model_cost,
                                feature_name=feature_name)
                            res[(rule_base, feature_name, cut, cut_option)] = \
                                {"mdl_gain": mdl_gain,
                                 "total_cl": total_cl,
                                 "indices": indices,
                                 "indices_for_score": indices_for_score,
                                 "feature_name": feature_name,
                                 "cut": cut,
                                 "cut_option": cut_option,
                                 "feature_type": feature_type,
                                 "cl_model": cl_model,
                                 "regret": reg,
                                 "total_cl_per_data": total_cl/len(indices_for_score),
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
                indices_for_score_here = values_here["indices_for_score"]
                reg_here = values_here["regret"]
                model_cost_here = values_here["cl_model"]

                cl_model_list_here = rule_base_here.cl_model_list
                cl_model_list_here.append(values_here["cl_model"])

                indice_list_here = rule_base_here.indices # deprecated!!!
                indices_for_score_list_here = indice_list_here # deprecated!!!

                encoded_feature_here = rule_base_here.encoded_feature
                encoded_feature_here[values_here["feature_name"]] += 1

                rule_beam.append(Subgroup(rule=rule_here, cuts=cuts_here, cuts_options=cuts_option_here,
                                          features_names=features_names_here, indices=indices_here, regret=reg_here,
                                          model_cost=model_cost_here, indices_for_score=indices_for_score_here,
                                          indices_for_score_list=indices_for_score_list_here,
                                          cl_model_list=cl_model_list_here, indices_list=indice_list_here,
                                          total_cl_per_data=values_here["total_cl_per_data"],
                                          encoded_feature=encoded_feature_here))
            rule_beam_list.append(rule_beam)

        if len(rule_beam_list) == 1: # i.e., no new rule is found
            return [[Subgroup(rule=[], cuts=[], cuts_options=[],
                              features_names=[], indices=[],
                              regret=[], model_cost=[],
                              cl_model_list=[], indices_list=[], total_cl_per_data=total_cl_per_data,
                              encoded_feature=[])], [0], [False]]

        best_rule_list = []
        incl_gain_list = []
        excl_gain_list = []
        norm_gain_list = []
        exclude_score_equal_include_score_list = []

        for rule_beam in rule_beam_list:
            incl_gain_per_beam_list = []
            excl_gain_per_beam_list = []
            norm_gain_per_beam_list = []
            exclude_score_equal_include_score_per_beam_list = []
            for r in rule_beam:
                if covered_indices is None:
                    indices_for_score = r.indices
                else:
                    indices_for_score = r.indices[~memberships_cumulative[r.indices]]
                if len(indices_for_score) == 0:
                    incl_gain_per_beam_list.append(-np.inf)
                    excl_gain_per_beam_list.append(-np.inf)
                    norm_gain_per_beam_list.append(-np.inf)
                    exclude_score_equal_include_score_per_beam_list.append(False)

                r.model_cost = r.get_model_cost(cached_model_cost=cached_model_cost)

                absolute_gain, normalized_gain, absolute_gain_excluding_tree, tree_cl_data, tree_regret = \
                    get_mdl_score_given_rule_for_list(subgroup_set, target, indices_for_score, r.model_cost,
                                                      default_MDL_score, memberships, parameters.num_class, mode,
                                                      features.values, covered_indices)
                incl_gain_per_beam_list.append(absolute_gain)
                excl_gain_per_beam_list.append(absolute_gain_excluding_tree)
                norm_gain_per_beam_list.append(normalized_gain)
                exclude_score_equal_include_score_per_beam_list.append( (absolute_gain == absolute_gain_excluding_tree) )

            # rank these rules by absolute_gain_excluding_tree;

            if use_surrogate_score:
                orders_here = np.argsort(-np.array(incl_gain_per_beam_list))
            else:
                orders_here = np.argsort(-np.array(excl_gain_per_beam_list))

            for jjj in range(len(orders_here)):
                if jjj > num_best_rules_return:
                    break
                best_rule_list.append(rule_beam[orders_here[jjj]])

                # the plus term is to make sure that when two rules have the same gain, the shorter one is picked.
                # 0.0001 should be small enough here as we only allow rules with max length 20, so it won't affect other things.
                incl_gain_list.append(incl_gain_per_beam_list[orders_here[jjj]] - 0.0001 * len(rule_beam[orders_here[jjj]].rule))
                excl_gain_list.append(excl_gain_per_beam_list[orders_here[jjj]] - 0.0001 * len(rule_beam[orders_here[jjj]].rule))
                norm_gain_list.append(norm_gain_per_beam_list[orders_here[jjj]] - 0.0001 * len(rule_beam[orders_here[jjj]].rule))
                exclude_score_equal_include_score_list.append(exclude_score_equal_include_score_per_beam_list[orders_here[jjj]])

        if use_surrogate_score:
            orders_incl_score = np.argsort(-np.array(incl_gain_list))[:num_best_rules_return]
        else:
            orders_incl_score = np.argsort(-np.array(excl_gain_list))[:num_best_rules_return]

        best_rules = [best_rule_list[i] for i in orders_incl_score]
        excl_gains = [excl_gain_list[i] for i in orders_incl_score]
        exclude_score_equal_include_score_s = [exclude_score_equal_include_score_list[i] for i in orders_incl_score]

        return [best_rules, excl_gains, exclude_score_equal_include_score_s]
