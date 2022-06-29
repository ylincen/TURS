import numpy as np
from constant import *

# len_feature: number of features in the datasets, i.e., number of columns in the tabular data.
def get_rule_length(subgroup_set):
    rule_lengths = []
    for s in subgroup_set:
        len_feature = max(s.features_names) + 1
        cut_option_left = np.zeros(len_feature, dtype=bool)
        cut_option_right = np.zeros(len_feature, dtype=bool)
        cut_option_within = np.zeros(len_feature, dtype=bool)
        rule_length = 0
        for fname, cutoption in zip(s.features_names, s.cuts_options):
            if cutoption == WITHIN_CUT and ~cut_option_within[fname]:
                cut_option_within[fname] = True
                rule_length += 1
            elif cutoption == RIGHT_CUT and ~cut_option_right[fname]:
                cut_option_right[fname] = True
                rule_length += 1
            elif cutoption == LEFT_CUT and ~cut_option_left[fname]:
                cut_option_left[fname] = True
                rule_length += 1
            else:
                pass # for cut_option_xxx is True
        rule_lengths.append(rule_length)
    return rule_lengths

