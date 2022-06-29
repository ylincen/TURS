# This file contains scripts related to predication
import sys
from utils import calc_probs
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from exp_get_rule_lengths import *
from get_covered_indices_bool import *

def get_test_membership(subgroup_set, test_X):
    # Assume X is recorded by row instead of column, i.e., test_X[0] is the first row
    n_test = len(test_X)
    membership_test_data = np.zeros((len(subgroup_set), n_test), dtype=bool)

    for i, x in enumerate(test_X):
        for si, subgroup in enumerate(subgroup_set):
            flag = 1
            for col, cut, cut_option in zip(subgroup.features_names, subgroup.cuts, subgroup.cuts_options):
                if cut_option == LEFT_CUT:
                    if x[col] <= cut:
                        pass
                    else:
                        flag = 0
                        break
                elif cut_option == RIGHT_CUT:
                    if x[col] > cut:
                        pass
                    else:
                        flag = 0
                        break
                elif cut_option == WITHIN_CUT:
                    if x[col] in cut:
                        pass
                    else:
                        flag = 0
                        break
                else:
                    sys.exit("error in which_rule!")
            # if NOT break for all col in subgroup.features_names, it means that x is in this subgroup!
            if flag == 1:
                membership_test_data[si, i] = 1
            else:
                pass

    return membership_test_data


def my_predication(subgroup_set, x_test, membership, y_train, y_test, write_to_file=True):

    train_membership = membership

    test_membership = get_test_membership(subgroup_set, x_test.T)
    unique_membership_list, corresponding_indices = np.unique(test_membership, axis=1, return_inverse=True)

    num_class = len(np.unique(np.append(y_test,y_train)))
    y_pred_probs = np.zeros((len(y_test), num_class), dtype=float)

    test_acc_per_group = {}
    for i, unique_membership in enumerate(unique_membership_list.T):
        update_indiecs = np.where(corresponding_indices == i)[0]
        if np.sum(unique_membership) == 0:
            covered_indices_bool = np.invert(np.bitwise_or.reduce(train_membership, axis=0))
            overlapping_probs = calc_probs(y_train[covered_indices_bool], num_class)
        else:
            # check whether some rule cover other rule
            if len(train_membership) == 1:  # the case whether there is only 1 rule in total
                covered_indices_bool = train_membership[0]
            else:
                covered_indices_bool = get_covered_indices_bool(unique_membership, train_membership)
            overlapping_probs = calc_probs(y_train[covered_indices_bool], num_class)
        y_pred_probs[update_indiecs] = overlapping_probs
        test_acc_per_group[tuple(unique_membership)] = \
            np.mean(y_test[update_indiecs] == np.argmax(y_pred_probs[update_indiecs], axis=1))
    if len(np.unique(y_test)) == 2:
        auc = roc_auc_score(y_test, y_pred_probs[:,1])
        print("ROC_AUC: ", auc)
    else:
        auc = roc_auc_score(y_test, y_pred_probs, average='weighted', multi_class='ovr')
        print("ROC_AUC: ", auc)

    train_overlapping = np.mean(np.sum(train_membership, axis=0) > 1)

    rule_lengths = get_rule_length(subgroup_set)
    mean_rule_lengths = np.mean(rule_lengths)

    if write_to_file:
        res = np.array([auc, len(subgroup_set), len(subgroup_set)/num_class, mean_rule_lengths,
                        train_overlapping])
        names = ["auc", "num rules", "num rules per class", "avg rule length", "train overlap"]

        return [res, names]


