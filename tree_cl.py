from sklearn import tree
import numpy as np

import nml_regret
from nml_regret import *
from calculate_NML import calc_probs
from numba import njit

def get_tree_cl(x_train, y_train, num_class):
    n = len(y_train)
    if n > 1000:
        min_samples = np.arange(10, max(50, int(n*0.01)), 20)
    else:
        min_samples = np.arange(10, 110, 20)

    best_tree_cl = np.inf
    probs = calc_probs(y_train, num_class)
    best_sum_cl_data = np.sum(-np.log2(probs[y_train]))
    best_sum_regrets = nml_regret.regret(len(y_train), num_class)

    for min_sample in min_samples:
        sum_cl_data, sum_regrets = get_tree_cl_individual(x_train, y_train, num_class, min_sample=min_sample)
        if sum_cl_data + sum_regrets <= best_tree_cl:
            best_tree_cl = sum_cl_data + sum_regrets
            best_sum_cl_data = sum_cl_data
            best_sum_regrets = sum_regrets

    return [best_sum_cl_data, best_sum_regrets]



# x_train, y_train: training data in the else-rule
def get_tree_cl_individual(x_train, y_train, num_class, min_sample=0.05):
    clf = tree.DecisionTreeClassifier(min_samples_leaf=min_sample, random_state=1)
    clf = clf.fit(x_train, y_train)

    num_rules = clf.get_n_leaves()

    which_paths_train = clf.apply(x_train)
    train_membership = np.zeros((num_rules, x_train.shape[0]), dtype=bool)

    which_paths_train_dic = {}
    counter = 0
    for i in range(x_train.shape[0]):
        if which_paths_train[i] in which_paths_train_dic:
            train_membership[which_paths_train_dic[which_paths_train[i]], i] = 1
        else:
            which_paths_train_dic[which_paths_train[i]] = counter
            train_membership[which_paths_train_dic[which_paths_train[i]], i] = 1
            counter += 1

    counts_per_leaf = np.sum(train_membership, axis=1)
    sum_regrets = 0
    for count in counts_per_leaf:
        sum_regrets += regret(count, num_class)

    sum_cl_data = 0  # this does NOT include regret!
    for train_membership_for_each_rule in train_membership:
        sum_cl_data += get_entropy(y_train[train_membership_for_each_rule], num_class)

    return [sum_cl_data, sum_regrets]


def get_entropy(target, num_class):
    probs = calc_probs(target, num_class)
    entropy = -np.sum(np.log2(probs[target]))

    return entropy



