from sklearn import preprocessing
from sklearn.model_selection import KFold, StratifiedKFold
import time
from findsubgroup_parrelle import *
from collect_res import *


def pre_run_cars(X, y, train_index, test_index, max_dp, max_iter, max_num_bin_numeric, max_num_subgroups, d):
    x_train, x_test = X[train_index, :], X[test_index, :]
    y_train, y_test = y[train_index], y[test_index]
    x_train, x_test = x_train.T, x_test.T

    feature_ncol = X.shape[1]
    max_num_bin = [max_num_bin_numeric] * feature_ncol # max_num_bin for NUMERIC variable
    feature_types = [NUMERIC for i in range(feature_ncol)] # feature type for NUMERIC variable

    # change feature type for integers with small number of unique values, as well as for str;
    for i, tp in enumerate(d.dtypes[:len(d.dtypes)-1]):
        if tp == int:
            if len(np.unique(X[:, i])) > 10:
                pass
            else: # treat integers with 5 or less unique values as CATEGORICAL variable
                feature_types[i] = CATEGORICAL

        if tp == str:
            feature_types[i] = CATEGORICAL

    features = Features(names=list(np.arange(feature_ncol)), values=x_train, types=feature_types)
    parameters = Parameters(max_depth=max_dp, max_levels_per_batch=5, max_num_subgroups=max_num_subgroups,
                            max_iter=max_iter,
                            max_num_bin=max_num_bin, num_class=len(np.unique(y)))
    return [x_train, x_test, y_train, y_test, features, parameters]


class RunExp:
    @staticmethod
    def run_cars_beam(d, random_st, max_dp, max_iter, max_num_bin_numeric,
                      max_num_subgroups, k_fold, data_name,
                      num_cores, nbeam, num_initial_rule_list, onlyonefold, cv_fold):
        le = preprocessing.LabelEncoder()
        for icol, tp in enumerate(d.dtypes):
            if tp != float:
                le.fit(d.iloc[:, icol])
                d.iloc[:, icol] = le.transform(d.iloc[:, icol])

        res_pd_this_data = pd.DataFrame()

        # k-fold cross validation
        kf = StratifiedKFold(n_splits=k_fold, random_state=random_st,
                             shuffle=True)  # indices for 10-fold cross validation

        nrow, ncol = d.shape
        X = d.iloc[:, 0:ncol - 1].to_numpy()
        y = d.iloc[:, ncol - 1].to_numpy()

        fold_number = 0

        for train_index, test_index in kf.split(X, y):
            fold_number += 1
            if cv_fold is not None:
                if fold_number != cv_fold:
                    continue
            print("fold_number: ", fold_number)
            x_train, x_test, y_train, y_test, features, parameters = \
                pre_run_cars(X, y, train_index, test_index, max_dp, max_iter, max_num_bin_numeric, max_num_subgroups, d)

            t0 = time.time()
            subgroup_set, membership, cl_total, absolute_gain = \
                find_rule_set_initby_list_paralle(y_train, features, parameters, CLASSIFICATION,
                                                  num_cores, nbeam, num_initial_rule_list)

            t1 = time.time() - t0
            print("time spent on this fold: ", t1)

            res_pd = CollectRes.collect_results_pd(subgroup_set, x_test, membership, y_train, y_test, t1,
                           max_num_bin_numeric, random_st, d, data_name, alg_name="TURS")
            res_pd_this_data = pd.concat([res_pd_this_data, res_pd], ignore_index=True)

            if onlyonefold:
                break

        return res_pd_this_data
