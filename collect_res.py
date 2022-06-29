from prediction import *
import pandas as pd
class CollectRes:
    @staticmethod
    def collect_results_pd(subgroup_set, x_test, membership, y_train, y_test, t1,
                           max_num_bin_numeric, random_st, d, data_name, alg_name):
        res, names = my_predication(subgroup_set, x_test, membership, y_train, y_test)

        res_pd = pd.DataFrame(data=[res], columns=names)


        res_pd['time'] = t1
        res_pd['max_num_bin_numeric'] = max_num_bin_numeric
        res_pd["random_state"] = random_st
        res_pd['nsample'] = len(d)
        res_pd['ncol'] = d.shape[1] - 1
        res_pd['data'] = data_name
        res_pd['alg'] = alg_name


        return res_pd
