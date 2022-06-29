from exp_utils import *
import os

# global parameters
max_num_bin_numeric = 100
flt_format = '%.3f'
max_dp = 20
append_to_current_csv = False

max_num_subgroups = 500
max_iter = 500
random_st = 1

k_fold = 10

print("start:", max_num_bin_numeric, " bins")


if len(sys.argv) == 3:
    data_chosen = sys.argv[1]
    cv_fold = int(sys.argv[2])
else:
    data_chosen = sys.argv[1]
    cv_fold = None

if cv_fold is None:
   file_name = "./res/" + data_chosen + ".csv"
else:
   file_name = "./res/" + data_chosen + "_fold" + str(cv_fold) + ".csv"

print("filename: ", file_name)
# full result pandas dataframe
res_pd_full = pd.DataFrame()


# get the datasets
all_files = os.listdir("./datasets")
print("all_files:", all_files)
datasets_path = []
datasets_names = []
for file in all_files:
    if '.csv' in file or '.txt' in file or '.data' in file:
        datasets_path.append("./datasets/" + file)
        datasets_names.append( str.split(file, ".")[0] )

datasets_without_header_row = ["waveform", "backnote", "chess", "contracept", "iris", "ionosphere",
                               "magic", "car", "tic-tac-toe", "wine", "breast", "transfusion", "poker"]
datasets_with_header_row = ["avila", "anuran", "diabetes", "sepsis"]

for data_name, data_path in zip(datasets_names, datasets_path):
    if data_name != data_chosen:
        continue

    print("start: ", data_name, "at ", data_path, "===================")
    if data_name in datasets_without_header_row:
        d = pd.read_csv(data_path, header=None)
    elif data_name in datasets_with_header_row:
        d = pd.read_csv(data_path)
    else:
        # sys.exit("error: data name not in the datasets lists that show whether the header should be included!")
        print(data_name, "not in the folder!")

    num_cores = 5

    res_pd_this_data = RunExp.run_cars_beam(d, random_st, max_dp, max_iter, max_num_bin_numeric, max_num_subgroups,
                                            k_fold=k_fold, data_name=data_name,
                                            num_cores=num_cores, nbeam=5, num_initial_rule_list=5,
                                            onlyonefold=False, cv_fold=cv_fold)
    print(res_pd_this_data)
    res_pd_full = pd.concat([res_pd_full, res_pd_this_data], ignore_index=True)
    res_pd_full.to_csv(path_or_buf=file_name, float_format=flt_format, index=False)




