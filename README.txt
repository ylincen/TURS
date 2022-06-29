All packages required are in requirement.txt. I use Python 3.8.10 for developing all TURS code. 

To get the AUC of TURS (for Experiment 1), and Rule lengths and Number of Rules (for Experiment 3), 
run 'python3 run_turs_given_data.py DATA_NAME FOLD_NUMBER' in the command line. 
E.g., 'python3 run_turs_given_data.py iris 1' runs the dataset iris for the 1st fold of the 10-fold cross-validation. 
The results will be stored in the folder './res/'; 

To get the AUC of TURS WITHOUT using the surrogate score (for Experiment 2), 
change the 'use_surrogate_score=True' in findsubgroup_parrelle.py to 'use_surrogate_score=False'. 
There are two places that need to be changed for this in this script, on Line 26 and Line 64 respectively.  
