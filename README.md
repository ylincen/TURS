# Updates
**Update 2023-Sep-04** Here is a refined version of our code: https://github.com/ylincen/TURS2 ;

**Update 2024-Jan-30** Check out our new "Arxiv journal version" of the paper: https://arxiv.org/abs/2401.09918v1 ;



# Algorithm & Experiment code for the paper:
Yang, L & van Leeuwen, M Truly Unordered Probabilistic Rule Sets for Multi-class Classification. In: Proceedings of the European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECMLPKDD 2022), Springer, 2022.

## Dependencies
All packages required are in requirement.txt. I use Python 3.8.10 for developing all TURS code. 

## Reproducing experiments in the paper 
To get the AUC of TURS (for Experiment 1), and Rule lengths and Number of Rules (for Experiment 3), 
run 'python3 run_turs_given_data.py DATA_NAME FOLD_NUMBER' in the command line. 
E.g., 'python3 run_turs_given_data.py iris 1' runs the dataset iris for the 1st fold of the 10-fold cross-validation. 
The results will be stored in the folder './res/'; SO CREATE A FOLDER called "./res/" locally in case you want to run the experiments.

To get the AUC of TURS WITHOUT using the surrogate score (for Experiment 2), 
change the 'use_surrogate_score=True' in findsubgroup_parrelle.py to 'use_surrogate_score=False'. 
There are two places that need to be changed for this in this script, on Line 26 and Line 64 respectively.  

Feel free to contact me for any questions or issues via l.yang at liacs dot leidenuniv dot nl; 
