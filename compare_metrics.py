import pandas as pd

# Load CSV files into pandas DataFrames
df1 = pd.read_csv("results/eval_data/baseline_MSE/result_DCASE2023T2gearbox_test_seed13711_masking_factor_10_roc.csv")
df2 = pd.read_csv("C:/Users/omar/projects/bachelor/results/dev_data/baseline/summarize/DCASE2023T2/baseline_MSE/result_DCASE2023T2gearbox_test_seed13711_roc.csv")


# Select only the columns of interest for comparison
cols_of_interest = ['pAUC', 'pAUC (source)', 'pAUC (target)', 'AUC (source)', 'AUC (target)']
df1_selected = df1[cols_of_interest]
df2_selected = df2[cols_of_interest]

# Compute the absolute differences between corresponding values
absolute_diff = df1_selected - df2_selected

# Display the absolute differences
print("Absolute Differences:")
print(absolute_diff)
