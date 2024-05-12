import pandas as pd
import os
import matplotlib.pyplot as plt
 # Import the AUC module if it's not already imported

dir_path = "results/dev_data/baseline_MSE/"
files = os.listdir(dir_path)
cols_of_interest = ['AUC (source)', 'AUC (target)', 'pAUC', 'pAUC (source)', 'pAUC (target)', 'precision (source)', 'precision (target)', 'recall (source)', 'recall (target)', 'F1 score (source)', 'F1 score (target)']

# Initialize dictionaries
dfs = {}
differences = {}

for file in files:
  if file.endswith(".csv") and file.startswith("result"):
    name = file.split("_")[-4:-1]
    name = "_".join(name)
    df = pd.read_csv(os.path.join(dir_path, file))
    df_selected = df[cols_of_interest]
    dfs[name] = df_selected

baseline_selected = dfs["baseline_omar_soubky"]
for factor, df in dfs.items():
        if factor != "baseline_omar_soubky":
            differences[factor] = df - baseline_selected

# Convert the differences dictionary to a DataFrame
differences_df = pd.concat(differences.values(), keys=differences.keys())

# Round the values to a certain precision (e.g., 6 decimal places)
rounded_differences_df = differences_df.round(6)

# Remove duplicate rows
differences_unique = rounded_differences_df.drop_duplicates()
sorted_differences = differences_unique.sort_values(by='pAUC')

# Print the unique differences
print("Unique Differences between baseline and masking factors:")
print(sorted_differences)