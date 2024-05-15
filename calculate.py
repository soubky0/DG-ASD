import pandas as pd
import os
import argparse
from scipy.stats.mstats import hmean  # Import the harmonic mean function

def compare_results(sort_by="hmean (source)"):  # Default sort by hmean
    dir_path = "results/dev_data/baseline_MSE/"
    files = os.listdir(dir_path)
    cols_of_interest = ['AUC (source)', 'AUC (target)', 'pAUC', 'pAUC (source)', 'pAUC (target)']
    non_percentage = ['precision (source)', 'precision (target)', 'recall (source)', 'recall (target)', 'F1 score (source)', 'F1 score (target)']

    dfs = {}
    differences = {}

    for file in files:
        if file.endswith(".csv") and file.startswith("result"):
            name = file.split("_")[-4:-1]
            name = "_".join(name)
            df = pd.read_csv(os.path.join(dir_path, file))
            df_selected = df[cols_of_interest] * 100
            df_selected[non_percentage] = df[non_percentage]
            
            # Calculate harmonic mean for source and target domains
            df_selected['hmean (source)'] = hmean([df_selected['AUC (source)'], df_selected['pAUC (source)']])
            df_selected['hmean (target)'] = hmean([df_selected['AUC (target)'], df_selected['pAUC (target)']])
            
            dfs[name] = df_selected

    baseline_selected = dfs["baseline_omar_soubky"]
    for factor, df in dfs.items():
        if factor != "baseline_omar_soubky":
            differences[factor] = df - baseline_selected

    # Convert differences to DataFrame
    differences_df = pd.concat(differences.values(), keys=differences.keys())

    # Round values to 6 decimal places
    rounded_differences_df = differences_df.round(6)

    # Remove duplicates and sort
    differences_unique = rounded_differences_df.drop_duplicates()
    sorted_differences = differences_unique.sort_values(by=sort_by)  
    reversed_sorted_differences = sorted_differences.iloc[::-1]

    print("Differences between baseline and masking factors (sorted by", sort_by, "):")
    print(reversed_sorted_differences[sort_by])
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare results of different factors with a baseline and save differences to a CSV file.')
    parser.add_argument('sort_by', nargs='?', type=str, help='Sorting metric. Available options: AUC (source), AUC (target), pAUC, pAUC (source), pAUC (target), precision (source), precision (target), recall (source), recall (target), F1 score (source), F1 score (target), hmean (source), hmean (target)')

    args = parser.parse_args()
    if args.sort_by:
        compare_results(args.sort_by)
    else:
        compare_results()
