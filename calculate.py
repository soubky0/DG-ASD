import pandas as pd
import os
import argparse
import glob
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
 

def compute_average_results(directory, output_filename):
    """
    Compute the average results from multiple CSV files in the specified directory.

    Parameters:
    directory (str): The directory containing the CSV files.

    Returns:
    pd.DataFrame: DataFrame containing the average results.
    """
    # Load all CSV files in the specified directory
    directory = os.path.join(os.getcwd(),"results", directory)
    file_paths = glob.glob(f'{directory}/result*.csv')

    if not file_paths:
        raise ValueError("No matching CSV files found in the specified directory.")

    # Read CSV files into DataFrames
    dfs = [pd.read_csv(file_path) for file_path in file_paths]

    # Check if all dataframes have the same columns
    if not all(df.columns.equals(dfs[0].columns) for df in dfs):
        raise ValueError("DataFrames do not have the same columns, cannot compute average.")

    # Concatenate all DataFrames
    concatenated_df = pd.concat(dfs)
    
    # Group by all columns except the result column, and compute the mean for the result column
    avg_df = concatenated_df.groupby(concatenated_df.columns[:-1].tolist()).mean().reset_index()
    output_path = os.path.join(directory, output_filename)
    avg_df.to_csv(output_path, index=False)
    print(f"Average results saved to {output_path}")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Compare results of different factors with a baseline and save differences to a CSV file.')
    # parser.add_argument('sort_by', nargs='?', type=str, help='Sorting metric. Available options: AUC (source), AUC (target), pAUC, pAUC (source), pAUC (target), precision (source), precision (target), recall (source), recall (target), F1 score (source), F1 score (target), hmean (source), hmean (target)')

    # args = parser.parse_args()
    # if args.sort_by:
    #     compare_results(args.sort_by)
    # else:
    #     compare_results()
    result = compute_average_results("baseline","average_result.csv")