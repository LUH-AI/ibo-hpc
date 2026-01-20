import pandas as pd
import glob
import os

def analyze_best_performance_per_file(directory_path="pd1_experiments/good", performance_column="test_performance"):
    """
    Reads all CSV files in a specified directory, finds the row with the 
    highest value in the specified performance column for EACH file, 
    and then computes the average of these best performances.
    """
    # 1. Construct the search pattern for all CSV files
    search_pattern = os.path.join(directory_path, '*.csv')
    
    # Use glob to find all matching files
    csv_files = glob.glob(search_pattern)

    if not csv_files:
        print(f"❌ No CSV files found in the directory: {directory_path}")
        return

    print(f"🔍 Found {len(csv_files)} CSV file(s) in {directory_path}.")
    
    # 2. List to hold the best performance (max test_performance) from each file
    best_performances = []

    # 3. Iterate through each file
    for file_path in csv_files:
        try:
            # Read the CSV. The data provided suggests the first column is an index.
            df = pd.read_csv(file_path, index_col=0) 
            
            # Find the highest performance in this file
            # .max() returns the single highest value in the column
            max_performance = df[performance_column].max()
            
            print(f" - Best '{performance_column}' in '{os.path.basename(file_path)}': {max_performance:.6f}")
            
            # 4. Store this maximum value
            best_performances.append(max_performance)
            
        except Exception as e:
            print(f"⚠️ Error reading file {file_path}: {e}")
            continue

    if not best_performances:
        print("❌ Could not extract best performance from any CSV files.")
        return
        
    # 5. Calculate the average of the best performances
    average_best_performance = sum(best_performances) / len(best_performances)

    # 6. Print the results
    print("\n--- ✅ Analysis Complete ---")
    print(f"Number of files analyzed: {len(best_performances)}")
    print(f"Average of the best '{performance_column}' from each file: **{average_best_performance:.6f}**")
    print("-" * 35)

if __name__ == "__main__":
    # ... (Setup code to create run_01.csv and run_02.csv with your data) ...
    
    # --- EXECUTE THE FUNCTION ---
    analyze_best_performance_per_file(directory_path="pd1_experiments/good")