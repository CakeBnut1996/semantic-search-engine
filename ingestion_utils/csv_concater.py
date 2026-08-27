import os
import pandas as pd


def load_and_combine_results(res_dir, keyword):
    """
    Scans a directory for CSVs containing a specific keyword, extracts
    model names from filenames, and merges them into consolidated DataFrames.

    Args:
        res_dir (str): The directory containing your .csv results.
        keyword (str): The string to look for in filenames (e.g., 'retrieval' or 'generation').

    Returns:
        tuple: (pos_df, neg_df, all_df) containing the combined data.
    """
    if not os.path.exists(res_dir):
        print(f"Error: Directory '{res_dir}' does not exist.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # 1. Identify relevant files using the dynamic keyword
    # Example: if keyword='retrieval', looks for '*all_data_retrieval*.csv'
    target_pattern = keyword
    files = [f for f in os.listdir(res_dir) if target_pattern in f and f.endswith('.csv')]

    pos_list = []
    neg_list = []

    # 2. Process and tag each file
    for file in files:
        # Extract model name: everything before it
        model_name = file.split(target_pattern)[0]

        file_path = os.path.join(res_dir, file)
        try:
            df = pd.read_csv(file_path)
            df['model'] = model_name

            # Sort by suffix
            if file.endswith('_pos.csv'):
                pos_list.append(df)
            elif file.endswith('_neg.csv'):
                neg_list.append(df)
        except Exception as e:
            print(f"Warning: Could not read {file}. Error: {e}")

    # 3. Efficient Concatenation
    pos_df = pd.concat(pos_list, ignore_index=True) if pos_list else pd.DataFrame()
    neg_df = pd.concat(neg_list, ignore_index=True) if neg_list else pd.DataFrame()

    # Create the master dataframe for global analysis
    all_df = pd.concat([pos_df, neg_df], ignore_index=True) if not (pos_df.empty and neg_df.empty) else pd.DataFrame()

    return pos_df, neg_df, all_df #