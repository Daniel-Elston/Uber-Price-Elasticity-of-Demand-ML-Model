from my_utils.dependencies import pd

def file_load(input_files):
    """
    Load data from CSV files into Pandas dataframes and store them in global variables.
    
    Args:
    - input_files (list of str): List of file paths of CSV files to be loaded.
    
    Returns:
    - df_store (list of pd.DataFrame): List of dataframes of the loaded CSV files.
    """
        
    for i, file in enumerate(input_files):
        # Load CSV file into a Pandas dataframe and store it in a global variable.
        print(file)
        globals()[f'df{i+1}'] = pd.read_csv(input_files[i])
    print(f'Total number of files loaded: {len(input_files)}')
    
    # Store all the dataframes in a list and return it.
    df_store = [eval(f'df{i+1}') for i in range(len(input_files))]
    return df_store

def trim_raw_data(df, required_cols, df_frac):
    """
    Trim a dataframe to the required columns and fraction of rows.
    
    Args:
    - df (pd.DataFrame): Dataframe to be trimmed.
    - required_cols (list of str): List of column names to keep in the trimmed dataframe.
    - df_frac (float): Fraction of rows to keep in the trimmed dataframe.
    
    Returns:
    - df (pd.DataFrame): Trimmed dataframe.
    """
    # Select the required columns and keep only the fraction of rows specified.
    df = df[required_cols]
    df = df.sample(frac=df_frac)
    return df
