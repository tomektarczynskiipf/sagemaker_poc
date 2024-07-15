
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

def create_random_dataframe_with_params(n_rows, n_cols, params, seed=None):
    """
    Create a DataFrame with random values and an additional binary target column based on the sum of products of values and parameters.
    
    Parameters:
    - n_rows: int, number of rows in the DataFrame
    - n_cols: int, number of columns in the DataFrame
    - params: list or array-like, parameters for each column
    - seed: int, random seed for reproducibility (default is None)
    
    Returns:
    - DataFrame with shape (n_rows, n_cols+1) where the last column is a binary target based on the sum of products.
    """
    if seed is not None:
        np.random.seed(seed)
        
    if len(params) != n_cols:
        raise ValueError("The length of params must be equal to the number of columns (n_cols).")
    
    data = np.random.rand(n_rows, n_cols)
    df = pd.DataFrame(data, columns=[f'col_{i+1}' for i in range(n_cols)])
    
    # Calculate the sum_product column
    df['sum_product'] = np.dot(df.values, params) + 0.5
    
    # Calculate the target column
    df['target'] = (np.random.rand(n_rows) < df['sum_product']).astype(int)
    
    # Drop the sum_product column
    df = df.drop(columns=['sum_product'])

    # Move target column to the first position
    columns = ['target'] + [col for col in df.columns if col != 'target']
    df = df[columns]    
    
    return df

if __name__ == '__main__':
    settings = os.environ
    
    # Create data
    params = [-1, -1, -0.5, 0, 0, 0.5, 1, 1]
    df = create_random_dataframe_with_params(n_rows = 100000, n_cols = 8, params = params, seed = 42)
    

    train, temp = train_test_split(df, test_size=0.4, random_state=42)
    test, valid = train_test_split(temp, test_size=0.5, random_state=42)

    train = train[['target', "col_1", "col_2", "col_3", "col_6", "col_7", "col_8"]]
    test = test[['target', "col_1", "col_2", "col_3", "col_6", "col_7", "col_8"]]
    valid = valid[['target', "col_1", "col_2", "col_3", "col_6", "col_7", "col_8"]]
    inference_train = train[["col_1", "col_2", "col_3", "col_6", "col_7", "col_8"]]

    train_path = os.path.join(settings["preprocessing_output_train"], "train.csv")
    train.to_csv(train_path, index=False, float_format='%.5f')

    test_path = os.path.join(settings["preprocessing_output_test"], "test.csv")
    test.to_csv(test_path, index=False, float_format='%.5f')

    valid_path = os.path.join(settings["preprocessing_output_valid"], "valid.csv")
    valid.to_csv(valid_path, index=False, float_format='%.5f')

    inference_train_path = os.path.join(settings["preprocessing_output_inference_train"], "inference_train.csv")
    inference_train.to_csv(inference_train_path, index=False, float_format='%.5f')
