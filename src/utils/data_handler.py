import numpy as np
import pandas as pd
from scipy import sparse


def load_csv_to_list(filepath: str, dtype=None) -> list:
    return pd.read_csv(filepath, dtype=dtype).to_numpy().flatten().tolist()


def load_csv_to_array(filepath: str, dtype=None) -> np.ndarray:
    return pd.read_csv(filepath, dtype=dtype).to_numpy().flatten()


def load_csv_to_series(filepath: str, dtype=None) -> pd.Series:
    return pd.Series(load_csv_to_list(filepath, dtype))


def save_df_as_csv(df: pd.DataFrame, filename: str):
    """
    Saves a pandas DataFrame without index as .csv file.
    :param df: DataFrame
    :param filename: name of the .csv file
    """
    df.to_csv(filename, sep=";", index=False)


def load_sparse_matrix(filepath):
    return sparse.load_npz(filepath)


def save_sparse_matrix(adj_matrix: sparse.csr_matrix, filepath: str):
    sparse.save_npz(filepath, adj_matrix)
