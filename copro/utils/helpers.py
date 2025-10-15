import numpy as np
import pandas as pd


def reconstruct_corrs_df_symmetric_from_long_df(df, var_a_col=0, var_b_col=1, corr_col=2):
    '''Reconstruct correlation dataframe in symmetrical matrix format.

    Reconstruct a full correlation matrix from a long DataFrame containing asymmetric correlation data.
    
    Args:
        df (pd.DataFrame): DataFrame with columns for peptide A, peptide B, and their correlation value
        var_a_col (str | int): Name of column containing first peptide identifier
        var_b_col (str | int): Name of column containing second peptide identifier
        corr_col (str | int): Name of column containing correlation values
        
    Returns:
        pd.DataFrame: Fully symmetric correlation matrix as a pd.DataFrame with peptide labels as columns and rows.
    '''
    if isinstance(var_a_col, int):
        var_a_col = df.columns[var_a_col]

    if isinstance(var_b_col, int):
        var_b_col = df.columns[var_b_col]

    if isinstance(corr_col, int):
        corr_col = df.columns[corr_col]

    all_peptides = set(df[var_a_col]).union(set(df[var_b_col]))
    all_peptides = sorted(list(all_peptides))
    n = len(all_peptides)
    
    pep_to_idx = {pep: i for i, pep in enumerate(all_peptides)}
    
    # Init
    corr_matrix = np.full((n, n), np.nan)
    np.fill_diagonal(corr_matrix, 1.0)
    
    # Fill in the known correlation values
    for _, row in df.iterrows():
        i = pep_to_idx[row[var_a_col]]
        j = pep_to_idx[row[var_b_col]]

        corr_matrix[i, j] = row[corr_col]
    
    # Fill in the symmetric values where possible
    for i in range(n):
        for j in range(i+1, n):

            if np.isnan(corr_matrix[i, j]) and not np.isnan(corr_matrix[j, i]):
                corr_matrix[i, j] = corr_matrix[j, i]
            elif np.isnan(corr_matrix[j, i]) and not np.isnan(corr_matrix[i, j]):
                corr_matrix[j, i] = corr_matrix[i, j]
            elif np.isnan(corr_matrix[j, i]) and np.isnan(corr_matrix[i, j]):
                rev = {i: pep for pep, i in pep_to_idx.items()}
                raise ValueError((
                    f'Logical bug. For combination of peptides: {rev[i]} and '
                    f'{rev[j]} there was no value found.'
                    ))
            elif not np.isnan(corr_matrix[j, i]) and not np.isnan(corr_matrix[i, j]):
                assert corr_matrix[i,j] == corr_matrix[j,i]
    

    corr_df = pd.DataFrame(corr_matrix, index=all_peptides, columns=all_peptides)
    
    return corr_df
