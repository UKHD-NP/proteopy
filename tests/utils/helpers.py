import numpy as np
import pandas as pd
import copy
from pytest import approx

def reconstruct_corr_df_sym(df, var_a_col='pepA', var_b_col='pepB', corr_col='correlation value'):
    '''Reconstruct correlation dataframe in symmetrical matrix format.

    Reconstruct a full correlation matrix from a long DataFrame containing asymmetric correlation data.
    
    Args:
        df (pd.DataFrame): DataFrame with columns for peptide A, peptide B, and their correlation value
        var_a_col (str): Name of column containing first peptide identifier
        var_b_col (str): Name of column containing second peptide identifier
        corr_col (str): Name of column containing correlation values
        
    Returns:
        pd.DataFrame: Fully symmetric correlation matrix as a pd.DataFrame with peptide labels as columns and rows.
    '''

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
                raise ValueError('Logical bug')
            elif not np.isnan(corr_matrix[j, i]) and not np.isnan(corr_matrix[i, j]):
                assert corr_matrix[i,j] == corr_matrix[j,i]
    

    corr_df = pd.DataFrame(corr_matrix, index=all_peptides, columns=all_peptides)
    
    return corr_df


def transform_dendogram_merge_arr_r2py(merge_arr: list):

    merge_new = np.array(merge_arr)
    n_samples = merge_new.shape[0] + 1

    merge_new = np.where(merge_new > 0, merge_new + n_samples, merge_new)
    merge_new = np.abs(merge_new)
    merge_new = merge_new - 1 # 1-based -> 0-based order

    return merge_new


def transform_dendogram_r2py(dendogram: dict):
    '''
    Parameters:
    -----------
    dendogram: dict
        Dictionary with the following structure: {..., merge = [[int, int], ...]}
    '''

    if not 'merge' in dendogram.keys():
        raise ValueError('Dendogram slot missing!')

    merge = dendogram['merge']
    merge_new = transform_dendogram_merge_arr_r2py(merge)

    dendogram_new = copy.deepcopy(dendogram)
    dendogram_new['merge'] = merge_new.tolist()

    return dendogram_new


def remap_dendogram_leaf_order(dendogram: dict, ref_labels: list):
    '''
    Remap nodes in dendogram['merge'] using a reference label order.
    
    Parameters:
    -----------
    - dendogram: dict
        - labels: list of ordered labels of length n_samples
        - merge: np.ndarray of shape (n_samples-1, 2)
        - heights: list of length n_samples
    - ref_annotation: list of labels in desired new leaf order
    
    Returns:
    --------
    - dendogram with updated node indices remapped to match ref_annotation order
    '''
    orig_labels = dendogram['labels']
    n_samples = len(orig_labels)

    assert set(orig_labels) == set(ref_labels), f'orig_labels: {orig_labels},\nref_labels: {ref_labels}'
    assert len(orig_labels) == len(ref_labels)
    assert len(orig_labels) == len(dendogram['merge']) + 1

    merge_arr = np.array(dendogram['merge'])
    
    # Mapping from original index to ref index
    orig_label_to_index = {label: i for i, label in enumerate(orig_labels)}
    ref_label_to_index = {label: i for i, label in enumerate(ref_labels)}
    
    # Create remapping array
    leaf_map = np.zeros(n_samples, dtype=int)
    for label in orig_labels:
        orig_idx = orig_label_to_index[label]
        ref_idx = ref_label_to_index[label]
        leaf_map[orig_idx] = ref_idx
    
    # Now remap only values < n_leaves
    merge_remapped = merge_arr.copy()

    for i in range(merge_remapped.shape[0]):
        for j in range(2):
            if merge_remapped[i, j] < n_samples:
                merge_remapped[i, j] = leaf_map[merge_remapped[i, j]]

    # Replace old merge
    dendogram_remapped = copy.deepcopy(dendogram)
    dendogram_remapped['labels'] = ref_labels
    dendogram_remapped['merge'] = merge_remapped.tolist()
    
    return dendogram_remapped


def check_dendogram_equality(dend, dend_ref, rel_tolerance=None, abs_tolerance=None):
    '''
    Note:
        To choose the tolerances view API: pytest.approx
    '''

    keys = ('labels', 'merge', 'heights')

    # Correct dict keys
    assert all([key in dend_ref.keys() for key in keys]), f'dend.keys: {list(dend.keys())}\nkeys:{keys}'
    assert all([key in dend.keys() for key in keys]), f'dend.keys: {list(dend.keys())}\nkeys:{keys}'

    # Equal labels
    labels_ref = dend_ref['labels']
    labels = dend['labels']
    assert labels_ref == labels

    # Equal merge arrays
    merge_arr_ref = dend_ref['merge']
    merge_arr = dend['merge']

    for i, (pair_ref, pair) in enumerate(zip(merge_arr, merge_arr_ref)):
        assert pair_ref == pair or pair_ref == pair[::-1], f'{i}'

    # Equal heights
    heights_ref = dend_ref['heights']
    heights = dend['heights']
    assert heights == approx(heights_ref, rel=rel_tolerance, abs=abs_tolerance)
