import numpy as np
import pandas as pd
import copy
from pytest import approx

from copro.utils.copf import reconstruct_corrs_df_symmetric_from_long_df

def test_reconstruct_corrs_df_symmetric_from_long_df():

    # labels: a-c
    #
    # [[ x, 0.1, 0.5],        [[ 1  , 0.1, 0.5],
    #  [ x, 1  , x  ],   ==>   [ 0.1, 1  , 0.4],
    #  [ x  0.4, 1  ]]         [ 0.5, 0.4, 1  ]]

    df = pd.DataFrame({
        'colA': ['a', 'a', 'b', 'c', 'c'],
        'colB': ['b', 'c', 'b', 'b', 'c'],
        'value':[0.1, 0.5, 1, 0.4, 1]
        })

    df_expected = pd.DataFrame({
        'a': [1, 0.1, 0.5],
        'b': [0.1, 1, 0.4],
        'c':[0.5, 0.4, 1]
        }, index=['a', 'b', 'c'])

    df_reconstructed = reconstruct_corrs_df_symmetric_from_long_df(df, 'colA', 'colB', 2)
    assert np.isclose(df_reconstructed, df_expected, atol=1e-4).all().all()
    assert all(df_reconstructed.index == df_reconstructed.columns)


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
