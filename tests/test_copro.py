import json
from pathlib import Path
import pytest
from pytest import approx
import pandas as pd
import copy

from copro.copro import (cluster_peptides,
                         pairwise_peptide_correlations
                         )

from tests.utils.helpers import (
        transform_dendogram_r2py,
        remap_dendogram_leaf_order,
        reconstruct_corr_df_sym,
        check_dendogram_equality
        )

TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR / "data"

@pytest.fixture
def traces_preproc():
    '''
    Get COPF mouse tissue pre-processed traces df.
    '''
    traces_path = DATA_DIR / 'mouse_tissue/traces_pre-processed_rcopf.tsv'
    traces = pd.read_csv(traces_path, sep='\t', header=0)
    traces = traces.rename(columns={'id': 'peptide_id'})

    return traces

@pytest.fixture
def traces_preproc_anns():
    '''
    Get COPF mouse tissue pre-processed traces annotations df.
    '''
    anns_path = (
            DATA_DIR / 'mouse_tissue/traces_pre-processed_trace-annotations_rcopf.tsv'
            )
    anns = pd.read_csv(anns_path, sep='\t', header=0)
    anns = anns.rename(columns={'id': 'peptide_id'})

    return anns


@pytest.fixture
def traces_preproc_ext(traces_preproc, traces_preproc_anns):
    '''
    Extend pre-processed traces with annotations.
    '''
    anns_select = traces_preproc_anns[['peptide_id', 'protein_id']]
    traces_ext = traces_preproc.merge(anns_select, on='peptide_id')
    traces_ext = pd.melt(traces_ext, id_vars=('protein_id', 'peptide_id'))
    traces_ext = traces_ext.rename(columns={'value': 'intensity', 'variable': 'sample'})

    return traces_ext


@pytest.fixture
def traces_corrs():
    '''
    Get COPF mouse tissue correlations df.
    '''
    traces_corrs_path = DATA_DIR / 'mouse_tissue/traces_correlations_rcopf.tsv'
    col_names = ['pepA', 'pepB', 'PCC', 'protein_id']
    df = pd.read_csv(traces_corrs_path, sep='\t', names=col_names)

    return df


@pytest.fixture
def traces_corrs_ref(traces_corrs):
    '''
    Filter COPF mouse tissue correlations df for 
    unique (non-symmetrical) correlation values.
    '''
    corrs_ref = traces_corrs.set_index('protein_id')
    corrs_ref = corrs_ref[corrs_ref['PCC'] != 1]

    sort_peps_ab = lambda row: tuple(sorted([row['pepA'], row['pepB']]))

    corrs_ref['sorted_pair'] = corrs_ref.apply(sort_peps_ab, axis=1)
    corrs_ref = corrs_ref.drop_duplicates(subset=['sorted_pair'])
    corrs_ref = corrs_ref.drop(columns=['sorted_pair'])
    corrs_ref = corrs_ref.sort_values(['pepA', 'pepB']).sort_index()

    return corrs_ref


def test_pairwise_peptide_correlations_vs_rcopf(traces_preproc_ext, traces_corrs_ref):
    '''
    Test pairwise_peptide_correlations() application for equality to rCOPF correlations df.
    Uses COPF mouse tissue dataset as reference results.
    '''

    # Apply pairwise_peptide_correlations on the entire mouse tissue df
    pep_corrs = lambda x: pairwise_peptide_correlations(x,
                                                        sample_column='sample',
                                                        peptide_column='peptide_id',
                                                        value_column='intensity')


    corrs = traces_preproc_ext.groupby('protein_id').apply(pep_corrs, include_groups=False)
    corrs = corrs.droplevel(1, axis=0)
    corrs = corrs.sort_values(['pepA', 'pepB']).sort_index()

    # Compare to rCOPF reference output
    pep_cols = ['pepA', 'pepB']
    assert corrs[pep_cols].equals(traces_corrs_ref[pep_cols]) # Both prev. sorted

    abs_tolerance = 1e-14 # loaded reference corrs precision 1e-15
    assert corrs['PCC'].values == approx(traces_corrs_ref['PCC'].values, abs=abs_tolerance)


@pytest.fixture
def traces_clustered_map_ref():

    clusts_ref_path = DATA_DIR / 'mouse_tissue/traces_clustered_rcopf.json'

    with open(clusts_ref_path, 'r') as f:
        dends_R = json.load(f)

    # Reformat to match python sklearn dendograms
    dends = {}

    for prot_id in dends_R:
    
        dend = dends_R[prot_id]
        dend = transform_dendogram_r2py(dend)

        if isinstance(dend['height'], float):
            dend['height'] = [dend['height']]

        dends[prot_id] = dend

    return dends


def test_cluster_peptides_vs_rcopf(traces_corrs, traces_clustered_map_ref):

    # Construct map: {protein: dendogram}
    dends = {}

    for protein_id, df in traces_corrs.groupby('protein_id'):

        corr_df_sym = reconstruct_corr_df_sym(df, var_a_col='pepA', var_b_col='pepB', corr_col='PCC')
        corr_dists = 1 - corr_df_sym
        dends[protein_id] = cluster_peptides(corr_dists)

    dends_ref = copy.deepcopy(traces_clustered_map_ref)

    # Remap 
    for prot_id, dend in dends_ref.items():

        dend_corrected = remap_dendogram_leaf_order(dend, ref_labels=dends[prot_id]['labels'])
        dends_ref[prot_id] = dend_corrected

    # Equal dendogram dict structure
    assert set(dends_ref.keys()) == set(dends.keys())
    assert len(dends_ref.keys()) == len(dends.keys())

    for prot_id in dends_ref.keys():
        abs_tolerance = 1e-4 # loaded reference heights precision = 1e-4
        check_dendogram_equality(dends[prot_id],
                                 dends_ref[prot_id],
                                 abs_tolerance=abs_tolerance)
