import json
from pathlib import Path
import pytest
import pandas as pd
import copy

from copro.copro import cluster_peptides
from tests.utils.helpers import transform_dendogram_r2py, remap_dendogram_leaf_order, reconstruct_corr_df_sym, check_dendogram_equality

TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR / "data"


@pytest.fixture
def traces_corr_df():

    traces_corr_df_path = DATA_DIR / 'mouse_tissue/traces_correlations_rcopf.tsv'
    df = pd.read_csv(traces_corr_df_path, sep='\t', names=['pepA', 'pepB', 'PCC', 'protein_id'])

    return df


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


def test_cluster_peptides_on_obj(traces_corr_df, traces_clustered_map_ref):

    # Construct map: {protein: dendogram}
    dends = {}

    for protein_id, df in traces_corr_df.groupby('protein_id'):

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
        check_dendogram_equality(dends[prot_id], dends_ref[prot_id], abs_tolerance=1e-4)
