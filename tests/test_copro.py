import json
from pathlib import Path
import pytest
from pytest import approx
import pandas as pd
import copy
from sklearn.cluster import AgglomerativeClustering


from copro.copro import (cluster_peptides,
                         pairwise_peptide_correlations,
                         cut_clusters_in_n_real_
                         )

from tests.utils.helpers import (
        transform_dendogram_r2py,
        remap_dendogram_leaf_order,
        reconstruct_corr_df_sym,
        check_dendogram_equality
        )

TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR / "data"

noise = 1e6

def compare_clusters_dsVlist(ds, lst):
    groups = ds.groupby(ds).groups
    clusters_ds = [v.tolist() for _, v in groups.items()]
    clusters_ds = [tuple(sorted(c)) for c in clusters_ds]
    lst = [tuple(sorted(c)) for c in lst]
    lst_log = set(lst)

    counter=0
    for c in clusters_ds:
        counter+=1
        assert c in lst
        lst_log.remove(c)

    assert not lst_log

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
def traces_clusters_ref():

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


def test_cluster_peptides_vs_rcopf(traces_corrs, traces_clusters_ref):

    # Construct map: {protein: dendogram}
    dends = {}

    for protein_id, df in traces_corrs.groupby('protein_id'):

        corr_df_sym = reconstruct_corr_df_sym(df, var_a_col='pepA', var_b_col='pepB', corr_col='PCC')
        corr_dists = 1 - corr_df_sym
        dends[protein_id] = cluster_peptides(corr_dists)

    dends_ref = copy.deepcopy(traces_clusters_ref)

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


@pytest.fixture
def traces_clusters(traces_clusters_ref):
    return traces_clusters_ref

#@pytest.fixture
#def traces_

def test_cut_clusters_in_n_real_():
    '''Test protein-level cut_clusters_in_n_real_() on a single peptide group.'''
    # Using dendogram-based toy data
    #
    #                                     (11)
    #                                       |
    #                     (9)------------------------------(10)
    #                      |                                |
    #          (6)--------------------(7)                 /   \
    #           |                   /    \               |     |
    #         /   \                 |    |               |     |
    #  pepA (0)    pepB (1)  pepC (2)    pepD (3)  pepE (4)    pepF (5)
    #        0           0         0           0         0           0   n_clust=1, min_pep=1
    #        0           0         0           0         0           0   n_clust=1, min_pep=2
    #        1           1         1           1         0           0   n_clust=2, min_pep=1
    #        1           1         1           1         0           0   n_clust=2, min_pep=2
    #        2           2         1           1         0           0   n_clust=3, min_pep=1
    #        2           2         1           1         0           0   n_clust=3, min_pep=2
    # Expected cluster after cutting with different configurations, above:
    #   cluster numbers may be different order, which is accounted for in test comparisons.

    dendogram = {
        'type': 'sklearn_agglomerative_clustering',
        'labels': ['pepA', 'pepB', 'pepC', 'pepD', 'pepE', 'pepF'],
        'merge': [[0,1], [2,3], [4,5], [6,7], [8,9]],
        'heights': [0.1, 0.2, 0.4, 0.8, 0.9]
        }

    # Config 1
    clusters = cut_clusters_in_n_real_(dendogram,
                                       n_clusters = 1,
                                       min_peptides_per_cluster=1)
    expected_clusters = [['pepA', 'pepB', 'pepC', 'pepD', 'pepE', 'pepF']]
    compare_clusters_dsVlist(clusters, expected_clusters)

    # Config 2
    clusters = cut_clusters_in_n_real_(dendogram,
                                       n_clusters = 1,
                                       min_peptides_per_cluster=2)
    expected_clusters = [['pepA', 'pepB', 'pepC', 'pepD', 'pepE', 'pepF']]
    compare_clusters_dsVlist(clusters, expected_clusters)

    # Config 3
    clusters = cut_clusters_in_n_real_(dendogram,
                                       n_clusters = 2,
                                       min_peptides_per_cluster=1)
    expected_clusters = [['pepA', 'pepB', 'pepC', 'pepD'], ['pepE', 'pepF']]
    compare_clusters_dsVlist(clusters, expected_clusters)

    # Config 4
    clusters = cut_clusters_in_n_real_(dendogram,
                                       n_clusters = 2,
                                       min_peptides_per_cluster=2)
    expected_clusters = [['pepA', 'pepB', 'pepC', 'pepD'], ['pepE', 'pepF']]
    compare_clusters_dsVlist(clusters, expected_clusters)

    # Config 5
    clusters = cut_clusters_in_n_real_(dendogram,
                                       n_clusters = 3,
                                       min_peptides_per_cluster=1)
    expected_clusters = [['pepA', 'pepB'], ['pepC', 'pepD'], ['pepE', 'pepF']]
    compare_clusters_dsVlist(clusters, expected_clusters)

    # Config 6
    clusters = cut_clusters_in_n_real_(dendogram,
                                       n_clusters = 3,
                                       min_peptides_per_cluster=2)
    expected_clusters = [['pepA', 'pepB'], ['pepC', 'pepD'], ['pepE', 'pepF']]
    compare_clusters_dsVlist(clusters, expected_clusters)


    # Using dendogram-based toy data
    #
    #                                     (8)
    #                                      |
    #                     (7)-----------------------------
    #                      |                             |
    #          (5)--------------------(6)                |
    #           |                   /    \               |
    #         /   \                 |    |               |
    #  pepa (0)    pepb (1)  pepc (2)    pepd (3)  pepe (4)
    #        0           0         0           0         0   n_clust=1, min_pep=1
    #        0           0         0           0         0   n_clust=1, min_pep=2
    #        1           1         1           1         0   n_clust=2, min_pep=1
    #        1           1         0           0         x   n_clust=2, min_pep=2
    #        2           2         1           1         0   n_clust=3, min_pep=1
    #        2           1         0           0         x   n_clust=3, min_pep=2

    dendogram = {
        'type': 'sklearn_agglomerative_clustering',
        'labels': ['pepA', 'pepB', 'pepC', 'pepD', 'pepE'],
        'merge': [[0,1], [2,3], [5,6], [4,7]],
        'heights': [0.1, 0.2, 0.4, 0.8]
        }

    # Config 1
    clusters = cut_clusters_in_n_real_(dendogram,
                                       n_clusters = 1,
                                       min_peptides_per_cluster=1)
    expected_clusters = [['pepA', 'pepB', 'pepC', 'pepD', 'pepE']]
    compare_clusters_dsVlist(clusters, expected_clusters)

    # Config 2
    clusters = cut_clusters_in_n_real_(dendogram,
                                       n_clusters = 1,
                                       min_peptides_per_cluster=2)
    expected_clusters = [['pepA', 'pepB', 'pepC', 'pepD', 'pepE']]
    compare_clusters_dsVlist(clusters, expected_clusters)

    # Config 3
    clusters = cut_clusters_in_n_real_(dendogram,
                                       n_clusters = 2,
                                       min_peptides_per_cluster=1)
    expected_clusters = [['pepA', 'pepB', 'pepC', 'pepD'], ['pepE']]
    compare_clusters_dsVlist(clusters, expected_clusters)

    # Config 4
    clusters = cut_clusters_in_n_real_(dendogram,
                                       n_clusters = 2,
                                       min_peptides_per_cluster=2)
    expected_clusters = [['pepA', 'pepB'], ['pepC', 'pepD'], ['pepE']]
    compare_clusters_dsVlist(clusters, expected_clusters)
    assert clusters['pepE'] == noise

    # Config 5
    clusters = cut_clusters_in_n_real_(dendogram,
                                       n_clusters = 3,
                                       min_peptides_per_cluster=1)
    expected_clusters = [['pepA', 'pepB'], ['pepC', 'pepD'], ['pepE']]
    compare_clusters_dsVlist(clusters, expected_clusters)

    # Config 6
    clusters = cut_clusters_in_n_real_(dendogram,
                                       n_clusters = 3,
                                       min_peptides_per_cluster=2)
    expected_clusters = [['pepA', 'pepB', 'pepC', 'pepD', 'pepE']]
    compare_clusters_dsVlist(clusters, expected_clusters)
    assert clusters.nunique() == 1
    assert clusters.iloc[0] == noise


    
    # Using correlation based toy data
    #
    #                                     (8)
    #                                      |
    #                     (7)-----------------------------
    #                      |                             |
    #          (5)--------------------(6)                |
    #           |                   /    \               |
    #         /   \                 |    |               |
    #  pepa (0)    pepb (1)  pepc (2)    pepd (3)  pepe (4)
    #        0           0         0           0         0   n_clust=1, min_pep=1
    #        0           0         0           0         0   n_clust=1, min_pep=2
    #        1           1         1           1         0   n_clust=2, min_pep=1
    #        1           1         0           0         x   n_clust=2, min_pep=2
    #        2           2         1           1         0   n_clust=3, min_pep=1
    #        2           1         0           0         x   n_clust=3, min_pep=2

    corrs = pd.DataFrame({
        'pepA': [0, 1, 3, 3, 4],
        'pepB': [1, 0, 3, 3, 4],
        'pepC': [3, 3, 0, 2, 4],
        'pepD': [3, 3, 2, 0, 4],
        'pepE': [4, 4, 4, 4, 0],
        }, index=['pepA', 'pepB', 'pepC', 'pepD', 'pepE'])

    model = AgglomerativeClustering(n_clusters=None,
                                    metric='precomputed',
                                    linkage='average',
                                    distance_threshold=0,
                                    compute_distances=True)

    model.fit(corrs)

    dendogram = {
        'type': 'sklearn_agglomerative_clustering',
        'labels': model.feature_names_in_.tolist(),
        'heights': model.distances_.tolist(),
        'merge': model.children_.tolist()
    }

    # Same as previous dendogram as same structure

    # Config 1
    clusters = cut_clusters_in_n_real_(dendogram,
                                       n_clusters = 1,
                                       min_peptides_per_cluster=1)
    expected_clusters = [['pepA', 'pepB', 'pepC', 'pepD', 'pepE']]
    compare_clusters_dsVlist(clusters, expected_clusters)

    # Config 2
    clusters = cut_clusters_in_n_real_(dendogram,
                                       n_clusters = 1,
                                       min_peptides_per_cluster=2)
    expected_clusters = [['pepA', 'pepB', 'pepC', 'pepD', 'pepE']]
    compare_clusters_dsVlist(clusters, expected_clusters)

    # Config 3
    clusters = cut_clusters_in_n_real_(dendogram,
                                       n_clusters = 2,
                                       min_peptides_per_cluster=1)
    expected_clusters = [['pepA', 'pepB', 'pepC', 'pepD'], ['pepE']]
    compare_clusters_dsVlist(clusters, expected_clusters)

    # Config 4
    clusters = cut_clusters_in_n_real_(dendogram,
                                       n_clusters = 2,
                                       min_peptides_per_cluster=2)
    expected_clusters = [['pepA', 'pepB'], ['pepC', 'pepD'], ['pepE']]
    compare_clusters_dsVlist(clusters, expected_clusters)
    assert clusters['pepE'] == noise

    # Config 5
    clusters = cut_clusters_in_n_real_(dendogram,
                                       n_clusters = 3,
                                       min_peptides_per_cluster=1)
    expected_clusters = [['pepA', 'pepB'], ['pepC', 'pepD'], ['pepE']]
    compare_clusters_dsVlist(clusters, expected_clusters)

    # Config 6
    clusters = cut_clusters_in_n_real_(dendogram,
                                       n_clusters = 3,
                                       min_peptides_per_cluster=2)
    expected_clusters = [['pepA', 'pepB', 'pepC', 'pepD', 'pepE']]
    compare_clusters_dsVlist(clusters, expected_clusters)
    assert clusters.nunique() == 1
    assert clusters.iloc[0] == noise
