import anndata as ad
import pandas as pd
import itertools
from scipy import stats
from sklearn.cluster import AgglomerativeClustering
from copro.utils.helpers import reconstruct_corr_df_sym

NOISE = 1e6


def pairwise_peptide_correlations_(
    df,
    sample_column="filename",
    peptide_column="peptide_id",
    value_column="intensity",
    ):
    '''
    Calculate pairwise peptide correlations.
    Only outputs unique (non-symmetrical) correlations.

    Parameters:
    - df (pandas.DataFrame): The input DataFrame containing the data.
    - sample_column (str): The name of the column in `df` representing the samples.
    - peptide_column (str): The name of the column in `df` representing the peptides.
    - value_column (str): The name of the column in `df` representing the values.

    Returns:
    - result (pandas.DataFrame): A DataFrame containing the pairwise peptide
        correlations. Columns: 'pepA', 'pepB', 'PCC' (Pearson correlation coefficient).
        Only outputs unique (non-symmetrical) correlations (AB, not AB, B-A, AA, BB).
    '''

    # TODO: modify df input to be obs x vars. Here we have redundant steps with
    # AnnDataTrces pairwise_peptide_correlations()
    df = df[[sample_column, peptide_column, value_column]]

    pivot_df = df.pivot_table(index=sample_column, columns=peptide_column, values=value_column)
    columns = pivot_df.columns.tolist()

    corr_dict = {}

    for col_a, col_b in itertools.combinations(columns, 2):

        pivot_col_a = pivot_df.loc[:, col_a]
        pivot_col_b = pivot_df.loc[:, col_b]
        corr_dict[col_a + '_' + col_b] = stats.pearsonr(pivot_col_a, pivot_col_b)

    corr_df = pd.DataFrame.from_dict(corr_dict, orient='index')
    corr_df.columns = ['PCC', 'p-value']
    corr_df['peptide_pair'] = corr_df.index
    corr_df[['pepA', 'pepB']] = corr_df['peptide_pair'].str.split('_', expand=True)
    corr_df = corr_df[["pepA","pepB","PCC"]]
    corr_df = corr_df.reset_index(drop=True)

    return corr_df


def pairwise_peptide_correlations(
    adata,
    protein_id='protein_id',
    inplace=True,
    copy=False,
    ):

    if inplace and copy:
        raise ValueError('Arguments raise and copy are mutually exclusive')

    if protein_id not in adata.var.columns:
        raise ValueError(f'protein_id: {protein_id} not in .var.columns')

    def compute_corrs(df):
        corrs = pairwise_peptide_correlations_(
            df,
            sample_column='obs_id',
            peptide_column='var_id',
            value_column='intensity')

        return corrs

    anns = adata.var[['protein_id']].reset_index()
    traces_df = adata.to_df().T.reset_index()
    traces_df = traces_df.merge(anns, on='index')
    traces_df = traces_df.rename(columns={'index': 'var_id'})

    # TODO: remove unnecessary step of melting which gets unmelted
    #   in protein-level function

    traces_df = pd.melt(
        traces_df,
        id_vars=['protein_id', 'var_id'],
        var_name='obs_id',
        value_name='intensity')

    corrs = traces_df.groupby('protein_id', observed=True).apply(compute_corrs, include_groups=False)
    corrs = corrs.droplevel(1, axis=0)
    corrs = corrs.sort_values(['pepA', 'pepB']).sort_index()

    if inplace:
        adata.uns['pairwise_peptide_correlations'] = corrs

    elif copy:
        adata_new = adata.copy()
        adata_new.uns['pairwise_peptide_correlations'] = corrs
        return adata_new

    else:
        return corrs


def cluster_peptides_(
    df,
    method: str = 'agglomerative-hierarchical-clustering',
    ):
    '''
    Perform peptide clustering grouped by protein annotation.


    Parameters:
    ----------
    df : pandas.DataFrame
        Data frame with pairwise correlations annotated with the protein they belong to.]

    method : str
        Which clustering method to apply.

    Returns:
    -------
    dict
        Dictionary with clustering method output.
        - 'agglomerative-hierarchical-clustering'
            => {protein_id: {'labels': list, 'height': list, 'merge': list(list)}}
            - labels: list of peptides
            - merge: steps in which different peptides are merged.
                     n_steps == n_samples - 1
                     The two ids included for every step represent the index of the peptide in 'labels'.
            - heights: The height of each merging step in 'merge'.
                       The idx of the height corresponds to the index of the step in 'merge'.
    '''

    assert all(df.index == df.columns)

    model = AgglomerativeClustering(n_clusters=None,
                                    metric='precomputed',
                                    linkage='average',
                                    distance_threshold=0,
                                    compute_distances=True)

    model.fit(df)

    # pylint: disable=no-member
    dendogram = {
        'type': 'sklearn_agglomerative_clustering',
        'labels': model.feature_names_in_.tolist(),
        'heights': model.distances_.tolist(),
        'merge': model.children_.tolist()
    }
    # pylint: enable=no-member

    return dendogram


def cluster_peptides(
    adata,
    method='agglomerative-hierarchical-clustering',
    inplace=True,
    copy=False,
    ):

    if inplace and copy:
        raise ValueError('Arguments raise and copy are mutually exclusive')


    if 'pairwise_peptide_correlations' not in adata.uns:
        raise ValueError(f'pairwise_peptide_correlations not in .uns')


    corrs = adata.uns['pairwise_peptide_correlations'].copy()

    dends = {}

    for protein_id, df in corrs.groupby('protein_id', observed=True):

        corr_sym = reconstruct_corr_df_sym(
            df,
            var_a_col='pepA',
            var_b_col='pepB',
            corr_col='PCC')

        corr_dists = 1 - corr_sym

        dends[protein_id] = cluster_peptides_(
            corr_dists,
            method= 'agglomerative-hierarchical-clustering')

    if inplace:
        adata.uns['dendograms'] = dends

    elif copy:
        adata_new = adata.copy()
        adata_new.uns['dendograms'] = dends
        return adata_new
    
    else:
        return dends
