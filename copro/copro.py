import itertools
import pandas as pd
from scipy import stats
from sklearn.cluster import AgglomerativeClustering

def pairwise_peptide_correlations(
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


def cluster_peptides(df,
                     column_map: dict = {'peptide_a': 'pepA',
                                         'peptide_b': 'pepB',
                                         'correlation_values': 'PCC',
                                         'protein_id': 'protein_id'},
                     method: str = 'agglomerative-hierarchical-clustering') -> dict:
    '''
    Perform peptide clustering grouped by protein annotation.


    Parameters:
    ----------
    df : pandas.DataFrame
        Data frame with pairwise correlations annotated with the protein they belong to.]
    column_map: dict

    method : str
        Which clustering method to apply.

    Returns:
    -------
    dict
        Dictionary with clustering method output.
        - 'agglomerative-hierarchical-clustering' => {protein_id: {'labels': list, 'height': list, 'merge': list(list)}}
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
        'labels': model.feature_names_in_.tolist(),
        'height': model.distances_.tolist(),
        'merge': model.children_.tolist()
    }
    # pylint: enable=no-member

    return dendogram
