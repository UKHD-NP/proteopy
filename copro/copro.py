import itertools
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm
from sklearn.cluster import AgglomerativeClustering
from copro.utils.data_structures import BinaryClusterTree

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
                     method: str = 'agglomerative-hierarchical-clustering') -> dict:
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


def cut_dendograms_in_n_real_(
        dendogram,
        n_clusters=2,
        min_peptides_per_cluster=2,
        noise=1e6,
        ):
    '''
    Cut clusters from cluster_peptides into N clusters with more than 1 peptide. 
    '''
    n_peptides = len(dendogram['labels'])
    n_real_clusters = 0
    k = n_clusters
    cluster_tree = BinaryClusterTree(constructor=dendogram)

    while n_real_clusters < n_clusters:
        clusters = cluster_tree.cut(k, use_labels=True)
        n_per_cluster = clusters.value_counts()
        is_multipep = n_per_cluster >= min_peptides_per_cluster
        n_real_clusters = is_multipep.sum()
        k += 1

        single_pep_clusters = n_per_cluster[~is_multipep].index
        clusters[clusters.isin(single_pep_clusters)] = noise

        if k >= n_peptides:
            clusters[:] = noise
            break

    # Rename cluster_ids to systematic format
    max_cluster = clusters.max()
    cats = clusters.astype('category').cat.categories
    n_clusters = len(cats)

    if max_cluster != n_clusters:
        for i in range(n_clusters):
            clusters[clusters == cats[i]] = i

    if noise in cats:
        clusters[clusters == max(clusters)] = noise

    return clusters


def proteoform_scores_(
        corrs,
        clusters,
        n_fractions,
        summary_func=np.mean,
        noise=1e6
        ):
    '''
    Calculates a score for proteoforms based on the difference of within
    cluster distances and between cluster distances.

    IMPORTANT: currently only implemented properly for n_clusters = 2

    Args:
        corrs (pd.DataFrame): correlation between peptides.
            In symmetrical matrix form (index == columns)
        clusters (pd.Series | pd.DataFrame): vector of cluster_ids with indexes
            corresponding to the peptides for a specific protein.
        n_fractions (int): Number of samples.
        summary_func (Callable): Summary function to apply to intra- and inter-
            cluster correlation coefficients.
    '''
    def replace_upper_triangle(df, replacement, k=0):
        arr = df.to_numpy().astype(float)
        rows, cols = np.triu_indices_from(arr, k=k)
        arr[rows, cols] = replacement

        new_df = pd.DataFrame(arr, columns=df.columns, index=df.index)

        return new_df

    if isinstance(clusters, pd.DataFrame):
        clusters = clusters['cluster']

    if np.issubdtype(clusters.dtype, np.floating):
        clusters = clusters.astype(int)

    assert any(corrs.index == corrs.columns)
    assert all([i in clusters.index for i in corrs.index]), \
        f'clusters.index = {clusters.index}' \
        f'\ncorrs_index = {corrs.index}'

    if (clusters == noise).all().all():
        return np.array([0, np.nan, np.nan, np.nan])

    cluster_ids = clusters.unique()
    cluster_ids = cluster_ids[cluster_ids != noise].tolist()

    if len(cluster_ids) > 2:

        raise ValueError('Functionality with n_clusters > 2 not implemented yet.')

        mat = corrs.copy(deep=True)
        stat_v = []

        for c in cluster_ids:
            cluster_ids_inv = cluster_ids[cluster_ids != c]
            clust1_ids = clusters[clusters == cluster_ids_inv[0]]
            clust2_ids = clusters[clusters == cluster_ids_inv[1]]
            clust_ids_ord = clust1_ids + clust2_ids
            mat_inv = corrs.loc[clust_ids_ord, clust_ids_ord]

            cross = mat_inv.loc[clust1_ids, clust2_ids] # QUESTION: why no diagonal removal as below?
            values = cross.to_numpy().flatten()
            values = values[~np.isnan(values)]
            stat_across = np.apply_along_axis(summary_func, 0, cross)

            rows, cols = np.triu_indices_from(mat_inv, k=0)  # k=1 excludes diagonal
            mat_inv.to_numpy()[rows, cols] = np.nan

            within_c1 = mat_inv.loc[clust1_ids, clust1_ids]
            values = within_c1.to_numpy().flatten()
            values = values[~np.isnan(values)]
            stat_within_c1 = np.apply_along_axis(summary_func, 0, values)

            within_c2 = mat_inv.loc[clust2_ids, clust2_ids]
            values = within_c2.to_numpy().flatten()
            values = values[~np.isnan(values)]
            stat_within_c2 = np.apply_along_axis(summary_func, 0, values)

            stat_within = min([stat_within_c1, stat_within_c2])

            diff_stat = stat_within - stat_across

            z_stat_within = np.atanh(stat_within)
            z_stat_across = np.atanh(stat_across)
            z_diff_stat = z_stat_within - z_stat_across

            dz = z_diff_stat / (np.sqrt((1 / (n_fractions-3)) + (1 / (n_fractions-3))))
            pval = 2 * (1 - norm.cdf(np.abs(dz)))

            stat_v.append([diff_stat, z_diff_stat, dz, pval])

        diff_stats = [i[0] for i in stat_v]
        sel_min_diff = np.which(diff_stats == diff_stats.min(skip_na=True))[0]

        return stat_v[sel_min_diff]

    else:
        clust1_ids = clusters[clusters == cluster_ids[0]].index.to_list()
        clust2_ids = clusters[clusters == cluster_ids[1]].index.to_list()
        clust_ids_ord = clust1_ids + clust2_ids
        mat = corrs.loc[clust_ids_ord, clust_ids_ord]

        # Cross-cluster statistic
        cross = corrs.loc[clust1_ids, clust2_ids]
        values = cross.to_numpy().flatten()
        stat_across = np.apply_along_axis(summary_func, 0, values)

        mat = replace_upper_triangle(mat, np.nan, k=0)

        # Within cluster statistic
        within_c1 = mat.loc[clust1_ids, clust1_ids]
        wc1_values = within_c1.to_numpy().flatten()
        wc1_values = wc1_values[~np.isnan(wc1_values)]
        stat_within_c1 = np.apply_along_axis(summary_func, 0, wc1_values)

        within_c2 = mat.loc[clust2_ids, clust2_ids]
        wc2_values = within_c2.to_numpy().flatten()
        wc2_values = wc2_values[~np.isnan(wc2_values)]
        stat_within_c2 = np.apply_along_axis(summary_func, 0, wc2_values)

        stat_within = min([stat_within_c1, stat_within_c2])

        diff_stat = stat_within - stat_across

        # Fisher's z-transformation to norm distr. and rationally scaled values
        z_stat_within = np.atanh(stat_within)
        z_stat_across = np.atanh(stat_across)
        z_diff_stat = z_stat_within - z_stat_across

        # T-test: intra-cluster peptide correlations are significantly different
        #   from cross-cluster peptide correlations
        dz = z_diff_stat / np.sqrt((1 / (n_fractions-3)) + (1 / (n_fractions-3)))
        pval = 2 * (1 - norm.cdf(np.abs(dz)))

        return np.array([diff_stat, z_diff_stat, dz, pval])
