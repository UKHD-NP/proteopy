import itertools
import copy
import pandas as pd
import numpy as np
import anndata as ad
from scipy import stats
from scipy.stats import norm
from sklearn.cluster import AgglomerativeClustering
from statsmodels.stats.multitest import multipletests
from copro.utils.data_structures import BinaryClusterTree
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


def cluster_peptides_(
        df,
        method: str = 'agglomerative-hierarchical-clustering'
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


class AnnDataTraces(ad.AnnData):

    def pairwise_peptide_correlations(self, protein_id='protein_id'):

        if protein_id not in self.var.columns:
            raise ValueError(f'protein_id: {protein_id} not in .var.columns')

        @staticmethod
        def compute_corrs(df):
            corrs = pairwise_peptide_correlations_(
                df,
                sample_column='obs_id',
                peptide_column='var_id',
                value_column='intensity')

            return corrs

        anns = self.var[['protein_id']].reset_index()
        traces_df = self.to_df().T.reset_index()
        traces_df = traces_df.merge(anns, on='index')
        traces_df = traces_df.rename(columns={'index': 'var_id'})

        # TODO: remove unnecessary step of melting which gets unmelted
        #   in protein-level function

        traces_df = pd.melt(
            traces_df,
            id_vars=['protein_id', 'var_id'],
            var_name='obs_id',
            value_name='intensity')

        corrs = traces_df.groupby('protein_id').apply(compute_corrs, include_groups=False)
        corrs = corrs.droplevel(1, axis=0)
        corrs = corrs.sort_values(['pepA', 'pepB']).sort_index()

        self.uns['pairwise_peptide_correlations'] = corrs


    def cluster_peptides(self, method = 'agglomerative-hierarchical-clustering'):

        if 'pairwise_peptide_correlations' not in self.uns:
            raise ValueError(f'pairwise_peptide_correlations not in .uns')


        corrs = self.uns['pairwise_peptide_correlations']

        dends = {}

        for protein_id, df in corrs.groupby('protein_id'):

            corr_sym = reconstruct_corr_df_sym(
                df,
                var_a_col='pepA',
                var_b_col='pepB',
                corr_col='PCC')

            corr_dists = 1 - corr_sym

            dends[protein_id] = cluster_peptides_(
                corr_dists,
                method= 'agglomerative-hierarchical-clustering')

        self.uns['dendograms'] = dends


    def cut_dendograms_in_n_real(self, n_clusters=2, min_peptides_per_cluster=2, noise=NOISE):

        if 'dendograms' not in self.uns:
            raise ValueError(f'dendograms not in .uns')

        var = self.var
        var['cluster_id'] = -1

        clusters_ann = {}

        dends = self.uns['dendograms']
        for prot, dend in dends.items():
            dend_upd = copy.deepcopy(dend)
            dend_upd['type'] = 'sklearn_agglomerative_clustering'

            clusters = cut_dendograms_in_n_real_(
                dend_upd,
                n_clusters=2,
                min_peptides_per_cluster=2,
                noise=noise)

            mask = (var['protein_id'] == prot) & (var.index.isin(clusters.index))
            var.loc[mask, 'cluster_id'] = clusters.reindex(var.index[mask])

            clusters_ann[prot] = clusters

        self.uns['clusters'] = clusters_ann

        assert not any((var['cluster_id'] == -1).tolist())


    def proteoform_scores(self, alpha=0.05, summary_func=np.mean, noise=NOISE):

        if 'pairwise_peptide_correlations' not in self.uns:
            raise ValueError(f'pairwise_peptide_correlations not in .uns')

        if 'dendograms' not in self.uns:
            raise ValueError(f'dendograms not in .uns')

        columns = [
            'protein_id',
            'proteoform_score',
            'proteoform_score_z',
            'proteoform_score_dz',
            'proteoform_score_pval',
            ]

        corrs = self.uns['pairwise_peptide_correlations'].copy().reset_index()
        var = self.var
        n_fractions = self.n_obs

        proteoform_scores_list = []

        for prot, corrs_prot in corrs.groupby('protein_id'):

            corrs_mat = reconstruct_corr_df_sym(
                corrs_prot,
                var_a_col='pepA',
                var_b_col='pepB',
                corr_col='PCC')

            clusters = var.loc[var['protein_id'] == prot, 'cluster_id']

            scores = proteoform_scores_(
                corrs_mat,
                clusters,
                n_fractions,
                summary_func=np.mean)

            scores_entry = {column:value for column, value in zip(columns[1:5], scores)}
            scores_entry['protein_id'] = prot
            scores_entry = pd.DataFrame([scores_entry])
            proteoform_scores_list.append(scores_entry)

        proteoform_scores = pd.concat(proteoform_scores_list, ignore_index=True)
        proteoform_scores = proteoform_scores[columns]

        # Perform multiple-testing correction
        mask_nonan = proteoform_scores['proteoform_score_pval'].notna()
        pvals = proteoform_scores.loc[mask_nonan, 'proteoform_score_pval']

        rejected, corrected_pvals, _, _ = multipletests(pvals, alpha=alpha, method='fdr_bh')

        proteoform_scores['proteoform_score_pval_adj'] = np.nan
        proteoform_scores['is_proteoform'] = np.nan
        
        proteoform_scores.loc[pvals.index, 'proteoform_score_pval_adj'] = corrected_pvals
        proteoform_scores.loc[pvals.index, 'is_proteoform'] = rejected

        # Add all new scores to .var
        var_upd = pd.merge(
            var,
            proteoform_scores,
            on='protein_id',
            how='left',
            validate='many_to_one')

        var_upd = var_upd.set_index('peptide_id', drop=False)
        var_upd.index.name = None

        assert (var.index == var_upd.index).all()
        self.var = var_upd
