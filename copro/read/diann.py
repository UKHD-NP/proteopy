import re
import warnings
import gc
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns


def diann(
    diann_output_path,
    aggr_level,
    precursor_pval_max,
    gene_pval_max,
    global_precursor_pval_max,
    show_input_stats = False,
    run_parser = None,
    fill_na = None,
    ):
    '''dia-NN input reader

    Load dia-nn output to anndata format.
    Notes:
        Filters for only proteotypic precursors - peptide maps to a single
        Protein.Ids.
    '''

    # Check args
    aggr_level_options = [
        'Stripped.Sequence',
        'Modified.Sequence',
        'Precursor.Id',
        ]

    if aggr_level not in aggr_level_options:
        raise ValueError((
            f'Wrong option passsed to aggr_level argument: {aggr_level}.'
            ))

    if run_parser is not None and not callable(run_parser):
        raise ValueError((
            'run_parser arg must either be a function or None.'
            ))

    data = pd.read_csv(diann_output_path, sep='\t', header=0)

    if run_parser:
        data['Run'] = data['Run'].apply(run_parser)

    if show_input_stats:
        print('Before Q-value and proteotypicity filtering\n------')
        proteotypic_fraction = (data['Proteotypic'] == 1).sum() / len(data)
        print(f'Proteotypic peptide fraction: {proteotypic_fraction:.2f}')

        multimapper_fraction = (
            (data['Protein.Ids'].str.split(';').apply(len) == 1)
            .sum() / len(data)
            )
        print(f'Multimapper peptide fraction: {multimapper_fraction:.2f}')

        # Q value distr. plots 
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16,4))
        plt.subplots_adjust(wspace=0.3)

        sns.histplot(data['Q.Value'], bins=100, ax=axes[0])
        axes[0].set_title('Q.Value distr.')

        if precursor_pval_max:
            axes[0].axvline(x=precursor_pval_max, color='red', linestyle='--', linewidth=2)

        sns.histplot(data['Global.Q.Value'], bins=100, ax=axes[1])
        axes[1].set_title('Gobal.Q.Value distr.')
        
        if global_precursor_pval_max:
            axes[1].axvline(x=global_precursor_pval_max, color='red', linestyle='--', linewidth=2)

        sns.histplot(data['Protein.Q.Value'], bins=100, ax=axes[2])
        axes[2].set_title('Protein.Q.Value distr.')

        if gene_pval_max:
            axes[2].axvline(x=gene_pval_max, color='red', linestyle='--', linewidth=2)
        plt.show()

        # Q values stats
        q_stats = data[['Q.Value', 'Protein.Q.Value', 'Global.Q.Value']].describe()
        print(q_stats)


    # Filter ds
    data_sub = data[
        (data['Proteotypic'] == 1)
        & (data['Protein.Ids'].str.split(';').apply(len).eq(1))
        ].copy()
    del data
    gc.collect()

    # ToDo: change to < instead of <=
    if precursor_pval_max:
        data_sub = data_sub[data_sub['Q.Value'] <= precursor_pval_max]
    if global_precursor_pval_max:
        data_sub = data_sub[data_sub['Global.Q.Value'] <= global_precursor_pval_max]
    if gene_pval_max:
        data_sub = data_sub[data_sub['Protein.Q.Value'] <= gene_pval_max]

    if len(data_sub) == 0:
        raise ValueError('Dataframe after filtering empty')

    if show_input_stats:
        # Q values stats
        q_stats = data_sub[['Q.Value', 'Protein.Q.Value', 'Global.Q.Value']].describe()
        print('\nAfter Q-value and proteotypicity filtering\n------')
        print(q_stats)

    # Check: how peptides map to proteins
    is_pep_multiprots = (
            data_sub.groupby([aggr_level, 'Run'], observed=True)
            ['Protein.Ids'].nunique() > 1
            )

    if is_pep_multiprots.any():
        raise ValueError(
            f'Peptides at aggregation level {aggr_level} map to multiple proteins. '
            'Not implemented yet.'
        )

    # Aggregate precursors
    data_cols = [
        'Run',
        aggr_level,
        'Protein.Ids',
        'Precursor.Quantity',
        ]

    precursor_data = data_sub[data_cols].copy()


    precursor_data_summed = (
        precursor_data.groupby([aggr_level,'Protein.Ids','Run'], observed=True) # In theory grouping by protein.ids not necessary
        ['Precursor.Quantity']
        .sum()
        .reset_index()
        )

    # Check: proteotypicity
    assert ((
        precursor_data_summed
        .groupby('Stripped.Sequence', observed=True)['Protein.Ids']
        .nunique().le(1).all()
        )), "Error: Some peptides map to multiple proteins!"

    X = pd.pivot(
        precursor_data_summed,
        index='Run',
        columns=aggr_level,
        values='Precursor.Quantity',
        )

    X = X.sort_index(axis=0).sort_index(axis=1)

    if fill_na:
        X.fillna(fill_na, inplace=True)

    X.columns.name = None
    X.index.name = None

    del precursor_data
    gc.collect()

    # obs
    obs = pd.DataFrame({'run_id': X.index}, index = X.index)
    obs.index.name = None

    meta_cols = [
        aggr_level,
        'Protein.Ids',
        'Protein.Group',
        'Genes',
        'Protein.Names',
        ]

    if aggr_level == 'Modified.Sequence':
        meta_cols.append('Stripped.Sequence')

    if aggr_level == 'Precursor.Id':
        meta_cols.extend(['Stripped.Sequence', 'Modified.Sequence', 'Precursor.Charge'])

    # todo: add column for if aggr_level is stripped sequence or modified id, which
    # collects the combinations of precursor_ids and modified sequences that were
    # combined in aggregation

    precursor_meta = data_sub[meta_cols].copy()

    # Groups contain identical rows
    assert (
        precursor_meta.groupby(aggr_level, observed=True)
        .apply(lambda x: x.nunique().eq(1).all(), include_groups=False)
        .all()
        )

    var = precursor_meta.groupby(aggr_level, observed=True).first()
    var = var.loc[X.columns]
    var[aggr_level] = var.index
    var['peptide_id'] = var.index
    var.index.name = None

    del precursor_meta
    del data_sub
    gc.collect()

    adata = ad.AnnData(
        X = X,
        var = var,
        obs = obs,
        )

    adata.strings_to_categoricals()

    if len(adata.obs_names.unique()) < adata.n_obs:
        adata.obs_names_make_unique()
        warnings.warn(
            'Repeated obs names were present in the data. '
            'They were made unique by numbered suffixes.'
            )
    
    if len(adata.var_names.unique()) < adata.n_vars:
        adata.var_names_make_unique()
        warnings.warn(
            'Repeated var names were present in the data. '
            'They were made unique by numbered suffixes.'
            )

    return adata
