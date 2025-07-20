import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import anndata as ad


def plot_proteoform_scores(
    adata,
    adj=True,
    pval_threshold=None,
    score_threshold=None,
    ax=False,
    ):

    if adj:
        pval_col = 'proteoform_score_pval_adj'
        ylabel = '-log10(adj. p-value)'
    else:
        pval_col = 'proteoform_score_pval'
        ylabel = '-log10(p-value)'

    var = adata.var[['proteoform_score', pval_col]].copy()
    var = var.drop_duplicates() # pval and proteoform_score are repeated 
                             # for peptides of same protein

    mask_nonan = var[pval_col].notna()
    df = var.loc[mask_nonan,['proteoform_score', pval_col]]
    df['neg_log10_pval'] = -np.log10(df[pval_col].replace(0, np.nan))

    # Mask for pval and score thresholds
    if pval_threshold and score_threshold:
        mask = (
            (df['proteoform_score'] > score_threshold) &
            (df['neg_log10_pval'] > -np.log10(pval_threshold))
            )
    elif score_threshold:
        mask = df['proteoform_score'] > score_threshold
    elif pval_threshold:
        mask = df['neg_log10_pval'] > -np.log10(pval_threshold)
    else:
        mask = pd.Series(False, index=df.index)

    df['is_above_threshold'] = mask

    # Rel plot
    g = sns.relplot(
        data=df,
        x='proteoform_score',
        y='neg_log10_pval',
        hue='is_above_threshold',
        palette={
            np.True_: '#008A1D',  # green
            np.False_: '#BDBDBD'   # grey
        },
        alpha=0.5,
        edgecolor=None,
        aspect=1.2,
        s=30,
        legend=False,
    )
    ax = g.ax

    # Add threshold lines
    if pval_threshold:
        ax.axhline(
            y=-np.log10(pval_threshold),
            color='#A2A2A2',    # grey
            linestyle='--',
            label=pval_threshold)

    if score_threshold:
        ax.axvline(
            x=score_threshold,
            color='#A2A2A2',    # grey
            linestyle='--',
            label=score_threshold)

    ax.set_xlabel('Proteoform Score')
    ax.set_ylabel(ylabel)
    g.tight_layout()

    if ax:
        return ax
    else:
        plt.show()
        return


def plot_proteoforms(
    adata,
    protein_ids=None,
    groupby=None,
    groupby_cat_order=None,
    rotation=0,
    ax=False
    ):

    if isinstance(protein_ids, str):
        protein_ids = [protein_ids]

    var = adata.var[['protein_id', 'cluster_id']].copy()
    var = var.reset_index().rename(columns={'index': 'var_index'}) # peptide_ids

    obs = adata.obs[[groupby]].copy()
    obs = obs.reset_index().rename(columns={'index': 'obs_index'}) # samples
    obs[groupby] = obs[groupby].astype('category')

    df = adata.to_df().copy()
    df = df.reset_index().rename(columns={'index': 'obs_index'}) # samples

    # Construct df: cols == [
    #   'obs_index',
    #   'var_index'==peptides,
    #   'intensity',
    #   'protein_id',
    #   'cluster_id',
    #   groupby]

    df = df.melt(id_vars='obs_index', var_name='var_index', value_name='intensity')
    df = pd.merge(df, var, on='var_index', how='left')
    df = pd.merge(df, obs, on='obs_index', how='left')

    # Explicitly order the x axis observations to group by groupby
    groupby_series = obs[groupby]
    cat_index_map = {cat: sorted(obs[obs[groupby] == cat]['obs_index'].to_list())
        for cat in obs[groupby].cat.categories}

    if groupby_cat_order:
        obs_index_ordered = [idx
            for cat in groupby_cat_order for idx in cat_index_map[cat]]
    else:
        obs_index_ordered = [idx
            for idx_list in cat_index_map.values() for idx in idx_list]

    df['obs_index'] = pd.Categorical(
        df['obs_index'],
        categories=obs_index_ordered,
        ordered=True)

    axes = []

    for prot_id in protein_ids:
        sub_df = df[df['protein_id'] == prot_id].copy()

        fig, _ax = plt.subplots(figsize=[15, 6])

        sns.lineplot(
            data=sub_df,
            x='obs_index',
            y='intensity',
            units='var_index',
            estimator=None,
            hue='cluster_id',
            ax=_ax
        )

        # Add group separator lines
        obs_idxpos_map = {obs: i for i, obs in enumerate(obs_index_ordered)}
        cats_ordered = groupby_cat_order if groupby_cat_order else list(cat_index_map.keys())
        for cat in cats_ordered[:-1]:
            last_obs_in_cat = cat_index_map[cat][-1]

            _ax.axvline(
                x=obs_idxpos_map[last_obs_in_cat] + 0.5,
                ymin=0.02,
                ymax=0.95,
                color='#D8D8D8',
                linestyle='--'
                )
        
        # Add group labels above each group section
        for cat in cats_ordered:
            group_obs = cat_index_map[cat]

            if not group_obs:
                continue

            start = obs_idxpos_map[group_obs[0]]
            end = obs_idxpos_map[group_obs[-1]]
            mid = (start + end) / 2

            _ax.text(
                x=mid,
                y=sub_df['intensity'].max() * 1.05,
                s=cat,
                ha='center',
                va='bottom',
                fontsize=12,
                fontweight='bold',
                )

        _ax.tick_params(axis='x', rotation=rotation)
        plt.tight_layout()

        if ax:
            axes.append(_ax)
        else:
            plt.show()
            plt.close(fig)

    if ax:
        return axes[0] if len(axes) == 1 else axes
