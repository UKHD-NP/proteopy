import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import anndata as ad

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
