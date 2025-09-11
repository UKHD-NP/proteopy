from functools import partial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def var_completeness(
    adata,
    zero_to_na = False,
    show = True,
    ax = False,
    save = False,
    ):
    vals = adata.to_df().copy()

    if zero_to_na:
        vals = vals.replace(0, np.nan)

    completeness = vals.count() / adata.n_obs

    fig, _ax = plt.subplots(figsize=(6,5))

    sns.histplot(
        completeness,
        ax=_ax
        )

    if save:
        fig.savefig(save, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    if ax:
        return _ax
    if not save and not show and not ax:
        raise ValueError((
            'Args show, ax and save all set to False, function does nothing.'
            ))


def n_detected_var(
    adata,
    group_by=None,
    group_by_order=None,
    zero_to_na=False,
    ylabel='Nr. vars detected',
    xlabel_rotation=90,
    group_by_label_rotation=0,
    show=True,
    ax=False,
    save=False
    ):

    df = pd.melt(
        adata.to_df().reset_index(names='obs'),
        id_vars='obs',
        var_name='var',
        value_name='intensity'
    )

    if zero_to_na:
        df.loc[df['intensity'] == 0, 'intensity'] = np.nan
    df = df[~df['intensity'].isna()]

    if group_by and group_by != 'obs':
        obs = adata.obs[[group_by]].reset_index(names='obs')
        df = pd.merge(df, obs, on='obs')
    if group_by == 'obs':
        df[group_by] = df['obs']
    if not group_by:
        group_by = 'all'
        df[group_by] = 'all'

    counts = df.groupby(['obs', group_by], observed=True).size().reset_index(name='count')

    obs_df = adata.obs.reset_index().rename(columns={'index': 'obs'})
    if group_by in obs_df.columns and isinstance(obs_df[group_by].dtype, pd.CategoricalDtype):
        obs_df[group_by] = obs_df[group_by].astype('category')

    if group_by_order:
        cat_index_map = {cat: sorted(obs_df[obs_df[group_by] == cat]['obs'].to_list())
                         for cat in group_by_order}
    else:
        cat_index_map = {cat: sorted(obs_df[obs_df[group_by] == cat]['obs'].to_list())
                         for cat in obs_df[group_by].unique()}

    x_ordered = [obs for cat in cat_index_map.values() for obs in cat]
    counts['obs'] = pd.Categorical(counts['obs'], categories=x_ordered, ordered=True)
    counts = counts.sort_values('obs')

    counts[group_by] = counts[group_by].astype(str)

    unique_groups = list(cat_index_map.keys())
    palette = sns.color_palette("Set2", n_colors=len(unique_groups))
    color_map = {str(grp): palette[i] for i, grp in enumerate(unique_groups)}
    bar_colors = counts[group_by].map(color_map)

    fig, _ax = plt.subplots(figsize=(6,4))
    counts.plot(kind='bar', x='obs', y='count', ax=_ax, color=bar_colors, legend=False)

    plt.setp(_ax.get_xticklabels(), rotation=xlabel_rotation, ha='right')
    _ax.set_xlabel('')
    _ax.set_ylabel(ylabel)

    obs_idx_map = {obs: i for i, obs in enumerate(x_ordered)}
    ymax = counts['count'].max()
    for cat, obs_list in cat_index_map.items():
        if not obs_list:
            continue
        start_idx = obs_idx_map[obs_list[0]]
        end_idx = obs_idx_map[obs_list[-1]]
        mid_idx = (start_idx + end_idx) / 2

        _ax.text(
            x=mid_idx,
            y=ymax * 1.05,
            s=cat,
            ha='center',
            va='bottom',
            fontsize=8,
            fontweight='bold',
            rotation=group_by_label_rotation
        )

    plt.tight_layout()

    if save:
        fig.savefig(save, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    if ax:
        return _ax

n_detected_peptides_per_sample = partial(
    n_detected_var,
    ylabel='Nr. peptides detected',
)

def n_elements_per_category(
    adata,
    category_col,
    elements_col = None,
    bin_width = None,
    bin_range = None,
    ):
    var = adata.var.copy()
    cats = [category_col]

    if elements_col:
        cats.append(elements_col)
    else:
        elements_col = 'index'
        var = var.reset_index()
        cats.append('index')

    var = var.drop_duplicates(subset=cats, keep='first')
    counts = var.groupby(category_col, observed=False).size()

    sns.histplot(
        counts,
        binwidth=bin_width if bin_width else None,
        binrange=bin_range if bin_range else None,
        )

    plt.show()

n_peptides_per_gene = partial(
    n_elements_per_category,
    category_col = 'protein_id',
    bin_width = 5,
    )

n_proteoforms_per_gene = partial(
    n_elements_per_category,
    elements_col = 'proteoform_id',
    category_col = 'protein_id',
    )
