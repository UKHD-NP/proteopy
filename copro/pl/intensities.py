import warnings
from functools import partial
import numpy as np
import pandas as pd
from pandas.api.types import is_categorical_dtype
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import anndata as ad

from .utils import _resolve_color_scheme

def peptide_intensities(
    adata,
    protein_ids=None,
    protein_id_key='protein_id', # Implement
    group_by=None,
    group_by_order=None,
    color=None,
    log_transform=None,
    z_transform=False,
    show_zeros=True,
    xlab_rotation=0,
    group_by_label_rotation=0,
    figsize=(15,6),
    show=True,
    save=None,
    ax=False,
    color_scheme=None,
    ):
    '''
    Args:
        log_transform (float, optional): Base for log transformation of the data. 1 will
            be added to each value before transformation.
        z_transform (float, optional): Transform values to have 0-mean and 1-variance
            along the peptide axis. Always uses zeros instead of NaNs if present, even
            if show_zeros=False.
        show_zeros (bool, optional): Don't display zeros if False.
    Returns:
    '''

    # Check input
    if isinstance(protein_ids, str):
        protein_ids = [protein_ids]

    assert protein_id_key in adata.var.columns

    # Format input
    var_cols = [protein_id_key]

    if color:
        var_cols.append(color)

    var = adata.var[var_cols].copy()
    var = var.reset_index().rename(columns={'index': 'var_index'})

    obs = adata.obs[[group_by]].copy()
    obs = obs.reset_index().rename(columns={'index': 'obs_index'})

    if is_categorical_dtype(obs[group_by]):
        obs[group_by] = obs[group_by].astype('category')

    X = adata.to_df().copy()

    X_zeros = X  # backup
    if not show_zeros:
        X = X.replace(0, np.nan)

    if log_transform:
        X = X.apply(lambda x: np.log(x+1) / np.log(log_transform))

    if z_transform:
        arr_zeros = X_zeros.to_numpy()
        arr_z = (
            (arr_zeros - np.mean(arr_zeros, axis=0, keepdims=True)) / 
            np.std(arr_zeros, axis=0, keepdims=True)
            )

        if not show_zeros:
            arr_z = np.where((arr_zeros == 0) & (arr_z != 0), np.nan, arr_z)
                         
        X = pd.DataFrame(arr_z, columns=X.columns, index=X.index)

    X = X.reset_index().rename(columns={'index': 'obs_index'})

    df = X.melt(id_vars='obs_index', var_name='var_index', value_name='intensity')
    df = pd.merge(df, var, on='var_index', how='left')
    df = pd.merge(df, obs, on='obs_index', how='left')

    # Explicitly order the x axis observations to group by group_by
    cat_index_map = {
        cat: sorted(obs[obs[group_by] == cat]['obs_index'].to_list())
        for cat in obs[group_by].cat.categories
        }

    if group_by_order:
        obs_index_ordered = [
            idx
            for cat in group_by_order
            for idx in cat_index_map[cat]
            ]

    else:
        obs_index_ordered = [
            idx
            for idx_list in cat_index_map.values()
            for idx in idx_list
            ]

    df['obs_index'] = pd.Categorical(
        df['obs_index'],
        categories=obs_index_ordered,
        ordered=True)

    axes = []

    if save and len(protein_ids) > 1:
        save_path = save if save.endswith('.pdf') else f'{save}.pdf'
        pdf_pages = PdfPages(save_path)

    for prot_id in protein_ids:
        sub_df = df[df[protein_id_key] == prot_id]
        fig, _ax = plt.subplots(figsize=figsize)

        if sub_df.empty:
            warnings.warn(f'No data found for protein: {prot_id}')
            _ax.text(
                0.5,
                0.5,
                f'No data found for protein: {prot_id}',
                ha='center', va='center', transform=_ax.transAxes,
                fontsize=14,
                color='gray'
                )
            _ax.set_xlim(0, 1)
            _ax.set_ylim(0, 1)
            _ax.set_xticks([])
            _ax.set_yticks([])

        else:

            #sub_df = sub_df.sort_values(by=group_by)

            if color:
                sns.lineplot(
                    data=sub_df,
                    x='obs_index',
                    y='intensity',
                    hue=color,
                    style=group_by,
                    units='var_index',
                    estimator=None,
                    errorbar=None,
                    marker='o',
                    dashes=False,
                    palette='Set2',
                    legend='brief',
                    ax=_ax,
                    )
            else:
                sns.lineplot(
                    data=sub_df,
                    x='obs_index',
                    y='intensity',
                    hue='var_index',
                    style=group_by,
                    marker='o',
                    dashes=False,
                    palette='Set2',
                    legend='brief',
                    ax=_ax
                    )

            # Legend
            handles, labels = _ax.get_legend_handles_labels()

            # Determine which labels correspond to the hue only (ignore style)
            if color:
                hue_values = sub_df[color].unique().astype(str)
            else:
                hue_values = sub_df['var_index'].unique().astype(str)

            # Keep only legend entries whose label matches a hue value
            new_handles_labels = [(h, l) for h, l in zip(handles, labels) if l in hue_values]

            if new_handles_labels:
                handles, labels = zip(*new_handles_labels)  # unzip back into separate lists
                _ax.legend(
                    handles,
                    labels,
                    bbox_to_anchor=(1.01, 1),
                    loc='upper left',
                    title=color if color else 'Peptide',
                    )

            # Add group separator lines
            obs_idxpos_map = {obs: i for i, obs in enumerate(obs_index_ordered)}
            cats_ordered = group_by_order if group_by_order else list(cat_index_map.keys())

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

                # Determine x-axis group regions
                start = obs_idxpos_map[group_obs[0]]
                end = obs_idxpos_map[group_obs[-1]]
                mid = (start + end) / 2

                rot = group_by_label_rotation if group_by_label_rotation else 0
                ha_for_rot = 'center' if (rot % 360 == 0) else 'left'

                # Determine padded y-axis limits
                ymax = sub_df['intensity'].max()
                ymin = sub_df['intensity'].min()
                ypad_top = (ymax - ymin) * 0.15
                ypad_bottom = (ymax - ymin) * 0.10
                _ax.set_ylim(ymin - ypad_bottom, ymax + ypad_top)

                _ax.text(
                    x=mid,
                    y=ymax + ypad_top * 0.4,
                    s=cat,
                    ha=ha_for_rot,
                    va='bottom',
                    fontsize=12,
                    fontweight='bold',
                    rotation=rot,
                    rotation_mode='anchor',
                    )
 
        plt.xticks(rotation=xlab_rotation, ha='right')
        _ax.set_title(prot_id)
        _ax.set_xlabel('Sample')
        _ax.set_ylabel('Intensity')

        plt.tight_layout()

        if ax:
            axes.append(_ax)

            if show:
                plt.show()


        elif save:

            if len(protein_ids) == 1:
                fig.savefig(save, bbox_inches='tight', dpi=300)

            else:
                pdf_pages.savefig(fig, bbox_inches='tight')

            if show:
                plt.show()

            plt.close(fig)

        elif show:
            plt.show()
            plt.close(fig)

        else:
            print("Warning: Plot created but not displayed, saved, or returned")
            plt.close(fig)

    if save and len(protein_ids) > 1:
        pdf_pages.close()

    if ax:
        return axes[0] if len(axes) == 1 else axes


proteoform_intensities = partial(
    peptide_intensities,
    color = 'proteoform_id',
    )


def intensity_distribution_per_obs(
    adata,
    group_by=None,
    group_by_order=None,
    zero_to_na=False,
    log_transform = False,
    z_transform = False,
    ylabel='Intensity',
    xlabel_rotation=90,
    group_by_label_rotation=0,
    show=True,
    ax=False,
    save=False,
    figsize=(8,5),
    color_scheme=None,
    ):
    
    df = adata.to_df()

    if zero_to_na:
        df = df.replace(0, np.nan)

    if log_transform:
        df = df.apply(lambda x: np.log(x+1) / np.log(log_transform))

    if z_transform:
        df = df.apply(lambda row: (row - row.mean(skipna=True)) / row.std(skipna=True), axis=1)

    df = pd.melt(
        df.reset_index(names='obs'),
        id_vars='obs',
        var_name='var',
        value_name='intensity'
    )
    df = df[~df['intensity'].isna()]

    # Merge group info
    if group_by and group_by != 'obs':
        obs = adata.obs[[group_by]].reset_index(names='obs')
        df = pd.merge(df, obs, on='obs')
    if group_by == 'obs':
        df[group_by] = df['obs']
    if not group_by:
        group_by = 'all'
        df[group_by] = 'all'

    # Determine x-axis order
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
    df['obs'] = pd.Categorical(df['obs'], categories=x_ordered, ordered=True)

    # Assign colors per group
    df[group_by] = df[group_by].astype(str)
    unique_groups = list(cat_index_map.keys())
    color_scheme
    colors = _resolve_color_scheme(color_scheme, unique_groups)
    color_map = {str(grp): colors[i] for i, grp in enumerate(unique_groups)}

    sample_palette = {obs: color_map[df.loc[df['obs'] == obs, group_by].iloc[0]] for obs in x_ordered}

    fig, _ax = plt.subplots(figsize=figsize)
    sns.boxplot(
        data=df,
        x='obs',
        hue='obs',
        y='intensity',
        palette=sample_palette,
        ax=_ax
    )

    plt.setp(_ax.get_xticklabels(), rotation=xlabel_rotation, ha='right')
    _ax.set_xlabel('')
    _ax.set_ylabel(ylabel)

    # Add group labels above groups
    obs_idx_map = {obs: i for i, obs in enumerate(x_ordered)}
    ymax = df['intensity'].max()
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
            fontsize=12,
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
