import numpy as np
import pandas as pd
from pandas.api.types import is_categorical_dtype
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import anndata as ad

def peptide_intensities(
    adata,
    protein_ids=None,
    protein_id_key='protein_id', # Implement
    group_by=None,
    group_by_order=None,
    color=None,
    rotation=0,
    group_by_label_rotation=0,
    figsize=(15,6),
    show=True,
    save=None,
    ax=False
    ):

    # Check input
    if isinstance(protein_ids, str):
        protein_ids = [protein_ids]

    assert protein_id_key in adata.var.columns

    # Format input
    var = adata.var[protein_id_key].copy()
    var = var.reset_index().rename(columns={'index': 'var_index'})

    obs = adata.obs[[group_by]].copy()
    obs = obs.reset_index().rename(columns={'index': 'obs_index'})

    if is_categorical_dtype(obs[group_by]):
        obs[group_by] = obs[group_by].astype('category')

    X = adata.to_df().copy()
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

        sns.lineplot(
            data=sub_df,
            x='obs_index',
            y='intensity',
            units='var_index',
            estimator=None,
            hue=color if color else 'var_index',
            ax=_ax,
        )

        _ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')

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
                rotation=group_by_label_rotation,
                )

        _ax.tick_params(axis='x', rotation=rotation)
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

            plt.close()

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

