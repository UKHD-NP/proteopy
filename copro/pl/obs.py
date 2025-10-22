import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage
from pandas.api.types import is_string_dtype, is_categorical_dtype

from .utils import _resolve_color_scheme

def n_samples_by_category(
    adata,
    category_cols,
    ignore_na=False,
    sort_by_counts=True,
    x_label_rotation=45,
    show=True,
    save=False,
    ax=False,
    color_scheme=None,
    ):
    if isinstance(category_cols, str):
        category_cols = [category_cols]

    first_cat_col = category_cols[0]

    obs = adata.obs[category_cols].copy()

    for col in category_cols:
        if not (is_string_dtype(obs[col]) or is_categorical_dtype(obs[col])):
            obs[col] = obs[col].astype(str)

    cats_order = (
        obs[first_cat_col].cat.categories.tolist()
        if is_categorical_dtype(obs[first_cat_col])
        else obs[first_cat_col].unique().tolist()
        )

    for n_col, col in enumerate(category_cols):
        if not ignore_na and obs[col].isna().any():
            obs[col] = obs[col].cat.add_categories(['missing'])
            obs[col] = obs[col].fillna('missing')


    # Plot
    if len(category_cols) == 1:
        freq = obs[first_cat_col].value_counts()

        if not sort_by_counts:
            freq = freq[cats_order]

        colors = _resolve_color_scheme(color_scheme, freq.index)
        _ax = freq.plot(kind='bar', color=colors)

    elif len(category_cols) == 2:
        df = obs.groupby(category_cols, observed=False).size().unstack(fill_value=0)

        if sort_by_counts:
            new_order = df.sum(axis=1).sort_values(ascending=False).index.tolist()
            df = df.loc[new_order]

        colors = _resolve_color_scheme(color_scheme, df.columns)
        _ax = df.plot(kind='bar', stacked=True, color=colors)
        _ax.legend(loc='center right', bbox_to_anchor=(2,0.5))
        
    else:
        print('nr of categories > 2 not implemented yet.')

    _ax.set_xlabel(first_cat_col)
    _ax.set_ylabel('#')

    ha = (
        'right' if x_label_rotation > 0
        else 'left' if x_label_rotation < 0
        else 'center'
        )
    plt.setp(_ax.get_xticklabels(), rotation=x_label_rotation, ha=ha)

    if show:
        plt.show()
    if save:
        _ax.figure.savefig(save, dpi=300, bbox_inches='tight')
    if ax:
        return _ax
    if not show and not save and not ax:
        warnings.warn((
            'Function does not do anything. Set at least one argument to True:'
            ' show, save, ax'
            ))

def obs_correlation_matrix(
    adata,
    method: str = "pearson",          # "pearson" or "spearman"
    zero_to_na: bool = False,         # set zeros -> NaN before correlating
    groupby: str | None = None,       # obs column for sample colors
    color_scheme=None,                # list/tuple/dict (e.g. adata.uns['colors_area_short'])
    figsize=(9, 7),
    cmap: str = "coolwarm",
    linkage_method: str = "average",
    xticklabels: bool = False,
    yticklabels: bool = False,
    show: bool = True,
    ax: bool = False,
    save: str | None = None,
):
    """
    Compute obs×obs correlations from adata.X and plot a clustered heatmap,
    with the colormap centered at the off-diagonal mean correlation.
    """
    # ---- values from adata.X (obs × var)
    vals = adata.to_df()  # ALWAYS uses adata.X
    if zero_to_na:
        vals = vals.replace(0, np.nan)

    # ---- obs×obs correlation (pairwise complete)
    corr_df = vals.T.corr(method=method)  # (obs × obs)
    corr_df.index = adata.obs_names
    corr_df.columns = adata.obs_names

    # ---- compute off-diagonal mean for color center
    A = corr_df.values.astype(float, copy=False)
    n = A.shape[0]
    if n > 1:
        offdiag = A[~np.eye(n, dtype=bool)]
        center_val = np.nanmean(offdiag)
    else:
        center_val = float(np.nanmean(A))  # degenerate case

    # ---- optional row/col colors from obs[groupby]
    row_colors = None
    legend_handles = None
    if groupby is not None:
        groups = adata.obs.loc[corr_df.index, groupby]
        cats = pd.Categorical(groups).categories

        # normalize provided color scheme to dict keyed by string labels
        if color_scheme is None:
            base = sns.color_palette(n_colors=len(cats))
            palette = {str(cat): base[i] for i, cat in enumerate(cats)}
        elif isinstance(color_scheme, (list, tuple)):
            if len(color_scheme) < len(cats):
                raise ValueError(
                    f"color_scheme has {len(color_scheme)} colors but {len(cats)} groups in '{groupby}'"
                )
            palette = {str(cat): color_scheme[i] for i, cat in enumerate(cats)}
        elif isinstance(color_scheme, dict):
            palette = {str(k): v for k, v in color_scheme.items()}
        else:
            raise TypeError("color_scheme must be None, list/tuple, or dict.")

        groups_str = groups.astype(str)
        row_colors = groups_str.map(palette).values
        # build legend handles
        legend_handles = [
            #plt.Line2D([0], [0], marker='o', color='none',
            #           markerfacecolor=palette[str(cat)], markersize=6, label=str(cat))
            Patch(facecolor=palette[str(cat)], edgecolor='none', label=str(cat))
            for cat in cats
        ]

    # ---- hierarchical clustering on (1 - r)
    dist = 1 - corr_df.values
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0, 2)  # numerical guard
    Z = linkage(squareform(dist, checks=False), method=linkage_method)

    # ---- clustermap (center at off-diagonal mean)
    g = sns.clustermap(
        corr_df,
        row_linkage=Z,
        col_linkage=Z,
        row_colors=row_colors,
        col_colors=row_colors if row_colors is not None else None,
        cmap=cmap,
        center=center_val,          
        figsize=figsize,
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        cbar_kws={"label": f"{method.capitalize()}"},
    )

    # ---- add legend for groupby colors
    if legend_handles is not None:
        g.ax_heatmap.legend(
            handles=legend_handles,
            title=groupby,
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            borderaxespad=0.,
            frameon=False,
        )

    g.ax_heatmap.set_xlabel("Samples")
    g.ax_heatmap.set_ylabel("Samples")

    plt.tight_layout()

    if show:
        plt.show()

    if ax:
        return g.ax_heatmap

    if save:
        g.savefig(save, dpi=300, bbox_inches="tight")