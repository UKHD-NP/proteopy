from functools import partial
import warnings
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import anndata as ad
from pandas.api.types import is_string_dtype, is_categorical_dtype
from scipy import sparse
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Patch
import seaborn as sns
import matplotlib as mpl

from copro.utils.anndata import check_proteodata
from copro.utils.matplotlib import _resolve_color_scheme
from copro.utils.functools import partial_with_docsig


def completeness(
    adata: ad.AnnData,
    axis: int,
    layer: str | None = None,
    zero_to_na: bool = False,
    groups: Iterable[Any] | str | None = None,
    group_by: str | None = None,
    xlabel_rotation: float = 0.0,
    figsize: tuple[float, float] = (6.0, 5.0),
    show: bool = True,
    ax: bool = False,
    save: bool | str | Path | None = False,
) -> Axes | None:
    """
    Plot a histogram of completeness across observations or variables.

    Parameters
    ----------
    adata : AnnData
        :class:`~anndata.AnnData` object with proteomics annotations.
    axis
        `0` plots completeness per variable, `1` per observation.
    layer
        Name of the layer to use instead of `.X`. Defaults to `.X`.
    zero_to_na
        Treat zero entries as missing values when True.
    groups
        Optional iterable of group labels to include.
    group_by
        Column name in `.var` (axis 0) or `.obs` (axis 1) used to stratify
        completeness into groups. Triggers a boxplot when provided.
    xlabel_rotation
        Rotation angle in degrees applied to x-axis tick labels.
    figsize
        Tuple ``(width, height)`` controlling figure size in inches.
    show
        Display the plot with `plt.show()` when True.
    ax
        Return the Matplotlib Axes object instead of displaying the plot.
    save
        File path or truthy value to trigger `fig.savefig`.
    """
    check_proteodata(adata)

    if axis not in (0, 1):
        raise ValueError("`axis` must be either 0 (var) or 1 (obs).")

    if layer is None:
        matrix = adata.X
    else:
        if layer not in adata.layers:
            raise KeyError(f"Layer '{layer}' not found in adata.layers.")
        matrix = adata.layers[layer]

    if matrix is None:
        raise ValueError("Selected matrix is empty; cannot compute completeness.")

    n_obs, n_vars = matrix.shape
    axis_length = n_obs if axis == 0 else n_vars

    if axis_length == 0:
        raise ValueError("Cannot compute completeness on empty axis.")

    if sparse.issparse(matrix):
        matrix_coo = matrix.tocoo()
        data = matrix_coo.data
        rows = matrix_coo.row
        cols = matrix_coo.col

        if zero_to_na:
            valid_mask = (~np.isnan(data)) & (data != 0)
            if axis == 0:
                counts = np.bincount(
                    cols[valid_mask],
                    minlength=n_vars,
                )
            else:
                counts = np.bincount(
                    rows[valid_mask],
                    minlength=n_obs,
                )
        else:
            nan_mask = np.isnan(data)
            if axis == 0:
                nan_counts = np.bincount(
                    cols[nan_mask],
                    minlength=n_vars,
                )
                counts = n_obs - nan_counts
            else:
                nan_counts = np.bincount(
                    rows[nan_mask],
                    minlength=n_obs,
                )
                counts = n_vars - nan_counts
    else:
        values = np.asarray(matrix)
        valid_mask = ~np.isnan(values)
        if zero_to_na:
            valid_mask &= values != 0
        counts = valid_mask.sum(axis=axis)

    counts = np.asarray(counts, dtype=float)
    completeness = counts / axis_length

    if axis == 0:
        index = adata.var_names
        axis_labels = ("var", "obs")
        grouping_frame = adata.var
    else:
        index = adata.obs_names
        axis_labels = ("obs", "var")
        grouping_frame = adata.obs

    completeness_series = pd.Series(completeness, index=index)

    if group_by is None:
        fig, _ax = plt.subplots(figsize=figsize)
        sns.histplot(
            completeness_series,
            ax=_ax,
        )
        _ax.set_xlabel(
            f"Fraction of non-missing {axis_labels[1]} values per {axis_labels[0]}",
        )
        plt.setp(_ax.get_xticklabels(), rotation=xlabel_rotation)
    else:
        if group_by not in grouping_frame.columns:
            raise KeyError(
                f"Column '{group_by}' not found in "
                f"{'.var' if axis == 0 else '.obs'}",
            )
        group_series = grouping_frame[group_by].reindex(index, copy=False)
        plot_df = pd.DataFrame(
            {
                "completeness": completeness_series,
                group_by: group_series,
            },
            index=index,
        )
        plot_df = plot_df.dropna(subset=[group_by])

        if groups is not None:
            if isinstance(groups, str):
                groups = [groups]
            else:
                groups = list(groups)
            plot_df = plot_df[plot_df[group_by].isin(groups)]
            order = [grp for grp in groups if grp in plot_df[group_by].unique()]
        else:
            order = None

        if plot_df.empty:
            raise ValueError(
                "No data available for the requested grouping combination.",
            )

        if isinstance(plot_df[group_by].dtype, pd.CategoricalDtype):
            plot_df[group_by] = plot_df[group_by].cat.remove_unused_categories()
            if order is None:
                order = list(plot_df[group_by].cat.categories)

        fig, _ax = plt.subplots(figsize=figsize)
        sns.boxplot(
            data=plot_df,
            x=group_by,
            y="completeness",
            order=order,
            ax=_ax,
        )
        _ax.set_ylabel(
            f"Fraction of non-missing {axis_labels[1]} values per {axis_labels[0]}",
        )
        _ax.set_xlabel(group_by)
        plt.setp(_ax.get_xticklabels(), rotation=xlabel_rotation)

    if save:
        fig.savefig(save, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    if ax:
        return _ax
    if not save and not show and not ax:
        raise ValueError(
            "Args show, ax and save all set to False, function does nothing.",
        )

docstr_header="Plot a histogram of completeness per variable.\n"
completeness_per_var = partial_with_docsig(
    completeness,
    axis=0,
    docstr_header=docstr_header,
    )

docstr_header="Plot a histogram of completeness per observation.\n"
completeness_per_obs = partial_with_docsig(
    completeness,
    axis=1,
    docstr_header=docstr_header,
    )


def n_var_per_obs(
    adata,
    group_by=None,
    group_by_order=None,
    zero_to_na=False,
    ylabel='Nr. vars detected',
    xlabel_rotation=90,
    group_by_label_rotation=0,
    show=True,
    ax=False,
    save=False,
    color_scheme=None,
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
    if group_by not in obs_df.columns:
        obs_df[group_by] = 'all'
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
    colors = _resolve_color_scheme(color_scheme, unique_groups)
    plot_kwargs = {}

    if colors is not None:
        color_map = {str(grp): colors[i] for i, grp in enumerate(unique_groups)}
        plot_kwargs['color'] = counts[group_by].map(color_map).to_list()

    fig, _ax = plt.subplots(figsize=(6,4))
    counts.plot(kind='bar', x='obs', y='count', ax=_ax, legend=False, **plot_kwargs)

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

n_peptides_per_obs = partial(
    n_var_per_obs,
    ylabel='Nr. peptides detected',
)

n_proteins_per_obs = partial(
    n_var_per_obs,
    ylabel='Nr. proteins detected',
)


def n_obs_per_category(
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

# TODO: make a better wrapper
n_samples_per_category = n_obs_per_category


def n_cat1_per_cat2_hist(
    adata,
    category_col,
    entries_col = None,
    bin_width = None,
    bin_range = None,
    ):
    var = adata.var.copy()
    cats = [category_col]

    if entries_col:
        cats.append(entries_col)
    else:
        entries_col = 'index'
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
    n_cat1_per_cat2_hist,
    category_col = 'protein_id',
    bin_width = 5,
    )

n_proteoforms_per_gene = partial(
    n_cat1_per_cat2_hist,
    entries_col = 'proteoform_id',
    category_col = 'protein_id',
    )


def cv_by_category(
    adata,
    color_scheme=None,   
    figsize: tuple = (6, 4),
    alpha: float = 0.8,
    hline: float | None = None,
    show_points: bool = False,
    point_alpha: float = 0.7,
    point_size: float = 1,
    order: list | None = None,
    group_label_rotation: int | float = 0,
    show: bool = True,
    ax: bool = False,
    save: str | None = None,
):
    """
    Plot violin plots of all 'cv_' columns stored in adata.var,
    optionally showing all individual variable CVs as points.

    Compatible with Seaborn >= 0.14.
    """

    # --- find CV columns
    cv_cols = [c for c in adata.var.columns if c.startswith("cv_")]
    if not cv_cols:
        raise ValueError("No 'cv_' columns found in adata.var.")

    # --- reshape data
    df = adata.var[cv_cols].copy()
    df_melted = df.melt(var_name="Group", value_name="CV")
    df_melted["Group"] = df_melted["Group"].str.replace("^cv_", "", regex=True)

    # --- determine order
    if order is None:
        order = df_melted["Group"].unique().tolist()

    # --- build palette
    if color_scheme is None:
        palette = dict(zip(order, sns.color_palette("Set2", n_colors=len(order))))
    elif isinstance(color_scheme, (list, tuple)):
        palette = dict(zip(order, color_scheme))
    elif isinstance(color_scheme, dict):
        palette = color_scheme
    else:
        raise TypeError(
            "color_scheme must be None, list/tuple, or dict (e.g. adata.uns['colors_area_short'])."
        )

    # --- create figure
    fig, ax_plot = plt.subplots(figsize=figsize, dpi=150)

    # --- plot violins
    sns.violinplot(
        data=df_melted,
        x="Group",
        y="CV",
        hue="Group",
        order=order,
        palette=palette,
        cut=0,
        inner="box",
        alpha=alpha,
        legend=False,
        ax=ax_plot,
    )

    # --- optionally overlay points
    if show_points:
        sns.stripplot(
            data=df_melted,
            x="Group",
            y="CV",
            order=order,
            color="black",
            alpha=point_alpha,
            size=point_size,
            jitter=0.2,
            dodge=False,
            ax=ax_plot,
        )

    # --- optional horizontal dashed line
    if hline is not None:
        ax_plot.axhline(
            y=hline,
            color="black",
            linestyle="--",
            linewidth=1,
            alpha=0.8,
        )
        # add annotation for clarity
        ax_plot.text(
            x=-0.4,
            y=hline,
            s=f"{hline:.2f}",
            color="black",
            va="bottom",
            ha="left",
            fontsize=8,
        )

    # --- finalize axes
    ax_plot.set_xlabel("")
    ax_plot.set_ylabel("Coefficient of Variation (CV)")
    for label in ax_plot.get_xticklabels():  # ✅ safe label rotation
        label.set_rotation(group_label_rotation)
    ax_plot.set_title("Distribution of CV across groups")
    sns.despine()
    plt.tight_layout()

    # --- save figure if requested
    if save:
        fig.savefig(save, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save}")

    # --- show or return
    if show:
        plt.show()

    if ax:
        return ax_plot


def obs_correlation_matrix(
    adata,
    method: str = "pearson",          # "pearson" or "spearman"
    zero_to_na: bool = False,         # set zeros -> NaN before correlating
    group_by: str | None = None,       # obs column for sample colors
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

    # ---- optional row/col colors from obs[group_by]
    row_colors = None
    legend_handles = None
    if group_by is not None:
        groups = adata.obs.loc[corr_df.index, group_by]
        cats = pd.Categorical(groups).categories

        # normalize provided color scheme to dict keyed by string labels
        if color_scheme is None:
            base = sns.color_palette(n_colors=len(cats))
            palette = {str(cat): base[i] for i, cat in enumerate(cats)}
        elif isinstance(color_scheme, (list, tuple)):
            if len(color_scheme) < len(cats):
                raise ValueError(
                    f"color_scheme has {len(color_scheme)} colors but {len(cats)} groups in '{group_by}'"
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

    # ---- add legend for group_by colors
    if legend_handles is not None:
        g.ax_heatmap.legend(
            handles=legend_handles,
            title=group_by,
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
