import warnings
from functools import partial
from typing import Any, Sequence
from collections.abc import Sequence as SequenceABC

import numpy as np
import pandas as pd
from pandas.api.types import is_categorical_dtype
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
import seaborn as sns
import matplotlib as mpl
import anndata as ad
import math
import os
from scipy import sparse

from copro.utils.anndata import check_proteodata
from copro.utils.matplotlib import _resolve_color_scheme
from copro.utils.array import is_log_transformed
from copro.utils.functools import partial_with_docsig


def peptide_intensities(
    adata: ad.AnnData,
    protein_ids: str | Sequence[str] | None = None,
    order_by: str | None = None,
    order: Sequence[str] | None = None,
    groups: str | Sequence[str] | None = None,
    color: str | None = None,
    group_by: str | None = None,
    log_transform: float | None = None,
    fill_na: float | None = None,
    z_transform: bool = False,
    show_zeros: bool = True,
    xlab_rotation: float = 0,
    order_by_label_rotation: float = 0,
    figsize: tuple[float, float] = (15, 6),
    show: bool = True,
    save: str | os.PathLike[str] | None = None,
    ax: bool = False,
    color_scheme: Any = None,
) -> Axes | list[Axes] | None:
    """
    Plot peptide intensities across samples for the requested proteins.

    Parameters
    ----------
    adata : AnnData
        Proteomics :class:`~anndata.AnnData`.
    protein_ids : str | Sequence[str]
        Show peptides mapping to this protein_id.
    order_by : str, optional
        Column in ``adata.obs`` used to group and order observations on the x-axis.
        When ``None``, observations follow ``adata.obs_names``.
    order : Sequence[str], optional
        Explicit order of groups (when ``order_by`` is set) or observations
        (when ``order_by`` is ``None``).
    groups : str | Sequence[str], optional
        Restrict ``order_by`` to selected categorical levels (requires
        ``order_by``). The provided order determines the plotting order unless
        ``order`` is supplied.
    color : str, optional
        ``adata.var`` column used for per-peptide coloring.
    group_by : str, optional
        ``adata.var`` column whose categories are aggregated into a single line.
        Mutually exclusive with ``color``; each group is colored via
        ``color_scheme``.
    log_transform : float, optional
        Logarithm base (>0 and !=1). Values are transformed as
        ``log(value + 1, base)``.
    fill_na : float, optional
        Replace missing intensities before zero/log/z transformations when set.
    z_transform : bool, optional
        Standardize each peptide across observations after optional log transform.
        Skips NA.
    show_zeros : bool, optional
        Display zero intensities when ``True``; otherwise zeros become ``NaN``.
    xlab_rotation : float, optional
        Rotation angle (degrees) applied to x-axis tick labels.
    order_by_label_rotation : float, optional
        Rotation angle for the group labels drawn above grouped sections.
    figsize : tuple[float, float], optional
        Size of each generated figure passed to :func:`matplotlib.pyplot.subplots`.
    color_scheme : Any, optional
        Palette specification forwarded to
        :func:`copro.utils.matplotlib._resolve_color_scheme` for either the
        per-peptide ``color`` or aggregated ``group_by`` categories.
    show : bool, optional
        Display the generated figure(s) with :func:`matplotlib.pyplot.show`.
    save : str | os.PathLike, optional
        Path for saving the figure(s). Multi-protein selections are written to a
        PDF stack.
    ax : bool, optional
        When ``True``, return the underlying Axes objects instead of closing them.

    Returns
    -------
    Axes | list[Axes] | None
        Axes handle(s) when ``ax`` is ``True``; otherwise ``None``.
    """

    # Check input
    check_proteodata(adata)

    if protein_ids is None:
        raise ValueError(
            "peptide_intensities requires at least one protein_id; "
            "pass a string or an iterable of IDs."
        )

    if isinstance(protein_ids, str):
        protein_ids = [protein_ids]

    if not protein_ids:
        raise ValueError("protein_ids cannot be empty.")

    if color and group_by:
        raise ValueError("`color` and `group_by` are mutually exclusive.")

    if groups is not None and order_by is None:
        raise ValueError("`groups` can only be used when `order_by` is provided.")

    if groups is None:
        group_levels = None
    elif isinstance(groups, str):
        group_levels = [groups]
    elif isinstance(groups, SequenceABC):
        group_levels = list(groups)
    else:
        raise TypeError("`groups` must be a string or a sequence of strings.")

    if group_levels is not None:
        if not group_levels:
            raise ValueError("`groups` cannot be empty.")
        seen_groups: set[Any] = set()
        deduped_groups: list[Any] = []
        for grp in group_levels:
            if grp in seen_groups:
                continue
            seen_groups.add(grp)
            deduped_groups.append(grp)
        group_levels = deduped_groups

    # Format input
    if log_transform is not None:
        if log_transform <= 0:
            raise ValueError("log_transform must be positive.")
        if log_transform == 1:
            raise ValueError("log_transform cannot be 1.")
        log_base = float(log_transform)
    else:
        log_base = None

    var_cols = ['protein_id']

    if color:
        if color not in adata.var.columns:
            raise KeyError(
                f"Column '{color}' is not present in adata.var; "
                "peptide coloring must use a .var annotation."
            )
        var_cols.append(color)
    if group_by:
        if group_by not in adata.var.columns:
            raise KeyError(
                f"Column '{group_by}' is not present in adata.var; "
                "grouping requires a .var annotation."
            )
        if group_by not in var_cols:
            var_cols.append(group_by)

    var = adata.var[var_cols].copy()
    var = var.reset_index(names='var_index')
    var = var[var['protein_id'].isin(protein_ids)]
    if color and color in var and is_categorical_dtype(var[color]):
        var[color] = var[color].cat.remove_unused_categories()
    if group_by and group_by in var and is_categorical_dtype(var[group_by]):
        var[group_by] = var[group_by].cat.remove_unused_categories()

    selected_vars = var['var_index'].tolist()
    palette_map = None

    if group_by:
        hue_labels = (
            pd.Series(pd.unique(var[group_by]))
            .dropna()
            .tolist()
        )
    elif color:
        hue_labels = pd.Series(pd.unique(var[color])).dropna().tolist()
    else:
        hue_labels = pd.Series(pd.unique(var['var_index'])).dropna().tolist()

    if hue_labels:
        palette_values = _resolve_color_scheme(color_scheme, hue_labels)
        if palette_values:
            palette_map = dict(zip(hue_labels, palette_values))

    obs = adata.obs.reset_index(names='obs_index')

    if order_by:
        if order_by not in obs.columns:
            raise KeyError(f"'{order_by}' is not present in adata.obs")

        if not is_categorical_dtype(obs[order_by]):
            obs[order_by] = obs[order_by].astype('category')

        obs = obs[['obs_index', order_by]]

        if group_levels is not None:
            available_groups = set(obs[order_by].dropna().unique())
            missing_groups = [grp for grp in group_levels if grp not in available_groups]
            if missing_groups:
                raise ValueError(
                    "Items in 'groups' are not present in the selected "
                    f"'{order_by}' categories: {sorted(missing_groups)}"
                )
            obs = obs[obs[order_by].isin(group_levels)].copy()
            if obs.empty:
                raise ValueError(
                    "No observations remain after filtering with `groups`."
                )
            if is_categorical_dtype(obs[order_by]):
                obs[order_by] = obs[order_by].cat.remove_unused_categories()
    else:
        obs = obs[['obs_index']]

    if selected_vars:
        adata_subset = adata[:, selected_vars]
        X_matrix = adata_subset.X
        was_sparse = sparse.issparse(X_matrix)
        if was_sparse:
            data_matrix = X_matrix.toarray()
        else:
            data_matrix = np.asarray(X_matrix)
        data_matrix = np.array(data_matrix, dtype=float, copy=True)
        var_names = list(adata_subset.var_names)
    else:
        data_matrix = np.empty((adata.n_obs, 0), dtype=float)
        var_names = []

    if fill_na is not None:
        if not np.isfinite(fill_na):
            raise ValueError("fill_na must be a finite float.")
        data_matrix = data_matrix.copy()
        data_matrix[np.isnan(data_matrix)] = float(fill_na)

    zero_mask = data_matrix == 0
    X_processed = data_matrix.copy()

    if log_base is not None:
        with np.errstate(divide='ignore', invalid='ignore'):
            X_processed = np.log1p(X_processed) / np.log(log_base)

    if z_transform and selected_vars:
        with np.errstate(divide='ignore', invalid='ignore'):
            arr_mean = np.nanmean(X_processed, axis=0, keepdims=True)
            arr_std = np.nanstd(X_processed, axis=0, keepdims=True)
        arr_std[arr_std == 0] = 1.0
        X_processed = (X_processed - arr_mean) / arr_std

    if not show_zeros and zero_mask.size:
        X_processed[zero_mask] = np.nan

    expr_df = pd.DataFrame(
        X_processed,
        columns=var_names,
        index=adata.obs_names,
    )

    expr_df = expr_df.reset_index(names='obs_index')

    df = expr_df.melt(
        id_vars='obs_index',
        var_name='var_index',
        value_name='intensity',
    )
    df = pd.merge(df, var, on='var_index', how='left')
    df = pd.merge(df, obs, on='obs_index', how='left')

    # Explicitly order the x axis observations
    cat_index_map = {}
    cats_ordered = []
    if order_by:
        if is_categorical_dtype(obs[order_by]):
            base_categories = list(obs[order_by].cat.categories)
        else:
            base_categories = list(pd.unique(obs[order_by]))
        base_categories_set = set(base_categories)

        if group_levels is not None:
            categories = [
                cat for cat in group_levels
                if cat in base_categories_set
            ]
        else:
            categories = base_categories

        cat_index_map = {
            cat: obs.loc[obs[order_by] == cat, 'obs_index'].to_list()
            for cat in categories
        }

        if order:
            missing = set(order) - set(cat_index_map)
            if missing:
                raise ValueError(
                    "Items in 'order' are not present in the selected "
                    f"'{order_by}' categories: {sorted(missing)}"
                )
            cats_ordered = list(order)
            seen_order = set(cats_ordered)
            cats_ordered.extend(
                cat for cat in categories if cat not in seen_order
            )
        else:
            cats_ordered = categories

        obs_index_ordered = [
            idx
            for cat in cats_ordered
            for idx in cat_index_map[cat]
        ]
    else:
        if order:
            missing = set(order) - set(obs['obs_index'])
            if missing:
                raise ValueError(
                    "Items in 'order' are not present in adata.obs_names: "
                    f"{sorted(missing)}"
                )
            obs_index_base = obs['obs_index'].tolist()
            seen_order = set(order)
            obs_index_ordered = list(order) + [
                idx for idx in obs_index_base if idx not in seen_order
            ]
        else:
            obs_index_ordered = obs['obs_index'].tolist()

    df['obs_index'] = pd.Categorical(
        df['obs_index'],
        categories=obs_index_ordered,
        ordered=True)

    axes = []

    if save and len(protein_ids) > 1:
        save_path = save if save.endswith('.pdf') else f'{save}.pdf'
        pdf_pages = PdfPages(save_path)

    for prot_id in protein_ids:
        sub_df = df[df['protein_id'] == prot_id]
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

            #sub_df = sub_df.sort_values(by=order_by)

            lineplot_kwargs = dict(
                data=sub_df,
                x='obs_index',
                y='intensity',
                marker='o',
                dashes=False,
                legend='brief',
                ax=_ax,
            )

            if palette_map:
                lineplot_kwargs['palette'] = palette_map

            if order_by:
                lineplot_kwargs['style'] = order_by

            if group_by:
                lineplot_kwargs.update(
                    hue=group_by,
                )
            elif color:
                lineplot_kwargs.update(
                    hue=color,
                    units='var_index',
                    estimator=None,
                    errorbar=None,
                )
            else:
                lineplot_kwargs.update(hue='var_index')

            sns.lineplot(**lineplot_kwargs)

            # Legend
            handles, labels = _ax.get_legend_handles_labels()

            # Determine which labels correspond to the hue only (ignore style)
            if group_by:
                hue_values = sub_df[group_by].dropna().unique().astype(str)
            elif color:
                hue_values = sub_df[color].unique().astype(str)
            else:
                hue_values = sub_df['var_index'].unique().astype(str)

            # Keep only legend entries whose label matches a hue value
            new_handles_labels = [(h, l) for h, l in zip(handles, labels) if l in hue_values]

            if new_handles_labels:
                handles, labels = zip(*new_handles_labels)  # unzip back into separate lists
                legend_title = (
                    group_by
                    if group_by
                    else color
                    if color
                    else 'Peptide'
                )
                _ax.legend(
                    handles,
                    labels,
                    bbox_to_anchor=(1.01, 1),
                    loc='upper left',
                    title=legend_title,
                )

            # Add group separator lines
            obs_idxpos_map = {obs: i for i, obs in enumerate(obs_index_ordered)}

            if order_by:
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

                    rot = order_by_label_rotation if order_by_label_rotation else 0
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

docstr_header = (
    "Plot peptide intensities colored by proteoforms across samples for the "
    "requested proteins."
    )
proteoform_intensities = partial_with_docsig(
    peptide_intensities,
    color = 'proteoform_id',
    )


def intensity_box_per_obs(
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
    df['obs'] = pd.Categorical(df['obs'], categories=x_ordered, ordered=True)

    # Assign colors per group
    df[group_by] = df[group_by].astype(str)
    unique_groups = list(cat_index_map.keys())
    if group_by!= 'all':
        if color_scheme is not None:
            colors = _resolve_color_scheme(color_scheme, unique_groups)
        else:
            colors = mpl.colormaps['Set2'](range(len(unique_groups))).tolist()
        color_map = {str(grp): colors[i] for i, grp in enumerate(unique_groups)}
    else:
        color_map = {'all': 'C0'}

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

    if _ax.get_legend() is not None:
        _ax.get_legend().remove()

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


def intensity_hist_imputed(
    adata,
    layer: str | None = None,
    bool_layer: str = "bool_imputed",
    bins: int = 60,
    density: bool = True,
    kde: bool = False,
    figsize=(7, 4),
    palette: dict | None = None,      # {'Measured': '#4C78A8', 'Imputed': '#F58518'}
    alpha: float = 0.6,
    title: str | None = None,
    legend_loc: str = "upper right",
    per_sample: bool = False,
    samples: list | None = None,      # list of obs names or integer indices (subset/order)
    ncols: int = 4,
    sharex: bool = True,
    sharey: bool = True,
    show: bool = True,
    save: bool | str | os.PathLike = False,
):
    """
    Plot histogram(s) of intensities in log2 scale colored by imputation status.

    - Auto-detects log; if raw, plots in log2 (<=0 -> NA).
    - Colors: 'Measured' vs 'Imputed' from layers[bool_layer].
    - If per_sample=False: one combined histogram.
      per_sample=True : grid of per-sample subplots (shared bins & one legend).

    Parameters
    ----------
    show : bool
        If True, call plt.show() at the end.
    save : bool | str | Path
        If True, save to a default filename.
        If str/Path, save to that path. If False, do not save.
    """
    # --- pull data ---
    Xsrc = adata.layers[layer] if layer is not None else adata.X
    X = Xsrc.toarray() if sparse.issparse(Xsrc) else np.asarray(Xsrc, dtype=float)

    if bool_layer not in adata.layers:
        raise KeyError(f"'{bool_layer}' not found in adata.layers")
    Bsrc = adata.layers[bool_layer]
    B = Bsrc.toarray() if sparse.issparse(Bsrc) else np.asarray(Bsrc)
    if B.shape != X.shape:
        raise ValueError(f"Shape mismatch: data {X.shape} vs {bool_layer} {B.shape}")

    # log2 for plotting (if needed)
    is_log, _ = is_log_transformed(adata, layer=layer)
    Y = X.copy()
    if not is_log:
        Y[~np.isfinite(Y) | (Y <= 0)] = np.nan
        Y = np.log2(Y)
    else:
        Y[~np.isfinite(Y)] = np.nan

    # palette & order
    if palette is None:
        palette = {"Measured": "#4C78A8", "Imputed": "#F58518"}
    hue_order = ["Measured", "Imputed"]

    # ------- Single (combined) histogram -------
    if not per_sample:
        vals = Y.ravel()
        flags = B.astype(bool).ravel()
        m = np.isfinite(vals)
        vals = vals[m]
        flags = flags[m]
        if vals.size == 0:
            raise ValueError("No finite values to plot after preprocessing.")

        status = np.where(flags, "Imputed", "Measured")
        df = pd.DataFrame({"intensity_log2": vals, "status": status})

        bin_edges = np.histogram_bin_edges(df["intensity_log2"].to_numpy(), bins=bins)

        fig, ax = plt.subplots(figsize=figsize)
        sns.histplot(
            data=df,
            x="intensity_log2",
            hue="status",
            hue_order=[h for h in hue_order if (df["status"] == h).any()],
            bins=bin_edges,
            stat=("density" if density else "count"),
            multiple="layer",
            common_norm=False,
            palette=palette,
            alpha=alpha,
            edgecolor=None,
            ax=ax,
            legend=False,
        )

        if kde:
            for k, g in df.groupby("status"):
                if len(g) > 1:
                    sns.kdeplot(g["intensity_log2"], ax=ax, color=palette.get(k), lw=1.5)

        ax.set_xlabel("Intensity (log2)")
        ax.set_ylabel("Density" if density else "Count")
        ax.set_title(title or "Intensity histogram (log2; colored by imputation)")

        present = [h for h in hue_order if (df["status"] == h).any()]
        handles = [Patch(facecolor=palette[h], edgecolor="none", alpha=alpha, label=h) for h in present]
        ax.legend(handles=handles, title="Status", loc=legend_loc, frameon=False)

        # save/show
        if save:
            path = save if isinstance(save, (str, os.PathLike)) else "intensity_hist_imputed.png"
            fig.savefig(path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        return  # nothing else returned

    # ------- Per-sample small multiples -------
    # select samples
    if samples is None:
        idx = np.arange(adata.n_obs)
        labels = adata.obs_names.to_numpy()
    else:
        idx, labels = [], []
        for s in samples:
            if isinstance(s, (int, np.integer)):
                idx.append(int(s)); labels.append(adata.obs_names[int(s)])
            else:
                where = np.where(adata.obs_names == str(s))[0]
                if where.size == 0:
                    raise KeyError(f"Sample '{s}' not in adata.obs_names")
                idx.append(int(where[0])); labels.append(str(s))
        idx = np.asarray(idx, dtype=int)
        labels = np.asarray(labels, dtype=object)

    # global bins across all selected samples
    all_vals = []
    for i in idx:
        vi = Y[i, :]
        m = np.isfinite(vi)
        if m.any():
            all_vals.append(vi[m])
    if len(all_vals) == 0:
        raise ValueError("No finite values to plot after preprocessing across selected samples.")
    bin_edges = np.histogram_bin_edges(np.concatenate(all_vals), bins=bins)

    n = len(idx)
    ncols = max(1, int(ncols))
    nrows = int(math.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False, sharex=sharex, sharey=sharey)

    present_any = set()
    for k, i in enumerate(idx):
        r, c = divmod(k, ncols)
        ax = axes[r, c]

        vi = Y[i, :]
        bi = B[i, :].astype(bool)

        m = np.isfinite(vi)
        vi = vi[m]
        bi = bi[m]
        if vi.size == 0:
            ax.set_visible(False)
            continue

        status = np.where(bi, "Imputed", "Measured")
        df_i = pd.DataFrame({"intensity_log2": vi, "status": status})
        present = [h for h in hue_order if (df_i["status"] == h).any()]
        present_any.update(present)

        sns.histplot(
            data=df_i,
            x="intensity_log2",
            hue="status",
            hue_order=present,
            bins=bin_edges,
            stat=("density" if density else "count"),
            multiple="layer",
            common_norm=False,
            palette=palette,
            alpha=alpha,
            edgecolor=None,
            ax=ax,
            legend=False,
        )

        if kde:
            for lab in present:
                g = df_i[df_i["status"] == lab]
                if len(g) > 1:
                    sns.kdeplot(g["intensity_log2"], ax=ax, color=palette.get(lab), lw=1.2)

        ax.set_title(str(labels[k]))
        if r == nrows - 1:
            ax.set_xlabel("Intensity (log2)")
        else:
            ax.set_xlabel("")
        if c == 0:
            ax.set_ylabel("Density" if density else "Count")
        else:
            ax.set_ylabel("")

    # hide any extra axes
    for k in range(n, nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r, c].set_visible(False)

    # global legend (figure-level unless user asked for 'best')
    present_any = [h for h in hue_order if h in present_any]
    handles = [Patch(facecolor=palette[h], edgecolor="none", alpha=alpha, label=h) for h in present_any]
    if legend_loc == "best":
        axes[0, 0].legend(handles=handles, title="Status", loc="best", frameon=False)
    else:
        fig.legend(handles=handles, title="Status", loc=legend_loc, frameon=False)

    plt.suptitle(title or "Intensity histograms per sample (log2; colored by imputation)", y=0.995, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    # save/show
    if save:
        path = save if isinstance(save, (str, os.PathLike)) else "intensity_hist_imputed-per-sample.png"
        fig.savefig(path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    return  # nothing returned
