import warnings
from functools import partial
from typing import Any, Sequence
from collections.abc import Sequence as SequenceABC
from numbers import Real

import numpy as np
import pandas as pd
from pandas.api.types import is_categorical_dtype
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
from adjustText import adjust_text
import seaborn as sns
import matplotlib as mpl
import anndata as ad
import math
import os
from scipy import sparse

from proteopy.utils.anndata import check_proteodata
from proteopy.utils.matplotlib import _resolve_color_scheme
from proteopy.utils.array import is_log_transformed
from proteopy.utils.functools import partial_with_docsig


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
        :func:`proteopy.utils.matplotlib._resolve_color_scheme` for either the
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


def intensity_box_per_sample(
    adata: ad.AnnData,
    layer: str | None = None,
    order_by: str | None = None,
    order: Sequence[str] | None = None,
    group_by: str | None = None,
    zero_to_na: bool = False,
    fill_na: float | int | None = None,
    log_transform: float | None = None,
    z_transform: bool = False,
    ylabel: str = "Intensity",
    xlabel_rotation: float = 90,
    order_by_label_rotation: float = 0,
    show: bool = True,
    ax: Axes | None = None,
    save: str | os.PathLike[str] | None = None,
    figsize: tuple[float, float] = (8, 5),
    color_scheme: Any | None = None,
) -> Axes:
    """
    Plot intensity distributions as boxplots, either per observation or pooled
    by a categorical grouping.

    Parameters
    ----------
    adata : AnnData
        Proteomics :class:`~anndata.AnnData`.
    layer : str, optional
        Key in ``adata.layers`` providing the matrix to plot. When ``None``,
        use ``adata.X``.
    order_by : str, optional
        Column in ``adata.obs`` whose categories annotate observation groups in
        per-observation mode. Mutually exclusive with ``group_by``.
    order : Sequence[str], optional
        Explicit order of ``order_by`` categories. When ``group_by`` is
        provided, these values order the grouped categories on the x-axis.
    group_by : str, optional
        Column in ``adata.obs`` used to pool observations into a single box per
        category. When ``None``, each observation is shown individually.
    zero_to_na : bool, optional
        Convert zero intensities to ``NaN`` before other transforms.
    fill_na : float, optional
        Replace missing intensities with this value before transformations.
    log_transform : float, optional
        Logarithm base (>0 and !=1). Applies ``log(value + 1, base)``.
    z_transform : bool, optional
        Standardize intensities per observation after the log transform.
    ylabel : str, optional
        Label for the y-axis.
    xlabel_rotation : float, optional
        Rotation angle for observation tick labels.
    order_by_label_rotation : float, optional
        Rotation angle for the group labels drawn above the axis.
    show : bool, optional
        Display the figure when ``True`` and a new axis is created.
    ax : Axes, optional
        Axis to draw on. When ``None``, a new figure and axis are created.
    save : str | os.PathLike, optional
        Path to save the figure. ``None`` skips saving.
    figsize : tuple of float, optional
        Figure size used when creating a new axis.
    color_scheme : Any, optional
        Palette forwarded to :func:`proteopy.utils.matplotlib._resolve_color_scheme`
        for either grouped categories (``group_by``) or observation-level
        annotations (``order_by``).

    Returns
    -------
    Axes
        Axis containing the rendered boxplot.
    """

    check_proteodata(adata)

    if order_by is not None and group_by is not None:
        raise ValueError("`order_by` and `group_by` cannot be combined.")

    # Validate save target early for clearer error messages
    if save is not None and not isinstance(save, (str, os.PathLike)):
        raise TypeError("`save` must be a string, PathLike, or None.")

    # Select the matrix to plot (layer vs X) while preserving dense/sparse inputs
    if layer is not None:
        if layer not in adata.layers:
            raise KeyError(f"Layer '{layer}' not found in adata.layers.")
        Xsrc = adata.layers[layer]
        X = Xsrc.toarray() if sparse.issparse(Xsrc) else np.asarray(Xsrc, dtype=float)
        df = pd.DataFrame(X, index=adata.obs_names, columns=adata.var_names)
    else:
        df = adata.to_df()

    if zero_to_na and fill_na is not None:
        raise ValueError("`zero_to_na` and `fill_na` are mutually exclusive.")

    # Optional NA imputation happens before any log or z transforms
    fill_value: float | None = None
    if fill_na is not None:
        if isinstance(fill_na, bool):
            raise TypeError("fill_na expects a numeric scalar, not boolean.")
        if not np.isscalar(fill_na):
            raise TypeError("fill_na must be a scalar numeric value.")
        fill_value = float(fill_na)
        df = df.fillna(fill_value)

    if zero_to_na:
        df = df.replace(0, np.nan)

    # Apply log and z-scaling sequentially if requested
    if log_transform is not None:
        if isinstance(log_transform, bool):
            raise TypeError("log_transform expects a numeric base, not boolean.")
        if log_transform <= 0:
            raise ValueError("log_transform must be positive.")
        if log_transform == 1:
            raise ValueError("log_transform cannot be 1.")
        log_base = float(log_transform)
        df = np.log1p(df) / np.log(log_base)

    if z_transform:
        row_means = df.mean(axis=1, skipna=True)
        row_stds = df.std(axis=1, skipna=True).replace(0, np.nan)
        df = df.sub(row_means, axis=0).div(row_stds, axis=0)

    # Long-form table with one row per (obs, var) intensity
    df_long = (
        df.assign(obs=df.index)
        .melt(id_vars="obs", var_name="var", value_name="intensity")
        .dropna(subset=["intensity"])
    )
    if df_long.empty:
        raise ValueError("No intensities remain after preprocessing; nothing to plot.")

    if group_by is not None and group_by != "obs" and group_by not in adata.obs.columns:
        raise KeyError(
            f"Column '{group_by}' not found in adata.obs; "
            "pass a valid metadata column or use group_by='obs'."
        )

    obs_index = pd.DataFrame({"obs": adata.obs_names})

    if group_by:
        # --- Grouped plotting branch: pool observations per category ---
        group_df = obs_index.copy()
        if group_by == "obs":
            group_df[group_by] = group_df["obs"]
        else:
            group_df[group_by] = adata.obs[group_by].reset_index(drop=True)
        df_long = df_long.merge(group_df, on="obs", how="left")
        group_series = group_df[group_by]

        df_long = df_long.dropna(subset=[group_by])
        if df_long.empty:
            raise ValueError(
                "No intensities remain after assigning group_by categories; nothing to plot."
            )

        if order:
            ordered_groups = list(dict.fromkeys(order))
            if is_categorical_dtype(group_series):
                default_order = list(group_series.cat.categories)
            else:
                default_order = group_series.dropna().drop_duplicates().tolist()
            for cat in default_order:
                if cat not in ordered_groups:
                    ordered_groups.append(cat)
        else:
            if is_categorical_dtype(group_series):
                ordered_groups = list(group_series.cat.categories)
            else:
                ordered_groups = group_series.dropna().drop_duplicates().tolist()

        ordered_groups = [
            cat for cat in ordered_groups if cat in df_long[group_by].unique()
        ]
        if not ordered_groups:
            raise ValueError("No group_by categories remain after preprocessing.")

        df_long[group_by] = pd.Categorical(
            df_long[group_by],
            categories=ordered_groups,
            ordered=True,
        )

        palette_values = None
        if color_scheme is not None:
            palette_values = _resolve_color_scheme(color_scheme, ordered_groups)
        if not palette_values:
            cmap = mpl.colormaps["Set2"]
            palette_values = cmap(np.linspace(0, 1, len(ordered_groups))).tolist()
        palette = dict(zip(ordered_groups, palette_values))

        created_fig = False
        if ax is None:
            fig, _ax = plt.subplots(figsize=figsize)
            created_fig = True
        else:
            _ax = ax
            fig = _ax.figure

        sns.boxplot(
            data=df_long,
            x=group_by,
            y="intensity",
            hue=group_by,
            order=ordered_groups,
            hue_order=ordered_groups,
            palette=palette,
            legend=False,
            flierprops=dict(marker='.', markersize=1),
            ax=_ax,
        )

        plt.setp(_ax.get_xticklabels(), rotation=xlabel_rotation, ha="right")
        _ax.set_xlabel(group_by)
        _ax.set_ylabel(ylabel)

        if created_fig:
            fig.tight_layout()

        if save is not None:
            fig.savefig(save, dpi=300, bbox_inches="tight")
        if show and created_fig:
            plt.show()
        return _ax

    # --- Per-observation plotting branch (default) ---
    order_col = order_by if order_by else "all"
    order_df = obs_index.copy()

    if order_by:
        if order_by not in adata.obs.columns:
            raise KeyError(
                f"Column '{order_by}' not found in adata.obs; "
                "pass a valid metadata column."
            )
        order_df[order_col] = adata.obs[order_by].reset_index(drop=True)
    else:
        order_df[order_col] = "all"

    if order_by:
        df_long = df_long.merge(order_df, on="obs", how="left")
    else:
        df_long[order_col] = "all"

    if df_long[order_col].isna().any():
        missing_obs = df_long.loc[df_long[order_col].isna(), "obs"].unique()
        preview = ", ".join(map(str, missing_obs[:5]))
        suffix = "..." if len(missing_obs) > 5 else ""
        raise ValueError(
            f"Missing '{order_col}' annotations for observations: {preview}{suffix}"
        )

    group_series = order_df[order_col]

    if order:
        ordered_groups = list(dict.fromkeys(order))
        if is_categorical_dtype(group_series):
            default_order = list(group_series.cat.categories)
        else:
            default_order = group_series.drop_duplicates().tolist()
        for cat in default_order:
            if cat not in ordered_groups:
                ordered_groups.append(cat)
    else:
        if is_categorical_dtype(group_series):
            ordered_groups = list(group_series.cat.categories)
        else:
            ordered_groups = group_series.drop_duplicates().tolist()

    def _obs_in_category(category):
        mask = group_series == category
        return order_df.loc[mask, "obs"].tolist()

    cat_index_map = {cat: _obs_in_category(cat) for cat in ordered_groups}

    available_obs = set(df_long["obs"])
    filtered_map: dict[Any, list[Any]] = {}
    for cat, obs_list in cat_index_map.items():
        pruned = [obs for obs in obs_list if obs in available_obs]
        if pruned:
            filtered_map[cat] = pruned
    cat_index_map = filtered_map

    if not cat_index_map:
        raise ValueError("No observations remain after preprocessing; nothing to plot.")

    x_ordered = [obs for obs_list in cat_index_map.values() for obs in obs_list]
    df_long["obs"] = pd.Categorical(df_long["obs"], categories=x_ordered, ordered=True)

    unique_groups = list(cat_index_map.keys())
    if order_col != "all":
        colors = None
        if color_scheme is not None:
            colors = _resolve_color_scheme(color_scheme, unique_groups)
        if not colors:
            colors = mpl.colormaps["Set2"](np.linspace(0, 1, len(unique_groups))).tolist()
        color_map = {grp: colors[i] for i, grp in enumerate(unique_groups)}
    else:
        color_map = {"all": "C0"}

    sample_palette: dict[Any, Any] = {}
    for obs in x_ordered:
        group_val = df_long.loc[df_long["obs"] == obs, order_col]
        if group_val.empty:
            continue
        color_key = group_val.iloc[0]
        sample_palette[obs] = color_map[color_key]

    created_fig = False
    if ax is None:
        fig, _ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        _ax = ax
        fig = _ax.figure

    sns.boxplot(
        data=df_long,
        x="obs",
        y="intensity",
        hue="obs",
        order=x_ordered,
        hue_order=x_ordered,
        palette=sample_palette,
        flierprops=dict(marker='.', markersize=1),
        ax=_ax,
    )

    if _ax.get_legend() is not None:
        _ax.get_legend().remove()

    plt.setp(_ax.get_xticklabels(), rotation=xlabel_rotation, ha="right")
    _ax.set_xlabel("")
    _ax.set_ylabel(ylabel)

    obs_idx_map = {obs: i for i, obs in enumerate(x_ordered)}
    ymax = df_long["intensity"].max()
    for cat, obs_list in cat_index_map.items():
        start_idx = obs_idx_map[obs_list[0]]
        end_idx = obs_idx_map[obs_list[-1]]
        mid_idx = (start_idx + end_idx) / 2
        _ax.text(
            x=mid_idx,
            y=ymax * 1.05,
            s=str(cat),
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            rotation=order_by_label_rotation,
        )

    if created_fig:
        fig.tight_layout()

    if save is not None:
        fig.savefig(save, dpi=300, bbox_inches="tight")
    if show and created_fig:
        plt.show()
    return _ax


def intensity_hist(
    adata,
    layer: str | None = None,
    color_imputed: bool = False,
    bool_layer: str | None = None,
    log_transform: float | int | None = None,
    ignore_warning: bool = False,
    fill_na: float | int | None = None,
    zero_to_nan: bool = False,
    bins: int = 60,
    density: bool = True,
    kde: bool = False,
    color_scheme: Any = None,
    xlim: tuple[float | int, float | int] | None = None,
    alpha: float = 0.6,
    title: str | None = None,
    legend_loc: str = "upper right",
    figsize=(7, 4),
    per_obs: bool = False,
    samples: list | None = None,
    ncols: int = 4,
    sharex: bool = True,
    sharey: bool = True,
    show: bool = True,
    ax: bool = False,
    save: str | os.PathLike[str] | None = None,
) -> Axes | None:
    """
    Plot histogram(s) of var intensities, optionally colored by imputation status.

    Parameters
    ----------
    adata : AnnData
        Proteomics :class:`~anndata.AnnData` containing the intensity matrix.
    layer : str | None, optional
        Layer to use for intensities; defaults to ``adata.X`` when ``None``.
    color_imputed : bool, optional
        When ``True``, overlay imputed values using the layer-specific mask;
        otherwise plot a single distribution of all intensities.
    bool_layer : str | None, optional
        Layer key containing the imputation mask. When ``None`` and
        ``color_imputed`` is ``True``, use ``imputation_mask_<layer>`` (with
        ``layer="X"`` for the main matrix).
    log_transform : float | int | None, optional
        Logarithm base (>0 and !=1). When ``None`` skip log-transforming;
        otherwise apply ``log(value + 1, base)`` after validating the input 
        (Defaut = None).
    ignore_warning : bool, optional
        When ``True``, force the log transform even if the data already appear
        log-transformed according to :func:`copro.utils.array.is_log_transformed`.
    fill_na : float | int | None, optional
        Replace missing values with this constant before log transformation.
    zero_to_nan : bool, optional
        When True, replace exact zeros with ``NaN`` before plotting.
    bins : int, optional
        Number of histogram bins passed to ``numpy.histogram_bin_edges``.
    density : bool, optional
        Plot density instead of counts.
    kde : bool, optional
        Overlay kernel density estimate curves.
    color_scheme : Any, optional
        Palette specification forwarded to
        :func:`proteopy.utils.matplotlib._resolve_color_scheme`.
    xlim : tuple[float | int, float | int] | None, optional
        Explicit x-axis limits ``(xmin, xmax)`` applied to all histograms.
    alpha : float, optional
        Histogram transparency (0-1 range).
    title : str | None, optional
        Custom title for the plot; defaults are auto-generated.
    legend_loc : str, optional
        Location argument forwarded to :func:`matplotlib.axes.Axes.legend`.
    figsize : tuple[float, float], optional
        Figure size passed to :func:`matplotlib.pyplot.subplots`.
    per_obs : bool, optional
        When ``True``, draw per-observation facets; otherwise aggregate all values.
    samples : list | None, optional
        Optional ordered subset of observations (by index or name) for
        ``per_obs`` plots.
    ncols : int, optional
        Number of columns in the per-observation facet grid.
    sharex, sharey : bool, optional
        Whether subplots share their x- or y-axes in per-observation mode.
    show : bool, optional
        If True, call plt.show() at the end.
    ax : bool, optional
        When ``True`` and ``per_obs`` is ``False``, return the Axes handle and
        skip automatic plotting even if ``show`` is ``True``.
    save : str | Path | None, optional
        Path where the figure should be written. When ``None``, no file is saved.

    Returns
    -------
    Axes | None
        Axes handle when ``ax`` is ``True`` (single histogram mode); otherwise
        ``None``.
    """
    # Basic parameter validation to fail fast on misconfigured inputs
    bool_params = {
        "color_imputed": color_imputed,
        "ignore_warning": ignore_warning,
        "zero_to_nan": zero_to_nan,
        "density": density,
        "kde": kde,
        "per_obs": per_obs,
        "sharex": sharex,
        "sharey": sharey,
        "show": show,
        "ax": ax,
    }
    for name, value in bool_params.items():
        if not isinstance(value, bool):
            raise TypeError(f"`{name}` must be a boolean.")

    if layer is not None and not isinstance(layer, str):
        raise TypeError("`layer` must be a string or None.")
    if bool_layer is not None and not isinstance(bool_layer, str):
        raise TypeError("`bool_layer` must be a string or None.")
    if not isinstance(bins, int) or bins <= 0:
        raise ValueError("`bins` must be a positive integer.")
    if not isinstance(ncols, int) or ncols <= 0:
        raise ValueError("`ncols` must be a positive integer.")
    if not isinstance(alpha, Real):
        raise TypeError("`alpha` must be numeric.")
    if not 0 <= float(alpha) <= 1:
        raise ValueError("`alpha` must be between 0 and 1.")
    if title is not None and not isinstance(title, str):
        raise TypeError("`title` must be a string or None.")
    if not isinstance(legend_loc, str):
        raise TypeError("`legend_loc` must be a string.")
    if figsize is None or not isinstance(figsize, SequenceABC) or len(figsize) != 2:
        raise TypeError("`figsize` must be a length-2 sequence of numbers.")
    try:
        figsize = (float(figsize[0]), float(figsize[1]))
    except (TypeError, ValueError):
        raise TypeError("`figsize` entries must be numeric.")
    if samples is not None:
        if (
            not isinstance(samples, SequenceABC)
            or isinstance(samples, (str, bytes))
        ):
            raise TypeError("`samples` must be a sequence of indices or names.")
    if save is not None and not isinstance(save, (str, os.PathLike)):
        raise TypeError("`save` must be a path-like object or None.")
    if xlim is not None:
        if not isinstance(xlim, SequenceABC) or len(xlim) != 2:
            raise TypeError("`xlim` must be a tuple (xmin, xmax).")
        xmin, xmax = xlim
        if not isinstance(xmin, Real) or not isinstance(xmax, Real):
            raise TypeError("`xlim` values must be numeric.")
        xmin = float(xmin)
        xmax = float(xmax)
        if not np.isfinite(xmin) or not np.isfinite(xmax):
            raise ValueError("`xlim` values must be finite.")
        if xmin >= xmax:
            raise ValueError("`xlim` must satisfy xmin < xmax.")
        x_limits = (xmin, xmax)
    else:
        x_limits = None
    if ax and per_obs:
        raise ValueError("`ax` can only be used when per_obs is False.")

    check_proteodata(adata)

    # --- pull data ---
    Xsrc = adata.layers[layer] if layer is not None else adata.X
    X = Xsrc.toarray() if sparse.issparse(Xsrc) else np.asarray(Xsrc, dtype=float)

    # Resolve the imputation mask layer if coloring measured vs imputed
    mask_layer_key: str | None = None
    if color_imputed:
        mask_layer_key = bool_layer
        if mask_layer_key is None:
            mask_target = str(layer) if layer is not None else "X"
            mask_layer_key = f"imputation_mask_{mask_target}"
        if mask_layer_key not in adata.layers:
            raise KeyError(
                f"'{mask_layer_key}' not found in adata.layers. "
                "Set color_imputed=False or provide `bool_layer` explicitly."
            )
        Bsrc = adata.layers[mask_layer_key]
        B = Bsrc.toarray() if sparse.issparse(Bsrc) else np.asarray(Bsrc)
        if B.shape != X.shape:
            raise ValueError(f"Shape mismatch: data {X.shape} vs {mask_layer_key} {B.shape}")
    else:
        B = None

    # Determine log transform base and guard against double-logging
    if log_transform is None:
        log_base = None
    else:
        if not isinstance(log_transform, Real):
            raise TypeError("log_transform must be a numeric value or None.")
        log_base = float(log_transform)
        if log_base <= 0:
            raise ValueError("log_transform must be positive.")
        if math.isclose(log_base, 1.0):
            raise ValueError("log_transform cannot be 1.")

        is_log, stats = is_log_transformed(adata, layer=layer)
        if is_log and not ignore_warning:
            raise ValueError(
                "Input appears already log-transformed. Stats: "
                f"{stats}. Pass ignore_warning=True to force another log."
            )

    Y = X.copy()
    Y = Y.astype(float, copy=False)
    Y[~np.isfinite(Y)] = np.nan

    if fill_na is not None and zero_to_nan:
        raise ValueError("fill_na and zero_to_nan cannot be used together.")

    if fill_na is not None:
        if not isinstance(fill_na, Real):
            raise TypeError("fill_na must be a numeric value or None.")
        fill_value = float(fill_na)
        if not np.isfinite(fill_value):
            raise ValueError("fill_na must be a finite numeric value.")
        np.nan_to_num(Y, copy=False, nan=fill_value)

    if zero_to_nan:
        Y[Y == 0] = np.nan

    if log_base is not None:
        Y[Y <= -1] = np.nan
        Y = np.log1p(Y) / math.log(log_base)

    # palette & order
    # Resolve colors for the measured/imputed legend
    default_palette = {"Measured": "#4C78A8", "Imputed": "#F58518"}
    status_labels = ["Measured", "Imputed"] if color_imputed else ["Measured"]
    resolved_colors = _resolve_color_scheme(color_scheme, status_labels)
    if not resolved_colors:
        resolved_colors = [default_palette[label] for label in status_labels]
    palette_map = dict(zip(status_labels, resolved_colors))
    hue_order = status_labels if color_imputed else None
    measured_color = palette_map.get("Measured", default_palette["Measured"])

    value_col = "intensity_value"
    if log_base is None:
        xlabel = "Intensity"
        descriptor = ""
    else:
        base_str = f"{log_base:g}"
        xlabel = f"Intensity (log{base_str}(x + 1))"
        descriptor = f"log{base_str}(x + 1)"

    # ------- Single (combined) histogram -------
    if not per_obs:
        # Flatten matrix for a single overall histogram
        vals = Y.ravel()
        m = np.isfinite(vals)
        vals = vals[m]
        if vals.size == 0:
            raise ValueError("No finite values to plot after preprocessing.")

        bin_edges = np.histogram_bin_edges(vals, bins=bins)

        fig, _ax = plt.subplots(figsize=figsize)
        if B is not None:
            flags = B.astype(bool).ravel()[m]
            status = np.where(flags, "Imputed", "Measured")
            df = pd.DataFrame({value_col: vals, "status": status})
            present = [h for h in hue_order if (df["status"] == h).any()]
            sns.histplot(
                data=df,
                x=value_col,
                hue="status",
                hue_order=present,
                bins=bin_edges,
                stat=("density" if density else "count"),
                multiple="layer",
                common_norm=False,
                palette=palette_map,
                alpha=alpha,
                edgecolor=None,
                ax=_ax,
                legend=False,
            )

            if kde:
                for k, g in df.groupby("status"):
                    if len(g) > 1:
                        sns.kdeplot(g[value_col], ax=_ax, color=palette_map.get(k), lw=1.5)

            handles = [
                Patch(
                    facecolor=palette_map[h],
                    edgecolor="none",
                    alpha=alpha,
                    label=h,
                )
                for h in present
            ]
            _ax.legend(handles=handles, title="Status", loc=legend_loc, frameon=False)
        else:
            df = pd.DataFrame({value_col: vals})
            sns.histplot(
                data=df,
                x=value_col,
                bins=bin_edges,
                stat=("density" if density else "count"),
                color=measured_color,
                alpha=alpha,
                edgecolor=None,
                ax=_ax,
            )
            if kde and len(df) > 1:
                sns.kdeplot(df[value_col], ax=_ax, color=measured_color, lw=1.5)

        _ax.set_xlabel(xlabel)
        _ax.set_ylabel("Density" if density else "Count")
        default_title = f"Intensity histogram ({descriptor})"
        _ax.set_title(title or default_title)
        if x_limits is not None:
            _ax.set_xlim(x_limits)

        # save/show
        if save is not None:
            fig.savefig(save, dpi=300, bbox_inches='tight')
        if show and not ax:
            plt.show()
        if ax:
            return _ax
        return None

    # ------- Per-observation small multiples -------
    # select observations
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
        raise ValueError("No finite values to plot after preprocessing across selected observations.")
    bin_edges = np.histogram_bin_edges(np.concatenate(all_vals), bins=bins)

    n = len(idx)
    ncols = max(1, int(ncols))
    nrows = int(math.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        squeeze=False,
        sharex=sharex,
        sharey=sharey,
    )

    present_any = set() if B is not None else None
    for k, i in enumerate(idx):
        r, c = divmod(k, ncols)
        _ax = axes[r, c]

        vi = Y[i, :]
        bi = B[i, :].astype(bool) if B is not None else None

        m = np.isfinite(vi)
        vi = vi[m]
        if bi is not None:
            bi = bi[m]
        if vi.size == 0:
            _ax.set_visible(False)
            continue

        if bi is not None:
            status = np.where(bi, "Imputed", "Measured")
            df_i = pd.DataFrame({value_col: vi, "status": status})
            present = [h for h in hue_order if (df_i["status"] == h).any()]
            present_any.update(present)

            sns.histplot(
                data=df_i,
                x=value_col,
                hue="status",
                hue_order=present,
                bins=bin_edges,
                stat=("density" if density else "count"),
                multiple="layer",
                common_norm=False,
                palette=palette_map,
                alpha=alpha,
                edgecolor=None,
                ax=_ax,
                legend=False,
            )

            if kde:
                for lab in present:
                    g = df_i[df_i["status"] == lab]
                    if len(g) > 1:
                        sns.kdeplot(g[value_col], ax=_ax, color=palette_map.get(lab), lw=1.2)
        else:
            df_i = pd.DataFrame({value_col: vi})
            sns.histplot(
                data=df_i,
                x=value_col,
                bins=bin_edges,
                stat=("density" if density else "count"),
                color=measured_color,
                alpha=alpha,
                edgecolor=None,
                ax=_ax,
                legend=False,
            )
            if kde and len(df_i) > 1:
                sns.kdeplot(df_i[value_col], ax=_ax, color=measured_color, lw=1.2)

        _ax.set_title(str(labels[k]))
        if r == nrows - 1:
            _ax.set_xlabel(xlabel)
        else:
            _ax.set_xlabel("")
        if c == 0:
            _ax.set_ylabel("Density" if density else "Count")
        else:
            _ax.set_ylabel("")
        if x_limits is not None:
            _ax.set_xlim(x_limits)

    # hide any extra axes
    for k in range(n, nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r, c].set_visible(False)

    # global legend (figure-level unless user asked for 'best')
    if present_any is not None and present_any:
        present_any = [h for h in hue_order if h in present_any]
        handles = [
            Patch(facecolor=palette_map[h], edgecolor="none", alpha=alpha, label=h)
            for h in present_any
        ]
        if legend_loc == "best":
            axes[0, 0].legend(handles=handles, title="Status", loc="best", frameon=False)
        else:
            fig.legend(handles=handles, title="Status", loc=legend_loc, frameon=False)

    per_obs_title = f"Intensity histograms per observation ({descriptor})"
    plt.suptitle(title or per_obs_title, y=0.995, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    # save/show
    if save is not None:
        fig.savefig(save, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    return None

docstr_header = (
    "Plot a histogram of var intensities per observation, optionally colored "
    "by imputation status."
    )
intensity_hist_per_obs = partial_with_docsig(
    intensity_hist,
    per_obs=True,
    docstr_header=docstr_header,
    )


def abundance_rank(
    adata: ad.AnnData,
    color: str | None = None,
    highlight_vars: Sequence[str] | None = None,
    var_labels_key: str | None = None,
    var_label_fontsize: float = 8,
    layer: str | None = None,
    summary_method: str = "average",
    log_transform: float | None = 10,
    input_space: str = "auto",
    force: bool = False,
    zero_to_na: bool = False,
    fill_na: float | int | None = None,
    figsize: tuple[float, float] = (8, 6),
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    alpha: float = 0.6,
    s: float = 20,
    show: bool = True,
    save: str | os.PathLike[str] | None = None,
    ax: bool = False,
    color_scheme: Any = None,
) -> Axes | None:
    """
    Plot variable intensities vs their abundance rank.

    A typical MS proteomics plot to assess dynamic range and intensity
    distribution. Each point represents a variable (protein/peptide) with
    its y-value being the summary statistic (e.g., average, median) across
    observations (computation ignores NaNs).

    When ``color`` is specified, summary values and ranks are computed
    separately for each group. The plot shows one dot per variable per
    group, with all groups superimposed. When ``color`` is ``None``, a
    single dot per variable is plotted based on the global summary.

    Parameters
    ----------
    adata : AnnData
        Proteomics :class:`~anndata.AnnData`.
    color : str, optional
        Column in ``adata.obs`` used to subset observations into groups.
        Summary values and ranks are computed separately per group, and
        all groups are plotted superimposed. When ``None``, global summary
        across all observations is used.
    highlight_vars : Sequence[str], optional
        List of variable names to highlight with text labels using adjustText.
        When ``color`` is specified, each variable is labeled once per group
        at its group-specific position.
    var_labels_key : str, optional
        Column in ``adata.var`` containing alternative labels for highlighted
        variables. When specified, these labels are displayed instead of the
        variable names. Useful for displaying gene symbols instead of IDs.
    var_label_fontsize : float, optional
        Font size for highlighted variable labels.
    layer : str, optional
        Key in ``adata.layers`` providing the intensity matrix. When ``None``,
        uses ``adata.X``.
    summary_method : str, optional
        Method to summarize intensities across observations per variable.
        Options: ``'sum'``, ``'average'``, ``'median'``, ``'max'``.
        NAs are ignored; if all values are NA, the result is NA.
    log_transform : float, optional
        Base for log transformation of intensities. Defines the target space.
        When ``None``, no transformation is applied (linear space).
    input_space : str, optional
        Specifies the input data space: ``'log'``, ``'linear'``, or ``'auto'``.
        When ``'auto'``, uses heuristics to infer whether data is already
        log-transformed.
    force : bool, optional
        When ``True``, suppress warnings about input space mismatches.
    zero_to_na : bool, optional
        Convert zero intensities to ``NaN`` before transformations.
    fill_na : float | int, optional
        Replace missing values with this constant before transformations.
    figsize : tuple[float, float], optional
        Figure dimensions (width, height) in inches.
    title : str, optional
        Plot title. Defaults to ``"Abundance Rank Plot"``.
    xlabel : str, optional
        Label for x-axis. Defaults to ``"Rank"``.
    ylabel : str, optional
        Label for y-axis. Auto-generated based on transformation applied.
    alpha : float, optional
        Point transparency (0-1 range).
    s : float, optional
        Point size for the scatter plot.
    show : bool, optional
        Display the figure with ``matplotlib.pyplot.show()``.
    save : str | os.PathLike, optional
        Path to save the figure. ``None`` skips saving.
    ax : bool, optional
        Return the underlying Axes object instead of ``None``.
    color_scheme : Any, optional
        Palette specification forwarded to
        :func:`proteopy.utils.matplotlib._resolve_color_scheme`.

    Returns
    -------
    Axes | None
        The Matplotlib Axes object if ``ax=True``, otherwise ``None``.

    Raises
    ------
    ValueError
        If ``input_space`` is ``'log'`` and ``log_transform`` is ``None``
        (cannot convert log to linear without knowing the base), if both
        ``zero_to_na`` and ``fill_na`` are set, or if no valid data remains.
    KeyError
        If ``color`` column is not in ``adata.obs``, if ``layer`` is not in
        ``adata.layers``, or if ``highlight_vars`` contains variables not in
        ``adata.var_names``.

    Notes
    -----
    **Input/Output Space Logic:**

    - ``input_space='auto'``: Heuristically infers whether data is
      log-transformed. Prints an informational message about the inference.

    - If input is inferred as log and target is also log (``log_transform``
      is set): No transformation; prints a message that the log base is
      ignored since data is already in log space.

    - If input is inferred as log and target is linear (``log_transform=None``):
      Raises an error because the original log base is unknown.

    - If ``input_space`` is explicitly set (not ``'auto'``):
      - ``input_space='linear'`` with ``log_transform`` set: Applies log
        transformation.
      - ``input_space='log'`` with ``log_transform=None``: Raises a warning
        (or error if ``force=False``).
      - Matching spaces: No transformation.

    - When ``force=False`` and the inferred space doesn't match the declared
      ``input_space``, a warning is raised.

    Examples
    --------
    Basic abundance rank plot:

    >>> proteopy.pl.abundance_rank(adata)

    Color by sample condition:

    >>> proteopy.pl.abundance_rank(adata, color="condition")

    Highlight specific proteins:

    >>> proteopy.pl.abundance_rank(
    ...     adata,
    ...     highlight_vars=["ProteinA", "ProteinB"],
    ... )
    """
    check_proteodata(adata)

    # --- Parameter validation ---
    if input_space not in ("auto", "log", "linear"):
        raise ValueError(
            "input_space must be 'auto', 'log', or 'linear'."
        )

    if zero_to_na and fill_na is not None:
        raise ValueError("`zero_to_na` and `fill_na` are mutually exclusive.")

    if color is not None and color not in adata.obs.columns:
        raise KeyError(f"Column '{color}' not found in adata.obs.")

    if layer is not None and layer not in adata.layers:
        raise KeyError(f"Layer '{layer}' not found in adata.layers.")

    if highlight_vars is not None:
        missing_vars = [v for v in highlight_vars if v not in adata.var_names]
        if missing_vars:
            raise KeyError(
                f"Variables not found in adata.var_names: {missing_vars}"
            )

    if var_labels_key is not None and var_labels_key not in adata.var.columns:
        raise KeyError(f"Column '{var_labels_key}' not found in adata.var.")

    if log_transform is not None:
        if not isinstance(log_transform, (int, float)):
            raise TypeError("log_transform must be a numeric value or None.")
        if log_transform <= 0:
            raise ValueError("log_transform must be positive.")
        if log_transform == 1:
            raise ValueError("log_transform cannot be 1.")

    if not isinstance(alpha, (int, float)) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be a number between 0 and 1.")

    if not isinstance(s, (int, float)) or s <= 0:
        raise ValueError("s must be a positive number.")

    valid_summary_methods = ("sum", "average", "median", "max")
    if summary_method not in valid_summary_methods:
        raise ValueError(
            f"summary_method must be one of {valid_summary_methods}, "
            f"got '{summary_method}'."
        )

    # --- Get data matrix ---
    if layer is not None:
        Xsrc = adata.layers[layer]
    else:
        Xsrc = adata.X

    if sparse.issparse(Xsrc):
        X = Xsrc.toarray()
    else:
        X = np.asarray(Xsrc, dtype=float)

    X = X.copy()

    # --- Handle NA and zero values ---
    if fill_na is not None:
        if not isinstance(fill_na, (int, float)):
            raise TypeError("fill_na must be a numeric value.")
        if not np.isfinite(fill_na):
            raise ValueError("fill_na must be a finite numeric value.")
        X[np.isnan(X)] = float(fill_na)

    if zero_to_na:
        X[X == 0] = np.nan

    # --- Determine input space and apply transformations ---
    inferred_is_log, infer_stats = is_log_transformed(adata, layer=layer)

    # Determine target space
    target_is_log = log_transform is not None

    # Handle input_space='auto'
    if input_space == "auto":
        if inferred_is_log:
            actual_input_is_log = True
        else:
            actual_input_is_log = False
    else:
        # Explicit input_space
        actual_input_is_log = (input_space == "log")

        # Check for mismatch between declared and inferred
        if actual_input_is_log != inferred_is_log and not force:
            inferred_str = "log" if inferred_is_log else "linear"
            warnings.warn(
                f"Declared input_space='{input_space}' but data appears to be "
                f"'{inferred_str}' (p95={infer_stats['p95']:.2f}, "
                f"frac_negative={infer_stats['frac_negative']:.4f}). "
                f"Set force=True to suppress this warning.",
                UserWarning,
            )

    # Apply transformation logic
    transform_applied = None

    if actual_input_is_log and target_is_log:
        # Both log: no transformation needed
        print(
            "Input is already log-transformed; ignoring log_transform "
            f"parameter (base={log_transform})."
        )
        transform_applied = "none (already log)"
        ylabel_default = "Intensity (log)"

    elif actual_input_is_log and not target_is_log:
        # Log to linear: cannot do without knowing the original base
        raise ValueError(
            "Cannot convert log-transformed data to linear space without "
            "knowing the original log base. Either set log_transform to keep "
            "data in log space, or provide data in linear space."
        )

    elif not actual_input_is_log and target_is_log:
        # Linear to log: apply transformation
        log_base = float(log_transform)
        with np.errstate(divide='ignore', invalid='ignore'):
            X = np.log1p(X) / np.log(log_base)
        transform_applied = f"log{log_base:g}p1"
        ylabel_default = f"Intensity ({transform_applied})"

    else:
        # Both linear: no transformation
        transform_applied = "none (linear)"
        ylabel_default = "Intensity"

    # --- Define summary function ---
    def compute_summary(arr, method):
        """Compute summary statistic ignoring NAs."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "All-NaN slice encountered")
            warnings.filterwarnings("ignore", "Mean of empty slice")
            if method == "sum":
                return np.nansum(arr, axis=0)
            elif method == "average":
                return np.nanmean(arr, axis=0)
            elif method == "median":
                return np.nanmedian(arr, axis=0)
            elif method == "max":
                return np.nanmax(arr, axis=0)

    # --- Compute summary values and ranks ---
    var_names = adata.var_names.to_numpy()
    n_vars = len(var_names)

    # Store summary values and ranks per group (or None for global)
    group_summary_values = {}
    group_rank_positions = {}

    if color is not None:
        # Get unique groups from the color column
        groups = adata.obs[color].dropna().unique().tolist()

        for group in groups:
            # Get observations belonging to this group
            group_obs_mask = adata.obs[color] == group
            group_obs_indices = np.where(group_obs_mask)[0]

            if len(group_obs_indices) == 0:
                continue

            # Compute summary using only this group's observations
            X_group = X[group_obs_indices, :]
            summary_vals = compute_summary(X_group, summary_method)

            # Create ranking for this group (descending by summary value)
            # Handle NaN values in ranking
            rank_position = np.full(n_vars, np.nan)
            valid_mask = np.isfinite(summary_vals)
            if valid_mask.any():
                valid_indices = np.where(valid_mask)[0]
                valid_summary = summary_vals[valid_mask]
                rank_order = np.argsort(-valid_summary)
                for rank, idx in enumerate(rank_order):
                    rank_position[valid_indices[idx]] = rank

            group_summary_values[group] = summary_vals
            group_rank_positions[group] = rank_position

    else:
        # Global summary across all observations
        summary_vals = compute_summary(X, summary_method)

        valid_mask = np.isfinite(summary_vals)
        if not valid_mask.any():
            raise ValueError("No valid intensities remain after preprocessing.")

        rank_position = np.full(n_vars, np.nan)
        valid_indices = np.where(valid_mask)[0]
        valid_summary = summary_vals[valid_mask]
        rank_order = np.argsort(-valid_summary)
        for rank, idx in enumerate(rank_order):
            rank_position[valid_indices[idx]] = rank

        group_summary_values[None] = summary_vals
        group_rank_positions[None] = rank_position

    # --- Build plotting DataFrame ---
    records = []

    if color is not None:
        for group in groups:
            if group not in group_summary_values:
                continue
            summary_vals = group_summary_values[group]
            rank_pos = group_rank_positions[group]

            for j, var_name in enumerate(var_names):
                summary_val = summary_vals[j]
                rank_val = rank_pos[j]

                if not np.isfinite(summary_val) or not np.isfinite(rank_val):
                    continue

                records.append({
                    'var': var_name,
                    'intensity': summary_val,
                    'rank': rank_val,
                    'color_group': group,
                })
    else:
        summary_vals = group_summary_values[None]
        rank_pos = group_rank_positions[None]

        for j, var_name in enumerate(var_names):
            summary_val = summary_vals[j]
            rank_val = rank_pos[j]

            if not np.isfinite(summary_val) or not np.isfinite(rank_val):
                continue

            records.append({
                'var': var_name,
                'intensity': summary_val,
                'rank': rank_val,
            })

    if not records:
        raise ValueError("No valid data points remain after computing summaries.")

    plot_df = pd.DataFrame(records)

    # --- Create plot ---
    fig, _ax = plt.subplots(figsize=figsize)

    if color is not None:
        palette_values = _resolve_color_scheme(color_scheme, groups)
        if not palette_values:
            cmap = mpl.colormaps.get_cmap("tab10")
            palette_values = [cmap(i % 10) for i in range(len(groups))]
        palette = dict(zip(groups, palette_values))

        for group in groups:
            group_df = plot_df[plot_df['color_group'] == group]
            _ax.scatter(
                group_df['rank'],
                group_df['intensity'],
                c=[palette[group]],
                alpha=alpha,
                s=s,
                label=str(group),
            )
        _ax.legend(
            title=color,
            bbox_to_anchor=(1.02, 1),
            loc='upper left',
        )
    else:
        _ax.scatter(
            plot_df['rank'],
            plot_df['intensity'],
            alpha=alpha,
            s=s,
            color='steelblue',
        )

    # --- Add highlighted variable labels ---
    if highlight_vars is not None and len(highlight_vars) > 0:
        # Build label mapping
        if var_labels_key is not None:
            var_to_label = dict(zip(var_names, adata.var[var_labels_key]))
        else:
            var_to_label = {v: v for v in var_names}

        texts = []
        highlight_x = []
        highlight_y = []

        if color is not None:
            # When coloring by groups, highlight each var in each group
            for group in groups:
                if group not in group_summary_values:
                    continue
                summary_vals = group_summary_values[group]
                rank_pos = group_rank_positions[group]
                grp_color = palette.get(group, 'red')

                for var in highlight_vars:
                    var_idx = np.where(var_names == var)[0]
                    if len(var_idx) == 0:
                        continue
                    var_idx = var_idx[0]
                    var_rank = rank_pos[var_idx]
                    var_summary = summary_vals[var_idx]

                    if not np.isfinite(var_summary) or not np.isfinite(var_rank):
                        continue

                    label = str(var_to_label.get(var, var))
                    highlight_x.append(var_rank)
                    highlight_y.append(var_summary)
                    texts.append(
                        _ax.text(
                            var_rank,
                            var_summary,
                            label,
                            fontsize=var_label_fontsize,
                        )
                    )
                    _ax.scatter(
                        [var_rank],
                        [var_summary],
                        color=grp_color,
                        s=s * 2,
                        zorder=10,
                        marker='o',
                        edgecolors='black',
                        linewidths=0.5,
                    )
        else:
            # Global: single marker per var
            summary_vals = group_summary_values[None]
            rank_pos = group_rank_positions[None]

            for var in highlight_vars:
                var_idx = np.where(var_names == var)[0]
                if len(var_idx) == 0:
                    continue
                var_idx = var_idx[0]
                var_rank = rank_pos[var_idx]
                var_summary = summary_vals[var_idx]

                if not np.isfinite(var_summary) or not np.isfinite(var_rank):
                    continue

                label = str(var_to_label.get(var, var))
                highlight_x.append(var_rank)
                highlight_y.append(var_summary)
                texts.append(
                    _ax.text(
                        var_rank,
                        var_summary,
                        label,
                        fontsize=var_label_fontsize,
                    )
                )
                _ax.scatter(
                    [var_rank],
                    [var_summary],
                    color='red',
                    s=s * 2,
                    zorder=10,
                    marker='o',
                    edgecolors='black',
                    linewidths=0.5,
                )

        if texts:
            adjust_text(
                texts,
                x=highlight_x,
                y=highlight_y,
                ax=_ax,
                arrowprops=dict(arrowstyle="-", color="0.4", lw=0.7),
                expand=(1.5,1.5),
                force_text=0.5,
                force_explode=(4.4, 4.4),
                avoid_self=True,
                only_move={'text': 'x+y'},
            )

    # --- Set labels and title ---
    _ax.set_xlabel(xlabel or "Rank")
    _ax.set_ylabel(ylabel or ylabel_default)
    _ax.set_title(title or "Abundance Rank Plot")

    plt.tight_layout()

    # --- Save/Show logic ---
    if save is not None:
        fig.savefig(save, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    if ax:
        return _ax
    if not save and not show and not ax:
        warnings.warn(
            "Plot created but not displayed, saved, or returned. "
            "Set show=True, save to a path, or ax=True.",
            UserWarning,
        )
        plt.close(fig)
    return None
