import warnings
from pathlib import Path
from typing import Any, Iterable, Sequence
import uuid

import numpy as np
import pandas as pd
import anndata as ad
from pandas.api.types import is_string_dtype, is_categorical_dtype
from scipy import sparse
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import seaborn as sns

from proteopy.utils.anndata import check_proteodata, is_proteodata
from proteopy.utils.matplotlib import _resolve_color_scheme
from proteopy.utils.functools import partial_with_docsig
from proteopy.utils.string import sanitize_string
from proteopy.pp.stats import calculate_cv


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

docstr_header="Plot a histogram of completeness per sample (observation).\n"
completeness_per_sample = partial_with_docsig(
    completeness,
    axis=1,
    docstr_header=docstr_header,
    )


def n_var_per_sample(
    adata: ad.AnnData,
    group_by: str | None = None,
    order_by: str | None = None,
    order: Sequence[str] | None = None,
    ascending: bool = False,
    zero_to_na: bool = False,
    layer: str | None = None,
    figsize: tuple[float, float] = (6.0, 4.0),
    level: str | None = None,
    ylabel: str = "Nr. vars detected",
    xlabel_rotation: float = 90,
    order_by_label_rotation: float = 0,
    show: bool = True,
    ax: bool = False,
    save: bool | str | Path | None = False,
    color_scheme: Any | None = None,
) -> Axes | None:
    """
    Plot the number of detected variables per sample (obs).

    Parameters
    ----------
    adata : AnnData
        AnnData object with proteomics annotations.
    group_by : str | None
        Optional column in ``adata.obs`` used to summarise observations into
        groups. When provided, a boxplot of detected variables per group is
        shown. Mutually exclusive with ``order_by``.
    order_by : str | None
        Optional column in ``adata.obs`` used to assign group labels.
    order : Sequence[str] | None
        Determines the x-axis order.
        Without ``group_by`` or ``order_by`` it should list observation names.
        With ``order_by`` it specifies the group order for the stacked bars.
        With ``group_by`` it specifies the group order for the boxplot.
    ascending : bool
        When both ``group_by`` and ``order_by`` are ``None`` and ``order`` is
        ``None``, sort observations by detected counts.
        ``False`` places higher counts to the left.
        ``True`` places lower counts to the left.
    zero_to_na : bool
        Treat zero entries as missing values when ``True``.
    layer : str | None
        Name of an alternate matrix in ``adata.layers``.
        Defaults to ``adata.X``.
    figsize : tuple[float, float]
        Figure size supplied to :func:`matplotlib.pyplot.subplots`.
    level : str | None
        Quantification level to count.
        ``"peptide"`` counts detected peptides.
        ``"protein"`` aggregates peptides to proteins.
        ``None`` follows the intrinsic level of the data.
    ylabel : str
        Label applied to the y-axis.
    xlabel_rotation : float
        Rotation in degrees applied to x tick labels.
    order_by_label_rotation : float
        Rotation in degrees applied to group labels drawn above the plot.
    show : bool
        Call :func:`matplotlib.pyplot.show` when ``True``.
    ax : bool
        Return the :class:`~matplotlib.axes.Axes` instead of displaying the plot.
    save : str | Path | None
        Path or truthy value triggering ``Figure.savefig``.
    color_scheme : Any | None
        Optional mapping or palette controlling bar colours.
    """
    _, data_level = is_proteodata(adata, raise_error=True)

    if level is not None:
        level = level.lower()
        if level not in {"peptide", "protein"}:
            raise ValueError(
                "level must be one of {'peptide', 'protein', None}."
            )

    if group_by is not None and order_by is not None:
        raise ValueError("`group_by` and `order_by` cannot be used together.")

    def _contains_value(seq, value) -> bool:
        for item in seq:
            if pd.isna(item) and pd.isna(value):
                return True
            if item == value:
                return True
        return False

    def _append_unique(seq, value) -> None:
        if not _contains_value(seq, value):
            seq.append(value)

    if layer is None:
        matrix = adata.X
    else:
        if layer not in adata.layers:
            raise KeyError(f"Layer '{layer}' not found in adata.layers.")
        matrix = adata.layers[layer]
        if matrix is None:
            raise ValueError(
                "Selected layer is empty; cannot compute variable counts."
            )

    n_obs, _ = matrix.shape

    if sparse.issparse(matrix):
        matrix_csr = matrix.tocsr()
        indptr = matrix_csr.indptr
        indices = matrix_csr.indices
        data = matrix_csr.data

        row_lengths = np.diff(indptr)
        row_indices = np.repeat(np.arange(n_obs), row_lengths)

        valid_entry_mask = ~np.isnan(data)
        if zero_to_na:
            valid_entry_mask &= data != 0

        valid_rows = row_indices[valid_entry_mask]
        valid_cols = indices[valid_entry_mask]
    else:
        values = np.asarray(matrix)
        valid_mask = ~np.isnan(values)
        if zero_to_na:
            valid_mask &= values != 0

        valid_rows, valid_cols = np.nonzero(valid_mask)

    if valid_rows.size:
        base_counts = np.bincount(valid_rows, minlength=n_obs)
    else:
        base_counts = np.zeros(n_obs, dtype=int)

    if level is None or level == data_level:
        counts_array = base_counts
    elif level == "protein" and data_level == "peptide":
        protein_ids = adata.var["protein_id"].to_numpy()
        protein_codes, protein_unique = pd.factorize(protein_ids, sort=False)
        n_proteins = len(protein_unique)

        if valid_rows.size:
            protein_cols = protein_codes[valid_cols]
            data_vals = np.ones(valid_rows.size, dtype=np.int8)
            protein_matrix = sparse.csr_matrix(
                (data_vals, (valid_rows, protein_cols)),
                shape=(n_obs, n_proteins),
            )
            protein_matrix.data[:] = 1
            counts_array = np.diff(protein_matrix.indptr)
        else:
            counts_array = np.zeros(n_obs, dtype=int)
    else:
        raise ValueError(
            f"Requested level '{level}' is incompatible with '{data_level}' data."
        )

    counts_series = pd.Series(counts_array, index=adata.obs_names, name="count")
    counts = counts_series.rename_axis("obs").reset_index()

    if group_by is not None:
        if group_by not in adata.obs.columns:
            raise KeyError(f"Column '{group_by}' not found in adata.obs.")

        group_df = adata.obs[[group_by]].copy()
        group_df = group_df.rename_axis("obs").reset_index()
        counts = pd.merge(counts, group_df, on="obs", how="left")
        counts = counts.dropna(subset=[group_by])
        if counts.empty:
            raise ValueError(
                "No observations remain after aligning `group_by` labels.",
            )

        group_values = counts[group_by]
        if isinstance(group_values.dtype, pd.CategoricalDtype):
            group_values = group_values.cat.remove_unused_categories()
            counts[group_by] = group_values
        available_groups: list[Any] = []
        for value in group_values:
            _append_unique(available_groups, value)

        if order:
            group_order: list[Any] = []
            for grp in order:
                if not _contains_value(group_order, grp):
                    group_order.append(grp)
            missing_groups = [
                grp for grp in group_order
                if not _contains_value(available_groups, grp)
            ]
            if missing_groups:
                missing_str = ", ".join(map(str, missing_groups))
                raise ValueError(
                    f"Unknown group(s) in order argument: {missing_str}.",
                )
        else:
            if isinstance(group_values.dtype, pd.CategoricalDtype):
                group_order = list(group_values.cat.categories)
            else:
                group_order = available_groups.copy()

        for value in available_groups:
            _append_unique(group_order, value)

        stats_df = (
            counts.groupby(group_by, observed=True)["count"]
            .agg(mean_count="mean", std_count="std")
            .reindex(group_order)
        )
        stats_df = stats_df.dropna(subset=["mean_count"])
        stats_df["std_count"] = stats_df["std_count"].fillna(0.0)
        stats_df = stats_df.reset_index()

        colors = None
        bar_colors = None
        if color_scheme is not None:
            colors = _resolve_color_scheme(color_scheme, group_order)
        if colors is not None:
            bar_colors = [
                colors[group_order.index(grp)] for grp in stats_df[group_by]
            ]

        fig, _ax = plt.subplots(figsize=figsize)
        bar_labels = stats_df[group_by].astype(str)
        bars = _ax.bar(
            bar_labels,
            stats_df["mean_count"],
            yerr=stats_df["std_count"],
            color=bar_colors,
            capsize=4.0,
            edgecolor="black",
        )
        plt.setp(_ax.get_xticklabels(), rotation=xlabel_rotation, ha="right")
        _ax.set_xlabel(group_by)
        _ax.set_ylabel(ylabel)
        plt.tight_layout()

        if save:
            fig.savefig(save, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        if ax:
            return _ax
        return None

    has_grouping = order_by is not None
    group_key = order_by if has_grouping else "_group"

    if has_grouping:
        if group_key != "obs":
            obs = adata.obs[[group_key]].copy()
            obs = obs.rename_axis("obs").reset_index()
            counts = pd.merge(counts, obs, on="obs", how="left")
        else:
            counts[group_key] = counts["obs"]
    else:
        counts[group_key] = "all"

    obs_df = adata.obs.copy()
    obs_df = obs_df.rename_axis("obs").reset_index()
    if group_key not in obs_df.columns:
        obs_df[group_key] = "all"
    if has_grouping and isinstance(obs_df[group_key].dtype, pd.CategoricalDtype):
        obs_df[group_key] = obs_df[group_key].astype("category")

    available_groups: list[Any] = []
    for value in obs_df[group_key]:
        _append_unique(available_groups, value)

    if has_grouping:
        if order:
            group_order = list(order)
            missing_groups = [
                grp for grp in group_order
                if not _contains_value(available_groups, grp)
            ]
            if missing_groups:
                missing_str = ", ".join(map(str, missing_groups))
                raise ValueError(
                    f"Unknown group(s) in order argument: {missing_str}."
                )
        else:
            group_order = available_groups.copy()

        for grp in available_groups:
            _append_unique(group_order, grp)

        cat_index_map: dict[str, list[str]] = {}
        for grp in group_order:
            obs_list = obs_df.loc[obs_df[group_key] == grp, "obs"].tolist()
            if obs_list:
                cat_index_map[str(grp)] = obs_list
        x_ordered = [obs for obs_list in cat_index_map.values() for obs in obs_list]
    else:
        if order:
            obs_order = list(order)
            available_obs = counts["obs"].tolist()
            missing_obs = [
                obs_name for obs_name in obs_order
                if not _contains_value(available_obs, obs_name)
            ]
            if missing_obs:
                missing_str = ", ".join(map(str, missing_obs))
                raise ValueError(f"Unknown obs in order argument: {missing_str}.")
            x_ordered: list[Any] = []
            for obs_name in obs_order:
                _append_unique(x_ordered, obs_name)
            for obs_name in counts["obs"]:
                _append_unique(x_ordered, obs_name)
        else:
            sorted_counts = counts.sort_values(
                "count",
                ascending=ascending,
                kind="mergesort",
            )
            x_ordered = sorted_counts["obs"].tolist()
        cat_index_map = {"all": x_ordered}
    counts["obs"] = pd.Categorical(
        counts["obs"],
        categories=x_ordered,
        ordered=True,
    )
    counts = counts.sort_values("obs")

    counts[group_key] = counts[group_key].astype(str)

    unique_groups = list(cat_index_map.keys())
    colors = _resolve_color_scheme(color_scheme, unique_groups)
    plot_kwargs = {}

    if colors is not None:
        color_map = {str(grp): colors[i] for i, grp in enumerate(unique_groups)}
        plot_kwargs["color"] = counts[group_key].map(color_map).to_list()

    fig, _ax = plt.subplots(figsize=figsize)
    counts.plot(
        kind="bar",
        x="obs",
        y="count",
        ax=_ax,
        legend=False,
        **plot_kwargs,
    )

    plt.setp(_ax.get_xticklabels(), rotation=xlabel_rotation, ha="right")
    _ax.set_xlabel("")
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
            rotation=order_by_label_rotation
        )

    plt.tight_layout()

    if save:
        fig.savefig(save, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    if ax:
        return _ax

docstr_header = "Plot the number of detected peptides per observation."
n_peptides_per_sample = partial_with_docsig(
    n_var_per_sample,
    level="peptide",
    ylabel="Nr. peptides detected",
    docstr_header=docstr_header,
)

docstr_header = "Plot the number of detected proteins per observation."
n_proteins_per_sample = partial_with_docsig(
    n_var_per_sample,
    level="protein",
    ylabel="Nr. proteins detected",
    docstr_header=docstr_header,
)


def n_samples_per_category(
    adata: ad.AnnData,
    category_key: str | Sequence[str],
    categories: Sequence[Any] | None = None,
    ignore_na: bool = False,
    ascending: bool = False,
    order: Sequence[Any] | None = None,
    xlabel_rotation: float = 45.0,
    color_scheme: Any | None = None,
    figsize: tuple[float, float] = (6.0, 4.0),
    show: bool = True,
    save: str | Path | None = None,
    ax: bool = False,
) -> Axes | None:
    """
    Plot sample (obs) counts per category (optionally stratified).

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix with categorical obs annotations.
    category_key : str | Sequence[str]
        One or two column names in ``adata.obs`` used to stratify observations.
    categories : Sequence[Any] | None
        Labels from the first category column to display on the x-axis. Rows
        whose first-column value is not listed are dropped.
    ignore_na : bool
        Drop observations with missing labels when ``True``; otherwise, missing
        values are shown as ``"missing"``.
    ascending : bool
        Sort categories by total counts when no explicit order is supplied.
        ``True`` places lower counts on the left.
    order : Sequence[Any] | None
        Explicit order for the x-axis labels (values of the first category
        column). Any levels not listed are appended afterwards in their intrinsic
        order. When provided, ``ascending`` is ignored.
    xlabel_rotation : float
        Rotation angle (degrees) applied to the x-axis tick labels.
    color_scheme : Any | None
        Mapping, sequence, colormap name, or callable used to colour categories.
    figsize : tuple[float, float]
        Figure size (width, height) in inches used for
        :func:`matplotlib.pyplot.subplots`.
    show : bool
        Call :func:`matplotlib.pyplot.show` when ``True``.
    save : str | Path | None
        Save the figure to the provided path (``str`` or :class:`~pathlib.Path``).
    ax : bool
        Return the :class:`~matplotlib.axes.Axes` instead of displaying the plot.
    """
    check_proteodata(adata)

    if isinstance(category_key, str):
        category_cols = [category_key]
    else:
        category_cols = list(category_key)
    if not category_cols:
        raise ValueError("category_key must contain at least one column name.")

    missing_label = "missing"
    unknown_cols = [col for col in category_cols if col not in adata.obs]
    if unknown_cols:
        raise KeyError(
            "Column(s) missing in adata.obs: "
            f"{', '.join(map(str, unknown_cols))}."
        )

    obs = adata.obs.loc[:, category_cols].copy()

    for col in category_cols:
        if not (is_string_dtype(obs[col]) or is_categorical_dtype(obs[col])):
            obs[col] = obs[col].astype("string")
        if ignore_na:
            continue
        if is_categorical_dtype(obs[col]):
            if missing_label not in obs[col].cat.categories:
                obs[col] = obs[col].cat.add_categories([missing_label])
            obs[col] = obs[col].fillna(missing_label)
        else:
            obs[col] = obs[col].fillna(missing_label)

    first_cat_col = category_cols[0]

    if ignore_na:
        obs = obs.dropna(subset=category_cols, how="any")

    first_cat_col = category_cols[0]

    selected_categories: list[Any] | None = None
    if categories is not None:
        if isinstance(categories, (str, bytes)):
            selected_categories = [categories]
        else:
            selected_categories = list(categories)
        if not selected_categories:
            raise ValueError("categories must contain at least one label.")
        mask = obs[first_cat_col].isin(selected_categories)
        if not mask.any():
            raise ValueError("No observations match the requested categories.")
        obs = obs.loc[mask].copy()

    if obs.empty:
        raise ValueError("No observations available after NA handling.")
    for col in category_cols:
        if is_categorical_dtype(obs[col]):
            obs[col] = obs[col].cat.remove_unused_categories()

    def _ordered_categories(series: pd.Series) -> list[Any]:
        if is_categorical_dtype(series):
            ordered = list(series.cat.categories)
        else:
            ordered = list(pd.unique(series))
        if not ignore_na and missing_label in ordered:
            ordered = [
                value for value in ordered if value != missing_label
            ] + [missing_label]
        return ordered

    first_level_order = _ordered_categories(obs[first_cat_col])

    if selected_categories is not None:
        first_level_order = [
            category for category in selected_categories if category in first_level_order
        ]
    if order is not None:
        if isinstance(order, str):
            specified = [order]
        else:
            specified = list(order)
        unknown_specified = [cat for cat in specified if cat not in first_level_order]
        if unknown_specified:
            raise ValueError(
                "Order values not present in the first category column: "
                f"{', '.join(map(str, unknown_specified))}."
            )
        remaining = [cat for cat in first_level_order if cat not in specified]
        first_level_order = specified + remaining

    use_count_sort = order is None and selected_categories is None

    fig, _ax = plt.subplots(figsize=figsize)

    if len(category_cols) == 1:
        freq = obs[first_cat_col].value_counts(dropna=False)

        if use_count_sort:
            freq = freq.sort_values(ascending=ascending)
        else:
            freq = freq.reindex(first_level_order, fill_value=0)
        plot_kwargs: dict[str, Any] = {}
        if color_scheme is not None:
            colors = _resolve_color_scheme(color_scheme, freq.index)
            if colors is not None:
                plot_kwargs["color"] = colors

        freq.plot(kind="bar", ax=_ax, **plot_kwargs)

    elif len(category_cols) == 2:
        second_cat_col = category_cols[1]
        second_level_order = _ordered_categories(obs[second_cat_col])
        df = (
            obs.groupby(category_cols, observed=False)
            .size()
            .unstack(fill_value=0)
        )
        df = df.reindex(first_level_order, fill_value=0)
        df = df.reindex(columns=second_level_order, fill_value=0)
        if use_count_sort:
            df = df.loc[df.sum(axis=1).sort_values(ascending=ascending).index]
        colors = _resolve_color_scheme(color_scheme, df.columns)
        plot_kwargs: dict[str, Any] = {}
        if colors is not None:
            plot_kwargs["color"] = colors
        df.plot(kind="bar", stacked=True, ax=_ax, **plot_kwargs)
        if df.shape[1] > 1:
            _ax.legend(loc="center right", bbox_to_anchor=(1.4, 0.5))
    else:
        raise NotImplementedError(
            "Plotting more than two category columns is not implemented."
        )

    _ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    _ax.set_xlabel(first_cat_col)
    _ax.set_ylabel('#')

    ha = (
        'right' if xlabel_rotation > 0
        else 'left' if xlabel_rotation < 0
        else 'center'
        )
    plt.setp(_ax.get_xticklabels(), rotation=xlabel_rotation, ha=ha)

    fig.tight_layout()

    save_path: Path | None = Path(save) if save is not None else None

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    if ax:
        return _ax

    if not show and save_path is None and not ax:
        warnings.warn(
            "Function does not do anything. Enable `show`, provide a `save` path, "
            "or set `ax=True`."
        )
        plt.close(fig)


def n_cat1_per_cat2_hist(
    adata: ad.AnnData,
    first_category: str,
    second_category: str,
    axis: int,
    bin_width: float | None = None,
    bin_range: tuple[float, float] | None = None,
    figsize: tuple[float, float] = (6.0, 4.0),
    show: bool = True,
    save: str | Path | None = None,
    ax: bool = False,
) -> Axes | None:
    """
    Plot the distribution of the number of first-category entries per second
    category.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    second_category : str
        Column name identifying the primary category. Resolved from
        ``adata.obs`` when ``axis == 0`` and ``adata.var`` when ``axis == 1``.
        Passing ``"index"`` is not supported.
    first_category : str
        Column providing the secondary category from the same axis as
        ``second_category``. Pass ``"index"`` to use ``adata.obs_names`` (``axis
        == 0``) or ``adata.var_names`` (``axis == 1``).
    axis : int
        ``0`` to work on ``adata.obs``, ``1`` to work on ``adata.var``.
    bin_width : float | None
        Optional histogram bin width. Must be positive when provided.
    bin_range : tuple[float, float] | None
        Optional tuple ``(lower, upper)`` limiting the histogram bins. ``lower``
        must be strictly smaller than ``upper``.
    figsize : tuple[float, float]
        Size (width, height) in inches passed to
        :func:`matplotlib.pyplot.subplots`.
    show : bool
        Call :func:`matplotlib.pyplot.show` when ``True``.
    save : str | Path | None
        Save the figure to the provided path when given.
    ax : bool
        Return the :class:`~matplotlib.axes.Axes` instance instead of displaying
        the plot.
    """
    check_proteodata(adata)
    # Ensures that the 'index' has unique values if used

    if axis not in (0, 1):
        raise ValueError("axis must be either 0 (.obs) or 1 (.var).")

    frame = adata.obs if axis == 0 else adata.var
    frame_label = ".obs" if axis == 0 else ".var"

    if second_category == "index":
        raise ValueError(
            "`second_category='index'` is not supported; pass 'index' via "
            "`first_category` instead."
        )
    if second_category not in frame:
        raise KeyError(
            f"Column '{second_category}' not found in adata{frame_label}."
        )
    if first_category != "index" and first_category not in frame:
        raise KeyError(
            f"Column '{first_category}' not found in adata{frame_label}."
        )

    if bin_width is not None:
        if bin_width <= 0:
            raise ValueError("bin_width must be a positive number.")
    if bin_range is not None:
        if (
            not isinstance(bin_range, tuple)
            or len(bin_range) != 2
            or not all(np.isfinite(bin_range))
        ):
            raise TypeError(
                "bin_range must be a tuple of two finite numbers (lower, upper)."
            )
        lower, upper = bin_range
        if lower >= upper:
            raise ValueError("bin_range lower bound must be less than upper bound.")

    temp_col = "__proteopy_axis_index__" if first_category == "index" else first_category
    data = frame[[second_category]].copy()
    if first_category == "index":
        index_values = adata.obs_names if axis == 0 else adata.var_names
        data[temp_col] = index_values
    else:
        data[temp_col] = frame[first_category]
    data = data.drop_duplicates(subset=[second_category, temp_col])
    counts = data.groupby(second_category, observed=False).size()

    if counts.empty:
        raise ValueError(
            "No entries available to compute counts for the requested categories."
        )

    fig, _ax = plt.subplots(figsize=figsize)
    if first_category == "index":
        entry_label = "observations" if axis == 0 else "variables"
    else:
        entry_label = first_category
    sns.histplot(
        counts,
        binwidth=bin_width,
        binrange=bin_range,
        ax=_ax,
    )
    _ax.set_xlabel(f"Number of {entry_label} per {second_category}")
    _ax.set_ylabel(f"# {second_category}")
    fig.tight_layout()

    if save is not None:
        fig.savefig(save, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    if ax:
        return _ax

    if not show and save is None and not ax:
        warnings.warn(
            "Function does not do anything. Enable `show`, provide a `save` path, "
            "or set `ax=True`."
        )
        plt.close(fig)

docstr_header = (
    "Plot the distribution of the number of first-category entries per second category."
    )
n_peptides_per_protein = partial_with_docsig(
    n_cat1_per_cat2_hist,
    first_category="peptide_id",
    second_category="protein_id",
    axis=1,
    docstr_header=docstr_header,
)

n_proteoforms_per_protein = partial_with_docsig(
    n_cat1_per_cat2_hist,
    second_category="proteoform_id",
    first_category="protein_id",
    axis=1,
    docstr_header=docstr_header,
)


def cv_by_group(
    adata: ad.AnnData,
    group_by: str,
    layer: str | None = None,
    zero_to_na: bool = False,
    min_samples: int = None,
    force: bool = False,
    order: list | None = None,
    color_scheme=None,
    alpha: float = 0.8,
    hline: float | None = None,
    show_points: bool = False,
    point_alpha: float = 0.7,
    point_size: float = 1,
    xlabel_rotation: int | float = 0,
    figsize: tuple[float, float] = (6, 4),
    show: bool = True,
    ax: bool = False,
    save: str | None = None,
):
    """
    Compute per-group coefficients of variation and plot their distributions.

    Parameters
    ----------
    adata : AnnData
        AnnData object that contains proteomics quantifications.
    group_by : str
        Column in ``adata.obs`` used to define observation groups for CV
        calculation.
    layer : str | None, optional
        AnnData layer to read intensities from. Defaults to ``adata.X``.
    zero_to_na : bool, optional
        Replace zero values with NaN before computing CVs. Default is ``False``.
    min_samples : int | None, optional
        Minimum number of observations per variable required to compute a CV.
        Variables with fewer non-NaN entries receive NaN. Default is ``3``.
        Ignored when using precomputed CV data from ``adata.varm``.
    force : bool, optional
        Force recomputation of CV values even if precomputed data exists in
        ``adata.varm``. When ``True``, uses a temporary slot that is deleted
        after extracting the data. Default is ``False``.
    order : list | None, optional
        Explicit order of group labels (without the ``cv_`` prefix) along the
        x-axis. When ``None`` the observed group order is used.
    color_scheme : sequence, dict | None, optional
        Color assignments for groups. When None, uses the Matplotlib default
        color cycle.
    alpha : float, optional
        Transparency for the violin bodies. Default is ``0.8``.
    hline : float | None, optional
        If set, draw a horizontal dashed line at this CV value.
    show_points : bool, optional
        Overlay individual variable CVs as a strip plot. Default is ``False``.
    point_alpha : float, optional
        Opacity for individual points when ``show_points`` is ``True``.
    point_size : float, optional
        Size of the individual CV points. Default is ``1``.
    xlabel_rotation : float, optional
        Rotation angle (degrees) for the x-axis group labels.
    figsize : tuple of float, optional
        Matplotlib figure size in inches. Default is ``(6, 4)``.
    show : bool, optional
        Call ``plt.show()`` when ``True``. Default is ``True``.
    ax : bool, optional
        Return the Matplotlib Axes if ``True``.
    save : str | None, optional
        Path to save the figure. When ``None`` the figure is not saved.
    """

    check_proteodata(adata)

    if group_by not in adata.obs.columns:
        raise KeyError(f"Column '{group_by}' not found in adata.obs.")
    if adata.n_obs == 0:
        raise ValueError(
            "AnnData object contains no observations; cannot compute CVs."
        )

    groups = adata.obs[group_by]
    if groups.dropna().empty:
        raise ValueError(
            f"Column '{group_by}' does not contain any non-missing group labels."
        )
    if isinstance(groups.dtype, pd.CategoricalDtype):
        observed_groups = groups.cat.remove_unused_categories().cat.categories
        unique_groups = [str(cat) for cat in observed_groups]
    else:
        unique_groups = pd.Index(groups.astype(str)).unique().tolist()

    if not unique_groups:
        raise ValueError(
            f"Column '{group_by}' does not contain any finite groups."
        )

    # Use existing CV data if available; otherwise compute temporarily
    layer_suffix = sanitize_string(layer) if layer is not None else "X"
    varm_key = f"cv_by_{sanitize_string(group_by)}_{layer_suffix}"

    key_existed = varm_key in adata.varm
    temp_key_name = None

    # Determine whether to use precomputed data or compute new
    use_precomputed = key_existed and not force

    if use_precomputed:
        # Check if min_samples was explicitly provided
        if min_samples:
            raise ValueError(
                f"Cannot use `min_samples={min_samples}` with precomputed CV "
                f"data in adata.varm['{varm_key}']. Either:\n"
                f"  - Use `force=True` to recompute CV values with the new "
                f"`min_samples` setting, or\n"
                f"  - Remove the precomputed data with "
                f"`del adata.varm['{varm_key}']` before calling this function."
            )
        print(f"Using existing CV data from adata.varm['{varm_key}'].")
        key_to_use = varm_key
    else:
        # Random key prevents overwriting existing varm slots
        temp_key_name = f"_temp_cv_{uuid.uuid4().hex[:8]}"
        default_min_samples = 3
        min_samples = min_samples or default_min_samples
        calculate_cv(
            adata,
            group_by=group_by,
            layer=layer,
            zero_to_na=zero_to_na,
            min_samples=min_samples,
            key_added=temp_key_name,
            inplace=True,
        )
        key_to_use = temp_key_name

    if key_to_use not in adata.varm:
        raise RuntimeError(
            f"Failed to compute CV data: adata.varm['{key_to_use}'] not found."
        )

    check_proteodata(adata)

    cv_df = adata.varm[key_to_use].copy()

    # Clean up temporary data immediately after extraction
    if temp_key_name is not None:
        del adata.varm[temp_key_name]

    df_melted = cv_df.melt(var_name="Group", value_name="CV", ignore_index=False)
    df_melted = df_melted.reset_index(drop=True)

    if order is None:
        order = unique_groups
    else:
        missing = [grp for grp in order if grp not in df_melted["Group"].unique()]
        if missing:
            raise ValueError(
                "Requested ordering includes groups with no CV data: "
                f"{', '.join(missing)}."
            )

    resolved_colors = _resolve_color_scheme(color_scheme, order)
    if resolved_colors is None:
        palette = None
    else:
        palette = dict(zip(order, resolved_colors))

    fig, ax_plot = plt.subplots(figsize=figsize, dpi=150)

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

    # Optionally overlay points
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

    # Optional horizontal dashed line
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

    ax_plot.set_xlabel("")
    ax_plot.set_ylabel("Coefficient of Variation (CV)")
    for label in ax_plot.get_xticklabels():
        label.set_rotation(xlabel_rotation)
    ax_plot.set_title("Distribution of CV across groups")
    sns.despine()
    plt.tight_layout()

    check_proteodata(adata)

    if save:
        fig.savefig(save, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save}")

    if show:
        plt.show()

    if ax:
        return ax_plot


def sample_correlation_matrix(
    adata: ad.AnnData,
    method: str = "pearson",
    zero_to_na: bool = False,
    layer: str | None = None,
    fill_na: float | None = None,
    margin_color: str | None = None,
    color_scheme=None,
    cmap: str = "coolwarm",
    linkage_method: str = "average",
    xticklabels: bool = False,
    yticklabels: bool = False,
    figsize: tuple[float, float] = (9.0, 7.0),
    show: bool = True,
    ax: bool = False,
    save: str | Path | None = None,
) -> Axes | None:
    """
    Plot a clustered correlation heatmap across samples (obs).

    Parameters
    ----------
    adata : AnnData
        :class:`~anndata.AnnData` with proteomics annotations.
    method : str
        Correlation estimator passed to :meth:`pandas.DataFrame.corr`.
    zero_to_na : bool
        Replace zeros with missing values before computing correlations.
    layer : str | None
        Optional ``adata.layers`` key to draw quantification values from.
        When ``None`` the primary matrix ``adata.X`` is used.
    fill_na : float | None
        Constant used to replace remaining ``NaN`` values prior to
        correlation. When ``None`` (default), a :class:`ValueError` is raised
        if missing values are detected (suggesting ``fill_na=0``).
    margin_color : str | None
        Optional column in ``adata.obs`` used to color dendrogram labels.
    color_scheme : Any
        Color palette specification understood by
        :func:`proteopy.utils.matplotlib._resolve_color_scheme`.
    cmap : str
        Continuous colormap for the heatmap body.
    linkage_method : str
        Linkage criterion handed to :func:`scipy.cluster.hierarchy.linkage`.
    xticklabels, yticklabels : bool
        Whether to show x- and y-axis tick labels.
    figsize : tuple[float, float]
        Matplotlib figure size in inches.
    show : bool
        Display the figure with :func:`matplotlib.pyplot.show`.
    ax : bool
        Return the heatmap :class:`matplotlib.axes.Axes` when ``True``.
    save : str | Path | None
        File path for saving the Seaborn cluster map. When ``None`` nothing is
        written.

    Returns
    -------
    Axes or None
        Heatmap axes when ``ax`` is ``True``; otherwise ``None``.

    Raises
    ------
    ValueError
        If the selected matrix still contains missing values after optional
        zero replacement and ``fill_na`` is ``None``.
    """
    check_proteodata(adata)
    # ---- values from adata.X or a specified layer (obs × var)
    expected_shape = (adata.n_obs, adata.n_vars)
    if layer is None:
        matrix = adata.X
    else:
        if layer not in adata.layers:
            raise KeyError(f"Layer '{layer}' not found in adata.layers.")
        matrix = adata.layers[layer]

    if matrix is None:
        raise ValueError("Selected matrix is empty; cannot compute correlations.")

    if matrix.shape != expected_shape:
        raise ValueError(
            "Selected matrix shape "
            f"{matrix.shape} does not match adata dimensions {expected_shape}."
        )

    if isinstance(matrix, pd.DataFrame):
        vals = matrix.reindex(index=adata.obs_names, columns=adata.var_names).copy()
    else:
        if sparse.issparse(matrix):
            # correlation requires dense values; convert temporarily
            dense_matrix = matrix.toarray()
        else:
            dense_matrix = np.asarray(matrix)

        vals = pd.DataFrame(
            dense_matrix,
            index=adata.obs_names,
            columns=adata.var_names,
        )
    if zero_to_na:
        vals = vals.replace(0, np.nan)

    if fill_na is not None:
        vals = vals.fillna(fill_na)

    if vals.isna().to_numpy().any():
        raise ValueError(
            "Input matrix contains missing values; provide `fill_na` (e.g., "
            "`fill_na=0`) to replace them before computing correlations."
        )

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

    # ---- optional row/col colors from obs[margin_color]
    row_colors = None
    legend_handles = None
    if margin_color is not None:
        if margin_color not in adata.obs.columns:
            raise KeyError(f"Column '{margin_color}' not found in adata.obs.")
        groups = adata.obs.loc[corr_df.index, margin_color]
        cats = pd.Categorical(groups.dropna()).categories

        resolved_colors = _resolve_color_scheme(color_scheme, cats)
        if resolved_colors is None:
            resolved_colors = (
                sns.color_palette(n_colors=len(cats)) if len(cats) > 0 else []
            )

        palette = {str(cat): color for cat, color in zip(cats, resolved_colors)}

        groups_str = groups.astype("string")
        row_color_series = groups_str.map(palette)

        missing_mask = row_color_series.isna() & groups.notna()
        if missing_mask.any():
            missing_cats = sorted(groups[missing_mask].astype(str).unique())
            raise ValueError(
                "No color provided for categories: "
                f"{', '.join(missing_cats)} in '{margin_color}'."
            )

        legend_handles = [
            Patch(facecolor=palette[str(cat)], edgecolor="none", label=str(cat))
            for cat in cats
        ]

        if groups.isna().any():
            na_color = mpl.colors.to_rgba("lightgray")
            row_color_series = row_color_series.astype(object)
            row_color_series[groups.isna()] = na_color
            legend_handles.append(
                Patch(facecolor=na_color, edgecolor="none", label="NA")
            )

        row_colors = (
            row_color_series.to_numpy() if row_color_series is not None else None
        )

    # ---- hierarchical clustering on (1 - r)
    dist = 1 - corr_df.values
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0, 2)  # numerical guard
    Z = linkage(squareform(dist), method=linkage_method)

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

    # ---- add legend for margin_color colors
    if legend_handles is not None:
        g.ax_heatmap.legend(
            handles=legend_handles,
            title=margin_color,
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

    if save:
        g.savefig(save, dpi=300, bbox_inches="tight")

    if ax:
        return g.ax_heatmap


def hclustv_profiles_heatmap(
    adata: ad.AnnData,
    selected_vars: list[str] | None = None,
    group_by: str | None = None,
    summary_method: str = "median",
    linkage_method: str = "average",
    distance_metric: str = "euclidean",
    layer: str | None = None,
    zero_to_na: bool = False,
    fill_na: float | int | None = None,
    skip_na: bool = True,
    cmap: str = "coolwarm",
    margin_color: bool = False,
    order_by: str | None = None,
    order: str | list | None = None,
    color_scheme: str | dict | Sequence | Colormap | None = None,
    row_cluster: bool = True,
    col_cluster: bool = True,
    xticklabels: bool = True,
    yticklabels: bool = False,
    figsize: tuple[float, float] = (10.0, 8.0),
    title: str | None = None,
    show: bool = True,
    ax: bool = False,
    save: str | Path | None = None,
) -> Axes | None:
    """
    Plot a clustered heatmap of variable profiles across samples or groups.

    Computes z-scores for each variable across samples (or group summaries),
    then applies hierarchical clustering to visualize variable expression
    patterns.

    Parameters
    ----------
    adata : AnnData
        :class:`~anndata.AnnData` with proteomics annotations.
    selected_vars : list[str] | None
        Explicit list of variables to include. When ``None``, all variables
        are used.
    group_by : str | None
        Column in ``adata.obs`` used to group observations. When provided,
        computes a summary statistic for each group rather than showing
        individual samples.
    summary_method : str
        Method for computing group summaries when ``group_by`` is specified.
        One of ``"median"`` or ``"mean"`` (alias ``"average"``).
    linkage_method : str
        Linkage criterion passed to :func:`scipy.cluster.hierarchy.linkage`.
    distance_metric : str
        Distance metric for clustering. One of ``"euclidean"``, ``"manhattan"``,
        or ``"cosine"``.
    layer : str | None
        Optional ``adata.layers`` key to draw quantification values from.
        When ``None`` the primary matrix ``adata.X`` is used.
    zero_to_na : bool
        Replace zeros with ``NaN`` before computing profiles.
    fill_na : float | int | None
        Replace ``NaN`` values with the specified constant.
    skip_na : bool
        Skip ``NaN`` values when computing group summaries and z-scores.
    cmap : str
        Colormap for the heatmap body.
    margin_color : bool
        Add a color bar between the column dendrogram and the heatmap.
        When ``True``, colors by sample (if ``group_by`` is ``None``) or by
        group (if ``group_by`` is set).
    order_by : str | None
        Column in ``adata.obs`` used to order samples (columns). When set,
        automatically disables column clustering and orders columns by the
        values of this column. Also displays a margin color bar colored by
        this column. Cannot be used with ``group_by``.
    order : str | list | None
        The order by which to present samples, groups, or categories. If
        ``order_by`` is ``None`` and ``order`` is ``None``, the existing order
        is used. If ``order_by`` is ``None`` and ``order`` is not ``None``,
        ``order`` specifies the column order (samples or groups). If
        ``order_by`` is not ``None`` and ``order`` is ``None``, the unique
        values in ``order_by`` are used (categorical order if categorical,
        sorted order otherwise). If ``order_by`` is not ``None`` and
        ``order`` is not ``None``, ``order`` defines the order of the unique
        ``order_by`` values. Values not in ``order`` are excluded.
    color_scheme : str | dict | Sequence | Colormap | None
        Palette specification for the margin color bar, forwarded to
        :func:`proteopy.utils.matplotlib._resolve_color_scheme`. Ignored
        when neither ``margin_color`` nor ``order_by`` is set.
    row_cluster : bool
        Perform hierarchical clustering on variables (rows).
    col_cluster : bool
        Perform hierarchical clustering on samples/groups (columns).
    xticklabels : bool
        Show x-axis tick labels (sample/group names).
    yticklabels : bool
        Show y-axis tick labels (variable names).
    figsize : tuple[float, float]
        Matplotlib figure size in inches.
    title : str | None
        Title for the plot.
    show : bool
        Display the figure with :func:`matplotlib.pyplot.show`.
    ax : bool
        Return the heatmap :class:`matplotlib.axes.Axes` when ``True``.
    save : str | Path | None
        File path for saving the figure.

    Returns
    -------
    Axes or None
        Heatmap axes when ``ax`` is ``True``; otherwise ``None``.
    """
    check_proteodata(adata)

    # Validate summary_method
    summary_method = summary_method.lower()
    if summary_method == "average":
        summary_method = "mean"
    if summary_method not in ("median", "mean"):
        raise ValueError(
            f"summary_method must be 'median' or 'mean', got '{summary_method}'."
        )

    # Validate distance_metric
    distance_metric = distance_metric.lower()
    if distance_metric not in ("euclidean", "manhattan", "cosine"):
        raise ValueError(
            f"distance_metric must be 'euclidean', 'manhattan', or 'cosine', "
            f"got '{distance_metric}'."
        )

    # Map metric names to scipy pdist names
    metric_map = {
        "euclidean": "euclidean",
        "manhattan": "cityblock",
        "cosine": "cosine",
    }
    scipy_metric = metric_map[distance_metric]

    # Validate order_by
    if order_by is not None:
        if group_by is not None:
            raise ValueError(
                "order_by cannot be used with group_by. When using group_by, "
                "columns represent groups, not individual samples."
            )
        if order_by not in adata.obs.columns:
            raise KeyError(f"Column '{order_by}' not found in adata.obs.")
        # order_by and col_cluster are mutually exclusive; disable clustering
        if col_cluster:
            print((
                "`order_by` parameter is incompatible with `col_cluster=True`. "
                "`col_cluster` has been overridden."
            ))
            col_cluster = False

    # Validate order parameter
    if order is not None:
        if col_cluster:
            print((
                "`order` parameter is incompatible with `col_cluster=True`. "
                "`col_cluster` has been overridden."
            ))
            col_cluster = False
        order = list(order)
        if order_by is None and group_by is None:
            # order specifies sample names
            available_samples = set(adata.obs_names)
            invalid_samples = [s for s in order if s not in available_samples]
            if invalid_samples:
                raise KeyError(
                    f"Samples not found in adata.obs_names: {invalid_samples}"
                )
        elif group_by is not None:
            # order specifies group names; validate against group_by column
            available_groups = set(adata.obs[group_by].dropna().unique())
            invalid_groups = [g for g in order if g not in available_groups]
            if invalid_groups:
                raise KeyError(
                    f"Groups not found in adata.obs['{group_by}']: {invalid_groups}"
                )
        # Validation for order_by case is done after we have the data

    # Extract matrix
    if layer is None:
        matrix = adata.X
    else:
        if layer not in adata.layers:
            raise KeyError(f"Layer '{layer}' not found in adata.layers.")
        matrix = adata.layers[layer]

    if matrix is None:
        raise ValueError("Selected matrix is empty.")

    # Densify if sparse
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    else:
        matrix = np.asarray(matrix)

    # Create DataFrame (obs x var)
    df = pd.DataFrame(
        matrix,
        index=adata.obs_names,
        columns=adata.var_names,
    )

    # Filter variables if specified
    if selected_vars is not None:
        missing_vars = [v for v in selected_vars if v not in df.columns]
        if missing_vars:
            raise KeyError(
                f"Variables not found in adata.var_names: {missing_vars}"
            )
        df = df[selected_vars]

    if zero_to_na:
        df = df.replace(0, np.nan)

    if fill_na is not None:
        df = df.fillna(fill_na)

    # Group by if specified
    if group_by is not None:
        if group_by not in adata.obs.columns:
            raise KeyError(f"Column '{group_by}' not found in adata.obs.")
        groups = adata.obs[group_by]
        df["__group__"] = groups.values

        # Compute group summaries
        # include_groups=False excludes __group__ from the lambda input
        if summary_method == "median":
            summary_df = df.groupby("__group__", observed=True).apply(
                lambda x: x.median(skipna=skip_na),
                include_groups=False,
            )
        else:
            summary_df = df.groupby("__group__", observed=True).apply(
                lambda x: x.mean(skipna=skip_na),
                include_groups=False,
            )

        # Transpose to get var x group
        profile_df = summary_df.T
    else:
        # Transpose to get var x obs
        profile_df = df.T

    # Drop variables with all NaN
    profile_df = profile_df.dropna(how="all")

    if profile_df.empty:
        raise ValueError("No variables remain after removing all-NaN rows.")

    # Compute z-scores per variable (row)
    row_means = profile_df.mean(axis=1, skipna=skip_na)
    row_stds = profile_df.std(axis=1, skipna=skip_na, ddof=0)
    row_stds = row_stds.replace(0, np.nan)  # avoid division by zero

    z_df = profile_df.sub(row_means, axis=0).div(row_stds, axis=0)

    # Fill NaN with 0 for clustering
    z_df_filled = z_df.fillna(0)

    # Order columns based on order_by and/or order
    if order_by is not None:
        # Get order based on obs column values
        order_col_values = adata.obs.loc[z_df_filled.columns, order_by]
        if order is not None:
            # Validate that order values exist in the order_by column
            available_values = set(order_col_values.unique())
            invalid_values = [v for v in order if v not in available_values]
            if invalid_values:
                raise KeyError(
                    f"Values not found in adata.obs['{order_by}']: {invalid_values}"
                )
            # Filter to samples whose order_by value is in order, then sort
            mask = order_col_values.isin(order)
            filtered_cols = z_df_filled.columns[mask]
            order_col_values = order_col_values.loc[filtered_cols]
            # Create categorical with specified order for sorting
            order_col_values = pd.Categorical(
                order_col_values,
                categories=order,
                ordered=True,
            )
            sorted_idx = (
                pd.Series(order_col_values, index=filtered_cols)
                .sort_values().index
                )
        else:
            # Use categorical order if categorical, sorted order otherwise
            if isinstance(order_col_values.dtype, pd.CategoricalDtype):
                cat_order = list(order_col_values.cat.categories)
                order_col_values = pd.Categorical(
                    order_col_values,
                    categories=cat_order,
                    ordered=True,
                )
                sorted_idx = pd.Series(
                    order_col_values,
                    index=z_df_filled.columns,
                ).sort_values().index
            else:
                sorted_idx = order_col_values.sort_values().index
        z_df_filled = z_df_filled[sorted_idx]
    elif order is not None:
        # order specifies sample or group names directly
        # Filter to only columns in order, maintaining order
        valid_cols = [c for c in order if c in z_df_filled.columns]
        z_df_filled = z_df_filled[valid_cols]

    # Build column colors for margin annotation
    col_colors = None
    col_names = z_df_filled.columns

    if order_by is not None:
        # Color by the order_by column
        categories = adata.obs.loc[col_names, order_by].values
    elif margin_color:
        # Color by sample or group
        categories = col_names
    else:
        categories = None

    if categories is not None:
        # Create color palette
        unique_cats = pd.Series(categories).unique()
        resolved_colors = _resolve_color_scheme(color_scheme, unique_cats)
        if resolved_colors is None:
            resolved_colors = (
                sns.color_palette("husl", n_colors=len(unique_cats))
                if len(unique_cats) > 0 else []
            )
        color_map = dict(zip(unique_cats, resolved_colors))
        col_colors = pd.Series(
            [color_map[c] for c in categories],
            index=col_names,
        )

    # Create clustermap
    g = sns.clustermap(
        z_df_filled,
        method=linkage_method,
        metric=scipy_metric,
        row_cluster=row_cluster,
        col_cluster=col_cluster,
        cmap=cmap,
        center=0,
        figsize=figsize,
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        cbar_kws={"label": "Z-score"},
        col_colors=col_colors,
    )

    g.ax_heatmap.set_xlabel("")

    # Remove y-axis ticks from the margin color bar if present
    if g.ax_col_colors is not None:
        g.ax_col_colors.set_yticks([])

    if title is not None:
        g.figure.suptitle(title, y=1.02)

    plt.tight_layout()

    if save:
        g.savefig(save, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    if ax:
        return g.ax_heatmap

    return None
