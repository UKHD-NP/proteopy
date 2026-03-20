from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns
import anndata as ad

from proteopy.utils.anndata import check_proteodata
from proteopy.utils.parsers import parse_stat_test_varm_slot
from proteopy.utils.matplotlib import _resolve_color_scheme
from proteopy.utils.stat_tests import (
    volcano_plot as _volcano_plot,
)


def _stat_test_title_from_varm_slot(
    adata: ad.AnnData,
    varm_slot: str,
) -> str:
    """
    Generate a human-readable plot title from a stat test varm slot.

    Parses the varm slot name to extract test type, group_by, design
    (group comparison), and optional layer information, then formats them
    into a concise title string. If parsing fails, returns the original
    slot name with a runtime warning.

    Returns
    -------
    str
        Formatted title string. Format:
        ``<Test Label> | <group_by>: <Design>`` or
        ``<Test Label> | <group_by>: <Design> | layer: <layer>``
        if a layer was specified. Returns the original ``varm_slot``
        unchanged if parsing fails.
    """
    try:
        parsed = parse_stat_test_varm_slot(varm_slot, adata=adata)
    except ValueError as exc:
        warnings.warn(
            f"Could not parse varm slot '{varm_slot}': {exc}",
            RuntimeWarning,
        )
        return varm_slot

    title = (
        f"{parsed['test_type_label']} | "
        f"{parsed['group_by']}: {parsed['design_label']}"
    )
    if parsed["layer"]:
        title = f"{title} | layer: {parsed['layer']}"
    return title


def _normalize_alt_color(
    alt_color: pd.Series | list[bool] | np.ndarray,
    adata: ad.AnnData,
    plot_index: pd.Index,
) -> pd.Series:
    """
    Validate and align alternative color boolean mask to plot data.

    Converts the user-provided ``alt_color`` input into a boolean
    Series indexed by ``adata.var_names``, then reindexes to match
    the plotting data (which may be a subset after filtering for
    finite values and positive p-values). This ensures the mask
    correctly aligns with proteins that will appear in the volcano
    plot.

    Parameters
    ----------
    alt_color : pd.Series | list[bool] | np.ndarray
        Boolean mask of length ``n_vars`` indicating which proteins
        should receive alternative coloring. Accepts a pandas
        Series (optionally indexed), a list, or a numpy array.
        Series indices are ignored and values are aligned to
        ``adata.var_names`` positionally.
    adata : ad.AnnData
        AnnData object used to determine expected length
        (``n_vars``) and indexing (``var_names``).
    plot_index : pd.Index
        Index of proteins remaining after filtering (subset of
        ``adata.var_names``). Used to align ``alt_color`` to the
        plotting subset.

    Returns
    -------
    pd.Series
        Boolean Series indexed by ``plot_index``, indicating which
        proteins should be colored with the alternative scheme.

    Raises
    ------
    ValueError
        If ``alt_color`` length does not match ``adata.n_vars``, is
        not boolean dtype, is not 1D, or contains missing values
        after reindexing.
    TypeError
        If ``alt_color`` is not a pandas Series, list, or numpy
        array.
    """
    if isinstance(alt_color, pd.Series):
        if len(alt_color) != adata.n_vars:
            raise ValueError(
                "alt_color must have length matching adata.n_vars."
            )
        if not pd.api.types.is_bool_dtype(alt_color):
            raise ValueError("alt_color must be boolean.")
        if alt_color.index.equals(adata.var_names):
            series = alt_color.copy()
        else:
            series = pd.Series(
                alt_color.to_numpy(),
                index=adata.var_names,
            )
    elif isinstance(alt_color, (list, np.ndarray)):
        values = np.asarray(alt_color)
        if values.ndim != 1:
            raise ValueError("alt_color must be a 1D boolean sequence.")
        if len(values) != adata.n_vars:
            raise ValueError(
                "alt_color must have length matching adata.n_vars."
            )
        if not np.issubdtype(values.dtype, np.bool_):
            raise ValueError("alt_color must be boolean.")
        series = pd.Series(values, index=adata.var_names)
    else:
        raise TypeError(
            "alt_color must be a pandas Series, list, or numpy array."
        )

    # Align to plot subset (may introduce NaNs if plot_index contains
    # proteins not in adata.var_names, which indicates a bug)
    series = series.reindex(plot_index)
    if series.isna().any():
        raise ValueError(
            "alt_color contains missing values after aligning to "
            "varm data."
        )
    return series


def _validate_thresholds(fc_thresh, pval_thresh):
    if (
        not isinstance(fc_thresh, (int, float))
        or fc_thresh <= 0
    ):
        raise ValueError(
            "fc_thresh must be a positive number."
        )
    if (
        not isinstance(pval_thresh, (int, float))
        or pval_thresh <= 0
        or pval_thresh > 1
    ):
        raise ValueError(
            "pval_thresh must be a number in (0, 1]."
        )


def _validate_labels(top_labels, highlight_labels):
    if top_labels is not None:
        if (
            not isinstance(top_labels, int)
            or top_labels < 0
        ):
            raise ValueError(
                "top_labels must be a non-negative integer."
            )
    if highlight_labels is not None:
        if not isinstance(highlight_labels, list):
            raise TypeError(
                "highlight_labels must be a list of strings."
            )
    if (
        top_labels is not None
        and highlight_labels is not None
    ):
        raise ValueError(
            "top_labels and highlight_labels are mutually "
            "exclusive."
        )


def _validate_figsize(figsize):
    if (
        not isinstance(figsize, (tuple, list))
        or len(figsize) != 2
        or not all(
            isinstance(v, (int, float)) and v > 0
            for v in figsize
        )
    ):
        raise ValueError(
            "figsize must be a tuple/list of 2 positive "
            "numbers."
        )


def _validate_volcano_inputs(
    adata: ad.AnnData,
    varm_slot: str,
    fc_col: str,
    pval_col: str,
    alt_labels_key: str | None,
    fc_thresh: float,
    pval_thresh: float,
    top_labels: int | None,
    highlight_labels: list[str] | None,
    figsize: tuple[float, float],
    yscale_log: bool,
) -> tuple[pd.DataFrame, str]:
    """
    Validate all input parameters for the volcano plot function.

    Checks the AnnData object, threshold values, label arguments,
    figure size, y-axis scale type, and the required columns in the
    varm slot. Resolves the p-value column name by preferring
    ``pval_col`` and falling back to ``"pval"`` when the adjusted
    column is absent.

    Parameters
    ----------
    adata : ad.AnnData
        :class:`~anndata.AnnData` containing differential abundance
        test results in ``.varm``.
    varm_slot : str
        Key in ``adata.varm`` holding the statistical test results
        as a :class:`pandas.DataFrame`.
    fc_col : str
        Column name in the varm DataFrame for log fold change values.
    pval_col : str
        Preferred column name in the varm DataFrame for p-values
        (typically adjusted). Falls back to ``"pval"`` if absent.
    alt_labels_key : str | None
        Column in ``adata.var`` used as alternative labels for
        rendering on the plot. ``None`` uses ``adata.var_names``.
    fc_thresh : float
        Absolute log fold change threshold; must be a positive number.
    pval_thresh : float
        P-value significance threshold; must be in the range (0, 1].
    top_labels : int | None
        Number of top significant proteins to label per fold-change
        direction. Must be a non-negative integer or ``None``.
        Mutually exclusive with ``highlight_labels``.
    highlight_labels : list[str] | None
        Explicit list of protein label strings to annotate.
        Must be a :class:`list` or ``None``. Mutually exclusive
        with ``top_labels``.
    figsize : tuple[float, float]
        Figure dimensions (width, height) in inches; both values
        must be positive numbers.
    yscale_log : bool
        Controls y-axis p-value representation. Must be a
        :class:`bool`.

    Returns
    -------
    tuple[pd.DataFrame, str]
        A 2-tuple of:

        - **results** (:class:`pandas.DataFrame`) — the validated
          varm slot DataFrame indexed by variable names.
        - **pval_col_used** (:class:`str`) — the resolved p-value
          column name (``pval_col`` if present, otherwise ``"pval"``).

    Raises
    ------
    KeyError
        If ``varm_slot`` is not found in ``adata.varm``, ``fc_col``
        is not a column in the varm DataFrame, neither ``pval_col``
        nor ``"pval"`` is a column in the varm DataFrame, or
        ``alt_labels_key`` is not a column in ``adata.var``.
    TypeError
        If ``adata.varm[varm_slot]`` is not a
        :class:`pandas.DataFrame`, or if ``highlight_labels`` is not
        a :class:`list`, or if ``yscale_log`` is not a :class:`bool`.
    ValueError
        If ``fc_thresh`` is not a positive number, ``pval_thresh``
        is not in (0, 1], ``figsize`` is not a 2-element
        tuple/list of positive numbers, ``top_labels`` is not a
        non-negative integer, or ``top_labels`` and
        ``highlight_labels`` are both non-``None``.
    """
    check_proteodata(adata)

    # -- Validate scalar parameters
    _validate_thresholds(fc_thresh, pval_thresh)
    _validate_labels(top_labels, highlight_labels)
    _validate_figsize(figsize)

    if not isinstance(yscale_log, bool):
        raise TypeError("yscale_log must be a bool.")

    # -- Validate varm slot exists and contains a DataFrame
    if varm_slot not in adata.varm:
        raise KeyError(
            f"varm_slot '{varm_slot}' not found in "
            f"adata.varm."
        )

    results = adata.varm[varm_slot]
    if not isinstance(results, pd.DataFrame):
        raise TypeError(
            "Expected adata.varm[varm_slot] to be a pandas "
            "DataFrame."
        )

    # -- Validate required columns exist
    if fc_col not in results.columns:
        raise KeyError(
            f"Column '{fc_col}' not found in varm slot "
            f"'{varm_slot}'."
        )

    # Prioritize adjusted p-values, fall back to unadjusted
    # if needed. This allows the function to work with test
    # results that may not include multiple testing correction.
    if pval_col in results.columns:
        pval_col_used = pval_col
    elif "pval" in results.columns:
        pval_col_used = "pval"
    else:
        raise KeyError(
            f"Columns '{pval_col}' or 'pval' not found in "
            f"'{varm_slot}'."
        )

    if (
        alt_labels_key is not None
        and alt_labels_key not in adata.var.columns
    ):
        raise KeyError(
            f"alt_labels_key '{alt_labels_key}' not found "
            f"in adata.var."
        )

    return results, pval_col_used


def volcano(
    adata: ad.AnnData,
    varm_slot: str,
    fc_thresh: float = 1.0,
    pval_thresh: float = 0.05,
    *,
    fc_col: str = "logfc",
    pval_col: str = "pval_adj",
    alt_labels_key: str | None = None,
    top_labels: int | None = None,
    highlight_labels: list[str] | None = None,
    figsize: tuple[float, float] = (6.0, 5.0),
    xlabel: str | None = None,
    ylabel: str | None = None,
    alt_color: pd.Series | list[bool] | np.ndarray | None = None,
    yscale_log: bool = True,
    title: str | None = None,
    show: bool = True,
    save: str | Path | None = None,
    ax: Axes | None = None,
) -> Axes:
    """
    Visualize differential abundance results as a volcano plot.

    Creates a scatter plot of log fold change (x-axis) versus p-value
    (y-axis) for proteins from a statistical test stored in
    ``adata.varm``.

    Parameters
    ----------
    adata : ad.AnnData
        :class:`~anndata.AnnData` containing differential abundance
        test results in ``.varm``.
    varm_slot : str
        Key in ``adata.varm`` containing the differential abundance
        test results as a DataFrame. Expected format produced by
        ``copro.tl.differential_abundance``.
    fc_thresh : float, optional
        Absolute log fold change threshold for significance. Proteins
        with ``|logfc| >= fc_thresh`` and ``pval <= pval_thresh`` are
        highlighted as significant.
    pval_thresh : float, optional
        P-value threshold for significance. Used in conjunction with
        ``fc_thresh`` to identify significant proteins.
    fc_col : str, optional
        Column name in the varm DataFrame containing log fold change
        values. Log base depends on the test method used.
    pval_col : str, optional
        Column name in the varm DataFrame containing adjusted p-values.
        If this column is not found, the function falls back to
        ``"pval"`` (unadjusted p-values).
    alt_labels_key : str | None, optional
        Column in ``adata.var`` used as alternative labels for
        rendering on the plot. When set, ``highlight_labels``
        accepts elements from this column. ``None`` uses
        ``adata.var_names``.
    top_labels : int | None, optional
        Number of top proteins to label on each side of the volcano
        plot (up to 0.5N labels total). For each direction (positive
        and negative fold change), selects the top N proteins that
        meet BOTH significance thresholds (``pval <= pval_thresh``
        AND ``|logfc| >= fc_thresh``). Proteins are ranked first by
        smallest p-value, then by largest absolute fold change.
        ``None`` disables automatic labeling.
    highlight_labels : list[str] | None, optional
        Sequence of protein label strings to annotate on the plot.
        Each entry must match a value in the label column
        (``alt_labels_key`` or ``adata.var_names``). Mutually exclusive
        with ``top_labels``. Labels not found after data filtering
        trigger a warning.
    figsize : tuple[float, float], optional
        Figure dimensions (width, height) in inches.
    xlabel : str | None, optional
        Label for the x-axis. Defaults to the value of ``fc_col`` if
        ``None``.
    ylabel : str | None, optional
        Label for the y-axis. When ``None``, defaults to the
        p-value column name (e.g., ``"pval_adj"``) if
        ``yscale_log=True``, or ``"-log10(pval_adj)"`` if
        ``yscale_log=False``.
    alt_color : pd.Series | list[bool] | np.ndarray | None, optional
        Boolean mask (length ``n_vars``) for alternative coloring
        scheme. When provided, this COMPLETELY OVERRIDES the default
        significance-based coloring: proteins with ``True`` are
        colored light purple (#8E54E5), proteins with ``False`` are
        colored gray (#808080). Significance thresholds (``fc_thresh``
        and ``pval_thresh``) are still visualized as dashed lines
        but do not influence point colors. ``None`` uses the default
        significance-based coloring (gray, blue, red).
    yscale_log : bool, optional
        Controls y-axis representation of p-values. When ``True``,
        plot raw p-values on a log10-scaled y-axis (inverted so
        smaller p-values appear higher). When ``False``, apply
        ``-log10`` transform to p-values and plot on a linear
        y-axis.
    title : str | None, optional
        Plot title. If ``None``, generates a title from the
        ``varm_slot`` name using the stat test metadata (test type,
        group comparison, layer).
    show : bool, optional
        Call ``matplotlib.pyplot.show()`` to display the plot.
    save : str | Path | None, optional
        File path to save the figure.
    ax : matplotlib.axes.Axes | None, optional
        Matplotlib Axes object to plot onto.

    Returns
    -------
    Axes
        The Matplotlib Axes object used for plotting.

    Raises
    ------
    KeyError
        If ``varm_slot`` is not in ``adata.varm``, ``fc_col`` or
        p-value columns are not in the varm DataFrame, or
        ``alt_labels_key`` is not in ``adata.var``.
    TypeError
        If ``adata.varm[varm_slot]`` is not a pandas DataFrame,
        ``highlight_labels`` is not a :class:`list`, or
        ``yscale_log`` is not a :class:`bool`.
    ValueError
        If ``fc_thresh`` is not a positive number, ``pval_thresh``
        is not in (0, 1], ``figsize`` is not a 2-element
        tuple/list of positive numbers, ``top_labels`` is not a
        non-negative integer, ``top_labels`` and
        ``highlight_labels`` are both non-``None``, or no valid
        (finite, positive p-value) results remain after filtering.

    Notes
    -----
    **Data Filtering**:
    Proteins are filtered before plotting to remove:
    - Missing values in fold change or p-value columns
    - Non-finite values (inf, -inf, nan)
    - Non-positive p-values

    Examples
    --------
    Plot differential abundance results with default settings:

    >>> pp.pl.volcano_plot(
    ...     adata,
    ...     varm_slot="welch;condition;treatment_vs_ctrl",
    ... )

    Label top 10 proteins per side and save to file:

    >>> pp.pl.volcano_plot(
    ...     adata,
    ...     varm_slot="welch;condition;treatment_vs_ctrl",
    ...     top_labels=10,
    ...     save="volcano.png",
    ... )

    Use custom coloring to highlight proteins of interest:

    >>> proteins_of_interest = adata.var["protein_id"].isin(
    ...     ["P12345", "Q67890"]
    ... )
    >>> pp.pl.volcano_plot(
    ...     adata,
    ...     varm_slot="welch;condition;treatment_vs_ctrl",
    ...     alt_color=proteins_of_interest,
    ... )
    """
    results, pval_col_used = _validate_volcano_inputs(
        adata, varm_slot, fc_col, pval_col, alt_labels_key,
        fc_thresh, pval_thresh, top_labels,
        highlight_labels, figsize, yscale_log,
    )

    fc_arr = results[fc_col].to_numpy()
    pvals_arr = results[pval_col_used].to_numpy()

    labels = None
    if top_labels is not None or highlight_labels is not None:
        if alt_labels_key is None:
            labels = adata.var_names.to_numpy()
        else:
            labels = adata.var[alt_labels_key].to_numpy()

    # Normalize alt_color via existing helper
    alt_arr = None
    if alt_color is not None:
        alt_series = _normalize_alt_color(
            alt_color, adata, results.index,
        )
        alt_arr = alt_series.to_numpy()

    if title is None:
        title = _stat_test_title_from_varm_slot(
            adata, varm_slot,
        )

    if ylabel is None:
        if yscale_log:
            ylabel = pval_col_used
        else:
            ylabel = f"-log10 {pval_col_used}"

    return _volcano_plot(
        fc_vals=fc_arr,
        pvals=pvals_arr,
        fc_thresh=fc_thresh,
        pval_thresh=pval_thresh,
        labels=labels,
        top_labels=top_labels,
        highlight_labels=highlight_labels,
        figsize=figsize,
        xlabel=fc_col if xlabel is None else xlabel,
        ylabel=ylabel,
        alt_color=alt_arr,
        yscale_log=yscale_log,
        title=title,
        show=show,
        save=save,
        ax=ax,
    )


def _validate_varm_slot(adata, varm_slot):
    if varm_slot not in adata.varm:
        raise KeyError(
            f"varm_slot '{varm_slot}' not found in "
            f"adata.varm."
        )
    results = adata.varm[varm_slot]
    if not isinstance(results, pd.DataFrame):
        raise TypeError(
            "Expected adata.varm[varm_slot] to be a "
            "pandas DataFrame."
        )
    return results


def _resolve_pval_column(results, varm_slot):
    if "pval_adj" in results.columns:
        return "pval_adj"
    if "pval" in results.columns:
        return "pval"
    raise KeyError(
        f"Neither 'pval_adj' nor 'pval' found in "
        f"varm slot '{varm_slot}'."
    )


def _validate_diff_abundance_inputs(
    adata,
    varm_slot,
    top_n,
    layer,
    show_pval,
    pval_fontsize,
):
    """
    Validate inputs for :func:`differential_abundance_box`.

    Checks the varm slot, ``top_n``, ``group_by`` column,
    layer, and ``pval_fontsize``. Parses the varm slot to
    resolve ``group_by`` and layer metadata.

    Parameters
    ----------
    adata : AnnData
        :class:`~anndata.AnnData`
    varm_slot : str
        Key in ``adata.varm`` holding a results DataFrame.
    top_n : int | None
        Number of top variables; defaults to 10 if ``None``.
    layer : str | None
        Layer key; auto-detected from varm metadata if
        ``None``.
    show_pval : bool
        Whether p-value annotations will be shown.
    pval_fontsize : float | int
        Font size for p-value annotations.

    Returns
    -------
    tuple
        ``(results, group_by, layer, pval_col, top_n)``
        where *results* is the varm DataFrame, *group_by*
        is the parsed grouping column, *layer* is the
        resolved layer key (or ``None``), *pval_col* is the
        resolved p-value column name, and *top_n* is the
        validated integer count.
    """
    check_proteodata(adata)

    results = _validate_varm_slot(adata, varm_slot)

    # -- Validate top_n
    if top_n is None:
        top_n = 10
    if not isinstance(top_n, int) or top_n <= 0:
        raise ValueError(
            "top_n must be a positive integer."
        )

    # -- Parse varm slot metadata
    parsed = parse_stat_test_varm_slot(
        varm_slot, adata=adata,
    )
    group_by = parsed["group_by"]

    if group_by not in adata.obs.columns:
        raise KeyError(
            f"Column '{group_by}' not found in "
            f"adata.obs."
        )

    # -- Resolve layer
    if layer is None:
        layer = parsed["layer"]
    if layer is not None and layer not in adata.layers:
        raise KeyError(
            f"Layer '{layer}' not found in "
            f"adata.layers."
        )

    pval_col = _resolve_pval_column(results, varm_slot)

    # -- Validate pval_fontsize
    if show_pval:
        if (
            not isinstance(
                pval_fontsize, (int, float),
            )
            or pval_fontsize <= 0
        ):
            raise ValueError(
                "pval_fontsize must be a positive "
                "number."
            )

    return results, group_by, layer, pval_col, top_n


def _prepare_diff_abundance_data(
    adata,
    results,
    top_n,
    pval_col,
    group_by,
    layer,
    order,
    show_pval,
):
    """
    Prepare long-format DataFrame for boxplot rendering.

    Sorts variables by p-value, extracts intensities for
    the top N variables, melts into long format, applies
    group ordering/filtering, and optionally extracts
    per-variable p-values for annotation.

    Parameters
    ----------
    adata : AnnData
        :class:`~anndata.AnnData`
    results : pd.DataFrame
        Varm results DataFrame with p-value column.
    top_n : int
        Number of top variables to include.
    pval_col : str
        Column name in *results* for p-values.
    group_by : str
        Column in ``adata.obs`` defining groups.
    layer : str | None
        Layer key for intensity data, or ``None`` for
        ``.X``.
    order : list[str] | None
        Desired group order/subset.
    show_pval : bool
        Whether to extract p-values for annotation.

    Returns
    -------
    tuple
        ``(df_long, top_vars, group_order, pvals_to_plot)``
        where *df_long* is the melted DataFrame,
        *top_vars* is the ordered list of variable names,
        *group_order* is the resolved group sequence, and
        *pvals_to_plot* is a Series of p-values (or
        ``None`` if ``show_pval`` is ``False``).
    """
    # -- Sort by p-value and select top N
    results_sorted = results.sort_values(
        by=pval_col, ascending=True,
    )
    top_vars = results_sorted.head(top_n).index.tolist()

    if not top_vars:
        raise ValueError(
            "No valid variables found after filtering."
        )

    pvals_to_plot = None
    if show_pval:
        pvals_to_plot = results.loc[
            top_vars, pval_col
        ].reindex(top_vars)

    # -- Extract intensity matrix
    if layer is not None:
        X = adata[:, top_vars].layers[layer]
    else:
        X = adata[:, top_vars].X

    if sparse.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=float)

    # -- Build long-format DataFrame
    df = pd.DataFrame(
        X,
        index=adata.obs_names,
        columns=top_vars,
    )
    df[group_by] = adata.obs[group_by].values
    df_long = df.melt(
        id_vars=[group_by],
        var_name="variable",
        value_name="intensity",
    )

    # -- Apply group ordering / filtering
    if order is not None:
        available_groups = set(
            df_long[group_by].unique(),
        )
        missing_in_data = set(order) - available_groups
        if missing_in_data:
            raise ValueError(
                f"Groups in 'order' not found in data: "
                f"{sorted(missing_in_data)}. "
                f"Available groups: "
                f"{sorted(available_groups)}"
            )
        df_long = df_long[
            df_long[group_by].isin(order)
        ]
        group_order = list(order)
    else:
        group_order = (
            df_long[group_by].dropna().unique().tolist()
        )

    if df_long.empty:
        raise ValueError(
            "No data remaining after filtering."
        )

    # -- Set variable categorical order (by significance)
    df_long["variable"] = pd.Categorical(
        df_long["variable"],
        categories=top_vars,
        ordered=True,
    )

    return df_long, top_vars, group_order, pvals_to_plot


def _annotate_boxplot_pvals(
    _ax,
    df_long,
    pvals_to_plot,
    top_vars,
    pval_fontsize,
):
    """
    Add per-variable p-value text annotations to boxplot axes.

    Computes a uniform label y-position above the data
    range and annotates each variable's p-value centered
    on its x-tick position.

    Parameters
    ----------
    _ax : matplotlib.axes.Axes
        Axes object containing the boxplots.
    df_long : pd.DataFrame
        Long-format DataFrame with an ``"intensity"``
        column used to determine the data range.
    pvals_to_plot : pd.Series
        Series mapping variable names to p-values.
    top_vars : list[str]
        Ordered list of variable names matching the
        x-tick positions.
    pval_fontsize : float | int
        Font size for the annotation text.
    """
    all_intensity = df_long["intensity"].to_numpy()
    finite_intensity = all_intensity[
        np.isfinite(all_intensity)
    ]
    data_max = (
        float(finite_intensity.max())
        if finite_intensity.size
        else 0.0
    )
    y_min, y_max = _ax.get_ylim()
    span = y_max - y_min if y_max != y_min else 1.0
    label_y = data_max + 0.05 * span
    needed_y_max = label_y + 2.5 * (0.05 * span)
    if needed_y_max > y_max:
        _ax.set_ylim(y_min, needed_y_max)

    for x_pos, var in zip(_ax.get_xticks(), top_vars):
        pval = pvals_to_plot.get(var)
        if pd.isna(pval):
            continue
        _ax.text(
            x_pos,
            label_y,
            f"p={pval:.2e}",
            ha="center",
            va="bottom",
            fontsize=pval_fontsize,
        )


def differential_abundance_box(
    adata: ad.AnnData,
    varm_slot: str,
    order: list[str] | None = None,
    top_n: int | None = None,
    layer: str | None = None,
    verbose: bool = False,
    figsize: tuple[float, float] = (10.0, 6.0),
    title: str | None = None,
    xlabel_rotation: float = 45,
    color_scheme: str | dict | list | None = None,
    show_pval: bool = True,
    pval_fontsize: float | int = 9,
    show: bool = True,
    save: str | Path | None = None,
    ax: bool | None = None,
) -> Axes | None:
    """
    Display boxplots of intensities for top differentially abundant variables.

    For each of the top N differentially abundant variables (sorted by
    p-value), shows side-by-side boxplots comparing intensities across
    groups defined by the statistical test and annotates the per-variable
    p-values.

    Parameters
    ----------
    adata : ad.AnnData
        :class:`~anndata.AnnData` containing differential abundance
        test results in ``.varm`` and expression data in ``.X`` or
        a specified layer.
    varm_slot : str
        Key in ``adata.varm`` containing the differential abundance
        test results. Expected format produced by
        ``proteopy.tl.differential_abundance``.
    order : list[str] | None, optional
        Order and/or subset groups within each variable's boxplots. When
        ``None``, uses all groups as they appear in the data. When
        provided, only groups listed in ``order`` are shown, in the
        given sequence. All values must exist in the ``group_by`` column.
    top_n : int, optional
        Number of top differentially abundant variables to display.
        Variables are ranked by ascending p-value (most significant
        first). Defaults to 10.
    layer : str | None, optional
        Key in ``adata.layers`` providing the intensity matrix. When
        ``None``, auto-detects the layer from the statistical test
        metadata.
    verbose : bool, optional
        If ``True``, print which layer or ``.X`` matrix is used for
        intensity data.
    figsize : tuple[float, float], optional
        Figure dimensions (width, height) in inches.
    title : str | None, optional
        Plot title. If ``None``, generates a title from the
        ``varm_slot`` name.
    xlabel_rotation : float, optional
        Rotation angle (degrees) for x-axis variable labels.
    color_scheme : str | dict | list | None, optional
        Color mapping for groups. Accepts a named Matplotlib colormap,
        a dict mapping group names to colors, or a list of colors.
        If ``None``, uses the default color cycle.
    show_pval : bool, optional
        If ``True``, annotate the per-variable p-values above the
        boxplots. If ``False``, omit the annotations.
    pval_fontsize : float | int, optional
        Font size for the p-value annotations when ``show_pval`` is
        ``True``.
    show : bool, optional
        Call ``matplotlib.pyplot.show()`` to display the plot.
    save : str | Path | None, optional
        File path to save the figure. Saved at 300 DPI with tight
        bounding box. ``None`` skips saving.
    ax : bool | None, optional
        Return the :class:`matplotlib.axes.Axes` object. When
        ``None`` or ``False``, returns ``None``. When ``True``,
        returns the Axes object for further customization.

    Returns
    -------
    Axes | None
        The Matplotlib Axes object if ``ax=True``, otherwise ``None``.

    Raises
    ------
    KeyError
        If ``varm_slot`` is not in ``adata.varm``, if ``layer`` is not
        in ``adata.layers``, or if the parsed ``group_by`` column is
        not in ``adata.obs``.
    TypeError
        If ``adata.varm[varm_slot]`` is not a pandas DataFrame.
    ValueError
        If ``top_n`` is not a positive integer, if no valid results
        remain after filtering, if ``order`` contains values not
        present in the ``group_by`` column, or if ``pval_fontsize``
        is not a positive number when ``show_pval`` is ``True``.

    Examples
    --------
    Plot top 10 differentially abundant proteins:

    >>> pp.pl.differential_abundance_box(
    ...     adata,
    ...     varm_slot="welch;condition;treated_vs_control",
    ... )

    Plot top 5 proteins for specific groups in a given order:

    >>> pp.pl.differential_abundance_box(
    ...     adata,
    ...     varm_slot="welch;condition;treated_vs_control",
    ...     order=["control", "treated"],
    ...     top_n=5,
    ... )
    """
    # -- Validate inputs
    results, group_by, layer, pval_col, top_n = (
        _validate_diff_abundance_inputs(
            adata, varm_slot, top_n, layer,
            show_pval, pval_fontsize,
        )
    )

    if verbose:
        src = f"layer: '{layer}'" if layer else ".X matrix"
        print(f"Using {src}")

    # -- Prepare data
    df_long, top_vars, group_order, pvals_to_plot = (
        _prepare_diff_abundance_data(
            adata, results, top_n, pval_col,
            group_by, layer, order, show_pval,
        )
    )

    # -- Resolve color scheme
    palette = None
    if color_scheme is not None:
        colors = _resolve_color_scheme(
            color_scheme, group_order,
        )
        if colors:
            palette = dict(zip(group_order, colors))

    # -- Create plot
    fig, _ax = plt.subplots(figsize=figsize)

    sns.boxplot(
        data=df_long,
        x="variable",
        y="intensity",
        hue=group_by,
        hue_order=group_order,
        palette=palette,
        gap=0.1,
        flierprops={'marker': '.', 'markersize': 1},
        ax=_ax,
    )

    # -- Annotate p-values
    if show_pval and pvals_to_plot is not None:
        _annotate_boxplot_pvals(
            _ax, df_long, pvals_to_plot,
            top_vars, pval_fontsize,
        )

    # -- Style axes
    _ax.set_xlabel("")
    _ax.set_ylabel("Intensity")
    plt.setp(
        _ax.get_xticklabels(),
        rotation=xlabel_rotation,
        ha="right",
    )

    if title is None:
        title = _stat_test_title_from_varm_slot(
            adata, varm_slot,
        )
    _ax.set_title(title)

    _ax.legend(
        title=group_by,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    plt.tight_layout()

    # -- Save / show / return
    if save:
        fig.savefig(save, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    if ax:
        return _ax
    if not save and not show and not ax:
        raise ValueError(
            "Args show, ax and save all set to "
            "False, function does nothing.",
        )
    return None
