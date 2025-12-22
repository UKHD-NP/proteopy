from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from adjustText import adjust_text
import anndata as ad

from copro.utils.anndata import check_proteodata
from copro.utils.parsers import parse_stat_test_varm_slot


def _stat_test_title_from_varm_slot(
    adata: ad.AnnData,
    varm_slot: str,
) -> str:
    """
    Generate a human-readable plot title from a stat test varm slot.

    Parses the varm slot name to extract test type, design (group
    comparison), and optional layer information, then formats them
    into a concise title string. If parsing fails, returns the
    original slot name with a runtime warning.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object containing the layers used for slot name
        parsing.
    varm_slot : str
        Stat test result slot name in ``adata.varm``. Expected
        format: ``<test_type>_<design>[_<layer>]`` (e.g.,
        ``welch_treatment_vs_control_raw``).

    Returns
    -------
    str
        Formatted title string. Format:
        ``<Test Label> | <Design>`` or
        ``<Test Label> | <Design> | layer: <layer>`` if a layer
        was specified. Returns the original ``varm_slot`` unchanged
        if parsing fails.
    """
    try:
        parsed = parse_stat_test_varm_slot(varm_slot, adata=adata)
    except ValueError as exc:
        warnings.warn(
            f"Could not parse varm slot '{varm_slot}': {exc}",
            RuntimeWarning,
        )
        return varm_slot

    title = f"{parsed['test_label']} | {parsed['design']}"
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


def volcano_plot(
    adata: ad.AnnData,
    varm_slot: str,
    fc_col: str = "logfc",
    pval_col: str = "pval_adj",
    fc_thresh: float = 1.0,
    pval_thresh: float = 0.05,
    top_labels: int | None = None,
    label_col: str | None = None,
    figsize: tuple[float, float] = (6.0, 5.0),
    xlabel: str | None = None,
    alt_color: pd.Series | list[bool] | np.ndarray | None = None,
    ylabel_logscale: bool = True,
    title: str | None = None,
    show: bool = True,
    save: str | Path | None = None,
    ax: bool | None = None,
) -> Axes | None:
    """
    Visualize differential abundance results as a volcano plot.

    Creates a scatter plot of log fold change (x-axis) versus p-value
    (y-axis) for proteins from a statistical test stored in
    ``adata.varm``. Points are colored by significance (exceeding both
    fold change and p-value thresholds), with options for custom
    coloring and automatic labeling of top hits.

    Parameters
    ----------
    adata : ad.AnnData
        :class:`~anndata.AnnData` containing differential abundance
        test results in ``.varm``.
    varm_slot : str
        Key in ``adata.varm`` containing the differential abundance
        test results as a DataFrame. Expected format produced by
        ``copro.tl.differential_abundance``.
    fc_col : str, optional
        Column name in the varm DataFrame containing log fold change
        values. Log base depends on the test method used.
    pval_col : str, optional
        Column name in the varm DataFrame containing adjusted p-values.
        If this column is not found, the function falls back to
        ``"pval"`` (unadjusted p-values).
    fc_thresh : float, optional
        Absolute log fold change threshold for significance. Proteins
        with ``|logfc| >= fc_thresh`` and ``pval <= pval_thresh`` are
        highlighted as significant.
    pval_thresh : float, optional
        P-value threshold for significance. Used in conjunction with
        ``fc_thresh`` to identify significant proteins.
    top_labels : int | None, optional
        Number of top proteins to label on each side of the volcano
        plot (up to 2N labels total). For each direction (positive
        and negative fold change), selects the top N proteins that
        meet BOTH significance thresholds (``pval <= pval_thresh``
        AND ``|logfc| >= fc_thresh``). Proteins are ranked first by
        smallest p-value, then by largest absolute fold change.
        ``None`` disables automatic labeling.
    label_col : str | None, optional
        Column in ``adata.var`` to use for labeling proteins.
        Defaults to ``adata.var_names`` if ``None``.
    figsize : tuple[float, float], optional
        Figure dimensions (width, height) in inches.
    xlabel : str | None, optional
        Label for the x-axis. Defaults to the value of ``fc_col`` if
        ``None``.
    alt_color : pd.Series | list[bool] | np.ndarray | None, optional
        Boolean mask (length ``n_vars``) for alternative coloring
        scheme. When provided, this COMPLETELY OVERRIDES the default
        significance-based coloring: proteins with ``True`` are
        colored light purple (#8E54E5), proteins with ``False`` are
        colored gray (#808080). Significance thresholds (``fc_thresh``
        and ``pval_thresh``) are still visualized as dashed lines
        but do not influence point colors. ``None`` uses the default
        significance-based coloring (gray, blue, red).
    ylabel_logscale : bool, optional
        Controls y-axis representation of p-values. When ``True``,
        plot raw p-values on a log10-scaled y-axis (inverted so
        smaller p-values appear higher); y-axis label shows
        ``pval_col`` name. When ``False``, apply ``-log10``
        transform to p-values and plot on a linear y-axis; y-axis
        label shows ``-log10(pval_col)``. Both representations
        produce visually similar plots with the same interpretation.
    title : str | None, optional
        Plot title. If ``None``, generates a title from the
        ``varm_slot`` name using the stat test metadata (test type,
        group comparison, layer).
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
        If ``varm_slot`` is not in ``adata.varm``, ``fc_col`` or
        p-value columns are not in the varm DataFrame, or
        ``label_col`` is not in ``adata.var``.
    TypeError
        If ``adata.varm[varm_slot]`` is not a pandas DataFrame.
    ValueError
        If no valid (finite, positive p-value) results remain after
        filtering, if ``top_labels`` is not a positive integer, or if
        all of ``show``, ``save``, and ``ax`` are ``False``.

    Notes
    -----
    **Data Filtering**:
    Proteins are filtered before plotting to remove:
    - Missing values in fold change or p-value columns
    - Non-finite values (inf, -inf, nan)
    - Non-positive p-values (cannot be log-transformed)

    **Color Schemes**:
    Default coloring (when ``alt_color=None``):
    - Gray: non-significant proteins (fail one or both thresholds)
    - Blue (#1f77b4): significantly downregulated
      (``logfc <= -fc_thresh`` AND ``pval <= pval_thresh``)
    - Red (#d62728): significantly upregulated
      (``logfc >= fc_thresh`` AND ``pval <= pval_thresh``)

    Alternative coloring (when ``alt_color`` is provided):
    - Gray (#808080): proteins with ``alt_color=False``
    - Light purple (#8E54E5): proteins with ``alt_color=True``
    - Significance thresholds do NOT affect colors, only threshold
      lines are drawn

    **Label Selection Algorithm** (when ``top_labels`` is set):
    1. Filter proteins to those meeting BOTH significance thresholds
    2. Separate into positive (``logfc > 0``) and negative
       (``logfc < 0``) groups
    3. Within each group, rank by: (1) smallest p-value, then
       (2) largest absolute fold change
    4. Select top N from each group (up to 2N total labels)
    5. Use ``adjustText`` library to prevent label overlap

    Examples
    --------
    Plot differential abundance results with default settings:

    >>> cp.pl.volcano_plot(adata, varm_slot="welch_treatment_vs_ctrl")

    Label top 10 proteins per side and save to file:

    >>> cp.pl.volcano_plot(
    ...     adata,
    ...     varm_slot="welch_treatment_vs_ctrl",
    ...     top_labels=10,
    ...     save="volcano.png",
    ... )

    Use custom coloring to highlight proteins of interest:

    >>> proteins_of_interest = adata.var["protein_id"].isin(
    ...     ["P12345", "Q67890"]
    ... )
    >>> cp.pl.volcano_plot(
    ...     adata,
    ...     varm_slot="welch_treatment_vs_ctrl",
    ...     alt_color=proteins_of_interest,
    ... )
    """
    check_proteodata(adata)

    # Validate varm slot exists and contains a DataFrame
    if varm_slot not in adata.varm:
        raise KeyError(
            f"varm_slot '{varm_slot}' not found in adata.varm."
        )

    results = adata.varm[varm_slot]
    if not isinstance(results, pd.DataFrame):
        raise TypeError(
            "Expected adata.varm[varm_slot] to be a pandas DataFrame."
        )

    # Validate required columns exist
    if fc_col not in results.columns:
        raise KeyError(
            f"Column '{fc_col}' not found in varm slot '{varm_slot}'."
        )

    # Prioritize adjusted p-values, fall back to unadjusted if needed.
    # This allows the function to work with test results that may not
    # include multiple testing correction.
    if pval_col in results.columns:
        pval_col_used = pval_col
    elif "pval" in results.columns:
        pval_col_used = "pval"
    else:
        raise KeyError(
            f"Columns '{pval_col}' or 'pval' not found in '{varm_slot}'."
        )

    if label_col is not None and label_col not in adata.var.columns:
        raise KeyError(
            f"Column '{label_col}' not found in adata.var."
        )

    # Extract only needed columns and filter invalid values
    plot_df = results.loc[:, [fc_col, pval_col_used]].copy()
    plot_df = plot_df.dropna(subset=[fc_col, pval_col_used])

    # Remove infinite values (inf, -inf) that cannot be plotted
    finite_mask = np.isfinite(plot_df[fc_col]) & np.isfinite(
        plot_df[pval_col_used]
    )
    if not finite_mask.all():
        warnings.warn(
            "Dropping entries with non-finite fold changes or p-values.",
            RuntimeWarning,
        )
        plot_df = plot_df.loc[finite_mask]

    # Remove non-positive p-values that cannot be log-transformed
    if (plot_df[pval_col_used] <= 0).any():
        warnings.warn(
            "Dropping non-positive p-values before log transform.",
            RuntimeWarning,
        )
        plot_df = plot_df.loc[plot_df[pval_col_used] > 0]

    if plot_df.empty:
        raise ValueError("No valid results available for plotting.")

    # Prepare plotting arrays
    fc_vals = plot_df[fc_col].to_numpy()
    pvals = plot_df[pval_col_used].to_numpy()
    neg_log_p = -np.log10(pvals)
    # Choose y-axis representation based on ylabel_logscale parameter.
    # Both approaches produce visually similar plots: log-scale shows
    # raw p-values with inverted axis, linear shows -log10 transform.
    y_vals = pvals if ylabel_logscale else neg_log_p
    # Keep y_map for label positioning (needed later if labels added).
    # This ensures labels use the same y-coordinates as the scatter
    # points regardless of the scale chosen.
    y_map = pd.Series(y_vals, index=plot_df.index)

    # Create masks for significance-based coloring.
    # Proteins must satisfy BOTH p-value AND fold change thresholds
    # to be considered significantly upregulated or downregulated.
    sig_mask = pvals <= pval_thresh
    up_mask = sig_mask & (fc_vals >= fc_thresh)
    down_mask = sig_mask & (fc_vals <= -fc_thresh)
    other_mask = ~(up_mask | down_mask)  # All non-significant proteins

    # Process alternative color scheme if provided.
    # When alt_color is set, it completely overrides significance-based
    # coloring. The masks are still used for threshold line drawing.
    alt_mask = None
    if alt_color is not None:
        alt_series = _normalize_alt_color(alt_color, adata, plot_df.index)
        alt_mask = alt_series.to_numpy()

    # Create scatter plot with significance-based or alternative coloring
    fig, _ax = plt.subplots(figsize=figsize)
    if alt_mask is None:
        # Default coloring: gray (non-sig), blue (down), red (up)
        _ax.scatter(
            fc_vals[other_mask],
            y_vals[other_mask],
            color="grey",
            alpha=0.5,
            s=12,
        )
        _ax.scatter(
            fc_vals[down_mask],
            y_vals[down_mask],
            color="#1f77b4",  # Blue for downregulated
            alpha=0.8,
            s=14,
        )
        _ax.scatter(
            fc_vals[up_mask],
            y_vals[up_mask],
            color="#d62728",  # Red for upregulated
            alpha=0.8,
            s=14,
        )
    else:
        # Alternative coloring: gray (False) vs purple (True)
        # This completely overrides significance-based coloring
        _ax.scatter(
            fc_vals[~alt_mask],
            y_vals[~alt_mask],
            color="grey",
            alpha=0.5,
            s=12,
        )
        _ax.scatter(
            fc_vals[alt_mask],
            y_vals[alt_mask],
            color="#8E54E5",  # Light purple for highlighted proteins
            alpha=0.8,
            s=14,
        )

    # Add threshold lines (vertical for fold change, horizontal for
    # p-value). These dashed lines visualize the significance cutoffs.
    _ax.axvline(fc_thresh, color="black", linestyle="--", linewidth=1)
    _ax.axvline(-fc_thresh, color="black", linestyle="--", linewidth=1)
    if ylabel_logscale:
        # Log-scale y-axis: plot raw p-values, invert so smaller
        # p-values appear higher (more significant = higher on plot)
        _ax.axhline(
            pval_thresh, color="black", linestyle="--", linewidth=1
        )
        _ax.set_yscale("log", base=10)
        _ax.invert_yaxis()
    else:
        # Linear y-axis: plot -log10(p-value), where larger values
        # represent more significant (smaller) p-values
        _ax.axhline(
            -np.log10(pval_thresh),
            color="black",
            linestyle="--",
            linewidth=1,
        )

    # Set axis labels and title
    _ax.set_xlabel(xlabel or fc_col)
    if ylabel_logscale:
        _ax.set_ylabel(pval_col_used)
    else:
        _ax.set_ylabel(f"-log10({pval_col_used})")
    if title is None:
        title = _stat_test_title_from_varm_slot(adata, varm_slot)
    _ax.set_title(title)

    # Add automatic labeling for top hits if requested.
    # This implements the top_labels selection algorithm described in
    # the docstring: N proteins per side, ranked by p-value then |FC|.
    if top_labels is not None:
        if not isinstance(top_labels, int) or top_labels <= 0:
            raise ValueError("top_labels must be a positive integer.")

        # Get label source (var_names or specified column in var)
        if label_col is None:
            labels = adata.var_names.to_series()
        else:
            labels = adata.var[label_col]

        # Align labels to plotting data (subset after filtering) and
        # prepare a working DataFrame with labels and metrics
        label_series = labels.reindex(plot_df.index)
        label_df = plot_df.copy()
        label_df["label"] = label_series
        label_df = label_df.dropna(subset=["label"])
        label_df["abs_fc"] = label_df[fc_col].abs()
        # Only label proteins that meet BOTH significance thresholds.
        # This ensures we only highlight genuinely significant hits.
        label_df = label_df[
            (label_df[pval_col_used] <= pval_thresh)
            & (label_df["abs_fc"] >= fc_thresh)
        ]

        # Select top N proteins per side (positive/negative FC).
        # Ranking criteria (in order of priority):
        # 1. Smallest p-value (most statistically significant)
        # 2. Largest absolute fold change (largest effect size)
        pos_df = label_df[label_df[fc_col] > 0].sort_values(
            by=[pval_col_used, "abs_fc"],
            ascending=[True, False],
        ).head(top_labels)
        neg_df = label_df[label_df[fc_col] < 0].sort_values(
            by=[pval_col_used, "abs_fc"],
            ascending=[True, False],
        ).head(top_labels)
        label_df = pd.concat([pos_df, neg_df])

        # Add text labels at protein positions and adjust to avoid
        # overlap using the adjustText library
        texts = []
        for _, row in label_df.iterrows():
            texts.append(
                _ax.text(
                    row[fc_col],
                    y_map.loc[row.name],
                    str(row["label"]),
                    fontsize=8,
                )
            )
        if texts:
            # Automatically adjust label positions with arrows pointing
            # to original data points
            adjust_text(
                texts,
                ax=_ax,
                arrowprops=dict(arrowstyle="->", color="0.4", lw=0.7),
            )

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
    return None
