from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator
import seaborn as sns
import anndata as ad
from matplotlib.axes import Axes
from seaborn.matrix import ClusterGrid
from adjustText import adjust_text
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from proteopy.utils.anndata import check_proteodata
from proteopy.utils.matplotlib import _resolve_color_scheme
from proteopy.utils.slot_parsers import (
    _pairwise_peptide_correlations_legacy_df_to_sym,
    _pairwise_var_correlations_df_to_sym,
    parse_pairwise_peptide_correlations_result_legacy,
    parse_pairwise_var_correlations_result,
    resolve_corrs_key,
    resolve_pairwise_var_correlations_key,
)


def _correlation_linkage_matrix(
    sym: pd.DataFrame,
    *,
    linkage_method: str = "average",
) -> np.ndarray:
    """Compute hierarchical-clustering linkage on a ``1 - corr`` distance.

    The input matrix must not contain NaN values; a
    :class:`ValueError` is raised otherwise.
    """
    if sym.isna().to_numpy().any():
        raise ValueError(
            "Pairwise correlation matrix contains NaN values; "
            "cannot perform hierarchical clustering. Clean the "
            "input data or re-run `pairwise_var_correlations` "
            "with parameters that avoid uncomparable pairs (e.g. "
            "stricter missing-value handling)."
        )
    dist = 1.0 - sym.to_numpy()
    # Symmetrize against floating-point asymmetry and zero the
    # diagonal so squareform accepts the matrix.
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    return linkage(condensed, method=linkage_method)


def _build_margin_colors(
    annotation_df: pd.DataFrame,
    index: pd.Index,
    margin_color: str | list[str],
    color_scheme=None,
) -> tuple[pd.DataFrame, dict[str, dict[str, tuple]]]:
    """Build a categorical color DataFrame from annotation columns.

    Each requested column of ``annotation_df`` (typically
    ``adata.var`` or ``adata.obs``) is mapped to a categorical
    palette resolved via
    :func:`proteopy.utils.matplotlib._resolve_color_scheme` and
    converted to per-row colors aligned with ``index``. The returned
    DataFrame is suitable for ``row_colors``/``col_colors`` arguments
    of :func:`seaborn.clustermap`. The returned mapping captures
    ``{column: {category: color}}`` in insertion order and can be
    used to render a categorical legend.
    """
    if isinstance(margin_color, str):
        keys = [margin_color]
    elif (
        isinstance(margin_color, list)
        and all(isinstance(k, str) for k in margin_color)
        and margin_color
    ):
        keys = list(margin_color)
    else:
        raise TypeError(
            "`margin_color` must be a string or a non-empty list "
            "of strings."
        )

    missing = [k for k in keys if k not in annotation_df.columns]
    if missing:
        raise ValueError(
            f"`margin_color` columns not found in annotation: {missing}"
        )

    annot = annotation_df.loc[index, keys].copy()
    color_df = pd.DataFrame(index=index)
    mapping: dict[str, dict[str, tuple]] = {}
    for k in keys:
        vals = annot[k].astype(str)
        cats = list(pd.unique(vals))
        resolved = _resolve_color_scheme(color_scheme, cats)
        if resolved is None:
            resolved = sns.color_palette(
                "tab10", n_colors=max(len(cats), 1)
            )
        cat_to_color = dict(zip(cats, resolved))
        color_df[k] = [cat_to_color[v] for v in vals]
        mapping[k] = cat_to_color
    return color_df, mapping


def _validate_peptide_intensity_heatmap_inputs(
    *,
    corrs_key,
    row_cluster,
    col_cluster,
    row_dendrogram,
    col_dendrogram,
    order_by,
    order,
) -> None:
    """Validate the boolean / key arguments of the intensity heatmap."""
    if not isinstance(corrs_key, str) or not corrs_key:
        raise ValueError("corrs_key must be a non-empty string.")
    for name, val in (
        ("row_cluster", row_cluster),
        ("col_cluster", col_cluster),
        ("row_dendrogram", row_dendrogram),
        ("col_dendrogram", col_dendrogram),
    ):
        if not isinstance(val, bool):
            raise TypeError(f"`{name}` must be a bool.")
    if col_cluster and order_by is not None:
        raise ValueError(
            "`order_by` has no effect when `col_cluster=True`. "
            "Pass `col_cluster=False` to use `order_by`."
        )
    if col_cluster and order is not None:
        raise ValueError(
            "`order` has no effect when `col_cluster=True`. "
            "Pass `col_cluster=False` to use `order`."
        )
    if order_by is not None and order is not None:
        raise ValueError(
            "`order` and `order_by` are mutually exclusive."
        )


def _resolve_sample_order(
    adata: ad.AnnData,
    sample_index: pd.Index,
    *,
    order_by: str | None,
    order: list[str] | None,
) -> pd.Index:
    """Resolve the column (sample) order for the intensity heatmap.

    ``order_by`` sorts ``sample_index`` by the values of ``adata.obs[
    order_by]`` (stable sort). ``order`` requires a list of strings
    that is a permutation of ``sample_index``. When both are ``None``
    the input order is returned unchanged.
    """
    if order_by is not None:
        if not isinstance(order_by, str) or not order_by:
            raise ValueError(
                "`order_by` must be a non-empty string or None."
            )
        if order_by not in adata.obs.columns:
            raise ValueError(
                f"`order_by` column '{order_by}' not found in "
                "adata.obs.columns."
            )
        ordered = adata.obs.loc[sample_index, order_by].sort_values(
            kind="stable"
        )
        return ordered.index
    if order is not None:
        if not isinstance(order, list) or not all(
            isinstance(v, str) for v in order
        ):
            raise TypeError("`order` must be a list of strings.")
        if len(order) != len(set(order)):
            raise ValueError("`order` must not contain duplicates.")
        expected = set(sample_index)
        given = set(order)
        if expected != given:
            missing = sorted(expected - given)
            extra = sorted(given - expected)
            raise ValueError(
                "`order` must be a permutation of adata.obs_names. "
                f"Missing: {missing}; Extra: {extra}."
            )
        return pd.Index(order)
    return sample_index


def _add_stacked_margin_legends(
    g: ClusterGrid,
    *,
    row_mapping: dict | None,
    col_mapping: dict | None,
    anchor_x: float,
    anchor_y_top: float,
    gap: float = 0.02,
) -> None:
    """Stack row- and column-margin legends vertically on the right.

    The row-margin legend is rendered first at ``anchor_y_top``; the
    column-margin legend (if any) is then placed below it with a
    figure-relative ``gap``. When only one mapping is provided the
    other is skipped. Requires a draw pass to measure the first
    legend's bounding box before placing the second.
    """
    next_y = anchor_y_top
    if row_mapping:
        leg = _add_margin_legend(
            g,
            row_mapping,
            anchor_x=anchor_x,
            anchor_y_top=next_y,
        )
        if col_mapping:
            g.figure.canvas.draw()
            bbox = leg.get_window_extent().transformed(
                g.figure.transFigure.inverted()
            )
            next_y = bbox.y0 - gap
    if col_mapping:
        _add_margin_legend(
            g,
            col_mapping,
            anchor_x=anchor_x,
            anchor_y_top=next_y,
        )


def _resolve_pairwise_corrs_source(
    adata: ad.AnnData,
    corrs_key: str | pd.DataFrame | None,
    *,
    group_id: str | None,
    legacy: bool,
) -> tuple[str | pd.DataFrame, str | None, pd.DataFrame]:
    """Resolve the correlation source and return ``(corrs_key, gid, sym)``.

    Accepts either a ``str``/``None`` key (looked up in
    ``adata.uns`` via :func:`resolve_corrs_key`) or a
    :class:`pandas.DataFrame` (consumed directly). Dispatches to the
    appropriate parser based on ``legacy`` and returns the (possibly
    inferred) ``corrs_key`` for downstream validation alongside the
    resolved group id and the symmetric correlation matrix.
    """
    if isinstance(corrs_key, pd.DataFrame):
        df_helper = (
            _pairwise_peptide_correlations_legacy_df_to_sym
            if legacy
            else _pairwise_var_correlations_df_to_sym
        )
        resolved_gid, sym = df_helper(
            corrs_key,
            group_id=group_id,
            source_label="corrs_key DataFrame",
        )
        return corrs_key, resolved_gid, sym

    resolved_key = resolve_corrs_key(adata, corrs_key, legacy=legacy)
    parser = (
        parse_pairwise_peptide_correlations_result_legacy
        if legacy
        else parse_pairwise_var_correlations_result
    )
    resolved_gid, sym = parser(
        adata,
        corrs_key=resolved_key,
        group_id=group_id,
    )
    return resolved_key, resolved_gid, sym


def _validate_pairwise_heatmap_inputs(
    *,
    corrs_key,
    cluster,
    row_dendrogram,
    col_dendrogram,
    var_order,
) -> None:
    """Validate the boolean / key arguments of the heatmap function."""
    is_str_key = isinstance(corrs_key, str) and corrs_key
    is_df_key = isinstance(corrs_key, pd.DataFrame)
    if not (is_str_key or is_df_key):
        raise ValueError(
            "corrs_key must be a non-empty string or a pandas DataFrame."
        )
    if not isinstance(cluster, bool):
        raise TypeError("`cluster` must be a bool.")
    if not isinstance(row_dendrogram, bool):
        raise TypeError("`row_dendrogram` must be a bool.")
    if not isinstance(col_dendrogram, bool):
        raise TypeError("`col_dendrogram` must be a bool.")
    if cluster and var_order is not None:
        raise ValueError(
            "`var_order` has no effect when `cluster=True`. "
            "Pass `cluster=False` to use `var_order`."
        )


def _print_correlation_stats(sym: pd.DataFrame) -> None:
    """Print global summary statistics of the upper-triangle values."""
    arr = sym.to_numpy()
    iu = np.triu_indices(arr.shape[0], k=1)
    upper = arr[iu]
    upper = upper[~np.isnan(upper)]
    if upper.size:
        stats_df = pd.DataFrame(
            {
                "mean": [float(np.mean(upper))],
                "std": (
                    [float(np.std(upper, ddof=1))]
                    if upper.size > 1
                    else [float("nan")]
                ),
                "median": [float(np.median(upper))],
                "min": [float(np.min(upper))],
                "max": [float(np.max(upper))],
            }
        )
    else:
        stats_df = pd.DataFrame(
            {
                "mean": [float("nan")],
                "std": [float("nan")],
                "median": [float("nan")],
                "min": [float("nan")],
                "max": [float("nan")],
            }
        )
    print("Global:")
    print(stats_df.to_string(index=False, float_format="%.3f"))


def _resolve_var_order(
    sym: pd.DataFrame,
    var_order: list[str],
) -> pd.DataFrame:
    """Reorder ``sym`` rows and columns by ``var_order``.

    ``var_order`` must be a permutation of ``sym.index``; missing or
    extra entries raise :class:`ValueError`.
    """
    if not isinstance(var_order, list) or not all(
        isinstance(v, str) for v in var_order
    ):
        raise TypeError("`var_order` must be a list of strings.")
    if len(var_order) != len(set(var_order)):
        raise ValueError("`var_order` must not contain duplicates.")
    expected = set(sym.index)
    given = set(var_order)
    if expected != given:
        missing = sorted(expected - given)
        extra = sorted(given - expected)
        raise ValueError(
            "`var_order` must be a permutation of the variables in "
            "the correlation matrix. "
            f"Missing: {missing}; Extra: {extra}."
        )
    return sym.loc[var_order, var_order]


def _toggle_dendrograms(
    g: ClusterGrid,
    *,
    row_cluster: bool,
    col_cluster: bool,
    row_dendrogram: bool,
    col_dendrogram: bool,
) -> None:
    """Hide row/column dendrograms when requested.

    A dendrogram is only drawn when its axis was clustered, so the
    visibility toggle only fires when both clustering and the
    dendrogram flag agree.
    """
    if row_cluster and not row_dendrogram:
        g.ax_row_dendrogram.set_visible(False)
    if col_cluster and not col_dendrogram:
        g.ax_col_dendrogram.set_visible(False)


def _finalize_clustermap(
    g: ClusterGrid,
    *,
    save: str | Path | None,
    show: bool,
) -> None:
    """Save and/or show the clustermap figure."""
    if save is not None:
        if not isinstance(save, (str, Path)):
            raise TypeError("`save` must be a path-like object or None.")
        g.savefig(save, dpi=300, bbox_inches="tight")
    if show:
        plt.show()


def _compress_heatmap_horizontally(
    g: ClusterGrid,
    *,
    scale: float,
) -> None:
    """Shrink all clustermap axes horizontally by ``scale``.

    Multiplies each managed axis's ``x0`` and ``width`` by ``scale``
    so the heatmap, dendrograms, color bars and colorbar collectively
    occupy ``scale`` of the figure width, freeing the right-hand
    margin for an external legend. The figure size itself is not
    modified.
    """
    axes = [
        g.ax_heatmap,
        g.ax_row_dendrogram,
        g.ax_col_dendrogram,
        g.ax_row_colors,
        g.ax_col_colors,
        g.ax_cbar,
    ]
    for ax in axes:
        if ax is None:
            continue
        pos = ax.get_position()
        ax.set_position(
            [pos.x0 * scale, pos.y0, pos.width * scale, pos.height]
        )


def _inset_margin_bars(g: ClusterGrid, *, gap: float = 0.003) -> None:
    """Shrink ``ax_row_colors``/``ax_col_colors`` by ``gap`` per edge.

    Insets each color bar inward from both of its long edges so a
    thin figure-relative gap separates it from its neighbouring
    dendrogram on one side and the heatmap on the other. ``gap`` is
    expressed in figure-relative units; the bar's perpendicular
    extent is unchanged.
    """
    if g.ax_row_colors is not None:
        pos = g.ax_row_colors.get_position()
        new_width = pos.width - 2 * gap
        if new_width > 0:
            g.ax_row_colors.set_position(
                [pos.x0 + gap, pos.y0, new_width, pos.height]
            )
    if g.ax_col_colors is not None:
        pos = g.ax_col_colors.get_position()
        new_height = pos.height - 2 * gap
        if new_height > 0:
            g.ax_col_colors.set_position(
                [pos.x0, pos.y0 + gap, pos.width, new_height]
            )


def _hide_margin_bar_labels(g: ClusterGrid) -> None:
    """Hide tick labels and axis labels on the margin color bars.

    Seaborn renders the ``margin_color`` column name as a tick label
    on each color bar (e.g. below ``ax_row_colors`` and to the right
    of ``ax_col_colors``). Those labels are redundant once a
    categorical legend is drawn separately, so this helper removes
    them and the tick marks for a cleaner appearance.
    """
    for ax in (g.ax_row_colors, g.ax_col_colors):
        if ax is None:
            continue
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")


def _place_title_above_top(
    g: ClusterGrid,
    title: str,
    *,
    reserved_height: float = 0.04,
) -> None:
    """Render ``title`` above the topmost visible top-side axis.

    If the gap between the topmost visible axis and the figure top
    is smaller than ``reserved_height``, that axis is shrunk
    vertically (from the top down) to free the required space, so
    the title does not collide with the column dendrogram, the
    column color bar, or the heatmap. The title is then centered
    horizontally over the heatmap and vertically within the freed
    strip.
    """
    candidates = [
        ax for ax in (
            g.ax_col_dendrogram,
            g.ax_col_colors,
            g.ax_heatmap,
        )
        if ax is not None and ax.get_visible()
    ]
    if not candidates:
        g.figure.suptitle(title)
        return
    topmost = max(candidates, key=lambda a: a.get_position().y1)
    pos = topmost.get_position()
    available = 1.0 - pos.y1
    if available < reserved_height:
        shrink = reserved_height - available
        new_height = max(pos.height - shrink, 0.01)
        topmost.set_position(
            [pos.x0, pos.y0, pos.width, new_height]
        )
        pos = topmost.get_position()
    title_y = (pos.y1 + 1.0) / 2
    heatmap_pos = g.ax_heatmap.get_position()
    title_x = heatmap_pos.x0 + heatmap_pos.width / 2
    g.figure.suptitle(title, x=title_x, y=title_y)


def _layout_right_side_panel(
    g: ClusterGrid,
    *,
    label_gap: float = 0.12,
    cbar_title: str = "Correlation",
) -> float:
    """Reposition ``ax_cbar`` into a right-side legend column.

    The colorbar is moved from its default top-left position to a
    vertical strip on the right of the figure, with its bottom edge
    aligned to ``ax_heatmap.y0`` (i.e. the heatmap's bottom,
    excluding column tick labels). The colorbar's existing width and
    height are preserved. The shared left ``x0`` of the right-side
    panel is returned so additional figure-level legends can be
    placed flush with the colorbar.

    ``label_gap`` is the figure-relative horizontal gap reserved
    between the heatmap's right edge and the right-side panel for
    the heatmap's row tick labels. ``cbar_title`` is rendered above
    the colorbar.
    """
    heatmap_pos = g.ax_heatmap.get_position()
    cbar_pos = g.ax_cbar.get_position()
    right_x = heatmap_pos.x1 + label_gap
    g.ax_cbar.set_position(
        [right_x, heatmap_pos.y0, cbar_pos.width, cbar_pos.height]
    )
    # Move the colorbar caption from the side to a title above it.
    g.ax_cbar.set_ylabel("")
    g.ax_cbar.set_title(cbar_title, fontsize=10)
    return right_x


def _add_margin_legend(
    g: ClusterGrid,
    mapping: dict[str, dict[str, tuple]],
    *,
    anchor_x: float,
    anchor_y_top: float,
):
    """Add a categorical legend for ``margin_color`` annotations.

    Renders one section per annotation column with the column name
    as a header above its category swatches. The legend's top-left
    corner is anchored at ``(anchor_x, anchor_y_top)`` in figure
    coordinates, so it can be aligned with the heatmap's top edge
    and left-aligned with the colorbar. Returns the
    :class:`~matplotlib.legend.Legend` so callers can measure its
    extent (e.g. to stack a second legend below).
    """
    handles: list[Patch] = []
    labels: list[str] = []
    for col_name, cat_to_color in mapping.items():
        # Section header rendered via a transparent patch.
        handles.append(Patch(facecolor="none", edgecolor="none"))
        labels.append(f"{col_name}:")
        for cat, color in cat_to_color.items():
            handles.append(Patch(facecolor=color, edgecolor="none"))
            labels.append(str(cat))
    return g.figure.legend(
        handles=handles,
        labels=labels,
        loc="upper left",
        bbox_to_anchor=(anchor_x, anchor_y_top),
        frameon=False,
        fontsize=10,
        handlelength=1.2,
        handleheight=1.0,
        borderaxespad=0.0,
    )


def pairwise_correlation_heatmap(
    adata: ad.AnnData,
    *,
    corrs_key: str | pd.DataFrame | None = None,
    group_id: str | None = None,
    legacy: bool = False,
    cluster: bool = True,
    linkage_method: str = "average",
    row_dendrogram: bool = True,
    col_dendrogram: bool = True,
    var_order: list[str] | None = None,
    margin_color: str | list[str] | None = None,
    cmap: str = "RdBu_r",
    color_scheme=None,
    label_size: int | float = 5,
    title: str | None = None,
    figsize: tuple[float, float] = (12, 10),
    print_stats: bool = False,
    show: bool = True,
    save: str | Path | None = None,
) -> ClusterGrid:
    """Clustered heatmap of pairwise variable correlations from ``.uns``.

    Reads a long-form correlation frame stored in
    ``adata.uns[corrs_key]`` by
    :func:`proteopy.tl.pairwise_var_correlations`, pivots it into
    a symmetric matrix and renders it as a diverging clustermap
    centered on zero with ``vmin=-1`` and ``vmax=1``. Hierarchical
    clustering on a ``1 - corr`` distance matrix orders the rows
    and columns; dendrograms appear on the top and left and row/
    column tick labels on the right and bottom. When
    ``margin_color`` is supplied, categorical annotation bars are
    rendered between the dendrograms and the heatmap (above the
    heatmap for columns, to the left of the heatmap for rows), and
    a legend of the annotation categories is drawn in the top-right
    of the figure.

    Parameters
    ----------
    adata : AnnData
        :class:`~anndata.AnnData` carrying the stored correlations.
    corrs_key : str | pandas.DataFrame | None
        Source of the long-form correlation frame. When a string,
        it is interpreted as a key into ``adata.uns`` (the default
        behavior). When a :class:`pandas.DataFrame`, the frame is
        consumed directly without consulting ``adata.uns``; the
        frame must follow the schema emitted by
        :func:`pairwise_var_correlations` (``varA``/``varB``/``corr``,
        with an optional ``group_id`` column) or, when
        ``legacy=True``, the legacy
        :func:`pairwise_peptide_correlations_legacy` schema
        (``pepA``/``pepB``/``PCC`` with ``protein_id`` as the index).
        When ``legacy=False`` and ``corrs_key`` is ``None``, the key
        is inferred automatically if exactly one slot produced by
        :func:`pairwise_var_correlations` is present in
        ``adata.uns``; a :class:`ValueError` is raised when no slot
        or multiple slots are present. When ``legacy=True`` and
        ``corrs_key`` is ``None``, it defaults to
        ``"pairwise_peptide_correlations"`` (the legacy default
        produced by :func:`pairwise_peptide_correlations_legacy`); a
        user-supplied value still overrides the default.
    group_id : str | None
        Selector for a single group when the stored frame contains a
        ``group_id`` column (e.g. ``protein_id`` for peptide-level
        results). Must be ``None`` for ungrouped frames. When
        ``legacy=True``, this is the ``protein_id`` stored as the
        legacy frame's index and is required.
    legacy : bool
        If True, parse ``adata.uns[corrs_key]`` using the legacy
        ``(pepA, pepB, PCC)`` schema produced by
        :func:`pairwise_peptide_correlations_legacy`. If False
        (default), use the modern ``(varA, varB, corr)`` schema.
    cluster : bool
        If True, hierarchically cluster both rows and columns using
        a ``1 - corr`` distance matrix with linkage method controlled
        by ``linkage_method``. The matrix must not contain NaN values
        when clustering; a :class:`ValueError` is raised otherwise.
        When False, ordering follows ``var_order`` (if provided) or
        the parser's natural order.
    linkage_method : str
        Linkage method passed to
        :func:`scipy.cluster.hierarchy.linkage` (e.g. ``"average"``,
        ``"single"``, ``"complete"``, ``"ward"``). Only used when
        ``cluster=True``.
    row_dendrogram : bool
        Show the row dendrogram on the left. Only takes effect when
        ``cluster=True``; ignored otherwise.
    col_dendrogram : bool
        Show the column dendrogram on the top. Only takes effect
        when ``cluster=True``; ignored otherwise.
    var_order : list[str] | None
        Explicit ordering of rows and columns when ``cluster=False``.
        Must be a permutation of the variables present in the
        correlation matrix; missing or extra entries raise
        :class:`ValueError`. Passing ``var_order`` together with
        ``cluster=True`` raises :class:`ValueError`.
    margin_color : str | list[str] | None
        Column name(s) in ``adata.var`` used to draw categorical
        annotation bars on the top (columns) and left (rows) of
        the heatmap, between the dendrograms and the heatmap,
        accompanied by a legend in the top-right of the figure.
        ``None`` disables the annotation bars and legend.
    cmap : str
        Diverging Matplotlib colormap name used by the heatmap.
    color_scheme : str | dict | Sequence | Colormap | callable | None
        Defines the color mapping for ``margin_color`` categories.
        Can be a named Matplotlib colormap, a single color, a list/
        tuple of colors, a dict mapping categories to colors, a
        :class:`~matplotlib.colors.Colormap` object, or a callable
        returning colors. If ``None``, the default Matplotlib color
        cycle is used. Resolved via
        :func:`proteopy.utils.matplotlib._resolve_color_scheme`.
        Ignored when ``margin_color`` is ``None``.
    label_size : int | float
        Font size for the heatmap row and column tick labels.
    title : str | None
        Figure title rendered above the column dendrogram (or
        whichever top-side axis is topmost when the dendrogram is
        hidden). When ``None``, a title is derived from the
        resolved group identifier (e.g.
        ``"Pairwise correlations: <group_id>"``).
    figsize : tuple[float, float]
        Matplotlib figure size in inches. The default is chosen
        empirically so that the heatmap rectangle renders close to
        square once the dendrograms, margin annotation bars, and
        right-side colorbar/legend column are laid out.
    print_stats : bool
        If True, print a pandas DataFrame with global summary
        statistics (mean, std, median, min, max) computed over the
        off-diagonal upper-triangle correlation values before showing
        the plot.
    show : bool
        Call :func:`matplotlib.pyplot.show` when ``True``.
    save : str | Path | None
        File path to save the figure. ``None`` skips saving.

    Returns
    -------
    seaborn.matrix.ClusterGrid
        The :class:`~seaborn.matrix.ClusterGrid` object containing
        the heatmap, dendrogram, and annotation axes.

    Examples
    --------
    >>> import proteopy as pr
    >>> adata = pr.datasets.example_peptide_data()
    >>> pr.tl.pairwise_peptide_correlations(adata)
    >>> pr.pl.pairwise_correlation_heatmap(
    ...     adata,
    ...     corrs_key="pairwise_peptide_correlations;;",
    ...     group_id="P12345",
    ...     margin_color="gene_name",
    ... )
    """
    check_proteodata(adata)

    # -- resolve the correlation source (string key or DataFrame)
    # and parse it into a symmetric matrix in one step
    corrs_key, resolved_gid, sym = _resolve_pairwise_corrs_source(
        adata,
        corrs_key,
        group_id=group_id,
        legacy=legacy,
    )

    _validate_pairwise_heatmap_inputs(
        corrs_key=corrs_key,
        cluster=cluster,
        row_dendrogram=row_dendrogram,
        col_dendrogram=col_dendrogram,
        var_order=var_order,
    )

    if print_stats:
        _print_correlation_stats(sym)

    # -- linkage (clustering) or explicit ordering
    if cluster:
        link = _correlation_linkage_matrix(
            sym,
            linkage_method=linkage_method,
        )
    else:
        link = None
        if var_order is not None:
            sym = _resolve_var_order(sym, var_order)

    # -- categorical margin annotation bars
    if margin_color is not None:
        margin_df, margin_mapping = _build_margin_colors(
            adata.var,
            sym.index,
            margin_color,
            color_scheme=color_scheme,
        )
    else:
        margin_df, margin_mapping = None, None

    # -- plot via clustermap
    g = sns.clustermap(
        sym,
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        row_cluster=cluster,
        col_cluster=cluster,
        row_linkage=link,
        col_linkage=link,
        row_colors=margin_df,
        col_colors=margin_df,
        cbar_kws={"label": "correlation"},
        xticklabels=True,
        yticklabels=True,
        figsize=figsize,
    )

    _toggle_dendrograms(
        g,
        row_cluster=cluster,
        col_cluster=cluster,
        row_dendrogram=row_dendrogram,
        col_dendrogram=col_dendrogram,
    )
    _compress_heatmap_horizontally(g, scale=0.82)
    if margin_df is not None:
        _inset_margin_bars(g)
        _hide_margin_bar_labels(g)
    right_x = _layout_right_side_panel(g)
    if margin_df is not None:
        anchor_y_top = g.ax_heatmap.get_position().y1
        _add_margin_legend(
            g,
            margin_mapping,
            anchor_x=right_x,
            anchor_y_top=anchor_y_top,
        )

    if title is None:
        title = (
            f"Pairwise correlations: {resolved_gid}"
            if resolved_gid is not None
            else "Pairwise correlations"
        )
    _place_title_above_top(g, title)
    g.ax_heatmap.set_xlabel("")
    g.ax_heatmap.set_ylabel("")
    g.ax_heatmap.tick_params(axis="both", labelsize=label_size)

    _finalize_clustermap(g, save=save, show=show)
    return g


def peptide_intensity_heatmap(
    adata: ad.AnnData,
    *,
    corrs_key: str | None = None,
    group_id: str | None = None,
    layer: str | None = None,
    row_cluster: bool = True,
    col_cluster: bool = True,
    linkage_method: str = "average",
    row_dendrogram: bool = True,
    col_dendrogram: bool = True,
    order_by: str | None = None,
    order: list[str] | None = None,
    zscore: bool = False,
    row_margin_color: str | list[str] | None = None,
    col_margin_color: str | list[str] | None = None,
    cmap: str | None = None,
    color_scheme=None,
    label_size: int | float = 5,
    title: str | None = None,
    figsize: tuple[float, float] = (12, 10),
    show: bool = True,
    save: str | Path | None = None,
) -> ClusterGrid:
    """Heatmap of peptide intensities across samples.

    Renders an ``n_peptides x n_samples`` intensity heatmap whose
    peptide rows are restricted to those present in the correlation
    frame stored at ``adata.uns[corrs_key]`` (as produced by
    :func:`proteopy.tl.pairwise_var_correlations`). The peptide
    ordering on the row axis can be driven by hierarchical clustering
    on a ``1 - corr`` distance matrix derived from the stored
    correlations, while the sample ordering on the column axis can be
    driven by hierarchical clustering on the intensities themselves,
    by sorting a ``.obs`` annotation, or by an explicit list.
    Categorical annotation bars from ``.var`` (rows) and ``.obs``
    (columns) are optional, with their legends stacked on the right
    side of the figure.

    Parameters
    ----------
    adata : AnnData
        :class:`~anndata.AnnData` carrying intensities in ``.X`` (or
        in ``.layers[layer]``) and the correlation frame in
        ``.uns[corrs_key]``.
    corrs_key : str | None
        Key in ``adata.uns`` holding the long-form correlation frame
        emitted by :func:`pairwise_var_correlations`. Used to
        determine the peptide subset rendered on the row axis and to
        compute the row linkage when ``row_cluster=True``. When
        ``None``, the key is inferred automatically if exactly one
        such slot exists in ``adata.uns``; a :class:`ValueError` is
        raised when no slot or multiple slots are present.
    group_id : str | None
        Selector for a single group when the stored frame contains a
        ``group_id`` column (e.g. ``protein_id`` for peptide-level
        results). Must be ``None`` for ungrouped frames.
    layer : str | None
        Optional key in ``adata.layers``; when set, intensities are
        read from that layer instead of ``.X``.
    row_cluster : bool
        If True, hierarchically cluster peptide rows using a
        ``1 - corr`` distance matrix derived from the stored
        correlations with linkage method ``linkage_method``. The
        correlation matrix must not contain NaN values; a
        :class:`ValueError` is raised otherwise. When False, rows
        follow the parser's natural order.
    col_cluster : bool
        If True, hierarchically cluster sample columns on the
        intensity matrix with linkage method ``linkage_method``. The
        intensity matrix must not contain NaN values when
        clustering. Incompatible with ``order_by`` and ``order``.
    linkage_method : str
        Linkage method passed to
        :func:`scipy.cluster.hierarchy.linkage` for both row and
        column clustering.
    row_dendrogram : bool
        Show the row dendrogram on the left. Only takes effect when
        ``row_cluster=True``.
    col_dendrogram : bool
        Show the column dendrogram on the top. Only takes effect when
        ``col_cluster=True``.
    order_by : str | None
        Column in ``.obs`` used to sort sample columns (stable sort
        on the column values). Incompatible with ``col_cluster=True``
        and with ``order``.
    order : list[str] | None
        Explicit list of sample IDs defining the column order. Must
        be a permutation of ``adata.obs_names``. Incompatible with
        ``col_cluster=True`` and with ``order_by``.
    zscore : bool
        If True, replace each peptide's intensities with row-wise
        z-scores computed across samples (per-peptide mean and std,
        NaN-tolerant). Peptides with zero variance across samples
        yield an all-NaN row, which is incompatible with
        ``col_cluster=True``; pre-filter such peptides with
        :func:`proteopy.pp.remove_zero_variance_vars`.
    row_margin_color : str | list[str] | None
        Column name(s) in ``adata.var`` used to draw categorical
        annotation bars on the left of the heatmap. Their legend is
        rendered in the top-right of the figure.
    col_margin_color : str | list[str] | None
        Column name(s) in ``adata.obs`` used to draw categorical
        annotation bars above the heatmap. Their legend is rendered
        below the row-margin legend.
    cmap : str | None
        Matplotlib colormap name used by the heatmap. When ``None``,
        defaults to ``"RdBu_r"`` (centered at 0) when ``zscore=True``
        and ``"viridis"`` otherwise.
    color_scheme : str | dict | Sequence | Colormap | callable | None
        Defines the color mapping for ``row_margin_color`` /
        ``col_margin_color`` categories. Resolved via
        :func:`proteopy.utils.matplotlib._resolve_color_scheme`. The
        same scheme is applied to row and column margins.
    label_size : int | float
        Font size for the heatmap row and column tick labels.
    title : str | None
        Figure title. When ``None``, derived from the resolved group
        identifier (e.g. ``"Peptide intensities: <group_id>"``).
    figsize : tuple[float, float]
        Matplotlib figure size in inches.
    show : bool
        Call :func:`matplotlib.pyplot.show` when ``True``.
    save : str | Path | None
        File path to save the figure. ``None`` skips saving.

    Returns
    -------
    seaborn.matrix.ClusterGrid
        The :class:`~seaborn.matrix.ClusterGrid` object containing
        the heatmap, dendrogram, and annotation axes.

    Examples
    --------
    >>> import proteopy as pr
    >>> adata = pr.datasets.example_peptide_data()
    >>> pr.tl.pairwise_var_correlations(adata, group_by="protein_id")
    >>> pr.pl.peptide_intensity_heatmap(
    ...     adata,
    ...     corrs_key="pairwise_correlations;protein_id;;;",
    ...     group_id="P12345",
    ...     order_by="condition",
    ...     row_margin_color="modification",
    ...     col_margin_color="condition",
    ... )
    """
    check_proteodata(
        adata,
        layers=[layer] if layer is not None else None,
    )

    corrs_key = resolve_pairwise_var_correlations_key(adata, corrs_key)

    _validate_peptide_intensity_heatmap_inputs(
        corrs_key=corrs_key,
        row_cluster=row_cluster,
        col_cluster=col_cluster,
        row_dendrogram=row_dendrogram,
        col_dendrogram=col_dendrogram,
        order_by=order_by,
        order=order,
    )
    if not isinstance(zscore, bool):
        raise TypeError("`zscore` must be a bool.")

    # -- parse correlations to determine peptide subset / row order
    resolved_gid, sym = parse_pairwise_var_correlations_result(
        adata,
        corrs_key=corrs_key,
        group_id=group_id,
    )
    peptide_index = sym.index

    # -- row linkage (peptides) from correlations
    if row_cluster:
        row_link = _correlation_linkage_matrix(
            sym,
            linkage_method=linkage_method,
        )
    else:
        row_link = None

    # -- build peptides x samples intensity frame
    mat = adata.X if layer is None else adata.layers[layer]
    mat = np.asarray(mat)
    pep_pos = adata.var_names.get_indexer(peptide_index)
    intensity_arr = mat[:, pep_pos].T
    intensity_df = pd.DataFrame(
        intensity_arr,
        index=peptide_index.copy(),
        columns=adata.obs_names.copy(),
    )

    # -- resolve sample order (only when col_cluster=False)
    if not col_cluster:
        sample_order = _resolve_sample_order(
            adata,
            intensity_df.columns,
            order_by=order_by,
            order=order,
        )
        intensity_df = intensity_df.loc[:, sample_order]

    # -- per-peptide z-score across samples (row-wise). Zero-variance
    # rows produce NaNs, which propagate to the col_cluster check.
    if zscore:
        row_means = intensity_df.mean(axis=1, skipna=True)
        row_stds = intensity_df.std(axis=1, skipna=True).replace(
            0, np.nan
        )
        intensity_df = intensity_df.sub(row_means, axis=0).div(
            row_stds, axis=0
        )

    # -- col clustering requires NaN-free intensities
    if col_cluster and np.isnan(intensity_df.to_numpy()).any():
        raise ValueError(
            "Intensity matrix contains NaN values; cannot perform "
            "column clustering. Set col_cluster=False or impute "
            "missing values first."
        )

    # -- categorical margin annotation bars
    if row_margin_color is not None:
        row_colors_df, row_mapping = _build_margin_colors(
            adata.var,
            intensity_df.index,
            row_margin_color,
            color_scheme=color_scheme,
        )
    else:
        row_colors_df, row_mapping = None, None

    if col_margin_color is not None:
        col_colors_df, col_mapping = _build_margin_colors(
            adata.obs,
            intensity_df.columns,
            col_margin_color,
            color_scheme=color_scheme,
        )
    else:
        col_colors_df, col_mapping = None, None

    # -- plot via clustermap
    if cmap is None:
        cmap = "RdBu_r" if zscore else "viridis"
    cbar_label = "z-score" if zscore else "intensity"
    cbar_title = "Z-score" if zscore else "Intensity"
    g = sns.clustermap(
        intensity_df,
        cmap=cmap,
        center=0 if zscore else None,
        row_cluster=row_cluster,
        col_cluster=col_cluster,
        row_linkage=row_link,
        col_linkage=None,
        row_colors=row_colors_df,
        col_colors=col_colors_df,
        cbar_kws={"label": cbar_label},
        xticklabels=True,
        yticklabels=True,
        figsize=figsize,
    )

    _toggle_dendrograms(
        g,
        row_cluster=row_cluster,
        col_cluster=col_cluster,
        row_dendrogram=row_dendrogram,
        col_dendrogram=col_dendrogram,
    )
    _compress_heatmap_horizontally(g, scale=0.82)
    if row_colors_df is not None or col_colors_df is not None:
        _inset_margin_bars(g)
        _hide_margin_bar_labels(g)
    right_x = _layout_right_side_panel(g, cbar_title=cbar_title)
    if row_mapping is not None or col_mapping is not None:
        anchor_y_top = g.ax_heatmap.get_position().y1
        _add_stacked_margin_legends(
            g,
            row_mapping=row_mapping,
            col_mapping=col_mapping,
            anchor_x=right_x,
            anchor_y_top=anchor_y_top,
        )

    if title is None:
        title = (
            f"Peptide intensities: {resolved_gid}"
            if resolved_gid is not None
            else "Peptide intensities"
        )
    _place_title_above_top(g, title)
    g.ax_heatmap.set_xlabel("")
    g.ax_heatmap.set_ylabel("")
    g.ax_heatmap.tick_params(axis="both", labelsize=label_size)

    _finalize_clustermap(g, save=save, show=show)
    return g


def proteoform_scores(
    adata: ad.AnnData,
    *,
    adj: bool = True,
    pval_threshold: float | int | None = None,
    score_threshold: float | int | None = None,
    yscale_log: bool = True,
    protein_id_key: str | None = None,
    highlight_prots: list[str] | None = None,
    protein_label_fontsize: int | float = 8,
    protein_label_color: str = "black",
    show: bool = True,
    save: str | Path | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Scatter plot of COPF proteoform scores vs. p-values.

    Parameters
    ----------
    adata : AnnData
        :class:`~anndata.AnnData` with COPF score annotations in ``.var``.
    adj : bool
        Use adjusted ``proteoform_score_pval_adj`` values when ``True``.
    pval_threshold : float | int | None
        Maximum p-value used to highlight points. ``None`` disables filtering
        by p-value.
    score_threshold : float | int | None
        Minimum proteoform score used to highlight points. ``None`` disables
        score-based filtering.
    yscale_log : bool
        When ``True``, plot p-values on a log10-scaled inverted
        y-axis. When ``False``, plot ``-log10(pval)`` on a linear
        y-axis.
    protein_id_key : str | None
        Column in ``.var`` whose values are used as display labels
        instead of ``protein_id``. 1-to-1 mapping between ``protein_id`` and
        ``protein_id_key`` is enforced.
    highlight_prots : list[str] | None
        Protein IDs to highlight with text labels on the scatter
        plot. When ``protein_id_key`` is set, values must come
        from the ``protein_id_key`` column.
    protein_label_fontsize : int | float
        Font size for the highlight labels.
    protein_label_color : str
        Color for the highlight labels and connector lines.
    show : bool
        Call :func:`matplotlib.pyplot.show` when ``True``.
    save : str | Path | None
        File path to save the figure. ``None`` skips saving.
    ax : matplotlib.axes.Axes | None
        Matplotlib Axes object to plot onto. If ``None``, a new
        figure and axes are created.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes object used for plotting.

    Examples
    --------
    Basic scatter plot of proteoform scores:

    >>> import proteopy as pr
    >>> adata = pr.read.long(...)
    >>> pr.tl.pairwise_peptide_correlations(adata)
    >>> pr.tl.peptide_dendograms_by_correlation(
    ...     adata,
    ...     corrs_key='pairwise_correlations;protein_id;;;',
    ...     method='agglomerative-hierarchical-clustering',
    ... )
    >>> pr.tl.peptide_clusters_from_dendograms(
    ...     adata,
    ...     n_clusters=2,
    ...     min_peptides_per_cluster=2,
    ... )
    >>> pr.tl.proteoform_scores(adata, min_pval_adj=0.4)
    >>> pr.pl.proteoform_scores(adata)

    Highlight specific proteins by ``protein_id``:

    >>> pr.pl.proteoform_scores(
    ...     adata,
    ...     highlight_prots=["P12345", "Q67890"],
    ... )

    Highlight proteins using an alternative label column:

    >>> pr.pl.proteoform_scores(
    ...     adata,
    ...     protein_id_key="gene_name",
    ...     highlight_prots=["GAPDH", "ACTB"],
    ...     protein_label_color="red",
    ...     protein_label_fontsize=10,
    ... )
    """

    check_proteodata(adata)

    if not isinstance(yscale_log, bool):
        raise TypeError("yscale_log must be a bool.")

    if adj:
        pval_col = "proteoform_score_pval_adj"
    else:
        pval_col = "proteoform_score_pval"

    required_cols = {"proteoform_score", pval_col}
    missing = required_cols.difference(adata.var.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(
            "Missing required columns in `adata.var`: " f"{missing_str}"
        )

    var = adata.var.loc[:, ["proteoform_score", pval_col]].copy()
    var = var.drop_duplicates()
    var = var.dropna(subset=["proteoform_score", pval_col])

    # Filter out invalid p-values before plotting.
    finite_mask = np.isfinite(var[pval_col])
    if not finite_mask.all():
        warnings.warn(
            "Dropping entries with non-finite p-values.",
            RuntimeWarning,
        )
        var = var.loc[finite_mask]

    positive_mask = var[pval_col] > 0
    if not positive_mask.all():
        warnings.warn(
            "Dropping non-positive p-values before plotting.",
            RuntimeWarning,
        )
        var = var.loc[positive_mask]

    if yscale_log:
        plot_pvals = var[pval_col]
        ylabel = "adj. p-value" if adj else "p-value"
    else:
        plot_pvals = -np.log10(var[pval_col])
        if adj:
            ylabel = "-log10(adj. p-value)"
        else:
            ylabel = "-log10(p-value)"

    if var.empty:
        raise ValueError("No valid proteoform scores available for plotting.")

    def _validate_threshold(
        value: float | int | None,
        *,
        name: str,
        allow_zero: bool = False,
        upper_bound: float | None = None,
    ) -> float | int | None:
        if value is None:
            return None
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a number, not bool.")
        if not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError(f"{name} must be a real number.")
        if not np.isfinite(value):
            raise ValueError(f"{name} must be a finite number.")
        if not allow_zero and value <= 0:
            raise ValueError(f"{name} must be greater than 0.")
        if upper_bound is not None and value > upper_bound:
            raise ValueError(
                f"{name} must be less than or equal to {upper_bound}."
            )
        return value

    pval_threshold = _validate_threshold(
        pval_threshold,
        name="pval_threshold",
        allow_zero=False,
        upper_bound=1.0,
    )
    score_threshold = _validate_threshold(
        score_threshold,
        name="score_threshold",
        allow_zero=True,
    )

    if pval_threshold is not None:
        if yscale_log:
            pval_threshold_line = pval_threshold
        else:
            pval_threshold_line = -np.log10(pval_threshold)
    else:
        pval_threshold_line = None

    mask = pd.Series(True, index=var.index)
    has_condition = False
    if score_threshold is not None:
        mask &= var["proteoform_score"] >= score_threshold
        has_condition = True
    if pval_threshold is not None:
        mask &= var[pval_col] <= pval_threshold
        has_condition = True
    if not has_condition:
        mask[:] = False

    var["is_above_threshold"] = mask
    var["plot_pval"] = plot_pvals

    if ax is not None:
        _ax = ax
        _fig = _ax.get_figure()
    else:
        _fig, _ax = plt.subplots()
    sns.scatterplot(
        data=var,
        x="proteoform_score",
        y="plot_pval",
        hue="is_above_threshold",
        palette={True: "#008A1D", False: "#BDBDBD"},
        alpha=0.5,
        s=30,
        edgecolor=None,
        legend=False,
        ax=_ax,
    )

    if yscale_log:
        _ax.set_yscale("log", base=10)
        _ax.invert_yaxis()
        _ax.yaxis.set_minor_locator(
            LogLocator(
                base=10,
                subs=np.arange(2, 10) * 0.1,
                numticks=12,
            )
        )
        _ax.yaxis.set_minor_formatter(plt.NullFormatter())

    # -- Highlight selected proteins with text labels --------
    if highlight_prots is not None:
        if not isinstance(highlight_prots, list) or not all(
            isinstance(v, str) for v in highlight_prots
        ):
            raise TypeError("`highlight_prots` must be a list of strings.")

        # Build protein_id <-> display label mapping.
        if protein_id_key is not None:
            if protein_id_key not in adata.var.columns:
                raise ValueError(
                    f"Column '{protein_id_key}' not found " "in `adata.var`."
                )
            # Validate 1-to-1 mapping.
            mapping_df = adata.var[
                ["protein_id", protein_id_key]
            ].drop_duplicates()
            dup_proteins = mapping_df.groupby("protein_id")[
                protein_id_key
            ].nunique()
            bad = dup_proteins[dup_proteins > 1]
            if not bad.empty:
                raise ValueError(
                    "1-to-1 mapping violation between "
                    f"'protein_id' and '{protein_id_key}' "
                    "for protein(s): "
                    f"{sorted(bad.index.tolist())}"
                )
            pid_to_label = dict(
                zip(
                    mapping_df["protein_id"],
                    mapping_df[protein_id_key],
                )
            )
            label_to_pid = dict(
                zip(
                    mapping_df[protein_id_key],
                    mapping_df["protein_id"],
                )
            )

            # highlight_prots may contain protein_id_key
            # values — resolve them to protein_ids.
            known_labels = set(mapping_df[protein_id_key])
            resolved_pids = set()
            unknown = set(highlight_prots) - known_labels
            if unknown:
                raise ValueError(
                    "The following values from "
                    "`highlight_prots` are not found in "
                    f"`adata.var['{protein_id_key}']`: "
                    f"{sorted(unknown)}"
                )
            highlight_pids = {label_to_pid[v] for v in highlight_prots}
        else:
            pid_to_label = None
            known_ids = set(adata.var["protein_id"])
            unknown = set(highlight_prots) - known_ids
            if unknown:
                raise ValueError(
                    "The following protein IDs from "
                    "`highlight_prots` are not found in "
                    "`adata.var['protein_id']`: "
                    f"{sorted(unknown)}"
                )
            highlight_pids = set(highlight_prots)

        # Map var index back to protein_id for the
        # deduplicated var DataFrame.
        pid_series = adata.var.loc[var.index, "protein_id"]
        highlight_mask = pid_series.isin(highlight_pids)
        var_highlight = var.loc[highlight_mask.values]

        if not var_highlight.empty:
            texts = []
            for idx in var_highlight.index:
                pid = pid_series.loc[idx]
                if pid_to_label is not None:
                    label = pid_to_label[pid]
                else:
                    label = pid
                texts.append(
                    _ax.text(
                        var_highlight.loc[idx, "proteoform_score"],
                        var_highlight.loc[idx, "plot_pval"],
                        label,
                        fontsize=protein_label_fontsize,
                        color=protein_label_color,
                    )
                )
            adjust_text(
                texts,
                x=var["proteoform_score"].values,
                y=var["plot_pval"].values,
                ax=_ax,
                force_points=(2.0, 2.0),
                force_text=(1.0, 1.0),
                expand=(2.0, 2.0),
                arrowprops=dict(
                    arrowstyle="-",
                    color="grey",
                    lw=0.5,
                ),
            )

    if score_threshold is not None:
        _ax.axvline(
            score_threshold,
            color="#A2A2A2",
            linestyle="--",
        )
    if pval_threshold_line is not None:
        _ax.axhline(
            pval_threshold_line,
            color="#A2A2A2",
            linestyle="--",
        )

    _ax.set_xlabel("Proteoform Score")
    _ax.set_ylabel(ylabel)
    _fig.tight_layout()

    if save is not None:
        if not isinstance(save, (str, Path)):
            raise TypeError("`save` must be a path-like object or None.")
        _fig.savefig(save, dpi=300, bbox_inches="tight")
    if show:
        plt.show()

    return _ax
