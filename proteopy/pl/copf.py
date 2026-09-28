from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
from matplotlib.patches import Patch
import seaborn as sns
import anndata as ad
from matplotlib.axes import Axes
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage as scipy_linkage
from adjustText import adjust_text

from proteopy.utils.anndata import check_proteodata
from proteopy.utils.matplotlib import _resolve_color_scheme
from proteopy.utils._matrix_wrangling import (
    reconstruct_symmetric_matrix_from_long,
)

NAN_HATCH = "////"
NAN_HATCH_COLOR = "#9e9e9e"


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


def pairwise_peptide_correlations_heatmap(
    adata: ad.AnnData,
    protein: str,
    *,
    corr_key: str = "pairwise_peptide_correlations",
    margin_color: str | list[str] = "proteoform_id",
    method: str = "average",
    cluster: bool = True,
    linkage=None,
    color_scheme=None,
    cmap: str = "coolwarm",
    xticklabels: bool | str = "auto",
    yticklabels: bool | str = "auto",
    figsize: tuple[float, float] = (8.0, 8.0),
    show: bool = True,
    ax: bool = False,
    print_stats: bool = False,
    save: str | Path | None = None,
) -> Axes | None:
    """Clustered peptide-correlation heatmap for a single protein.

    Draw the peptide x peptide Pearson-correlation matrix of one protein
    as a clustered heatmap, so that peptides of the same proteoform (COPF
    cluster) group into visible blocks. One or more categorical
    annotation strips (``margin_color``) are drawn alongside the heatmap,
    by default the per-protein ``cluster_id`` assignment.

    The correlations are read from
    ``adata.uns[corr_key]`` (produced by
    :func:`proteopy.tl.pairwise_peptide_correlations`) and reconstructed
    into a full symmetric matrix for the selected protein. The clustering
    is symmetric: the same tree is used for rows and columns.

    Parameters
    ----------
    adata : AnnData
        :class:`~anndata.AnnData` carrying COPF annotations.
    protein : str
        ``protein_id`` of the protein to plot.
    corr_key : str
        Key in ``adata.uns`` holding the long-form pairwise correlations
        (columns ``pepA``, ``pepB``, ``PCC``; indexed by ``protein_id``).
    margin_color : str | list[str]
        Column(s) in ``adata.var`` used for the annotation strip(s). A
        string is treated as a single-item list; a list draws one strip
        per entry, in the given order. The default shows the COPF
        proteoform labels (``<protein_id>_<cluster_id>``) written by
        :func:`proteopy.tl.peptide_clusters_from_dendograms`. Colours
        are assigned across all columns together, so strips never share
        a colour, and each column gets its own legend, placed right of
        all heatmap labels. Legend entries follow the column's category
        order if it is categorical, otherwise the lexicographic order of
        the ``str``-coerced values; store the column as an ordered
        :class:`pandas.Categorical` to change it. Missing values are
        drawn in light gray.
    method : str
        Linkage method handed to :func:`scipy.cluster.hierarchy.linkage`
        when the tree is recomputed from ``1 - correlation``.
    cluster : bool
        Cluster and draw dendrograms on both axes. When ``False``,
        peptides follow the category order of ``adata.var["peptide_id"]``
        if it is categorical, otherwise lexicographic order. ``NaN``
        correlations (e.g. from constant peptides) are drawn hatched and
        treated as ``r = 0`` when building the tree.
    linkage : numpy.ndarray | None
        Precomputed SciPy linkage matrix of shape ``(n_peptides - 1, 4)``.
        When provided it is used for both axes, shadowing the default
        ``1 - correlation`` computation. Leaf indices refer to the
        peptides in the default order described under ``cluster``.
    color_scheme : Any
        Color palette specification understood by
        :func:`proteopy.utils.matplotlib._resolve_color_scheme`.
    cmap : str
        Continuous colormap for the heatmap body.
    xticklabels, yticklabels : bool | str
        Peptide tick labels on each axis. ``"auto"`` labels as many
        peptides as fit without overlapping; ``True`` labels every
        peptide (enlarge ``figsize`` for large proteins); ``False``
        hides the labels.
    figsize : tuple[float, float]
        Matplotlib figure size in inches.
    show : bool
        Display the figure with :func:`matplotlib.pyplot.show`.
    ax : bool
        Return the heatmap :class:`matplotlib.axes.Axes` when ``True``.
    print_stats : bool
        Print correlation summary statistics before drawing.
    save : str | Path | None
        File path to save the figure. ``None`` skips saving.

    Returns
    -------
    Axes or None
        Heatmap axes when ``ax`` is ``True``; otherwise ``None``.

    Raises
    ------
    ValueError
        If ``protein`` is unknown, ``corr_key`` is missing, the protein
        has no pairwise correlations, the table holds several
        correlations per peptide pair (e.g. per-batch results),
        ``margin_color`` is empty or repeats a column, ``linkage`` does
        not match the number of peptides, or clustering is requested on
        a matrix containing missing values.
    KeyError
        If a ``margin_color`` column is not present in ``adata.var``.
    TypeError
        If ``method``, ``cluster``, ``margin_color`` or ``save`` has the
        wrong type.

    Warns
    -----
    UserWarning
        If peptides in ``adata.uns[corr_key]`` are no longer in
        ``adata.var`` (filtered after the correlations were computed;
        they are left out), if peptides of ``protein`` have no
        correlations (shown as empty rows and columns), if NaN
        correlations are treated as ``r = 0`` for clustering, or if
        ``adata`` has fewer than 3 samples.

    Notes
    -----
    The heatmap only displays correlations; it does not transform
    intensities. They are computed by
    :func:`proteopy.tl.pairwise_peptide_correlations` on ``adata.X`` as
    stored, so their scale is the scale of ``.X``: Pearson correlations
    of raw and log-transformed intensities differ, and a single extreme
    sample can dominate raw-scale correlations. Log-transform (and
    aggregate precursors to peptides) before computing correlations.
    ``scipy.stats.pearsonr`` is not NaN-aware, so a pair is ``NaN`` as
    soon as either peptide is missing in any sample, and there is no
    minimum-overlap cutoff.

    Examples
    --------
    >>> import proteopy as pr
    >>> adata = pr.read.long(...)
    >>> pr.tl.pairwise_peptide_correlations(adata)
    >>> pr.tl.peptide_dendograms_by_correlation(
    ...     adata,
    ...     method='agglomerative-hierarchical-clustering',
    ... )
    >>> pr.tl.peptide_clusters_from_dendograms(
    ...     adata,
    ...     n_clusters=2,
    ...     min_peptides_per_cluster=2,
    ... )
    >>> pr.pl.pairwise_peptide_correlations_heatmap(adata, protein="P12345")

    Annotate by proteoform and draw two strips:

    >>> pr.pl.pairwise_peptide_correlations_heatmap(
    ...     adata,
    ...     protein="P12345",
    ...     margin_color=["cluster_id", "proteoform_id"],
    ... )
    """
    check_proteodata(adata)

    # -- Validate arguments up front
    if not isinstance(method, str):
        raise TypeError("`method` must be a string.")
    if not isinstance(cluster, bool):
        raise TypeError("`cluster` must be a bool.")

    if isinstance(margin_color, str):
        margin_cols = [margin_color]
    elif isinstance(margin_color, list) and all(
        isinstance(c, str) for c in margin_color
    ):
        margin_cols = list(margin_color)
    else:
        raise TypeError("`margin_color` must be a string or list of strings.")
    if not margin_cols:
        raise ValueError("`margin_color` must name at least one column.")
    duplicated_cols = sorted(
        {c for c in margin_cols if margin_cols.count(c) > 1}
    )
    if duplicated_cols:
        raise ValueError(
            "Duplicate columns in `margin_color`: "
            f"{', '.join(duplicated_cols)}."
        )
    if save is not None and not isinstance(save, (str, Path)):
        raise TypeError("`save` must be a path-like object or None.")

    if protein not in set(adata.var["protein_id"]):
        raise ValueError(
            f"protein '{protein}' not found in adata.var['protein_id']."
        )

    if corr_key not in adata.uns:
        raise ValueError(
            f"'{corr_key}' not found in adata.uns; run "
            "pr.tl.pairwise_peptide_correlations() first."
        )

    missing_cols = [c for c in margin_cols if c not in adata.var.columns]
    if missing_cols:
        hint = ""
        if {"cluster_id", "proteoform_id"} & set(missing_cols):
            hint = (
                " These columns are written by "
                "pr.tl.peptide_clusters_from_dendograms(); run it first or "
                "pass another .var column."
            )
        raise KeyError(
            "margin_color column(s) not found in adata.var: "
            f"{', '.join(missing_cols)}.{hint}"
        )

    if adata.n_obs < 3:
        warnings.warn(
            f"adata has {adata.n_obs} sample(s); Pearson correlations "
            "from fewer than 3 samples are always +/-1 or undefined.",
            UserWarning,
            stacklevel=2,
        )

    corrs = adata.uns[corr_key]
    if protein not in corrs.index:
        raise ValueError(
            f"protein '{protein}' has no pairwise correlations in "
            f"adata.uns['{corr_key}'] (a protein needs at least two "
            "peptides)."
        )

    # -- Reconstruct the symmetric peptide x peptide correlation matrix
    corr_long = corrs.loc[[protein]]
    pairs = pd.DataFrame(
        np.sort(corr_long[["pepA", "pepB"]].to_numpy(dtype=str), axis=1)
    )
    if pairs.duplicated().any():
        raise ValueError(
            f"adata.uns['{corr_key}'] holds several correlations per "
            f"peptide pair for protein '{protein}' (e.g. per-batch "
            "results); pass a table with one correlation per pair, such "
            "as the pooled 'pairwise_peptide_correlations'."
        )
    corr_df = reconstruct_symmetric_matrix_from_long(
        corr_long,
        var_a_col="pepA",
        var_b_col="pepB",
        value_col="PCC",
        allow_missing=True,
    )

    # Peptides filtered out after pr.tl.pairwise_peptide_correlations
    # still sit in .uns; their correlations with the rest stay valid.
    stale = [p for p in corr_df.index if p not in adata.var.index]
    if stale:
        warnings.warn(
            f"{len(stale)} peptide(s) of protein '{protein}' in "
            f"adata.uns['{corr_key}'] are no longer in adata.var and are "
            "not shown.",
            UserWarning,
            stacklevel=2,
        )
        kept = [p for p in corr_df.index if p not in set(stale)]
        if len(kept) < 2:
            raise ValueError(
                f"Fewer than two peptides of protein '{protein}' remain "
                "in adata.var; recompute "
                "pr.tl.pairwise_peptide_correlations()."
            )
        corr_df = corr_df.loc[kept, kept]

    protein_peptides = adata.var.index[adata.var["protein_id"] == protein]
    without_corr = [p for p in protein_peptides if p not in corr_df.index]
    if without_corr:
        warnings.warn(
            f"{len(without_corr)} peptide(s) of protein '{protein}' have "
            f"no correlations in adata.uns['{corr_key}'] (e.g. all "
            "intensities missing); they are shown as empty rows and "
            "columns.",
            UserWarning,
            stacklevel=2,
        )
        full = list(corr_df.index) + without_corr
        corr_df = corr_df.reindex(index=full, columns=full)

    peptides = list(corr_df.index)

    # -- Labels come from the peptide_id column, never the index
    pep_ids = adata.var.loc[peptides, "peptide_id"].astype(str).tolist()

    # -- Categories per annotation column (category order, else
    # lexicographic)
    col_groups, col_cats = {}, {}
    for col in margin_cols:
        groups = adata.var.loc[peptides, col]
        if isinstance(groups.dtype, pd.CategoricalDtype):
            cats = [c for c in groups.cat.categories if c in set(groups)]
        else:
            cats = sorted(groups.dropna().unique(), key=str)
        col_groups[col], col_cats[col] = groups, cats

    # -- Colours are resolved across all columns at once, so several
    # strips never reuse the same colour
    all_cats = [cat for col in margin_cols for cat in col_cats[col]]
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if color_scheme is None and len(all_cats) > len(cycle):
        resolved = sns.color_palette("husl", len(all_cats))
    else:
        resolved = _resolve_color_scheme(color_scheme, all_cats)
        if resolved is None:
            resolved = sns.color_palette(n_colors=len(all_cats))

    annot_colors = pd.DataFrame(index=peptides)
    legend_groups: dict[str, list[Patch]] = {}
    offset = 0
    for col in margin_cols:
        groups, cats = col_groups[col], col_cats[col]
        colors = resolved[offset : offset + len(cats)]
        offset += len(cats)
        palette = {str(cat): color for cat, color in zip(cats, colors)}

        color_series = groups.astype("string").map(palette)
        bad = color_series.isna() & groups.notna()
        if bad.any():
            missing_cats = sorted(groups[bad].astype(str).unique())
            raise ValueError(
                f"No color provided for categories in '{col}': "
                f"{', '.join(missing_cats)}."
            )

        handles = [
            Patch(facecolor=palette[str(cat)], edgecolor="none", label=cat)
            for cat in cats
        ]

        if groups.isna().any():
            na_color = mpl.colors.to_rgba("lightgray")
            # Element-wise: a masked assignment would broadcast the tuple
            color_series = pd.Series(
                [
                    na_color if is_na else color
                    for color, is_na in zip(color_series, groups.isna())
                ],
                index=color_series.index,
                dtype=object,
            )
            handles.append(
                Patch(facecolor=na_color, edgecolor="none", label="NA")
            )

        annot_colors[col] = color_series.to_numpy()
        legend_groups[col] = handles

    annot_colors.index = pep_ids
    corr_df.index = pep_ids
    corr_df.columns = pep_ids

    # -- Default order (used when not clustering): peptide_id categories,
    # else lexicographic
    pep_col = adata.var.loc[peptides, "peptide_id"]
    if isinstance(pep_col.dtype, pd.CategoricalDtype):
        present = set(pep_ids)
        order = [str(c) for c in pep_col.cat.categories if str(c) in present]
    else:
        order = sorted(pep_ids)
    corr_df = corr_df.loc[order, order]
    annot_colors = annot_colors.loc[order]

    # -- Color center at the off-diagonal mean (as sample_correlation_matrix)
    A = corr_df.to_numpy(dtype=float)
    n = A.shape[0]
    offdiag = A[~np.eye(n, dtype=bool)]
    finite = offdiag[~np.isnan(offdiag)]
    center_val = float(finite.mean()) if finite.size else 0.0

    # -- Resolve the row/column linkage (symmetric)
    row_linkage = None
    col_linkage = None
    do_cluster = cluster
    if cluster:
        if linkage is not None:
            linkage = np.asarray(linkage, dtype=float)
            if linkage.shape != (n - 1, 4):
                raise ValueError(
                    f"`linkage` must have shape ({n - 1}, 4) for the {n} "
                    f"peptides of protein '{protein}'; got "
                    f"{linkage.shape}."
                )
            row_linkage = col_linkage = linkage
        else:
            n_nan = int(np.isnan(A[np.triu_indices(n, k=1)]).sum())
            if n_nan:
                warnings.warn(
                    f"{n_nan} correlation(s) of protein '{protein}' are "
                    "NaN (e.g. constant peptides or missing intensities); "
                    "they are drawn hatched and treated as r = 0 when "
                    "clustering.",
                    UserWarning,
                    stacklevel=2,
                )
            dist = 1.0 - np.nan_to_num(A, nan=0.0)
            np.fill_diagonal(dist, 0.0)
            dist = np.clip(dist, 0, 2)
            Z = scipy_linkage(squareform(dist, checks=False), method=method)
            row_linkage = col_linkage = Z

    # -- Optional statistics printout
    if print_stats and n > 1:
        values = finite if finite.size else np.array([np.nan])
        summary = pd.DataFrame(
            {
                "min": [values.min()],
                "max": [values.max()],
                "mean": [values.mean()],
                "median": [np.median(values)],
                "std": [values.std()],
            }
        )
        print(f"Peptide correlation summary (off-diagonal, {protein}):")
        print(summary.to_string(index=False))
        print()

        for col in margin_cols:
            groups = pd.Series(col_groups[col].to_numpy(), index=pep_ids)
            rows = []
            for gid in col_cats[col]:
                members = groups.index[groups == gid]
                block = corr_df.loc[members, members].to_numpy()
                within = block[~np.eye(len(members), dtype=bool)]
                within = within[~np.isnan(within)]
                mean_within = within.mean() if within.size else np.nan
                rows.append({col: gid, "mean_within": mean_within})
            print(f"Per {col} (within-group mean correlation):")
            print(pd.DataFrame(rows).to_string(index=False))
            print()

    # -- Draw the clustered heatmap
    clustermap_kwargs = dict(
        row_colors=annot_colors,
        col_colors=annot_colors,
        cmap=cmap,
        center=center_val,
        figsize=figsize,
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        cbar_kws={"label": "PCC"},
    )
    if do_cluster:
        clustermap_kwargs["row_linkage"] = row_linkage
        clustermap_kwargs["col_linkage"] = col_linkage
    else:
        clustermap_kwargs["row_cluster"] = False
        clustermap_kwargs["col_cluster"] = False

    g = sns.clustermap(corr_df, **clustermap_kwargs)

    # NaN cells are masked by seaborn; a hatched background keeps them
    # distinct from every colour on the correlation scale
    if np.isnan(A).any():
        g.ax_heatmap.patch.set_facecolor("white")
        g.ax_heatmap.patch.set_edgecolor(NAN_HATCH_COLOR)
        g.ax_heatmap.patch.set_hatch(NAN_HATCH)
        legend_groups["correlation"] = [
            Patch(
                facecolor="white",
                edgecolor=NAN_HATCH_COLOR,
                hatch=NAN_HATCH,
                label="NaN",
            )
        ]

    g.ax_heatmap.set_xlabel("Peptides")
    g.ax_heatmap.set_ylabel("Peptides")
    g.ax_col_dendrogram.set_title(protein)
    _place_margin_legends(g, legend_groups)

    if save is not None:
        g.savefig(save, dpi=300, bbox_inches="tight")
    if show:
        plt.show()

    if ax:
        return g.ax_heatmap
    return None


def _place_margin_legends(g, legend_groups):
    """Stack one legend per annotation column right of all heatmap text.

    The figure is widened when the legends would extend past its right
    edge; every axes keeps its size in inches, so nothing overlaps and
    nothing is cut off on screen or when saved.
    """
    fig = g.figure
    if hasattr(fig.canvas, "get_renderer"):
        renderer = fig.canvas.get_renderer()
    else:
        # Vector backends (pdf, svg) have no canvas renderer
        renderer = fig._get_renderer()
    to_fig = fig.transFigure.inverted()

    text_axes = [
        ax for ax in (g.ax_heatmap, g.ax_col_colors) if ax is not None
    ]
    right = max(
        ax.get_tightbbox(renderer).transformed(to_fig).x1 for ax in text_axes
    )
    x = right + 0.02
    y = g.ax_col_dendrogram.get_position().y1

    legends = []
    legends_right, legends_bottom = 1.0, 0.0
    for title, handles in legend_groups.items():
        legend = fig.legend(
            handles=handles,
            title=title,
            loc="upper left",
            bbox_to_anchor=(x, y),
            borderaxespad=0.0,
            frameon=False,
        )
        box = legend.get_window_extent(renderer).transformed(to_fig)
        legends.append((legend, y))
        legends_right = max(legends_right, box.x1)
        legends_bottom = min(legends_bottom, box.y0)
        y -= box.height + 0.02

    if legends_right <= 1.0 and legends_bottom >= 0.0:
        return
    width, height = fig.get_size_inches()
    new_width = width * max(legends_right + 0.01, 1.0)
    extra_bottom = height * max(-legends_bottom + 0.01, 0.0)
    new_height = height + extra_bottom
    fig.set_size_inches(new_width, new_height)

    def new_x(v):
        return v * width / new_width

    def new_y(v):
        return (v * height + extra_bottom) / new_height

    for ax in fig.axes:
        pos = ax.get_position()
        ax.set_position(
            [
                new_x(pos.x0),
                new_y(pos.y0),
                pos.width * width / new_width,
                pos.height * height / new_height,
            ]
        )
    for legend, top in legends:
        legend.set_bbox_to_anchor(
            (new_x(x), new_y(top)), transform=fig.transFigure
        )
