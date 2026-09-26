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
    margin_color: str | list[str] = "cluster_id",
    method: str = "average",
    cluster: bool = True,
    linkage=None,
    color_scheme=None,
    cmap: str = "coolwarm",
    xticklabels: bool = True,
    yticklabels: bool = True,
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
        per entry, in the given order. Proteoform labels are obtained by
        passing ``"proteoform_id"``.
    method : str
        Linkage method handed to :func:`scipy.cluster.hierarchy.linkage`
        when the tree is recomputed from ``1 - correlation``.
    cluster : bool
        Cluster and draw dendrograms on both axes. When ``False`` the
        matrix is shown in its input order without dendrograms.
    linkage : numpy.ndarray | None
        Precomputed SciPy linkage matrix. When provided it is used for
        both axes, shadowing the default ``1 - correlation`` computation.
    color_scheme : Any
        Color palette specification understood by
        :func:`proteopy.utils.matplotlib._resolve_color_scheme`.
    cmap : str
        Continuous colormap for the heatmap body.
    xticklabels, yticklabels : bool
        Whether to show peptide tick labels on each axis.
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
        has no pairwise correlations, or clustering is requested on a
        matrix containing missing values.
    KeyError
        If a ``margin_color`` column is not present in ``adata.var``.

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
        raise KeyError(
            "margin_color column(s) not found in adata.var: "
            f"{', '.join(missing_cols)}."
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
    corr_df = reconstruct_symmetric_matrix_from_long(
        corr_long,
        var_a_col="pepA",
        var_b_col="pepB",
        value_col="PCC",
    )

    peptides = list(corr_df.index)

    # -- Labels come from the peptide_id column, never the index
    pep_ids = adata.var.loc[peptides, "peptide_id"].astype(str).tolist()

    # -- Build annotation strip(s) and their legend handles
    annot_colors = pd.DataFrame(index=peptides)
    legend_handles: list[Patch] = []
    for col in margin_cols:
        groups = adata.var.loc[peptides, col]
        if isinstance(groups.dtype, pd.CategoricalDtype):
            cats = [c for c in groups.cat.categories if c in set(groups)]
        else:
            cats = sorted(map(str, groups.dropna().unique()))

        resolved = _resolve_color_scheme(color_scheme, cats)
        if resolved is None:
            resolved = (
                sns.color_palette(n_colors=len(cats)) if len(cats) else []
            )
        palette = {str(cat): color for cat, color in zip(cats, resolved)}

        color_series = groups.astype("string").map(palette)
        if color_series.isna().any() and groups.notna().any():
            bad = color_series.isna() & groups.notna()
            missing_cats = sorted(groups[bad].astype(str).unique())
            raise ValueError(
                f"No color provided for categories in '{col}': "
                f"{', '.join(missing_cats)}."
            )

        prefix = f"{col}: " if len(margin_cols) > 1 else ""
        legend_handles.extend(
            Patch(
                facecolor=palette[str(cat)],
                edgecolor="none",
                label=f"{prefix}{cat}",
            )
            for cat in cats
        )

        if groups.isna().any():
            na_color = mpl.colors.to_rgba("lightgray")
            color_series = color_series.astype(object)
            color_series[groups.isna()] = na_color
            legend_handles.append(
                Patch(
                    facecolor=na_color, edgecolor="none", label=f"{prefix}NA"
                )
            )

        annot_colors[col] = color_series.to_numpy()

    annot_colors.index = pep_ids
    corr_df.index = pep_ids
    corr_df.columns = pep_ids

    # -- Color center at the off-diagonal mean (as sample_correlation_matrix)
    A = corr_df.to_numpy(dtype=float)
    n = A.shape[0]
    if n > 1:
        offdiag = A[~np.eye(n, dtype=bool)]
        center_val = np.nanmean(offdiag)
    else:
        offdiag = A.ravel()
        center_val = float(np.nanmean(A))

    # -- Resolve the row/column linkage (symmetric)
    row_linkage = None
    col_linkage = None
    do_cluster = cluster
    if cluster:
        if linkage is not None:
            row_linkage = col_linkage = linkage
        else:
            if np.isnan(A).any():
                raise ValueError(
                    "Correlation matrix for protein "
                    f"'{protein}' contains missing values; clustering "
                    "is not possible. Pass cluster=False to skip "
                    "clustering."
                )
            dist = 1.0 - A
            np.fill_diagonal(dist, 0.0)
            dist = np.clip(dist, 0, 2)
            Z = scipy_linkage(squareform(dist, checks=False), method=method)
            row_linkage = col_linkage = Z

    # -- Optional statistics printout
    if print_stats and n > 1:
        summary = pd.DataFrame(
            {
                "min": [np.nanmin(offdiag)],
                "max": [np.nanmax(offdiag)],
                "mean": [np.nanmean(offdiag)],
                "median": [np.nanmedian(offdiag)],
                "std": [np.nanstd(offdiag)],
            }
        )
        print(f"Peptide correlation summary (off-diagonal, {protein}):")
        print(summary.to_string(index=False))
        print()

        clusters = adata.var.loc[peptides, "cluster_id"]
        rows = []
        for cid, grp in clusters.groupby(clusters, observed=True):
            idx = [peptides.index(p) for p in grp.index]
            block = A[np.ix_(idx, idx)]
            if len(idx) > 1:
                within = block[~np.eye(len(idx), dtype=bool)]
                mean_within = np.nanmean(within)
            else:
                mean_within = np.nan
            rows.append({"cluster_id": cid, "mean_within": mean_within})
        print(f"\nPer cluster_id (within-cluster mean correlation):")
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

    if legend_handles:
        g.ax_heatmap.legend(
            handles=legend_handles,
            title=", ".join(margin_cols),
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0.0,
            frameon=False,
        )

    g.ax_heatmap.set_xlabel("Peptides")
    g.ax_heatmap.set_ylabel("Peptides")
    g.figure.suptitle(protein)

    plt.tight_layout()

    if save is not None:
        if not isinstance(save, (str, Path)):
            raise TypeError("`save` must be a path-like object or None.")
        g.savefig(save, dpi=300, bbox_inches="tight")
    if show:
        plt.show()

    if ax:
        return g.ax_heatmap
    return None
