from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import seaborn as sns
import anndata as ad
from matplotlib.axes import Axes
from adjustText import adjust_text

from proteopy.utils.anndata import check_proteodata

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
                    f"Column '{protein_id_key}' not found "
                    "in `adata.var`."
                )
            # Validate 1-to-1 mapping.
            mapping_df = adata.var[
                ["protein_id", protein_id_key]
            ].drop_duplicates()
            dup_proteins = (
                mapping_df
                .groupby("protein_id")[protein_id_key]
                .nunique()
            )
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
            unknown = (set(highlight_prots) - known_labels)
            if unknown:
                raise ValueError(
                    "The following values from "
                    "`highlight_prots` are not found in "
                    f"`adata.var['{protein_id_key}']`: "
                    f"{sorted(unknown)}"
                )
            highlight_pids = {
                label_to_pid[v] for v in highlight_prots
            }
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
            raise TypeError(
                "`save` must be a path-like object or None."
            )
        _fig.savefig(save, dpi=300, bbox_inches="tight")
    if show:
        plt.show()

    return _ax
