from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import anndata as ad
from matplotlib.axes import Axes

from copro.utils.anndata import check_proteodata

def proteoform_scores(
    adata: ad.AnnData,
    *,
    adj: bool = True,
    pval_threshold: float | int | None = None,
    score_threshold: float | int | None = None,
    log_scores: bool = False,
    show: bool = True,
    save: str | Path | None = None,
    ax: bool = False,
) -> Axes | None:
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
    log_scores : bool
        Plot ``-log10(p-value)`` on the y-axis when ``True``; otherwise plot
        raw p-values.
    show : bool
        Call :func:`matplotlib.pyplot.show` when ``True``.
    save : str | Path | None
        File path to save the figure. ``None`` skips saving.
    ax : bool
        Return the created :class:`matplotlib.axes.Axes` instead of ``None``.
    """

    check_proteodata(adata)

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

    if log_scores:
        positive_mask = var[pval_col] > 0
        if not positive_mask.all():
            warnings.warn(
                "Dropping non-positive p-values before log-transforming.",
                RuntimeWarning,
            )
            var = var.loc[positive_mask]
        plot_pvals = -np.log10(var[pval_col])
        if adj:
            ylabel = "-log10(adj. p-value)"
        else:
            ylabel = "-log10(p-value)"
    else:
        non_negative = var[pval_col] >= 0
        if not non_negative.all():
            warnings.warn(
                "Dropping negative p-values before plotting.",
                RuntimeWarning,
            )
            var = var.loc[non_negative]
        plot_pvals = var[pval_col]
        ylabel = "adj. p-value" if adj else "p-value"

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
        if log_scores:
            pval_threshold_line = -np.log10(pval_threshold)
        else:
            pval_threshold_line = pval_threshold
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
    if ax:
        return _ax
    if not (show or save or ax):
        raise ValueError(
            "Function does nothing: set one of `show`, `save`, or `ax`."
        )
