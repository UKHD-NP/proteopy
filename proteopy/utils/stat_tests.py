from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from adjustText import adjust_text


def volcano_plot(
    fc_vals: np.ndarray | pd.Series,
    pvals: np.ndarray | pd.Series,
    fc_thresh: float | None = None,
    pval_thresh: float | None = None,
    labels: Sequence[str] | pd.Series | np.ndarray | None = None,
    top_labels: int | None = None,
    figsize: tuple[float, float] = (6.0, 5.0),
    xlabel: str | None = None,
    ylabel: str | None = None,
    alt_color: pd.Series | list[bool] | np.ndarray | None = None,
    yscale_log: bool = True,
    title: str | None = None,
    show: bool = True,
    save: str | Path | None = None,
    ax: bool | None = None,
) -> Axes | None:
    """
    Core volcano plot renderer (framework-agnostic).

    Draws a scatter plot of fold change (x-axis) versus p-value
    (y-axis). Points are colored by significance or by an optional
    alternative boolean mask. This is an internal helper; public
    callers should use ``proteopy.pl.volcano_plot``.

    Parameters
    ----------
    fc_vals : np.ndarray | pd.Series
        Fold change values (x-axis).
    pvals : np.ndarray | pd.Series
        P-values (y-axis). Must be same length as ``fc_vals``.
    fc_thresh : float | None, optional
        Absolute fold change threshold for significance. When
        ``None``, the fold change requirement for significance
        coloring is dropped. Threshold line is not drawn.
    pval_thresh : float | None, optional
        P-value threshold for significance. When ``None``, the
        p-value requirement for significance coloring is dropped.
        Threshold line is not drawn. When both thresholds are
        ``None``, all points are colored as significant (blue
        for negative FC, red for positive FC).
    labels : Sequence[str] | pd.Series | np.ndarray | None, optional
        Labels for each point, same length as ``fc_vals``.
        Required when ``top_labels`` is set.
    top_labels : int | None, optional
        Number of top proteins to label per side (up to 2N
        total). Ranked by smallest p-value, then largest
        ``|fc|``.
    figsize : tuple[float, float], optional
        Figure dimensions (width, height) in inches.
    xlabel : str | None, optional
        X-axis label. Defaults to ``"logFC"`` when ``None``.
    ylabel : str | None, optional
        Y-axis label. When ``None``, defaults to ``"pval"`` if
        ``yscale_log=True`` or ``"-log10(pval)"`` if
        ``yscale_log=False``.
    alt_color : pd.Series | list[bool] | np.ndarray | None, optional
        Boolean mask (same length as ``fc_vals``) for
        alternative coloring. ``True`` entries are colored
        purple, ``False`` gray. Overrides significance-based
        coloring.
    yscale_log : bool, optional
        When ``True``, plot raw p-values on a log10-scaled
        inverted y-axis. When ``False``, plot ``-log10(pval)``
        on a linear y-axis.
    title : str | None, optional
        Plot title. If ``None``, no title is set.
    show : bool, optional
        Call ``matplotlib.pyplot.show()`` to display the plot.
    save : str | Path | None, optional
        File path to save the figure at 300 DPI.
    ax : bool | None, optional
        If ``True``, return the Axes object.

    Returns
    -------
    Axes | None
        The Matplotlib Axes object if ``ax=True``, otherwise
        ``None``.

    Raises
    ------
    ValueError
        If ``fc_vals`` and ``pvals`` have different lengths, if
        no valid data remains after filtering, if ``top_labels``
        is set but ``labels`` is ``None``, if ``top_labels`` is
        not a positive integer, if ``alt_color`` fails
        validation, or if ``show``, ``save``, and ``ax`` are all
        falsy.
    """
    if not save and not show and not ax:
        raise ValueError(
            "Args show, ax and save all set to False, "
            "function does nothing.",
        )

    fc_vals = np.asarray(fc_vals, dtype=float)
    pvals = np.asarray(pvals, dtype=float)

    if fc_vals.shape != pvals.shape:
        raise ValueError(
            "fc_vals and pvals must have the same length."
        )

    if top_labels is not None:
        if labels is None:
            raise ValueError(
                "labels must be provided when "
                "top_labels is set."
            )
        if not isinstance(top_labels, int) or top_labels <= 0:
            raise ValueError(
                "top_labels must be a positive integer."
            )

    if labels is not None:
        labels = np.asarray(labels)
        if labels.shape[0] != fc_vals.shape[0]:
            raise ValueError(
                "labels must have the same length as "
                "fc_vals."
            )

    # Validate alt_color before filtering
    if alt_color is not None:
        alt_color = np.asarray(alt_color)
        if alt_color.ndim != 1:
            raise ValueError(
                "alt_color must be a 1D boolean sequence."
            )
        if alt_color.shape[0] != fc_vals.shape[0]:
            raise ValueError(
                "alt_color must have the same length as "
                "fc_vals."
            )
        if not np.issubdtype(alt_color.dtype, np.bool_):
            raise ValueError("alt_color must be boolean.")

    # ----- Filtering -----
    # Drop NaN
    nan_mask = np.isnan(fc_vals) | np.isnan(pvals)
    if nan_mask.any():
        warnings.warn(
            "Dropping entries with NaN fold changes or "
            "p-values.",
            RuntimeWarning,
        )

    # Drop non-finite (inf, -inf)
    nonfinite_mask = (
        ~np.isfinite(fc_vals) | ~np.isfinite(pvals)
    )
    inf_only = nonfinite_mask & ~nan_mask
    if inf_only.any():
        warnings.warn(
            "Dropping entries with non-finite fold changes "
            "or p-values.",
            RuntimeWarning,
        )

    # Drop non-positive p-values
    nonpos_mask = pvals <= 0
    nonpos_new = nonpos_mask & ~nonfinite_mask
    if nonpos_new.any():
        warnings.warn(
            "Dropping non-positive p-values before log "
            "transform.",
            RuntimeWarning,
        )

    keep = ~(nonfinite_mask | nonpos_mask)
    fc_vals = fc_vals[keep]
    pvals = pvals[keep]
    if labels is not None:
        labels = labels[keep]
    if alt_color is not None:
        alt_color = alt_color[keep]

    if len(fc_vals) == 0:
        raise ValueError(
            "No valid results available for plotting."
        )

    # ----- Prepare plotting arrays -----
    neg_log_p = -np.log10(pvals)
    y_vals = pvals if yscale_log else neg_log_p

    # Significance masks — when a threshold is None, that
    # requirement is dropped. If both are None, all points are
    # significant and colored by FC sign.
    sig_mask = (
        np.ones(len(pvals), dtype=bool)
        if pval_thresh is None
        else pvals <= pval_thresh
    )
    if fc_thresh is None:
        up_mask = sig_mask & (fc_vals > 0)
        down_mask = sig_mask & (fc_vals < 0)
    else:
        up_mask = sig_mask & (fc_vals >= fc_thresh)
        down_mask = sig_mask & (fc_vals <= -fc_thresh)
    other_mask = ~(up_mask | down_mask)

    # ----- Scatter plot -----
    fig, _ax = plt.subplots(figsize=figsize)

    if alt_color is None:
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
            color="#1f77b4",
            alpha=0.8,
            s=14,
        )
        _ax.scatter(
            fc_vals[up_mask],
            y_vals[up_mask],
            color="#d62728",
            alpha=0.8,
            s=14,
        )
    else:
        _ax.scatter(
            fc_vals[~alt_color],
            y_vals[~alt_color],
            color="grey",
            alpha=0.5,
            s=12,
        )
        _ax.scatter(
            fc_vals[alt_color],
            y_vals[alt_color],
            color="#8E54E5",
            alpha=0.8,
            s=14,
        )

    # Threshold lines — only drawn when the threshold is set
    if fc_thresh is not None:
        _ax.axvline(
            fc_thresh,
            color="black",
            linestyle="--",
            linewidth=1,
        )
        _ax.axvline(
            -fc_thresh,
            color="black",
            linestyle="--",
            linewidth=1,
        )
    if yscale_log:
        if pval_thresh is not None:
            _ax.axhline(
                pval_thresh,
                color="black",
                linestyle="--",
                linewidth=1,
            )
        _ax.set_yscale("log", base=10)
        _ax.invert_yaxis()
    else:
        if pval_thresh is not None:
            _ax.axhline(
                -np.log10(pval_thresh),
                color="black",
                linestyle="--",
                linewidth=1,
            )

    # Axis labels
    _ax.set_xlabel(xlabel or "logFC")
    if ylabel is not None:
        _ax.set_ylabel(ylabel)
    elif yscale_log:
        _ax.set_ylabel("pval")
    else:
        _ax.set_ylabel("-log10(pval)")

    if title is not None:
        _ax.set_title(title)

    # ----- Top labels -----
    if top_labels is not None and labels is not None:
        abs_fc = np.abs(fc_vals)
        label_mask = (
            sig_mask
            if fc_thresh is None
            else sig_mask & (abs_fc >= fc_thresh)
        )
        idx = np.where(label_mask)[0]

        if len(idx) > 0:
            lbl_fc = fc_vals[idx]
            lbl_pv = pvals[idx]
            lbl_abs = abs_fc[idx]
            lbl_labels = labels[idx]
            lbl_y = y_vals[idx]

            # Positive side
            pos_idx = np.where(lbl_fc > 0)[0]
            if len(pos_idx) > 0:
                order = np.lexsort(
                    (-lbl_abs[pos_idx], lbl_pv[pos_idx]),
                )
                pos_sel = pos_idx[order[:top_labels]]
            else:
                pos_sel = np.array([], dtype=int)

            # Negative side
            neg_idx = np.where(lbl_fc < 0)[0]
            if len(neg_idx) > 0:
                order = np.lexsort(
                    (-lbl_abs[neg_idx], lbl_pv[neg_idx]),
                )
                neg_sel = neg_idx[order[:top_labels]]
            else:
                neg_sel = np.array([], dtype=int)

            sel = np.concatenate([pos_sel, neg_sel])
            texts = []
            for i in sel:
                texts.append(
                    _ax.text(
                        lbl_fc[i],
                        lbl_y[i],
                        str(lbl_labels[i]),
                        fontsize=8,
                    )
                )
            if texts:
                adjust_text(
                    texts,
                    ax=_ax,
                    arrowprops=dict(
                        arrowstyle="->",
                        color="0.4",
                        lw=0.7,
                    ),
                )

    # ----- Save / show / return -----
    if save:
        fig.savefig(save, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    if ax:
        return _ax
    return None
