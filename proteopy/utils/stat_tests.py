from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from adjustText import adjust_text


def _validate_numeric_inputs(fc_vals, pvals):
    try:
        fc_vals = np.asarray(fc_vals, dtype=float)
    except (ValueError, TypeError):
        raise ValueError(
            "fc_vals must contain numeric values."
        )
    try:
        pvals = np.asarray(pvals, dtype=float)
    except (ValueError, TypeError):
        raise ValueError(
            "pvals must contain numeric values."
        )

    if fc_vals.ndim != 1:
        raise ValueError("fc_vals must be 1D.")
    if pvals.ndim != 1:
        raise ValueError("pvals must be 1D.")

    if fc_vals.shape != pvals.shape:
        raise ValueError(
            "fc_vals and pvals must have the same length."
        )

    return fc_vals, pvals


def _validate_highlight_labels(highlight_labels):
    if len(highlight_labels) != len(set(highlight_labels)):
        raise ValueError(
            "highlight_labels must not contain "
            "duplicates."
        )
    if len(highlight_labels) == 0:
        warnings.warn(
            "highlight_labels is empty.",
            UserWarning,
        )
    if not np.issubdtype(
        np.asarray(highlight_labels).dtype, np.str_
    ):
        raise ValueError(
            "highlight_labels must contain "
            "string values."
        )


def _validate_label_inputs(
    labels, top_labels, highlight_labels, n_points,
):
    if (
        top_labels is not None
        and highlight_labels is not None
    ):
        raise ValueError(
            "top_labels and highlight_labels are "
            "mutually exclusive."
        )

    if (
        labels is None
        and (top_labels is not None
             or highlight_labels is not None)
    ):
        raise ValueError(
            "labels must be provided when "
            "top_labels or highlight_labels is set."
        )

    if top_labels is not None:
        if (
            not isinstance(top_labels, int)
            or top_labels <= 0
        ):
            raise ValueError(
                "top_labels must be a positive integer."
            )

    if highlight_labels is not None:
        _validate_highlight_labels(highlight_labels)

    if labels is not None:
        labels = np.asarray(labels)
        if not np.issubdtype(labels.dtype, np.str_):
            raise ValueError(
                "labels must contain string values."
            )
        if labels.shape[0] != n_points:
            raise ValueError(
                "labels must have the same length as "
                "fc_vals."
            )

    return labels


def _validate_alt_color(alt_color, n_points):
    if alt_color is None:
        return None
    alt_color = np.asarray(alt_color)
    if alt_color.ndim != 1:
        raise ValueError(
            "alt_color must be a 1D boolean sequence."
        )
    if alt_color.shape[0] != n_points:
        raise ValueError(
            "alt_color must have the same length as "
            "fc_vals."
        )
    if not np.issubdtype(alt_color.dtype, np.bool_):
        raise ValueError("alt_color must be boolean.")
    return alt_color


def _validate_thresholds(fc_thresh, pval_thresh):
    if fc_thresh is not None and fc_thresh <= 0:
        raise ValueError(
            "fc_thresh must be a positive number."
        )
    if pval_thresh is not None:
        if pval_thresh < 0 or pval_thresh > 1:
            raise ValueError(
                "pval_thresh must be in [0, 1]."
            )


def _filter_volcano_data(fc_vals, pvals, labels, alt_color):
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

    return fc_vals, pvals, labels, alt_color


def _draw_scatter(
    _ax,
    fc_vals,
    y_vals,
    up_mask,
    down_mask,
    other_mask,
    alt_color,
):
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


def _draw_threshold_lines(_ax, fc_thresh, pval_thresh, yscale_log):
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


def _annotate_top_labels(
    _ax,
    fc_vals,
    pvals,
    y_vals,
    labels,
    top_labels,
    sig_mask,
    fc_thresh,
):
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
        pos_idx = np.where(lbl_fc >= 0)[0]
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


def _annotate_highlight_labels(
    _ax,
    fc_vals,
    y_vals,
    labels,
    highlight_labels,
):
    hl_set = set(highlight_labels)
    hl_idx = np.where(
        np.isin(labels, list(hl_set))
    )[0]

    # Warn about missing labels
    found = set(labels[hl_idx])
    missing = hl_set - found
    if missing:
        warnings.warn(
            "highlight_labels not found after "
            f"filtering: {sorted(missing)}",
            RuntimeWarning,
        )

    if len(hl_idx) > 0:
        texts = []
        for i in hl_idx:
            texts.append(
                _ax.text(
                    fc_vals[i],
                    y_vals[i],
                    str(labels[i]),
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


def volcano_plot(
    fc_vals: Sequence[float] | np.ndarray,
    pvals: Sequence[float] | np.ndarray,
    fc_thresh: float | None = None,
    pval_thresh: float | None = None,
    *,
    labels: Sequence[str] | np.ndarray | None = None,
    top_labels: int | None = None,
    highlight_labels: Sequence[str] | None = None,
    figsize: tuple[float, float] = (6.0, 5.0),
    xlabel: str | None = None,
    ylabel: str | None = None,
    alt_color: list[bool] | np.ndarray | None = None,
    yscale_log: bool = True,
    title: str | None = None,
    show: bool = True,
    save: str | Path | None = None,
    ax: Axes | None = None,
) -> Axes:
    """
    Volcano plot renderer (framework-agnostic).

    Draws a scatter plot of fold change (x-axis) versus p-value
    (y-axis). Points are colored by significance or by an optional
    alternative boolean mask.

    Parameters
    ----------
    fc_vals : Sequence[float] | np.ndarray
        Fold change values (x-axis).
    pvals : Sequence[float] | np.ndarray
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
    labels : Sequence[str] | np.ndarray | None, optional
        Labels for each point, same length as ``fc_vals``.
        Required when ``top_labels`` is set.
    top_labels : int | None, optional
        Number of top proteins to label per side (up to 2N
        total). Ranked by smallest p-value, then largest
        ``|fc|``.
    highlight_labels : Sequence[str] | None, optional
        Sequence of label strings to highlight on the plot.
        Each entry must match a value in ``labels``. Matched
        points are annotated with their label and a connecting
        arrow. Labels not found after filtering trigger a
        warning.
    figsize : tuple[float, float], optional
        Figure dimensions (width, height) in inches.
    xlabel : str | None, optional
        X-axis label. Defaults to ``"logFC"`` when ``None``.
    ylabel : str | None, optional
        Y-axis label. When ``None``, defaults to ``"pval"`` if
        ``yscale_log=True`` or ``"-log10(pval)"`` if
        ``yscale_log=False``.
    alt_color : list[bool] | np.ndarray | None, optional
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
    ax : matplotlib.axes.Axes | None, optional
        Matplotlib Axes to plot onto. If ``None``, a new figure
        and axes are created.

    Returns
    -------
    Axes
        The Matplotlib Axes object used for plotting.

    Raises
    ------
    ValueError
        If ``fc_vals`` or ``pvals`` are not 1D, contain
        non-numeric values, or have different lengths; if no
        valid data remains after filtering; if ``fc_thresh`` is
        not positive (when set); if ``pval_thresh`` is not in
        ``[0, 1]`` (when set); if ``top_labels`` is set but
        ``labels`` is ``None``, or is not a positive integer;
        if ``highlight_labels`` is set but ``labels`` is
        ``None``, contains duplicates, or contains non-string
        values; if both ``top_labels`` and
        ``highlight_labels`` are set; if ``labels`` contains
        non-string values; if ``ax`` is not a
        ``matplotlib.axes.Axes`` object (when set); if
        ``alt_color`` fails validation.

    Examples
    --------
    Basic usage with lists:

    >>> from proteopy.utils.stat_tests import volcano_plot
    >>> fc = [-2.1, -0.5, 0.3, 1.8, 3.0]
    >>> pv = [0.001, 0.3, 0.5, 0.04, 0.0005]
    >>> volcano_plot(fc, pv, show=False)
    <Axes: ...>

    With fold change and p-value thresholds:

    >>> volcano_plot(
    ...     fc, pv,
    ...     fc_thresh=1.5,
    ...     pval_thresh=0.05,
    ...     show=False,
    ... )
    <Axes: ...>

    With ``top_labels`` to annotate the most significant hits:

    >>> genes = ["GeneA", "GeneB", "GeneC", "GeneD", "GeneE"]
    >>> volcano_plot(
    ...     fc, pv,
    ...     fc_thresh=1.5,
    ...     pval_thresh=0.05,
    ...     labels=genes,
    ...     top_labels=2,
    ...     show=False,
    ... )
    <Axes: ...>

    With ``highlight_labels`` to annotate specific proteins:

    >>> volcano_plot(
    ...     fc, pv,
    ...     labels=genes,
    ...     highlight_labels=["GeneA", "GeneE"],
    ...     show=False,
    ... )
    <Axes: ...>
    """
    fc_vals, pvals = _validate_numeric_inputs(fc_vals, pvals)
    _validate_thresholds(fc_thresh, pval_thresh)
    if ax is not None and not isinstance(ax, Axes):
        raise ValueError(
            "ax must be a matplotlib Axes object."
        )
    labels = _validate_label_inputs(
        labels, top_labels, highlight_labels,
        fc_vals.shape[0],
    )
    alt_color = _validate_alt_color(
        alt_color, fc_vals.shape[0],
    )
    fc_vals, pvals, labels, alt_color = _filter_volcano_data(
        fc_vals, pvals, labels, alt_color,
    )

    # -- Prepare plotting arrays
    neg_log_p = -np.log10(pvals)
    y_vals = pvals if yscale_log else neg_log_p

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

    # -- Plot
    if ax is None:
        fig, _ax = plt.subplots(figsize=figsize)
    else:
        _ax = ax
        fig = _ax.get_figure()
    _draw_scatter(
        _ax, fc_vals, y_vals, up_mask, down_mask,
        other_mask, alt_color,
    )
    _draw_threshold_lines(_ax, fc_thresh, pval_thresh, yscale_log)

    # Axis labels
    _ax.set_xlabel("logFC" if xlabel is None else xlabel)
    if ylabel is None:
        ylabel = "pval" if yscale_log else "-log10(pval)"
    _ax.set_ylabel(ylabel)

    if title is not None:
        _ax.set_title(title)

    # -- Labels
    if top_labels is not None and labels is not None:
        _annotate_top_labels(
            _ax, fc_vals, pvals, y_vals, labels,
            top_labels, sig_mask, fc_thresh,
        )
    if highlight_labels is not None and labels is not None:
        _annotate_highlight_labels(
            _ax, fc_vals, y_vals, labels, highlight_labels,
        )

    if save:
        fig.savefig(save, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return _ax
