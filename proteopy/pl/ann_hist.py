from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import anndata as ad
from pandas.api.types import (
    is_bool_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns

from proteopy.utils.anndata import check_proteodata


_RESERVED_KWARGS = ("data", "ax")
_COLUMN_KWARGS = ("hue", "weights")
_MAX_REPORTED_GROUPS = 5
_LINE_DEFAULTS = {
    "value": None,
    "color": "red",
    "linestyle": "dashed",
    "text": None,
}


def _resolve_frame(
    adata: ad.AnnData,
    columns: list[str],
    axis: str | None,
) -> tuple[pd.DataFrame, str]:
    if axis is not None:
        if axis not in ("obs", "var"):
            raise ValueError("`axis` must be 'obs', 'var' or None.")
        frame = adata.obs if axis == "obs" else adata.var
        missing = [c for c in columns if c not in frame.columns]
        if missing:
            raise KeyError(f"Columns {missing} not found in adata.{axis}.")
        return frame, axis

    locations = {}
    for column in columns:
        found = [
            name
            for name, frame in (("obs", adata.obs), ("var", adata.var))
            if column in frame.columns
        ]
        if not found:
            raise KeyError(
                f"Column '{column}' not found in adata.obs or adata.var."
            )
        if len(found) == 2:
            raise ValueError(
                f"Column '{column}' exists in both adata.obs and "
                "adata.var, so the annotation axis is ambiguous. Pass "
                "`axis='obs'` or `axis='var'` to choose."
            )
        locations[column] = found[0]

    axes = set(locations.values())
    if len(axes) > 1:
        placement = ", ".join(
            f"'{c}' in adata.{a}" for c, a in locations.items()
        )
        raise ValueError(
            "All plotted columns must come from the same annotation "
            f"axis, got {placement}."
        )
    name = axes.pop()
    return (adata.obs if name == "obs" else adata.var), name


def _check_dtype(series: pd.Series, column: str) -> None:
    if is_bool_dtype(series):
        raise TypeError(
            f"Column '{column}' is boolean, which is not supported. "
            f"Convert it with `.astype('category')` first."
        )
    if isinstance(series.dtype, pd.CategoricalDtype):
        return
    if is_numeric_dtype(series):
        return
    if is_object_dtype(series) or is_string_dtype(series):
        return
    raise TypeError(
        f"Column '{column}' has unsupported dtype '{series.dtype}'. "
        "Supported dtypes are int, float, str, object and categorical."
    )


def _collapse_to_level(
    frame: pd.DataFrame,
    columns: list[str],
    level_key: list[str] | None,
) -> pd.DataFrame:
    if level_key is None:
        return frame[columns].copy()

    missing = [c for c in level_key if c not in frame.columns]
    if missing:
        raise KeyError(
            f"`level_key` columns {missing} not found in the frame "
            "holding the plotted columns."
        )
    overlap = [c for c in level_key if c in columns]
    if overlap:
        raise ValueError(
            f"`level_key` columns {overlap} are also plotted columns."
        )
    if frame[level_key].isna().any().any():
        raise ValueError("`level_key` columns must not contain NaN.")

    grouped = frame.groupby(level_key, observed=True)[columns]
    n_values = grouped.nunique(dropna=False)
    for column in columns:
        bad = n_values.index[n_values[column] > 1]
        if len(bad) == 0:
            continue
        examples = []
        for group in bad[:_MAX_REPORTED_GROUPS]:
            key = group if isinstance(group, tuple) else (group,)
            key_values = pd.Series(key, index=level_key)
            mask = (frame[level_key] == key_values).all(axis=1)
            values = frame.loc[mask, column].unique().tolist()
            examples.append(f"  {group}: {values}")
        raise ValueError(
            f"Column '{column}' is not constant within {len(bad)} "
            f"group(s) of `level_key={level_key}`. Examples:\n"
            + "\n".join(examples)
        )

    return frame.drop_duplicates(subset=level_key)[columns].copy()


def _apply_na_handling(
    df: pd.DataFrame,
    columns: list[str],
    zero_to_na: bool,
    fill_na: float | int | str | None,
) -> None:
    if zero_to_na:
        numeric = [c for c in columns if _is_plain_numeric(df[c])]
        if not numeric:
            raise ValueError(
                "`zero_to_na=True` requires at least one numeric "
                "plotted column."
            )
        for column in numeric:
            df[column] = df[column].mask(df[column] == 0)

    if fill_na is not None:
        for column in columns:
            series = df[column]
            if (
                isinstance(series.dtype, pd.CategoricalDtype)
                and fill_na not in series.cat.categories
            ):
                series = series.cat.add_categories([fill_na])
            df[column] = series.fillna(fill_na)


def _is_plain_numeric(series: pd.Series) -> bool:
    return is_numeric_dtype(series) and not isinstance(
        series.dtype, pd.CategoricalDtype
    )


def _to_ordered_categorical(series: pd.Series) -> pd.Series:
    if isinstance(series.dtype, pd.CategoricalDtype):
        return series
    as_str = series.where(series.isna(), series.astype(str))
    categories = sorted(map(str, as_str.dropna().unique()))
    return pd.Series(
        pd.Categorical(as_str, categories=categories),
        index=series.index,
        name=series.name,
    )


def _normalize_lines(
    lines: dict | list[dict] | None,
    name: str,
) -> list[dict]:
    if lines is None:
        return []
    if isinstance(lines, dict):
        lines = [lines]
    if not isinstance(lines, list) or not all(
        isinstance(line, dict) for line in lines
    ):
        raise TypeError(f"`{name}` must be a dict or a list of dicts.")

    normalized = []
    for line in lines:
        unknown = set(line) - set(_LINE_DEFAULTS)
        if unknown:
            raise ValueError(
                f"Unknown `{name}` keys {sorted(unknown)}; allowed keys "
                f"are {list(_LINE_DEFAULTS)}."
            )
        value = line.get("value")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float, np.number))
            or not np.isfinite(value)
        ):
            raise ValueError(
                f"Each `{name}` dict needs a finite numeric 'value', "
                f"got {value!r}."
            )
        normalized.append({**_LINE_DEFAULTS, **line})
    return normalized


def _draw_lines(
    ax: Axes,
    vlines: list[dict],
    hlines: list[dict],
) -> None:
    handles = []
    for lines, draw in ((vlines, ax.axvline), (hlines, ax.axhline)):
        for line in lines:
            handle = draw(
                line["value"],
                color=line["color"],
                linestyle=line["linestyle"],
                label=line["text"],
            )
            if line["text"] is not None:
                handles.append(handle)
    if not handles:
        return

    # ax.legend() replaces seaborn's hue legend, so re-attach it
    hue_legend = ax.get_legend()
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
        frameon=False,
    )
    if hue_legend is not None:
        ax.add_artist(hue_legend)


def ann_hist(
    adata: ad.AnnData,
    x: str | None = None,
    y: str | None = None,
    level_key: str | list[str] | None = None,
    axis: Literal["obs", "var"] | None = None,
    zero_to_na: bool = False,
    fill_na: float | int | str | None = None,
    vline: dict | list[dict] | None = None,
    hline: dict | list[dict] | None = None,
    show: bool = True,
    save: str | Path | None = None,
    **kwargs,
) -> Axes:
    """
    Plot a histogram of ``.obs`` or ``.var`` annotation columns.

    Thin wrapper around :func:`seaborn.histplot`. The annotation
    columns are collected from ``adata.obs`` or ``adata.var``,
    optionally reduced to one row per ``level_key`` group, and passed
    to ``sns.histplot`` together with all extra keyword arguments.

    Parameters
    ----------
    adata : AnnData
        :class:`~anndata.AnnData` holding the annotations.
    x : str | None
        Annotation column plotted on the x axis. Same meaning as
        ``x`` in :func:`seaborn.histplot`, except that it names a
        ``.obs``/``.var`` column.
    y : str | None
        Annotation column plotted on the y axis. Same meaning as
        ``y`` in :func:`seaborn.histplot`. Pass only ``y`` for a
        horizontal histogram, or both ``x`` and ``y`` for a bivariate
        histogram. At least one of ``x`` and ``y`` is required.
    level_key : str | list[str] | None
        Column(s) defining the level the annotation belongs to.
        Annotations often describe a higher level than the axis and
        are repeated for every row of that level, e.g.
        ``protein_q_value`` is repeated for all peptides of a protein
        in a peptide-level AnnData. With ``level_key="protein_id"``,
        each protein contributes one value instead of one per peptide.
    axis : {"obs", "var"} | None
        Annotation axis holding the columns. If ``None``, it is
        inferred from where the columns live.
    zero_to_na : bool
        If True, replace zeros with NaN in the numeric ``x`` / ``y``
        columns before plotting.
    fill_na : float | int | str | None
        If not None, replace NaN in the ``x`` / ``y`` columns with this
        value, e.g. ``"missing"`` for str or categorical annotations.
    vline : dict | list[dict] | None
        Vertical line(s) drawn with
        :meth:`matplotlib.axes.Axes.axvline`. A dict defines one
        line, a list of dicts several. Keys: ``value`` (required,
        numeric x position), ``color`` (default ``"red"``),
        ``linestyle`` (default ``"dashed"``) and ``text`` (default
        ``None``). Lines with ``text`` are listed in a legend to the
        right of the plot. On a categorical axis, category ``i`` sits
        at position ``i``.
    hline : dict | list[dict] | None
        Horizontal line(s) drawn with
        :meth:`matplotlib.axes.Axes.axhline`. Same format as
        ``vline``, with ``value`` as the y position.
    show : bool
        Call :func:`matplotlib.pyplot.show` at the end.
    save : str | Path | None
        Save the figure to this path. If None, do not save.
    **kwargs
        Passed verbatim to :func:`seaborn.histplot` (e.g. ``bins``,
        ``binwidth``, ``stat``, ``discrete``, ``shrink``, ``element``,
        ``kde``, ``log_scale``, ``hue``, ``multiple``, ``palette``).
        See https://seaborn.pydata.org/generated/seaborn.histplot.html
        for the full list and what each argument does. ``hue`` and
        ``weights`` given as strings name ``.obs``/``.var`` columns
        from the same frame as ``x``/``y``. ``data`` and ``ax`` are
        reserved.

    Returns
    -------
    Axes
        The Axes the histogram was drawn on.

    Raises
    ------
    KeyError
        If a referenced column is missing.
    ValueError
        If neither ``x`` nor ``y`` is given, ``axis`` is ``None`` and
        a plotted column exists in both ``.obs`` and ``.var``, the
        plotted columns come from different annotation axes, a
        plotted column is not constant within a ``level_key`` group,
        or a ``vline`` / ``hline`` dict has unknown keys or no finite
        numeric ``value``.
    TypeError
        If a reserved keyword is passed, a column has an unsupported
        dtype (e.g. bool or datetime), or ``vline`` / ``hline`` is not
        a dict or a list of dicts.

    Notes
    -----
    Non-numeric values are ordered by category order for categorical
    columns, otherwise lexicographically after ``str`` coercion. To
    impose a custom order, store the column as an ordered Categorical:

    >>> adata.var["fraction"] = pd.Categorical(
    ...     adata.var["fraction"],
    ...     categories=["F1", "F2", "F10"],
    ...     ordered=True,
    ... )

    Examples
    --------
    >>> import proteopy as pr
    >>> adata = pr.load.toy_simple_peptides(
    ...     n_proteins=10,
    ...     n_peptides_per_protein=(
    ...         [5] * 1 + [6] * 2 + [7] * 5 + [8] * 2
    ...     ),
    ... )
    >>> pr.pl.ann_hist(
    ...     adata,
    ...     x="Protein.Q.Value",
    ...     level_key="protein_id",
    ...     vline=[
    ...         {"value": 0.01, "text": "1% FDR"},
    ...         {"value": 0.05, "color": "grey", "text": "5% FDR"},
    ...     ],
    ...     bins=50,
    ... )
    >>> pr.pl.ann_hist(
    ...     adata,
    ...     x="Q.Value",
    ...     y="Q.Value.Global",
    ... )
    >>> pr.pl.ann_hist(adata, x="condition", shrink=0.8)
    """
    check_proteodata(adata)

    if x is None and y is None:
        raise ValueError("At least one of `x` and `y` must be given.")
    for name in _RESERVED_KWARGS:
        if name in kwargs:
            hint = " (`ax` is not supported yet)" if name == "ax" else ""
            raise TypeError(
                f"`{name}` is set by ann_hist and cannot be passed{hint}."
            )
    if isinstance(level_key, str):
        level_key = [level_key]
    vlines = _normalize_lines(vline, "vline")
    hlines = _normalize_lines(hline, "hline")

    # -- Collect and validate the referenced columns
    plot_columns = [c for c in (x, y) if c is not None]
    kwarg_columns = [
        kwargs[k] for k in _COLUMN_KWARGS if isinstance(kwargs.get(k), str)
    ]
    columns = list(dict.fromkeys(plot_columns + kwarg_columns))
    frame, _ = _resolve_frame(adata, columns, axis)
    for column in columns:
        _check_dtype(frame[column], column)

    # -- Reduce to one row per level and prepare the values
    df = _collapse_to_level(frame, columns, level_key)
    _apply_na_handling(df, plot_columns, zero_to_na, fill_na)
    for column in columns:
        if not _is_plain_numeric(df[column]):
            df[column] = _to_ordered_categorical(df[column])

    _, ax = plt.subplots()
    sns.histplot(data=df, x=x, y=y, ax=ax, **kwargs)
    _draw_lines(ax, vlines, hlines)

    if save is not None:
        ax.figure.savefig(save, bbox_inches="tight")
    if show:
        plt.show()
    return ax
