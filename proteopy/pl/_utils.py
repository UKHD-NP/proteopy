"""Shared helpers for the :mod:`proteopy.pl` plotting functions."""

from typing import Any
from collections.abc import Sequence

import pandas as pd


def contains_value(seq: Sequence, value: Any) -> bool:
    """Check if *value* is in *seq*, treating NaN as equal."""
    for item in seq:
        if pd.isna(item) and pd.isna(value):
            return True
        if item == value:
            return True
    return False


def append_unique(seq: list, value: Any) -> None:
    """Append *value* to *seq* only if not already present."""
    if not contains_value(seq, value):
        seq.append(value)


def dedupe(values) -> list:
    """Drop duplicates from *values*, keeping the given order."""
    out: list = []
    for value in values:
        append_unique(out, value)
    return out


def resolve_default_order(
    values,
    *,
    keep_unused_categories: bool = True,
) -> list:
    """
    Resolve the default plotting order of *values*.

    The order a plot uses whenever the caller has not imposed
    one through ``order``, ``ascending``, or a function-specific
    sort:

    - **Categorical**: the category order. This is what makes
      the order controllable — a user fixes it once by storing
      the annotation as an ordered Categorical.
    - **Any other dtype**: the unique values sorted
      lexicographically on their string representation, which
      is stable across functions, calls, and dtypes.

    An order is never derived from the position of rows in the
    AnnData object (``.obs_names`` / ``.var_names``) or from the
    sequence in which values appear along an axis. Both are
    artefacts of how the object was built and the user has no
    clean way to change either.

    Parameters
    ----------
    values : Series or array-like
        Annotation values labelling the axis, for example
        ``adata.obs["sample_id"]`` or an ``order_by`` column.
        Missing values are dropped.
    keep_unused_categories : bool, optional
        Keep categories that no value matches, so a subset of
        the data still plots against the full series. Pass
        ``False`` where an empty position cannot be drawn, and
        document that the function drops empty groups.

    Returns
    -------
    list
        The values in plotting order, with the original dtype
        preserved so they still match the data.

    Examples
    --------
    >>> import pandas as pd
    >>> from proteopy.pl._utils import resolve_default_order
    >>> resolve_default_order(pd.Series(["F10", "F2", "F1"]))
    ['F1', 'F10', 'F2']

    An ordered Categorical overrides the lexicographic order:

    >>> fractions = pd.Categorical(
    ...     ["F10", "F2", "F1"],
    ...     categories=["F1", "F2", "F10"],
    ...     ordered=True,
    ... )
    >>> resolve_default_order(pd.Series(fractions))
    ['F1', 'F2', 'F10']
    """
    series = values if isinstance(values, pd.Series) else pd.Series(values)

    if isinstance(series.dtype, pd.CategoricalDtype):
        order = list(series.cat.categories)
        if keep_unused_categories:
            return order
        present = set(series.dropna().unique())
        return [value for value in order if value in present]

    return sorted(series.dropna().unique(), key=str)
