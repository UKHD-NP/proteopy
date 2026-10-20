"""Resolvers shared by plotting functions."""

from __future__ import annotations

import pandas as pd


def _default_label_order(values: pd.Series) -> list:
    """Return the distinct non-missing labels in the default plot order.

    Category order if ``values`` is categorical (categories without
    values are dropped), else lexicographic order of the
    ``str``-coerced values. Plots that assign colors by position in
    this order color the same label identically.
    """
    present = values.dropna()
    if isinstance(values.dtype, pd.CategoricalDtype):
        used = set(present)
        return [c for c in values.cat.categories if c in used]
    return sorted(pd.unique(present), key=str)
