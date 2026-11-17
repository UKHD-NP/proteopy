"""Resolvers shared by plotting functions."""

from __future__ import annotations

import pandas as pd

from proteopy._parameter_resolvers import _resolve_group_order


def _default_label_order(values: pd.Series) -> list:
    """Return the distinct non-missing labels in the default plot order.

    Category order if ``values`` is categorical (categories without
    values are dropped), else lexicographic order of the
    ``str``-coerced values. Plots that assign colors by position in
    this order color the same label identically.
    """
    return _resolve_group_order(values)
