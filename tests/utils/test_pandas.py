"""Tests for shared pandas grouping helpers."""

import pandas as pd
import pytest

from proteopy.utils.pandas import _resolve_group_order


@pytest.mark.parametrize("ordered", [False, True])
def test_group_order_preserves_observed_category_order(ordered):
    series = pd.Series(
        pd.Categorical(
            ["a", None, "b", "a"],
            categories=["unused", "b", "a"],
            ordered=ordered,
        )
    )
    before = series.copy(deep=True)

    assert _resolve_group_order(series) == ["b", "a"]
    pd.testing.assert_series_equal(series, before)


def test_group_order_sorts_strings_lexicographically():
    series = pd.Series(["b", None, "a", "b"])
    assert _resolve_group_order(series) == ["a", "b"]


def test_group_order_sorts_numeric_labels_as_strings():
    series = pd.Series([2, 10, 1, pd.NA], dtype="Int64")
    assert _resolve_group_order(series) == [1, 10, 2]


@pytest.mark.parametrize("dtype", ["object", "category", "string"])
def test_group_order_excludes_all_missing_labels(dtype):
    series = pd.Series([None, None], dtype=dtype)
    assert _resolve_group_order(series) == []
