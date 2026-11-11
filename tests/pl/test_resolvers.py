import numpy as np
import pandas as pd

from proteopy.pl._resolvers import _default_label_order


class TestDefaultLabelOrder:
    def test_lexicographic(self):
        values = pd.Series(["b", "a", np.nan, "c", "a"])
        assert _default_label_order(values) == ["a", "b", "c"]

    def test_str_coerced(self):
        values = pd.Series([10, 2, 1])
        assert _default_label_order(values) == [1, 10, 2]

    def test_categorical_follows_category_order(self):
        # Category order is neither lexicographic nor order of appearance.
        values = pd.Series(
            pd.Categorical(
                ["F1", "F10", "F2", "F1"],
                categories=["F1", "F2", "F10"],
                ordered=True,
            )
        )
        assert _default_label_order(values) == ["F1", "F2", "F10"]

    def test_unordered_categorical_follows_category_order(self):
        values = pd.Series(
            pd.Categorical(["a", "m", "z"], categories=["z", "a", "m"])
        )
        assert _default_label_order(values) == ["z", "a", "m"]

    def test_categorical_drops_unused(self):
        values = pd.Series(
            pd.Categorical(["a", "c", "a"], categories=["c", "b", "a"])
        )
        assert _default_label_order(values) == ["c", "a"]

    def test_categorical_ignores_missing(self):
        values = pd.Series(
            pd.Categorical(["b", None, "a"], categories=["b", "a"])
        )
        assert _default_label_order(values) == ["b", "a"]
