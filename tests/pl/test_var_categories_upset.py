"""Tests for ``pr.pl.var_categories_upset``.

Sections
--------
* Fixtures and helpers
* Core intersection counts and thresholds
* Category order and names
* Feature counting
* Degenerate structures
* Sparse input and mutation
* Plot construction
* print_stats output
* save and show
* Threshold selection
* Negative cases
* Property and metamorphic relations
"""

from __future__ import annotations

import inspect
import math
import random

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from upsetplot import UpSet

from proteopy.pl import var_categories_upset

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

T = True
F = False


# ---------------------------------------------------------------------------
# AnnData builders
# ---------------------------------------------------------------------------


def _adata_from_columns(
    category_key,
    cat_values,
    columns,
    *,
    level="protein",
    protein_ids=None,
    categories=None,
):
    names = list(columns)
    n_obs = len(cat_values)
    X = np.empty((n_obs, len(names)), dtype=float)
    for j, name in enumerate(names):
        X[:, j] = columns[name]

    sample_ids = [f"s{i + 1}" for i in range(n_obs)]
    obs = pd.DataFrame({"sample_id": sample_ids}, index=pd.Index(sample_ids))
    if categories is None:
        obs[category_key] = pd.Series(cat_values, index=obs.index)
    else:
        obs[category_key] = pd.Categorical(cat_values, categories=categories)

    var = pd.DataFrame(index=pd.Index(names))
    if level == "protein":
        var["protein_id"] = names
    else:
        var["peptide_id"] = names
        var["protein_id"] = [protein_ids[name] for name in names]

    return ad.AnnData(X=X, obs=obs, var=var)


def _adata(
    category_key,
    cat_values,
    detections,
    value,
    *,
    level="protein",
    protein_ids=None,
    categories=None,
):
    """Build an AnnData from per-feature detected-observation counts."""
    positions = {}
    for i, cat in enumerate(cat_values):
        positions.setdefault(cat, []).append(i)

    columns = {}
    for name, per_cat in detections.items():
        column = [np.nan] * len(cat_values)
        for cat, n_detected in per_cat.items():
            for i in positions[cat][:n_detected]:
                column[i] = value
        columns[name] = column

    return _adata_from_columns(
        category_key,
        cat_values,
        columns,
        level=level,
        protein_ids=protein_ids,
        categories=categories,
    )


def _h1():
    return _adata(
        "organ",
        ["lung", "kidney", "spleen"] * 3,
        {
            "g01": {"kidney": 3},
            "g02": {"lung": 3},
            "g03": {"spleen": 3},
            "g04": {"kidney": 3, "lung": 3},
            "g05": {"kidney": 3, "spleen": 3},
            "g06": {"lung": 3, "spleen": 3},
            "g07": {"kidney": 3, "lung": 3, "spleen": 3},
            "g08": {"kidney": 3, "lung": 3, "spleen": 3},
            "g09": {},
            "g10": {"kidney": 1, "lung": 2},
            "g11": {"kidney": 3},
        },
        3.5,
    )


def _h3():
    names = [f"g{i}" for i in range(1, 7)]
    return _adata(
        "site",
        ["north"] * 5 + ["south"] * 5,
        {
            "g1": {"south": 5},
            "g2": {"north": 1, "south": 4},
            "g3": {"north": 2, "south": 3},
            "g4": {"north": 3, "south": 2},
            "g5": {"north": 4, "south": 1},
            "g6": {"north": 5},
        },
        7.0,
        level="peptide",
        protein_ids={name: "Q1" for name in names},
    )


def _h4():
    return _adata(
        "arm",
        ["p"] * 5 + ["q"] * 3,
        {
            "g1": {"p": 3, "q": 1},
            "g2": {"p": 2, "q": 2},
            "g3": {"p": 5, "q": 3},
            "g4": {"p": 2, "q": 1},
            "g5": {"p": 4},
        },
        8.0,
    )


def _h5():
    return _adata_from_columns(
        "arm",
        ["u"] * 3 + ["v"] * 3,
        {
            "g1": [0.0, 0.0, 7.5, np.nan, np.nan, np.nan],
            "g2": [np.nan, np.nan, np.nan, 0.0, 0.0, 0.0],
            "g3": [1.0, np.nan, 0.0, 0.0, np.nan, np.nan],
        },
    )


def _h6():
    return _adata(
        "batch",
        ["mid", "alpha", "zeta", "mid", "alpha", "zeta"],
        {
            "v1": {"zeta": 2, "alpha": 2, "mid": 2},
            "v2": {"mid": 1},
            "v3": {},
        },
        4.0,
        categories=["zeta", "beta", "alpha", "mid"],
    )


def _h7():
    return _adata(
        "fruit",
        ["kiwi", "apple", "fig", "apple"],
        {"w1": {"fig": 1}, "w2": {"apple": 2}},
        1.0,
    )


# ---------------------------------------------------------------------------
# Spies on the plotting library
# ---------------------------------------------------------------------------


def _bind(func, args, kwargs):
    bound = inspect.signature(func).bind(None, *args, **kwargs)
    bound.apply_defaults()
    arguments = dict(bound.arguments)
    arguments.pop("self", None)
    return arguments


class _Spy:
    def __init__(self):
        self.init = []
        self.style = []
        self.plot_returns = []

    @property
    def data(self):
        return self.init[0]["data"]


@pytest.fixture(autouse=True)
def show_calls(monkeypatch):
    matplotlib.use("Agg")
    recorded = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: recorded.append(1))
    yield recorded
    plt.close("all")


@pytest.fixture
def rng() -> random.Random:
    """Return a reproducible generator for randomized test cases."""
    return random.Random(0)


@pytest.fixture
def sorting_adata():
    """Return a builder for specified intersection counts."""

    def build(counts: dict, categories=None) -> ad.AnnData:
        columns = {}
        for members, count in counts.items():
            for _ in range(count):
                columns[f"p{len(columns)}"] = [
                    1.0 if category in members else np.nan
                    for category in ["C", "A", "B"]
                ]
        samples = ["s1", "s2", "s3"]
        proteins = list(columns)
        obs = pd.DataFrame(
            {"sample_id": samples, "group": ["C", "A", "B"]},
            index=samples,
        )
        if categories is not None:
            obs["group"] = pd.Categorical(
                obs["group"], categories=categories, ordered=True
            )
        return ad.AnnData(
            X=np.asarray(list(columns.values()), dtype=float).T,
            obs=obs,
            var=pd.DataFrame({"protein_id": proteins}, index=proteins),
        )

    return build


@pytest.fixture
def rendered_intersections():
    """Return a reader of memberships and counts from drawn artists."""

    def read(axes: dict) -> list:
        matrix = axes["matrix"]
        matrix.figure.canvas.draw()
        labels = {
            int(tick): label.get_text()
            for tick, label in zip(
                matrix.get_yticks(), matrix.get_yticklabels()
            )
        }
        bars = sorted(
            axes["intersections"].patches,
            key=lambda patch: patch.get_x(),
        )
        dots = matrix.collections[0]
        offsets = dots.get_offsets()
        colors = dots.get_facecolors()
        assert offsets.shape == (len(bars) * len(labels), 2)
        assert colors.shape == (len(offsets), 4)
        rows = []
        for bar in bars:
            center = bar.get_x() + bar.get_width() / 2
            members = [
                labels[int(y)]
                for (x, y), color in zip(offsets, colors)
                if np.isclose(x, center)
                and np.allclose(color, bar.get_facecolor())
            ]
            rows.append((tuple(sorted(members)), bar.get_height()))
        return rows

    return read


@pytest.fixture
def spy(monkeypatch):
    recorder = _Spy()
    original_init = UpSet.__init__
    original_style = UpSet.style_subsets
    original_plot = UpSet.plot

    def init(self, *args, **kwargs):
        recorder.init.append(_bind(original_init, args, kwargs))
        return original_init(self, *args, **kwargs)

    def style_subsets(self, *args, **kwargs):
        recorder.style.append(_bind(original_style, args, kwargs))
        return original_style(self, *args, **kwargs)

    def plot(self, *args, **kwargs):
        result = original_plot(self, *args, **kwargs)
        recorder.plot_returns.append(result)
        return result

    monkeypatch.setattr(UpSet, "__init__", init)
    monkeypatch.setattr(UpSet, "style_subsets", style_subsets)
    monkeypatch.setattr(UpSet, "plot", plot)
    return recorder


@pytest.fixture
def plot_axes(spy):
    """Inspect backend axes without relying on the public return value."""

    def plot(adata, category_key, **kwargs):
        kwargs.setdefault("show", False)
        assert var_categories_upset(adata, category_key, **kwargs) is None
        return spy.plot_returns[-1]

    return plot


def _series(spy, adata, category_key, **kwargs):
    kwargs.setdefault("show", False)
    var_categories_upset(adata, category_key=category_key, **kwargs)
    return spy.data


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


def _check_series(series, levels, expected):
    assert isinstance(series, pd.Series)
    assert series.name == "n_features"
    assert pd.api.types.is_integer_dtype(series.dtype)
    assert isinstance(series.index, pd.MultiIndex)
    assert list(series.index.names) == list(levels)
    for level_values in series.index.levels:
        assert level_values.dtype == np.dtype(bool)
    observed = {tuple(key): int(value) for key, value in series.items()}
    assert observed == {tuple(k): v for k, v in expected.items()}


def _vec(n_levels, members):
    return tuple(i in members for i in range(n_levels))


def _per_category(series, levels):
    return {
        name: int(sum(v for key, v in series.items() if key[position]))
        for position, name in enumerate(levels)
    }


def _legends(axes):
    figure = axes["matrix"].figure
    legends = list(figure.legends)
    for axis in figure.axes:
        if axis.get_legend() is not None:
            legends.append(axis.get_legend())
    return legends


def _bar_heights(axes):
    return sorted(
        patch.get_height() for patch in axes["intersections"].patches
    )


def _bar_widths(axes):
    return sorted(patch.get_width() for patch in axes["totals"].patches)


_TITLE_PREFIXES = ("Global:", "Intersections:", "Per ")


def _is_title(line):
    stripped = line.strip()
    return stripped.endswith(":") and stripped.startswith(_TITLE_PREFIXES)


def _section(out, title):
    lines = out.splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == title)
    body = []
    for line in lines[start + 1 :]:
        if _is_title(line):
            break
        if line.strip():
            body.append(line)
    return body


def _global_table(out):
    body = _section(out, "Global:")
    return body[0].split(), body[1].split()


def _intersections_table(out, n_categories):
    body = _section(out, "Intersections:")
    header = body[0].split()
    rows = [
        [field.strip() for field in line.split(None, n_categories + 1)]
        for line in body[1:]
    ]
    return header, rows


def _per_table(out, category_key):
    body = _section(out, f"Per {category_key}:")
    header = body[0].split()
    rows = [
        [field.strip() for field in line.rsplit(None, 2)] for line in body[1:]
    ]
    return header, rows


# ---------------------------------------------------------------------------
# Randomized-input generation and reference computation
# ---------------------------------------------------------------------------

_FRACTIONS = [0.0, 0.25, 1 / 3, 0.5, 2 / 3, 1.0]
_ROUNDS = 12


def _generate(rng):
    n_categories = rng.randint(1, 5)
    categories = [f"c{i}" for i in range(n_categories)]
    cat_values = []
    for category in categories:
        cat_values += [category] * rng.randint(1, 4)
    rng.shuffle(cat_values)

    columns = {}
    for j in range(rng.randint(1, 30)):
        column = []
        for _ in cat_values:
            draw = rng.random()
            if draw < 0.4:
                column.append(float("nan"))
            elif draw < 0.6:
                column.append(0.0)
            else:
                column.append(rng.uniform(0.1, 100.0))
        columns[f"q{j:02d}"] = column

    if rng.random() < 0.5:
        threshold = ("min_count", rng.randint(0, 5))
    else:
        threshold = ("min_fraction", rng.choice(_FRACTIONS))

    return {
        "categories": categories,
        "cat_values": cat_values,
        "columns": columns,
        "threshold": threshold,
        "zero_to_na": rng.choice([True, False]),
    }


def _case_adata(case):
    return _adata_from_columns("cond", case["cat_values"], case["columns"])


def _case_kwargs(case):
    kind, value = case["threshold"]
    return {kind: value, "zero_to_na": case["zero_to_na"], "show": False}


def _reference(case):
    categories = case["categories"]
    cat_values = case["cat_values"]
    names = list(case["columns"])
    kind, bound = case["threshold"]

    members = {}
    for category in categories:
        rows = [i for i, value in enumerate(cat_values) if value == category]
        flags = []
        for name in names:
            column = case["columns"][name]
            n_detected = 0
            for i in rows:
                entry = column[i]
                if math.isnan(entry):
                    continue
                if case["zero_to_na"] and entry == 0.0:
                    continue
                n_detected += 1
            if not rows:
                flags.append(False)
            elif kind == "min_count":
                flags.append(n_detected >= bound)
            else:
                flags.append(n_detected >= math.floor(bound * len(rows)))
        members[category] = flags

    counts = {}
    for j in range(len(names)):
        vector = tuple(members[category][j] for category in categories)
        counts[vector] = counts.get(vector, 0) + 1
    counts.setdefault(tuple(False for _ in categories), 0)
    return counts


def _mapping(series):
    return {tuple(key): int(value) for key, value in series.items()}


def _last_series(spy, adata, case):
    var_categories_upset(adata, "cond", **_case_kwargs(case))
    return spy.init[-1]["data"]


def _numeric_categories():
    return _adata("dose", [3, 20, 100], {"x1": {20: 1}}, 1.0)


class TestVarCategoriesUpset:
    # ── Intersection sorting ────────────────────────────────────────

    @pytest.mark.parametrize(
        "intersection_order, expected",
        [
            ("degree", [1, 4, 2, 3]),
            ("-degree", [3, 2, 4, 1]),
            ("cardinality", [4, 3, 2, 1]),
            ("-cardinality", [1, 2, 3, 4]),
        ],
    )
    def test_T77_intersection_order_orders_rendered_intersections(
        self,
        plot_axes,
        sorting_adata,
        rendered_intersections,
        intersection_order,
        expected,
    ):
        counts = {(): 1, ("A",): 4, ("A", "B"): 2, ("A", "B", "C"): 3}
        axes = plot_axes(
            sorting_adata(counts),
            "group",
            intersection_order=intersection_order,
            show=False,
        )
        rows = rendered_intersections(axes)
        assert len(rows) == len(counts)
        np.testing.assert_array_equal([count for _, count in rows], expected)
        assert dict(rows) == counts

    def test_T78_intersection_order_default_matches_explicit_degree(
        self, plot_axes, sorting_adata, rendered_intersections
    ):
        adata = sorting_adata({(): 1, ("B",): 4, ("A",): 2, ("A", "B"): 3})
        default = plot_axes(adata, "group", show=False)
        explicit = plot_axes(
            adata, "group", intersection_order="degree", show=False
        )
        assert rendered_intersections(default) == rendered_intersections(
            explicit
        )

    @pytest.mark.parametrize(
        "intersection_order",
        ["degree", "-degree", "cardinality", "-cardinality"],
    )
    @pytest.mark.parametrize(
        "categories, expected",
        [(None, ["A", "B", "C"]), (["C", "A", "B"], ["C", "A", "B"])],
        ids=["lexicographic", "categorical"],
    )
    def test_T79_intersection_order_preserves_category_order(
        self,
        plot_axes,
        sorting_adata,
        intersection_order,
        categories,
        expected,
    ):
        adata = sorting_adata({(): 1, ("A",): 2, ("B", "C"): 3}, categories)
        axes = plot_axes(
            adata, "group", intersection_order=intersection_order, show=False
        )
        labels = [t.get_text() for t in axes["matrix"].get_yticklabels()]
        assert labels == expected

    @pytest.mark.parametrize(
        "intersection_order",
        ["degree", "-degree", "cardinality", "-cardinality"],
    )
    @pytest.mark.parametrize(
        "counts",
        [
            {(): 1, ("A",): 3, ("A", "B"): 3, ("A", "B", "C"): 2},
            {(): 1, ("A",): 2, ("B",): 4, ("C",): 3, ("A", "B"): 5},
            {
                (): 1,
                ("A",): 2,
                ("B",): 2,
                ("A", "B"): 4,
                ("A", "C"): 4,
                ("A", "B", "C"): 3,
            },
            {(): 2, ("A",): 2, ("B",): 2, ("A", "B"): 2},
        ],
        ids=["equal-counts", "equal-degrees", "equal-both", "all-equal"],
    )
    def test_T80_intersection_order_ties_are_consistent(
        self,
        plot_axes,
        sorting_adata,
        rendered_intersections,
        counts,
        intersection_order,
    ):
        adata = sorting_adata(counts)
        first = plot_axes(
            adata, "group", intersection_order=intersection_order, show=False
        )
        rows = rendered_intersections(first)
        assert len(rows) == len(counts)
        assert dict(rows) == counts
        if "degree" in intersection_order:
            metric = [len(members) for members, _ in rows]
            descending = intersection_order.startswith("-")
        else:
            metric = [count for _, count in rows]
            descending = not intersection_order.startswith("-")
        assert metric == sorted(metric, reverse=descending)
        second = plot_axes(
            adata, "group", intersection_order=intersection_order, show=False
        )
        assert rows == rendered_intersections(second)

    @pytest.mark.parametrize("intersection_order", [None, True, 1, [], {}])
    def test_T81_intersection_order_rejects_non_strings_before_plotting(
        self, intersection_order
    ):
        before = set(plt.get_fignums())
        with pytest.raises(
            TypeError, match="`intersection_order` must be a str"
        ):
            var_categories_upset(
                _h1(),
                "organ",
                intersection_order=intersection_order,
                show=False,
            )
        assert set(plt.get_fignums()) == before

    @pytest.mark.parametrize(
        "intersection_order", ["", "size", "input", "Degree"]
    )
    def test_T82_intersection_order_rejects_unknown_modes_before_plotting(
        self, intersection_order
    ):
        before = set(plt.get_fignums())
        with pytest.raises(
            ValueError,
            match="`intersection_order` must be one of .*degree.*cardinality",
        ):
            var_categories_upset(
                _h1(),
                "organ",
                intersection_order=intersection_order,
                show=False,
            )
        assert set(plt.get_fignums()) == before

    # -- Core intersection counts and thresholds

    def test_T1_full_detection_membership(self, spy):
        series = _series(spy, _h1(), "organ", min_fraction=1.0)
        _check_series(
            series,
            ["kidney", "lung", "spleen"],
            {
                (T, F, F): 2,
                (F, T, F): 1,
                (F, F, T): 1,
                (T, T, F): 1,
                (T, F, T): 1,
                (F, T, T): 1,
                (T, T, T): 2,
                (F, F, F): 2,
            },
        )

    def test_T2_default_threshold_membership(self, spy):
        series = _series(spy, _h1(), "organ")
        _check_series(
            series,
            ["kidney", "lung", "spleen"],
            {
                (T, F, F): 2,
                (F, T, F): 1,
                (F, F, T): 1,
                (T, T, F): 2,
                (T, F, T): 1,
                (F, T, T): 1,
                (T, T, T): 2,
                (F, F, F): 1,
            },
        )

    def test_T3_count_threshold_inclusive(self, spy):
        series = _series(spy, _h3(), "site", min_count=3)
        _check_series(
            series,
            ["north", "south"],
            {(F, T): 3, (T, F): 3, (F, F): 0},
        )

    def test_T4_fraction_threshold_per_category(self, spy):
        series = _series(spy, _h4(), "arm", min_fraction=0.6)
        _check_series(
            series,
            ["p", "q"],
            {(T, F): 1, (F, T): 2, (T, T): 2, (F, F): 0},
        )

    def test_T5_zero_counts_as_detected(self, spy):
        series = _series(spy, _h5(), "arm", min_count=3)
        _check_series(
            series,
            ["u", "v"],
            {(T, F): 1, (F, T): 1, (F, F): 1},
        )

    def test_T6_zero_to_na_hides_zeros(self, spy):
        series = _series(spy, _h5(), "arm", zero_to_na=True)
        _check_series(series, ["u", "v"], {(T, F): 2, (F, F): 1})

    def test_T7_only_all_false_zero_entry(self, spy):
        series = _series(spy, _h3(), "site", min_count=3)
        assert len(series) == 3
        assert int(series[(F, F)]) == 0

    def test_T8_category_without_observations_is_dropped(self, spy):
        series = _series(spy, _h6(), "batch", min_fraction=0.0)
        _check_series(
            series,
            ["zeta", "alpha", "mid"],
            {(T, T, T): 3, (F, F, F): 0},
        )

    def test_T9_imbalanced_category_sizes(self, spy):
        adata = _adata(
            "grp",
            ["small"] * 2 + ["large"] * 20,
            {
                "r1": {"small": 2},
                "r2": {"small": 1, "large": 20},
                "r3": {"small": 2, "large": 2},
                "r4": {"large": 1},
                "r5": {"small": 2, "large": 19},
            },
            1.0,
        )
        series = _series(spy, adata, "grp", min_count=2)
        _check_series(
            series,
            ["large", "small"],
            {(F, T): 1, (T, F): 1, (T, T): 2, (F, F): 1},
        )

    def test_T73_fraction_resolves_to_an_integer_count(self, spy):
        adata = _adata(
            "grp",
            ["B"] * 25,
            {"e1": {"B": 14}, "e2": {"B": 13}, "e3": {"B": 25}},
            1.0,
        )
        series = _series(spy, adata, "grp", min_fraction=0.56)
        _check_series(series, ["B"], {(T,): 2, (F,): 1})

    # -- Category order and names

    def test_T10_categorical_level_order(self, spy):
        series = _series(spy, _h6(), "batch")
        _check_series(
            series,
            ["zeta", "alpha", "mid"],
            {
                (T, T, T): 1,
                (F, F, T): 1,
                (F, F, F): 1,
            },
        )

    def test_T11_plain_level_order(self, spy):
        series = _series(spy, _h7(), "fruit")
        _check_series(
            series,
            ["apple", "fig", "kiwi"],
            {(F, T, F): 1, (T, F, F): 1, (F, F, F): 0},
        )

    def test_T12_plain_values_str_coerced(self, spy):
        series = _series(spy, _numeric_categories(), "dose")
        _check_series(
            series,
            ["100", "20", "3"],
            {(F, T, F): 1, (F, F, F): 0},
        )

    def test_T13_categorical_values_str_coerced(self, spy):
        adata = _adata(
            "grade",
            [100, 4, 30],
            {"x1": {4: 1}},
            1.0,
            categories=[30, 4, 100],
        )
        series = _series(spy, adata, "grade")
        _check_series(
            series,
            ["30", "4", "100"],
            {(F, T, F): 1, (F, F, F): 0},
        )

    def test_T14_matrix_labels_from_categorical(self, plot_axes):
        axes = plot_axes(_h6(), "batch", show=False)
        labels = axes["matrix"].get_yticklabels()
        assert [label.get_text() for label in labels] == [
            "zeta",
            "alpha",
            "mid",
        ]

    def test_T15_matrix_labels_str_coerced(self, plot_axes):
        axes = plot_axes(_numeric_categories(), "dose", show=False)
        labels = axes["matrix"].get_yticklabels()
        assert [label.get_text() for label in labels] == ["100", "20", "3"]

    def test_T16_special_and_long_names(self, spy):
        values = ["Größe µ", "a b c", "_underscore"]
        adata = _adata(
            "site",
            values,
            {"f1": {value: 1 for value in values}},
            1.0,
        )
        series = _series(spy, adata, "site")
        _check_series(
            series,
            ["Größe µ", "_underscore", "a b c"],
            {(T, T, T): 1, (F, F, F): 0},
        )

    # -- Feature counting

    def test_T17_feature_ids_case_sensitive(self, spy):
        adata = _adata(
            "group",
            ["A", "A", "B", "B"],
            {
                "Prot1": {"B": 2},
                "PROT1": {"B": 2},
                "prot1": {"B": 2},
            },
            1.0,
        )
        series = _series(spy, adata, "group")
        _check_series(series, ["A", "B"], {(F, T): 3, (F, F): 0})

    def test_T18_repeated_detections_counted_once(self, spy):
        adata = _adata(
            "group",
            ["A"] * 4 + ["B"] * 4,
            {"f1": {"A": 4, "B": 4}, "f2": {"A": 4, "B": 4}},
            1.0,
        )
        series = _series(spy, adata, "group")
        _check_series(series, ["A", "B"], {(T, T): 2, (F, F): 0})

    # -- Degenerate structures

    def test_T19_identical_membership_vectors(self, spy):
        adata = _adata(
            "lane",
            ["x"] * 3 + ["y"] * 3,
            {
                "f1": {"x": 3, "y": 3},
                "f2": {"x": 3, "y": 3},
                "f3": {"x": 3, "y": 3},
                "f4": {},
            },
            1.0,
        )
        series = _series(spy, adata, "lane")
        _check_series(series, ["x", "y"], {(T, T): 3, (F, F): 1})

    def test_T20_category_without_members(self, spy):
        adata = _adata(
            "part",
            ["h", "h", "m", "m", "t", "t"],
            {
                "e1": {"h": 2, "t": 2},
                "e2": {"t": 1},
                "e3": {},
            },
            1.0,
        )
        series = _series(spy, adata, "part")
        _check_series(
            series,
            ["h", "m", "t"],
            {(T, F, T): 1, (F, F, T): 1, (F, F, F): 1},
        )

    def test_T21_single_category(self, plot_axes, spy):
        adata = _adata(
            "only",
            ["solo"] * 2,
            {"f1": {"solo": 2}, "f2": {"solo": 2}, "f3": {"solo": 2}},
            1.0,
        )
        axes = plot_axes(adata, "only", show=False)
        _check_series(spy.data, ["solo"], {(T,): 3, (F,): 0})
        assert set(axes) == {
            "matrix",
            "intersections",
            "totals",
            "shading",
        }

    def test_T22_many_categories(self, spy):
        categories = [f"k{i:02d}" for i in range(1, 13)]
        adata = _adata(
            "plate",
            categories,
            {
                "v1": {categories[0]: 1, categories[1]: 1},
                "v2": {categories[11]: 1},
                "v3": {},
            },
            1.0,
        )
        series = _series(spy, adata, "plate")
        _check_series(
            series,
            categories,
            {
                _vec(12, {0, 1}): 1,
                _vec(12, {11}): 1,
                _vec(12, set()): 1,
            },
        )

    # -- Sparse input and mutation

    def test_T23_sparse_warns_and_densifies(self, spy):
        adata = _h1()
        adata.X = sparse.csc_matrix(adata.X)
        with pytest.warns(UserWarning):
            series = _series(spy, adata, "organ", min_fraction=1.0)
        assert not isinstance(series.dtype, pd.SparseDtype)
        _check_series(
            series,
            ["kidney", "lung", "spleen"],
            {
                (T, F, F): 2,
                (F, T, F): 1,
                (F, F, T): 1,
                (T, T, F): 1,
                (T, F, T): 1,
                (F, T, T): 1,
                (T, T, T): 2,
                (F, F, F): 2,
            },
        )

    def test_T24_input_not_mutated(self):
        view = _h1()[:, :7]
        before_x = np.array(view.X, dtype=float)
        before_obs = view.obs.copy(deep=True)
        before_var = view.var.copy(deep=True)
        var_categories_upset(
            view,
            "organ",
            zero_to_na=True,
            print_stats=True,
            show=False,
        )
        np.testing.assert_array_equal(np.array(view.X, dtype=float), before_x)
        pd.testing.assert_frame_equal(view.obs, before_obs)
        pd.testing.assert_frame_equal(view.var, before_var)
        assert view.is_view

    def test_T25_sparse_arrays_not_mutated(self):
        adata = _h1()
        adata.X = sparse.csc_matrix(adata.X)
        before = {
            name: getattr(adata.X, name).copy()
            for name in ("data", "indices", "indptr")
        }
        with pytest.warns(UserWarning):
            var_categories_upset(adata, "organ", show=False)
        for name, values in before.items():
            np.testing.assert_array_equal(getattr(adata.X, name), values)

    # -- Plot construction

    def test_T27_upset_constructor_options(self, spy):
        var_categories_upset(_h1(), "organ", show=False)
        assert len(spy.init) == 1
        call = spy.init[0]
        assert call["subset_size"] == "sum"
        assert call["sort_by"] == "degree"
        assert call["sort_categories_by"] == "input"
        assert call["show_counts"] is True
        assert call["include_empty_subsets"] is False

    def test_T28_no_subset_styling(self, spy):
        var_categories_upset(_h6(), "batch", show=False)
        assert spy.style == []

    def test_T29_returns_none(self, spy):
        assert var_categories_upset(_h1(), "organ", show=False) is None
        assert len(spy.plot_returns) == 1

    def test_T30_axes_keys_share_figure(self, plot_axes):
        axes = plot_axes(_h3(), "site", show=False)
        assert {
            "matrix",
            "intersections",
            "totals",
            "shading",
        } <= set(axes)
        figure = axes["matrix"].figure
        assert all(ax.figure is figure for ax in axes.values())

    def test_T31_intersection_bar_heights(self, plot_axes):
        axes = plot_axes(_h1(), "organ", min_fraction=1.0, show=False)
        assert _bar_heights(axes) == [1, 1, 1, 1, 1, 2, 2, 2]

    def test_T32_totals_bar_widths(self, plot_axes):
        axes = plot_axes(_h1(), "organ", min_fraction=1.0, show=False)
        assert _bar_widths(axes) == [5, 5, 6]

    def test_T33_no_legend(self, plot_axes):
        axes = plot_axes(_h3(), "site", show=False)
        assert _legends(axes) == []

    # -- print_stats output

    def test_T34_stats_table_order(self, capsys):
        var_categories_upset(_h1(), "organ", print_stats=True, show=False)
        out = capsys.readouterr().out
        assert (
            out.index("Global:")
            < out.index("Intersections:")
            < out.index("Per organ:")
        )

    def test_T35_global_stats_table(self, capsys):
        var_categories_upset(
            _h1(),
            "organ",
            min_fraction=1.0,
            print_stats=True,
            show=False,
        )
        header, values = _global_table(capsys.readouterr().out)
        assert header == [
            "count",
            "mean",
            "median",
            "std",
            "min",
            "max",
        ]
        assert float(values[0]) == 8
        assert values[1] == "1.4"
        assert float(values[2]) == 1
        assert values[3] == "0.5"
        assert float(values[4]) == 1
        assert float(values[5]) == 2

    def test_T36_global_std_single_entry(self, capsys):
        adata = _adata(
            "axis",
            ["x", "y", "z"],
            {f"f{i}": {} for i in range(1, 6)},
            1.0,
        )
        var_categories_upset(adata, "axis", print_stats=True, show=False)
        header, values = _global_table(capsys.readouterr().out)
        assert header == [
            "count",
            "mean",
            "median",
            "std",
            "min",
            "max",
        ]
        assert float(values[0]) == 1
        assert values[1] == "5.0"
        assert float(values[2]) == 5
        assert math.isnan(float(values[3]))
        assert float(values[4]) == 5
        assert float(values[5]) == 5

    def test_T37_intersections_stats_table(self, capsys):
        var_categories_upset(
            _h1(),
            "organ",
            min_fraction=1.0,
            print_stats=True,
            show=False,
        )
        header, rows = _intersections_table(capsys.readouterr().out, 3)
        assert header == [
            "kidney",
            "lung",
            "spleen",
            "n_features",
            "label",
        ]
        assert rows == [
            ["False", "False", "False", "2", "No category"],
            ["True", "False", "False", "2", "kidney"],
            ["True", "True", "True", "2", "kidney & lung & spleen"],
            ["True", "True", "False", "1", "kidney & lung"],
            ["True", "False", "True", "1", "kidney & spleen"],
            ["False", "True", "False", "1", "lung"],
            ["False", "True", "True", "1", "lung & spleen"],
            ["False", "False", "True", "1", "spleen"],
        ]

    def test_T38_per_category_stats_table(self, capsys):
        var_categories_upset(
            _h1(),
            "organ",
            min_fraction=1.0,
            print_stats=True,
            show=False,
        )
        header, rows = _per_table(capsys.readouterr().out, "organ")
        assert header == ["organ", "n_features", "percent"]
        assert rows == [
            ["kidney", "6", "54.5"],
            ["lung", "5", "45.5"],
            ["spleen", "5", "45.5"],
        ]

    def test_T39_stats_printed_before_show(self, monkeypatch, capsys):
        seen = {}

        def show(*args, **kwargs):
            seen.setdefault("out", capsys.readouterr().out)

        monkeypatch.setattr(plt, "show", show)
        var_categories_upset(_h1(), "organ", print_stats=True, show=True)
        assert "Global:" in seen["out"]

    def test_T40_labels_follow_category_order(self, capsys):
        var_categories_upset(_h6(), "batch", print_stats=True, show=False)
        _, rows = _intersections_table(capsys.readouterr().out, 3)
        labels = [row[-1] for row in rows]
        assert "zeta & alpha & mid" in labels

    def test_T75_category_named_like_table_column(self, spy, capsys):
        adata = _adata(
            "site",
            ["n_features"] * 3 + ["alpha"] * 3,
            {
                "e1": {"n_features": 3},
                "e2": {"alpha": 1, "n_features": 2},
                "e3": {"alpha": 3},
                "e4": {"alpha": 2},
            },
            2.0,
        )
        series = _series(spy, adata, "site", min_count=2, print_stats=True)
        _check_series(
            series,
            ["alpha", "n_features"],
            {(T, F): 2, (F, T): 2, (F, F): 0},
        )
        out = capsys.readouterr().out
        header, rows = _intersections_table(out, 2)
        assert header == ["alpha", "n_features", "n_features", "label"]
        assert rows == [
            ["True", "False", "2", "alpha"],
            ["False", "True", "2", "n_features"],
            ["False", "False", "0", "No category"],
        ]
        header, rows = _per_table(out, "site")
        assert header == ["site", "n_features", "percent"]
        assert rows == [
            ["alpha", "2", "50.0"],
            ["n_features", "2", "50.0"],
        ]

    def test_T76_category_key_named_like_table_column(self, spy, capsys):
        adata = _adata(
            "n_features",
            ["x"] * 3 + ["y"] * 2,
            {
                "g1": {"x": 3, "y": 2},
                "g2": {"y": 1},
                "g3": {"x": 1},
                "g4": {},
            },
            3.0,
        )
        series = _series(
            spy, adata, "n_features", min_fraction=0.5, print_stats=True
        )
        _check_series(
            series,
            ["x", "y"],
            {(T, T): 1, (F, T): 1, (T, F): 1, (F, F): 1},
        )
        header, rows = _per_table(capsys.readouterr().out, "n_features")
        assert header == ["n_features", "n_features", "percent"]
        assert rows == [
            ["x", "2", "50.0"],
            ["y", "2", "50.0"],
        ]

    # -- Default output

    def test_T42_quiet_by_default(self, capsys, tmp_path):
        var_categories_upset(
            _h1(),
            "organ",
            show=False,
            save=str(tmp_path / "quiet.png"),
        )
        assert capsys.readouterr().out == ""

    # -- save and show

    def test_T44_save_writes_file(self, tmp_path):
        target = tmp_path / "fig.pdf"
        var_categories_upset(_h1(), "organ", show=False, save=target)
        assert target.exists()
        assert target.stat().st_size > 0

    def test_T45_show_calls_pyplot_show(self, show_calls):
        var_categories_upset(_h3(), "site", show=True)
        assert len(show_calls) == 1

    def test_T46_no_show_when_disabled(self, show_calls, tmp_path):
        var_categories_upset(
            _h3(),
            "site",
            show=False,
            save=str(tmp_path / "quiet.png"),
        )
        assert show_calls == []

    def test_T47_new_figure_per_call(self, plot_axes):
        first = plot_axes(_h1(), "organ", show=False)
        second = plot_axes(_h1(), "organ", show=False)
        assert first["matrix"].figure is not second["matrix"].figure

    def test_T48_figure_left_open(self, plot_axes):
        axes = plot_axes(_h1(), "organ", show=False)
        assert plt.fignum_exists(axes["matrix"].figure.number)

    # -- Threshold selection

    def test_T56_thresholds_none_explicitly(self, spy):
        series = _series(
            spy, _h1(), "organ", min_count=None, min_fraction=None
        )
        _check_series(
            series,
            ["kidney", "lung", "spleen"],
            {
                (T, F, F): 2,
                (F, T, F): 1,
                (F, F, T): 1,
                (T, T, F): 2,
                (T, F, T): 1,
                (F, T, T): 1,
                (T, T, T): 2,
                (F, F, F): 1,
            },
        )

    def test_T65_min_fraction_integer_accepted(self, spy):
        series = _series(spy, _h1(), "organ", min_fraction=0)
        _check_series(
            series,
            ["kidney", "lung", "spleen"],
            {(T, T, T): 11, (F, F, F): 0},
        )

    def test_T66_min_fraction_numpy_float_accepted(self, spy):
        series = _series(spy, _h4(), "arm", min_fraction=np.float64(0.5))
        _check_series(
            series,
            ["p", "q"],
            {(T, F): 1, (T, T): 4, (F, F): 0},
        )

    # -- Negative cases

    def test_T50_save_invalid_type(self):
        with pytest.raises(TypeError):
            var_categories_upset(_h1(), "organ", show=False, save=["plot.png"])

    def test_T51_save_unsupported_type(self):
        with pytest.raises(TypeError):
            var_categories_upset(_h1(), "organ", show=False, save=42)

    def test_T52_flag_not_bool(self):
        with pytest.raises(TypeError):
            var_categories_upset(_h1(), "organ", show="yes")

    def test_T53_flag_bool_like_rejected(self):
        with pytest.raises(TypeError):
            var_categories_upset(
                _h1(), "organ", show=False, zero_to_na=np.bool_(False)
            )

    def test_T54_flag_non_bool_value(self):
        with pytest.raises(TypeError):
            var_categories_upset(_h1(), "organ", show=0.5)

    def test_T55_both_thresholds_set(self):
        with pytest.raises(ValueError):
            var_categories_upset(
                _h1(),
                "organ",
                min_count=1,
                min_fraction=1.0,
                show=False,
            )

    def test_T57_min_count_not_int(self):
        with pytest.raises(TypeError):
            var_categories_upset(
                _h1(), "organ", min_count=np.int32(3), show=False
            )

    def test_T58_min_count_wrong_type(self):
        with pytest.raises(TypeError):
            var_categories_upset(_h1(), "organ", min_count=[1], show=False)

    def test_T59_min_count_bool_rejected(self):
        with pytest.raises(TypeError):
            var_categories_upset(_h1(), "organ", min_count=False, show=False)

    def test_T60_min_count_negative(self):
        with pytest.raises(ValueError):
            var_categories_upset(_h1(), "organ", min_count=-7, show=False)

    def test_T61_min_fraction_wrong_type(self):
        with pytest.raises(TypeError):
            var_categories_upset(
                _h1(),
                "organ",
                min_fraction=np.float16(0.25),
                show=False,
            )

    def test_T62_min_fraction_bool_rejected(self):
        with pytest.raises(TypeError):
            var_categories_upset(
                _h1(), "organ", min_fraction=False, show=False
            )

    def test_T63_min_fraction_out_of_range(self):
        with pytest.raises(ValueError):
            var_categories_upset(
                _h1(), "organ", min_fraction=-0.01, show=False
            )

    def test_T64_min_fraction_not_finite(self):
        with pytest.raises(ValueError):
            var_categories_upset(
                _h1(), "organ", min_fraction=float("-inf"), show=False
            )

    def test_T74_fraction_beyond_float_range_rejected(self):
        with pytest.raises(ValueError):
            var_categories_upset(
                _h1(), "organ", min_fraction=-(2**1100), show=False
            )

    def test_T67_category_key_missing(self):
        with pytest.raises(KeyError):
            var_categories_upset(_h1(), "Organ", show=False)

    def test_T68_category_key_with_missing_values(self, spy):
        adata = _h6()
        adata.obs.loc["s1", "batch"] = np.nan
        before_obs = adata.obs.copy(deep=True)
        series = _series(spy, adata, "batch")
        _check_series(
            series,
            ["zeta", "alpha", "mid"],
            {(T, T, T): 1, (F, F, F): 2},
        )
        pd.testing.assert_frame_equal(adata.obs, before_obs)

    @pytest.mark.parametrize("missing", [None, np.nan, pd.NA])
    def test_missing_labels_do_not_join_a_literal_nan_group(
        self, spy, missing
    ):
        adata = _adata_from_columns(
            "group",
            [missing, "nan", "A"],
            {
                "only_missing": [1.0, np.nan, np.nan],
                "only_nan": [np.nan, 1.0, np.nan],
                "only_A": [np.nan, np.nan, 1.0],
            },
        )
        series = _series(spy, adata, "group")
        _check_series(
            series,
            ["A", "nan"],
            {(F, F): 1, (F, T): 1, (T, F): 1},
        )

    def test_explicit_missing_label_is_plotted(self, spy):
        adata = _adata_from_columns(
            "group",
            ["A", None, "A"],
            {
                "known": [1.0, np.nan, 1.0],
                "missing": [np.nan, 1.0, np.nan],
                "none": [np.nan, np.nan, np.nan],
            },
            categories=["unused", "Missing", "A"],
        )
        adata.obs["group"] = adata.obs["group"].fillna("Missing")
        series = _series(spy, adata, "group")
        _check_series(
            series,
            ["Missing", "A"],
            {(F, T): 1, (T, F): 1, (F, F): 1},
        )

    @pytest.mark.parametrize("categorical", [False, True])
    def test_all_missing_category_labels_raise_before_plotting(
        self, categorical
    ):
        adata = _h6()
        values = [None] * adata.n_obs
        if categorical:
            values = pd.Categorical(values, categories=["unused"])
        adata.obs["batch"] = values
        before = set(plt.get_fignums())
        with pytest.raises(ValueError, match="at least one non-missing"):
            var_categories_upset(adata, "batch", show=False)
        assert set(plt.get_fignums()) == before

    def test_T69_empty_axis(self):
        with pytest.raises(ValueError):
            var_categories_upset(_h1()[:, :0], "organ", show=False)

    def test_T70_category_key_not_str(self):
        with pytest.raises(TypeError):
            var_categories_upset(_h1(), ("organ",), show=False)

    def test_T71_category_key_empty_string(self):
        adata = _h1()
        adata.obs[""] = ["valid"] * adata.n_obs
        with pytest.raises(ValueError):
            var_categories_upset(adata, "", show=False)

    def test_T72_category_name_collision(self):
        adata = _adata("flag", [True, "True"], {"x1": {True: 1}}, 1.0)
        with pytest.raises(ValueError):
            var_categories_upset(adata, "flag", show=False)

    def test_T26_proteodata_validated(self):
        adata = _h1()
        X = np.array(adata.X, dtype=float)
        X[0, 0] = np.inf
        adata.X = X
        with pytest.raises(ValueError):
            var_categories_upset(adata, "organ", show=False)

    # -- Property relations

    def test_P1_series_matches_reference(self, rng, spy):
        for _ in range(_ROUNDS):
            case = _generate(rng)
            series = _last_series(spy, _case_adata(case), case)
            _check_series(series, case["categories"], _reference(case))
            plt.close("all")

    def test_P2_counts_sum_to_n_vars(self, rng, spy):
        for _ in range(_ROUNDS):
            case = _generate(rng)
            adata = _case_adata(case)
            series = _last_series(spy, adata, case)
            values = [int(value) for value in series.to_numpy()]
            assert all(value >= 0 for value in values)
            assert sum(values) == adata.n_vars
            plt.close("all")

    def test_P3_deterministic_series(self, rng, spy):
        for _ in range(_ROUNDS):
            case = _generate(rng)
            first = _last_series(spy, _case_adata(case), case)
            second = _last_series(spy, _case_adata(case), case)
            pd.testing.assert_series_equal(first, second)
            plt.close("all")

    def test_P4_sparse_matches_dense(self, rng, spy):
        for _ in range(_ROUNDS):
            case = _generate(rng)
            dense = _last_series(spy, _case_adata(case), case)
            adata = _case_adata(case)
            adata.X = sparse.csr_matrix(adata.X)
            with pytest.warns(UserWarning):
                sparse_series = _last_series(spy, adata, case)
            assert not isinstance(sparse_series.dtype, pd.SparseDtype)
            assert _mapping(sparse_series) == _mapping(dense)
            plt.close("all")

    def test_P5_input_unchanged(self, rng):
        for _ in range(_ROUNDS):
            case = _generate(rng)
            adata = _case_adata(case)
            before_x = np.array(adata.X, dtype=float)
            before_obs = adata.obs.copy(deep=True)
            before_var = adata.var.copy(deep=True)
            var_categories_upset(adata, "cond", **_case_kwargs(case))
            np.testing.assert_array_equal(
                np.array(adata.X, dtype=float), before_x
            )
            pd.testing.assert_frame_equal(adata.obs, before_obs)
            pd.testing.assert_frame_equal(adata.var, before_var)
            plt.close("all")

    # -- Metamorphic relations

    def test_M1_higher_count_threshold_shrinks_membership(self, rng, spy):
        for _ in range(_ROUNDS):
            case = _generate(rng)
            case["threshold"] = ("min_count", rng.randint(0, 4))
            lower = _last_series(spy, _case_adata(case), case)
            raised = dict(case)
            raised["threshold"] = (
                "min_count",
                case["threshold"][1] + 1,
            )
            higher = _last_series(spy, _case_adata(raised), raised)
            low_counts = _per_category(lower, case["categories"])
            high_counts = _per_category(higher, case["categories"])
            for category in case["categories"]:
                assert high_counts[category] <= low_counts[category]
            plt.close("all")

    def test_M2_obs_permutation_invariant(self, rng, spy):
        for _ in range(_ROUNDS):
            case = _generate(rng)
            original = _last_series(spy, _case_adata(case), case)
            order = list(range(len(case["cat_values"])))
            rng.shuffle(order)
            shuffled = dict(case)
            shuffled["cat_values"] = [case["cat_values"][i] for i in order]
            shuffled["columns"] = {
                name: [column[i] for i in order]
                for name, column in case["columns"].items()
            }
            permuted = _last_series(spy, _case_adata(shuffled), shuffled)
            assert list(permuted.index.names) == list(original.index.names)
            assert _mapping(permuted) == _mapping(original)
            plt.close("all")

    def test_M3_var_permutation_invariant(self, rng, spy):
        for _ in range(_ROUNDS):
            case = _generate(rng)
            original = _last_series(spy, _case_adata(case), case)
            names = list(case["columns"])
            rng.shuffle(names)
            shuffled = dict(case)
            shuffled["columns"] = {
                name: case["columns"][name] for name in names
            }
            permuted = _last_series(spy, _case_adata(shuffled), shuffled)
            assert _mapping(permuted) == _mapping(original)
            plt.close("all")

    def test_M4_detected_value_irrelevant(self, rng, spy):
        for _ in range(_ROUNDS):
            case = _generate(rng)
            case["zero_to_na"] = False
            original = _last_series(spy, _case_adata(case), case)
            zeroed = dict(case)
            zeroed["columns"] = {
                name: [entry if math.isnan(entry) else 0.0 for entry in column]
                for name, column in case["columns"].items()
            }
            replaced = _last_series(spy, _case_adata(zeroed), zeroed)
            assert _mapping(replaced) == _mapping(original)
            plt.close("all")
