"""Contract tests for category-completeness UpSet plots.

The suite covers:
1. Public API and exact UpSetPlot adapter input.
2. Threshold, missing-value, and sparse-matrix semantics.
3. Deterministic category ordering and ProteoData levels.
4. Numerical reporting and rendered bar values.
5. Plot lifecycle, non-mutation, and input validation.
"""

from collections import Counter
import inspect

import anndata as ad
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype
import pytest
from scipy import sparse

import proteopy as pr

plt.switch_backend("Agg")


@pytest.mark.skip(reason="completeness_by_cat_upset is not implemented yet")
class TestCompletenessByCatUpset:
    # ------------------------------------------------------------------
    # Helper constructors and independent reference data
    # ------------------------------------------------------------------

    @staticmethod
    def _function():
        """Return the public function under test."""
        return getattr(pr.pl, "completeness_by_cat_upset")

    @staticmethod
    def _spy_upset(monkeypatch):
        """Capture real UpSet constructor inputs and created instances."""
        from upsetplot import UpSet

        original_init = UpSet.__init__
        captured = {
            "data": [],
            "kwargs": [],
            "instances": [],
        }

        def spy_init(instance, data, *args, **kwargs):
            captured["data"].append(data.copy())
            captured["kwargs"].append(kwargs.copy())
            original_init(instance, data, *args, **kwargs)
            captured["instances"].append(instance)

        monkeypatch.setattr(UpSet, "__init__", spy_init)
        return captured

    @staticmethod
    def _count_map(series):
        """Map boolean membership tuples to integer counts."""
        return {
            tuple(bool(value) for value in membership): int(count)
            for membership, count in series.items()
        }

    @staticmethod
    def _expected_default_counts():
        """Return the hand-calculated default intersection counts."""
        index = pd.MultiIndex.from_tuples(
            [
                (False, False, False),
                (False, False, True),
                (False, True, False),
                (False, True, True),
                (True, False, False),
                (True, False, True),
                (True, True, False),
                (True, True, True),
            ],
            names=["A", "B", "C"],
        )
        return pd.Series(
            [2, 1, 1, 1, 1, 1, 1, 1],
            index=index,
            dtype=int,
            name="n_features",
        )

    @staticmethod
    def _protein_adata(matrix, conditions, protein_ids=None):
        """Build valid protein-level data for focused edge cases."""
        matrix = np.asarray(matrix, dtype=float)
        obs_names = [f"s{index}" for index in range(matrix.shape[0])]
        if protein_ids is None:
            protein_ids = [f"P{index}" for index in range(matrix.shape[1])]
        obs = pd.DataFrame(
            {
                "sample_id": obs_names,
                "condition": conditions,
            },
            index=obs_names,
        )
        var = pd.DataFrame(
            {"protein_id": protein_ids},
            index=protein_ids,
        )
        return ad.AnnData(X=matrix, obs=obs, var=var)

    @pytest.fixture
    def peptide_adata(self):
        """Build a peptide dataset with all category intersections."""
        obs_names = ["A1", "A2", "B1", "B2", "C1", "C2"]
        var_names = [
            "f_abc",
            "f_ab",
            "f_ac",
            "f_bc",
            "f_a",
            "f_b",
            "f_c",
            "f_none",
            "f_partial",
        ]
        observed = 1.0
        missing = np.nan
        matrix = np.array(
            [
                [
                    observed,
                    observed,
                    observed,
                    missing,
                    observed,
                    missing,
                    missing,
                    missing,
                    observed,
                ],
                [
                    observed,
                    observed,
                    observed,
                    missing,
                    observed,
                    missing,
                    missing,
                    missing,
                    missing,
                ],
                [
                    observed,
                    observed,
                    missing,
                    observed,
                    missing,
                    observed,
                    missing,
                    missing,
                    observed,
                ],
                [
                    observed,
                    observed,
                    missing,
                    observed,
                    missing,
                    observed,
                    missing,
                    missing,
                    missing,
                ],
                [
                    observed,
                    missing,
                    observed,
                    observed,
                    missing,
                    missing,
                    observed,
                    missing,
                    observed,
                ],
                [
                    observed,
                    missing,
                    observed,
                    observed,
                    missing,
                    missing,
                    observed,
                    missing,
                    missing,
                ],
            ],
            dtype=float,
        )
        obs = pd.DataFrame(
            {
                "sample_id": obs_names,
                "condition": ["A", "A", "B", "B", "C", "C"],
            },
            index=obs_names,
        )
        var = pd.DataFrame(
            {
                "peptide_id": var_names,
                "protein_id": [f"p{i}" for i in range(len(var_names))],
            },
            index=var_names,
        )
        return ad.AnnData(X=matrix, obs=obs, var=var)

    # ── A. Public API and adapter contract ───────────────────

    def test_signature_exposes_the_locked_public_contract(self):
        function = self._function()
        signature = inspect.signature(function)

        assert list(signature.parameters) == [
            "adata",
            "cat_key",
            "min_count",
            "min_fraction",
            "zero_to_na",
            "print_stats",
            "verbose",
            "show",
            "save",
        ]
        cat_key_default = signature.parameters["cat_key"].default
        assert cat_key_default is inspect.Parameter.empty
        assert signature.parameters["min_count"].default is None
        assert signature.parameters["min_fraction"].default == 1.0
        assert signature.parameters["zero_to_na"].default is False
        assert signature.parameters["print_stats"].default is False
        assert signature.parameters["verbose"].default is False
        assert signature.parameters["show"].default is True
        assert signature.parameters["save"].default is None
        assert "layer" not in signature.parameters
        assert "ax" not in signature.parameters

    def test_passes_exact_hand_calculated_counts_to_upset(
        self,
        peptide_adata,
        monkeypatch,
    ):
        function = self._function()
        captured = self._spy_upset(monkeypatch)

        function(peptide_adata, cat_key="condition", show=False)

        actual = captured["data"][0]
        expected = self._expected_default_counts()
        assert isinstance(actual, pd.Series)
        assert is_integer_dtype(actual.dtype)
        pd.testing.assert_series_equal(
            actual.sort_index(),
            expected.sort_index(),
        )

    def test_passes_explicit_count_semantics_and_order_to_upset(
        self,
        peptide_adata,
        monkeypatch,
    ):
        function = self._function()
        captured = self._spy_upset(monkeypatch)

        function(peptide_adata, cat_key="condition", show=False)

        kwargs = captured["kwargs"][0]
        assert kwargs["subset_size"] == "sum"
        assert kwargs["sort_by"] == "degree"
        assert kwargs["sort_categories_by"] == "input"
        assert kwargs["show_counts"] is True
        assert kwargs.get("include_empty_subsets", False) is False

    def test_upset_processed_values_match_independent_oracle(
        self,
        peptide_adata,
        monkeypatch,
    ):
        function = self._function()
        captured = self._spy_upset(monkeypatch)

        function(peptide_adata, cat_key="condition", show=False)

        upset = captured["instances"][0]
        assert self._count_map(upset.intersections) == self._count_map(
            self._expected_default_counts()
        )
        pd.testing.assert_series_equal(
            upset.totals.reindex(["A", "B", "C"]),
            pd.Series([4, 4, 4], index=["A", "B", "C"]),
            check_names=False,
        )
        assert upset.total == 9

    def test_returns_native_multi_axes_dictionary(self, peptide_adata):
        function = self._function()

        axes = function(peptide_adata, cat_key="condition", show=False)

        assert set(axes) == {
            "matrix",
            "intersections",
            "totals",
            "shading",
        }
        assert all(isinstance(axis, Axes) for axis in axes.values())
        assert len({id(axis.figure) for axis in axes.values()}) == 1

    # ── B. Threshold and detection semantics ─────────────────

    def test_min_fraction_uses_inclusive_per_category_boundary(
        self,
        peptide_adata,
        monkeypatch,
    ):
        function = self._function()
        captured = self._spy_upset(monkeypatch)

        function(
            peptide_adata,
            cat_key="condition",
            min_fraction=0.5,
            show=False,
        )

        counts = self._count_map(captured["data"][0])
        assert counts[(True, True, True)] == 2
        assert counts[(False, False, False)] == 1
        assert sum(counts.values()) == peptide_adata.n_vars

    def test_min_count_uses_inclusive_per_category_boundary(
        self,
        peptide_adata,
        monkeypatch,
    ):
        function = self._function()
        captured = self._spy_upset(monkeypatch)

        function(
            peptide_adata,
            cat_key="condition",
            min_count=1,
            min_fraction=None,
            show=False,
        )

        counts = self._count_map(captured["data"][0])
        assert counts[(True, True, True)] == 2
        assert counts[(False, False, False)] == 1
        assert sum(counts.values()) == peptide_adata.n_vars

    def test_zero_fraction_detects_every_feature_in_every_category(
        self,
        peptide_adata,
        monkeypatch,
    ):
        function = self._function()
        captured = self._spy_upset(monkeypatch)

        function(
            peptide_adata,
            cat_key="condition",
            min_fraction=0.0,
            show=False,
        )

        assert self._count_map(captured["data"][0]) == {
            (False, False, False): 0,
            (True, True, True): peptide_adata.n_vars,
        }

    def test_zero_count_detects_every_feature_in_every_category(
        self,
        peptide_adata,
        monkeypatch,
    ):
        function = self._function()
        captured = self._spy_upset(monkeypatch)

        function(
            peptide_adata,
            cat_key="condition",
            min_count=0,
            min_fraction=None,
            show=False,
        )

        assert self._count_map(captured["data"][0]) == {
            (False, False, False): 0,
            (True, True, True): peptide_adata.n_vars,
        }

    def test_count_above_group_size_places_every_feature_in_no_category(
        self,
        peptide_adata,
        monkeypatch,
    ):
        function = self._function()
        captured = self._spy_upset(monkeypatch)

        function(
            peptide_adata,
            cat_key="condition",
            min_count=3,
            min_fraction=None,
            show=False,
        )

        assert self._count_map(captured["data"][0]) == {
            (False, False, False): peptide_adata.n_vars,
        }

    def test_fraction_uses_each_category_own_sample_count(self, monkeypatch):
        function = self._function()
        captured = self._spy_upset(monkeypatch)
        adata = self._protein_adata(
            matrix=[[1.0], [np.nan], [1.0], [np.nan], [np.nan]],
            conditions=["A", "A", "B", "B", "B"],
        )

        function(
            adata,
            cat_key="condition",
            min_fraction=0.5,
            show=False,
        )

        assert self._count_map(captured["data"][0]) == {
            (False, False): 0,
            (True, False): 1,
        }

    def test_zero_to_na_changes_zero_detection(self, monkeypatch):
        function = self._function()
        captured = self._spy_upset(monkeypatch)
        adata = self._protein_adata(
            matrix=[[0.0, np.nan], [0.0, np.nan]],
            conditions=["A", "B"],
            protein_ids=["P_zero", "P_nan"],
        )

        function(adata, cat_key="condition", show=False)
        function(
            adata,
            cat_key="condition",
            zero_to_na=True,
            show=False,
        )

        default_counts = self._count_map(captured["data"][0])
        zero_missing_counts = self._count_map(captured["data"][1])
        assert default_counts == {
            (False, False): 1,
            (True, True): 1,
        }
        assert zero_missing_counts == {(False, False): 2}

    def test_zero_is_present_but_none_and_nan_are_absent(
        self,
        monkeypatch,
    ):
        function = self._function()
        captured = self._spy_upset(monkeypatch)
        adata = self._protein_adata(
            matrix=[
                [0.0, None, np.nan, 2.0],
                [0.0, np.nan, None, np.nan],
            ],
            conditions=["A", "B"],
        )

        function(adata, cat_key="condition", show=False)

        assert self._count_map(captured["data"][0]) == {
            (False, False): 2,
            (True, False): 1,
            (True, True): 1,
        }

    def test_repeated_measurements_of_one_protein_count_once_per_set(
        self,
        monkeypatch,
    ):
        function = self._function()
        captured = self._spy_upset(monkeypatch)
        adata = self._protein_adata(
            matrix=[[1.0], [2.0], [np.nan]],
            conditions=["A", "A", "B"],
            protein_ids=["P12345"],
        )

        function(adata, cat_key="condition", show=False)

        assert self._count_map(captured["data"][0]) == {
            (False, False): 0,
            (True, False): 1,
        }

    def test_protein_ids_are_case_sensitive_while_repeats_collapse(
        self,
        monkeypatch,
    ):
        function = self._function()
        captured = self._spy_upset(monkeypatch)
        adata = self._protein_adata(
            matrix=[[1.0, 3.0], [2.0, 4.0]],
            conditions=["A", "A"],
            protein_ids=["P12345", "p12345"],
        )

        function(adata, cat_key="condition", show=False)

        assert self._count_map(captured["data"][0]) == {
            (False,): 0,
            (True,): 2,
        }

    # ── C. Sparse data, levels, set edges, and ordering ───────

    def test_dense_and_sparse_x_produce_identical_upset_input(
        self,
        peptide_adata,
        monkeypatch,
    ):
        function = self._function()
        captured = self._spy_upset(monkeypatch)
        sparse_adata = peptide_adata.copy()
        sparse_adata.X = sparse.csr_matrix(sparse_adata.X)

        function(peptide_adata, cat_key="condition", show=False)
        with pytest.warns(UserWarning, match="[Ss]parse.*dens"):
            function(sparse_adata, cat_key="condition", show=False)

        pd.testing.assert_series_equal(
            captured["data"][0].sort_index(),
            captured["data"][1].sort_index(),
        )
        assert sparse.isspmatrix_csr(sparse_adata.X)

    def test_accepts_protein_level_proteodata(self, peptide_adata):
        function = self._function()
        protein_adata = peptide_adata.copy()
        protein_adata.var = pd.DataFrame(
            {"protein_id": list(protein_adata.var_names)},
            index=protein_adata.var_names,
        )

        axes = function(
            protein_adata,
            cat_key="condition",
            show=False,
        )

        assert isinstance(axes["intersections"], Axes)

    def test_plain_string_categories_use_lexicographic_order(
        self,
        peptide_adata,
        monkeypatch,
    ):
        function = self._function()
        captured = self._spy_upset(monkeypatch)
        reordered = peptide_adata[[2, 3, 0, 1, 4, 5], :].copy()

        axes = function(reordered, cat_key="condition", show=False)

        assert captured["data"][0].index.names == ["A", "B", "C"]
        assert list(captured["instances"][0].totals.index) == [
            "A",
            "B",
            "C",
        ]
        labels = [tick.get_text() for tick in axes["matrix"].get_yticklabels()]
        assert labels == ["A", "B", "C"]

    def test_categorical_categories_use_declared_category_order(
        self,
        peptide_adata,
        monkeypatch,
    ):
        function = self._function()
        captured = self._spy_upset(monkeypatch)
        peptide_adata.obs["condition"] = pd.Categorical(
            peptide_adata.obs["condition"],
            categories=["C", "A", "B"],
            ordered=True,
        )

        axes = function(peptide_adata, cat_key="condition", show=False)

        assert captured["data"][0].index.names == ["C", "A", "B"]
        assert list(captured["instances"][0].totals.index) == [
            "C",
            "A",
            "B",
        ]
        labels = [tick.get_text() for tick in axes["matrix"].get_yticklabels()]
        assert labels == ["C", "A", "B"]

    def test_supports_a_single_observed_category(
        self,
        peptide_adata,
        monkeypatch,
    ):
        function = self._function()
        captured = self._spy_upset(monkeypatch)
        one_category = peptide_adata[:2, :].copy()

        function(one_category, cat_key="condition", show=False)

        assert self._count_map(captured["data"][0]) == {
            (False,): 5,
            (True,): 4,
        }

    def test_two_identical_sets_have_only_one_shared_region(
        self,
        monkeypatch,
    ):
        function = self._function()
        captured = self._spy_upset(monkeypatch)
        adata = self._protein_adata(
            matrix=[[1.0, 2.0], [3.0, 4.0]],
            conditions=["A", "B"],
        )

        function(adata, cat_key="condition", show=False)

        counts = self._count_map(captured["data"][0])
        assert counts == {
            (False, False): 0,
            (True, True): 2,
        }
        assert counts.get((True, False), 0) == 0
        assert counts.get((False, True), 0) == 0

    def test_fully_disjoint_sets_have_no_shared_region(
        self,
        monkeypatch,
    ):
        function = self._function()
        captured = self._spy_upset(monkeypatch)
        adata = self._protein_adata(
            matrix=[[1.0, np.nan], [np.nan, 2.0]],
            conditions=["A", "B"],
        )

        function(adata, cat_key="condition", show=False)

        counts = self._count_map(captured["data"][0])
        assert counts == {
            (False, False): 0,
            (False, True): 1,
            (True, False): 1,
        }
        assert counts.get((True, True), 0) == 0

    def test_retains_category_with_zero_detected_features(
        self,
        peptide_adata,
        monkeypatch,
    ):
        function = self._function()
        captured = self._spy_upset(monkeypatch)
        peptide_adata.X[4:, :] = np.nan

        function(peptide_adata, cat_key="condition", show=False)

        upset = captured["instances"][0]
        assert captured["data"][0].index.names == ["A", "B", "C"]
        assert upset.totals["C"] == 0

    def test_weird_category_names_survive_adapter_and_tick_labels(
        self,
        monkeypatch,
    ):
        function = self._function()
        captured = self._spy_upset(monkeypatch)
        names = ["A / β & [x]", "B:()?! + ="]
        adata = self._protein_adata(
            matrix=[[1.0], [1.0]],
            conditions=names,
        )

        axes = function(adata, cat_key="condition", show=False)

        assert captured["data"][0].index.names == names
        labels = [tick.get_text() for tick in axes["matrix"].get_yticklabels()]
        assert labels == names

    def test_very_long_category_name_is_not_truncated(
        self,
        monkeypatch,
    ):
        function = self._function()
        captured = self._spy_upset(monkeypatch)
        long_name = "condition_" + "very_long_" * 20 + "end"
        adata = self._protein_adata(
            matrix=[[1.0]],
            conditions=[long_name],
        )

        axes = function(adata, cat_key="condition", show=False)

        assert captured["data"][0].index.names == [long_name]
        labels = [tick.get_text() for tick in axes["matrix"].get_yticklabels()]
        assert labels == [long_name]

    def test_eleven_categories_are_not_dropped_or_truncated(
        self,
        monkeypatch,
    ):
        function = self._function()
        captured = self._spy_upset(monkeypatch)
        names = [f"set_{index:02d}" for index in range(11)]
        adata = self._protein_adata(
            matrix=np.ones((11, 1)),
            conditions=names,
        )

        axes = function(adata, cat_key="condition", show=False)

        assert captured["data"][0].index.names == names
        labels = [tick.get_text() for tick in axes["matrix"].get_yticklabels()]
        assert labels == names

    def test_adapter_order_is_byte_reproducible(
        self,
        peptide_adata,
        monkeypatch,
    ):
        function = self._function()
        captured = self._spy_upset(monkeypatch)

        function(peptide_adata, cat_key="condition", show=False)
        function(peptide_adata, cat_key="condition", show=False)

        first, second = captured["data"]
        pd.testing.assert_series_equal(first, second)
        first_bytes = first.to_csv().encode("utf-8")
        second_bytes = second.to_csv().encode("utf-8")
        assert first_bytes == second_bytes

    # ── D. No-category set and rendered values ──────────────

    def test_no_category_is_present_when_its_count_is_zero(
        self,
        peptide_adata,
        monkeypatch,
    ):
        function = self._function()
        captured = self._spy_upset(monkeypatch)
        all_categories = peptide_adata[:, ["f_abc"]].copy()

        axes = function(
            all_categories,
            cat_key="condition",
            show=False,
        )

        counts = self._count_map(captured["data"][0])
        assert counts[(False, False, False)] == 0
        assert len(axes["intersections"].patches) == len(
            captured["instances"][0].intersections
        )
        patches = axes["intersections"].patches
        heights = [patch.get_height() for patch in patches]
        assert 0 in heights

    def test_no_category_is_named_in_the_plot_legend(
        self,
        peptide_adata,
    ):
        function = self._function()

        axes = function(peptide_adata, cat_key="condition", show=False)

        legend = axes["intersections"].get_legend()
        assert legend is not None
        labels = [text.get_text() for text in legend.get_texts()]
        assert "No category" in labels

    def test_intersection_bar_heights_equal_hand_calculated_counts(
        self,
        peptide_adata,
    ):
        function = self._function()

        axes = function(peptide_adata, cat_key="condition", show=False)

        patches = axes["intersections"].patches
        heights = [patch.get_height() for patch in patches]
        assert sorted(heights) == [1, 1, 1, 1, 1, 1, 1, 2]

    def test_dwarfing_set_keeps_small_intersection_bar_exact(
        self,
        monkeypatch,
    ):
        function = self._function()
        captured = self._spy_upset(monkeypatch)
        matrix = np.full((2, 1001), np.nan)
        matrix[0, :] = 0.0
        matrix[1, 0] = 1.0
        adata = self._protein_adata(
            matrix=matrix,
            conditions=["A", "B"],
        )

        axes = function(adata, cat_key="condition", show=False)

        assert self._count_map(captured["data"][0]) == {
            (False, False): 0,
            (True, False): 1000,
            (True, True): 1,
        }
        patches = axes["intersections"].patches
        heights = [patch.get_height() for patch in patches]
        assert sorted(heights) == [0, 1, 1000]

    def test_marginal_bar_widths_equal_upset_totals(
        self,
        peptide_adata,
        monkeypatch,
    ):
        function = self._function()
        captured = self._spy_upset(monkeypatch)

        axes = function(peptide_adata, cat_key="condition", show=False)

        widths = [patch.get_width() for patch in axes["totals"].patches]
        expected = captured["instances"][0].totals.to_numpy()
        np.testing.assert_array_equal(sorted(widths), sorted(expected))

    def test_intersection_text_labels_equal_bar_values(
        self,
        peptide_adata,
        monkeypatch,
    ):
        function = self._function()
        captured = self._spy_upset(monkeypatch)

        axes = function(peptide_adata, cat_key="condition", show=False)

        labels = Counter(
            text.get_text().strip() for text in axes["intersections"].texts
        )
        expected = Counter(
            str(int(value)) for value in captured["instances"][0].intersections
        )
        assert labels == expected

    # ── E. Printed and verbose reporting ────────────────────

    def test_print_stats_reports_global_intersection_and_marginal_values(
        self,
        peptide_adata,
        capsys,
    ):
        function = self._function()

        function(
            peptide_adata,
            cat_key="condition",
            print_stats=True,
            show=False,
        )

        lines = {
            " ".join(line.split())
            for line in capsys.readouterr().out.splitlines()
            if line.strip()
        }
        output = "\n".join(lines)
        assert "Global:" in output
        assert "count mean median std min max" in output
        assert "8 1.1 1.0 0.4 1 2" in lines
        assert "Intersections:" in output
        assert "False False False 2 No category" in lines
        assert "True True True 1 A & B & C" in lines
        assert "Per condition:" in output
        assert "category n_features fraction" in output
        assert "A 4 0.4" in lines
        assert "B 4 0.4" in lines
        assert "C 4 0.4" in lines

    def test_print_stats_false_is_quiet(self, peptide_adata, capsys):
        function = self._function()

        function(peptide_adata, cat_key="condition", show=False)

        assert capsys.readouterr().out == ""

    def test_verbose_reports_input_threshold_and_dimensions(
        self,
        peptide_adata,
        capsys,
    ):
        function = self._function()

        function(
            peptide_adata,
            cat_key="condition",
            verbose=True,
            show=False,
        )

        output = capsys.readouterr().out
        assert ".X" in output
        assert "condition" in output
        assert "min_fraction=1.0" in output
        assert "9" in output
        assert "3" in output

    # ── F. Plot lifecycle and non-mutation ──────────────────

    def test_show_true_calls_pyplot_show(
        self,
        peptide_adata,
        monkeypatch,
    ):
        function = self._function()
        calls = []
        monkeypatch.setattr(plt, "show", lambda: calls.append(True))

        axes = function(peptide_adata, cat_key="condition", show=True)

        assert calls == [True]
        assert isinstance(axes["intersections"], Axes)

    def test_show_false_does_not_call_pyplot_show(
        self,
        peptide_adata,
        monkeypatch,
    ):
        function = self._function()
        calls = []
        monkeypatch.setattr(plt, "show", lambda: calls.append(True))

        function(peptide_adata, cat_key="condition", show=False)

        assert calls == []

    def test_save_path_writes_nonempty_figure(
        self,
        peptide_adata,
        tmp_path,
    ):
        function = self._function()
        destination = tmp_path / "upset.png"

        axes = function(
            peptide_adata,
            cat_key="condition",
            show=False,
            save=destination,
        )

        assert destination.is_file()
        assert destination.stat().st_size > 0
        assert isinstance(axes["matrix"], Axes)

    def test_plot_does_not_mutate_anndata_or_view(self, peptide_adata):
        function = self._function()
        view = peptide_adata[:, :]
        matrix_before = peptide_adata.X.copy()
        obs_before = peptide_adata.obs.copy(deep=True)
        var_before = peptide_adata.var.copy(deep=True)

        function(view, cat_key="condition", show=False)

        np.testing.assert_array_equal(
            peptide_adata.X,
            matrix_before,
        )
        pd.testing.assert_frame_equal(peptide_adata.obs, obs_before)
        pd.testing.assert_frame_equal(peptide_adata.var, var_before)

    # ── G. Validation ──────────────────────────────────

    def test_invalid_proteodata_is_rejected_before_plotting(self):
        function = self._function()
        adata = ad.AnnData(
            X=np.array([[1.0]]),
            obs=pd.DataFrame(
                {"sample_id": ["s1"], "condition": ["A"]},
                index=["s1"],
            ),
        )

        with pytest.raises(ValueError, match="protein_id|peptide_id"):
            function(adata, cat_key="condition", show=False)

    def test_missing_cat_key_is_rejected(self, peptide_adata):
        function = self._function()

        with pytest.raises(KeyError, match="missing.*obs|obs.*missing"):
            function(peptide_adata, cat_key="missing", show=False)

    def test_invalid_cat_key_type_is_rejected(self, peptide_adata):
        function = self._function()
        invalid_values = [None, 1]

        for invalid in invalid_values:
            with pytest.raises(TypeError, match="cat_key"):
                function(peptide_adata, cat_key=invalid, show=False)

    def test_empty_cat_key_is_rejected(self, peptide_adata):
        function = self._function()

        with pytest.raises(ValueError, match="cat_key"):
            function(peptide_adata, cat_key="", show=False)

    def test_missing_category_labels_are_rejected(self, peptide_adata):
        function = self._function()
        peptide_adata.obs.loc["A1", "condition"] = np.nan

        with pytest.raises(
            ValueError,
            match="condition.*missing|missing.*condition",
        ):
            function(peptide_adata, cat_key="condition", show=False)

    def test_empty_observation_axis_is_rejected(self, peptide_adata):
        function = self._function()
        empty = peptide_adata[:0, :].copy()

        with pytest.raises(ValueError, match="observation|sample|empty"):
            function(empty, cat_key="condition", show=False)

    def test_empty_variable_axis_is_rejected(self, peptide_adata):
        function = self._function()
        empty = peptide_adata[:, :0].copy()

        with pytest.raises(ValueError, match="variable|feature|empty"):
            function(empty, cat_key="condition", show=False)

    def test_completely_empty_input_is_rejected(self):
        function = self._function()
        empty = self._protein_adata(
            matrix=np.empty((0, 0)),
            conditions=[],
            protein_ids=[],
        )

        with pytest.raises(
            ValueError,
            match="observation|sample|variable|feature|empty",
        ):
            function(empty, cat_key="condition", show=False)

    def test_duplicate_protein_feature_ids_are_rejected(self):
        function = self._function()
        duplicate_ids = ["P12345", "P12345"]
        adata = self._protein_adata(
            matrix=[[1.0, 2.0]],
            conditions=["A"],
            protein_ids=duplicate_ids,
        )

        with pytest.raises(ValueError, match="unique|duplicate"):
            function(adata, cat_key="condition", show=False)

    def test_both_thresholds_are_rejected(self, peptide_adata):
        function = self._function()

        with pytest.raises(
            ValueError,
            match="mutually exclusive|one threshold",
        ):
            function(
                peptide_adata,
                cat_key="condition",
                min_count=1,
                min_fraction=0.5,
                show=False,
            )

    def test_min_count_requires_explicitly_disabling_default_fraction(
        self,
        peptide_adata,
    ):
        function = self._function()

        with pytest.raises(
            ValueError,
            match="mutually exclusive|one threshold",
        ):
            function(
                peptide_adata,
                cat_key="condition",
                min_count=1,
                show=False,
            )

    def test_missing_threshold_is_rejected(self, peptide_adata):
        function = self._function()

        with pytest.raises(ValueError, match="one threshold|required"):
            function(
                peptide_adata,
                cat_key="condition",
                min_count=None,
                min_fraction=None,
                show=False,
            )

    def test_invalid_min_fraction_values_are_rejected(self, peptide_adata):
        function = self._function()
        invalid_values = {
            "below range": -0.1,
            "above range": 1.1,
            "not finite": np.nan,
            "boolean": True,
            "string": "0.5",
        }

        for invalid in invalid_values.values():
            with pytest.raises((TypeError, ValueError), match="min_fraction"):
                function(
                    peptide_adata,
                    cat_key="condition",
                    min_fraction=invalid,
                    show=False,
                )

    def test_invalid_min_count_values_are_rejected(self, peptide_adata):
        function = self._function()
        invalid_values = {
            "negative": -1,
            "non-integral": 1.5,
            "boolean": True,
            "string": "1",
        }

        for invalid in invalid_values.values():
            with pytest.raises((TypeError, ValueError), match="min_count"):
                function(
                    peptide_adata,
                    cat_key="condition",
                    min_count=invalid,
                    min_fraction=None,
                    show=False,
                )

    @pytest.mark.parametrize(
        "argument,value",
        [
            ("zero_to_na", 1),
            ("print_stats", 1),
            ("verbose", 1),
            ("show", 1),
        ],
    )
    def test_invalid_boolean_arguments_are_rejected(
        self,
        peptide_adata,
        argument,
        value,
    ):
        function = self._function()
        kwargs = {"show": False, argument: value}

        with pytest.raises(TypeError, match=argument):
            function(
                peptide_adata,
                cat_key="condition",
                **kwargs,
            )

    def test_invalid_save_type_is_rejected(self, peptide_adata):
        function = self._function()
        invalid_values = [True, False, ["plot.png"]]

        for invalid in invalid_values:
            with pytest.raises(TypeError, match="save"):
                function(
                    peptide_adata,
                    cat_key="condition",
                    save=invalid,
                    show=False,
                )
