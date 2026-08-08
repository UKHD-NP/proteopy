"""Contract tests for quantile normalization preprocessing.

The suite is organized by behavioral category so a failing contract is
quick to locate:

1. Core normalization covers the classic complete-data algorithm and
   its distributional invariants.
2. Storage and returns cover ``.uns`` schema, keys, and in-place/copy
   semantics.
3. Missingness covers limma-style stretching of ragged sample
   distributions onto the full feature grid before pooling.
4. Ties cover the fixed average-rank policy and interpolation back from
   the reference distribution.
5. Grouping covers independent references, categorical edge cases, and
   groups without finite observations.
6. Numerical edge cases cover zero-valued references, singleton grids,
   singleton observations, and warning-free behavior.
7. Validation covers the public API, ProteoPy data requirements, and
   failures that must not partially mutate the input.
8. Verbose output covers operation, storage, and missingness messages.

Small protein-level AnnData matrices make both interpolation phases
readable by inspection. Exact hand-computed expectations distinguish
fractional-quantile stretching from integer-rank averaging. Private
helpers are intentionally exercised only through ``normalize_quantile``.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse

try:
    from proteopy.pp.normalize_quantile import normalize_quantile
except ModuleNotFoundError as error:
    if error.name != "proteopy.pp.normalize_quantile":
        raise
    normalize_quantile = None


@pytest.mark.skipif(
    normalize_quantile is None,
    reason="normalize_quantile is not implemented yet",
)
class TestNormalizeQuantile:
    """Contract tests for normalize_quantile."""

    # ------------------------------------------------------------------
    # Helper constructors
    # ------------------------------------------------------------------

    @staticmethod
    def make_protein_adata(
        X,
        obs_names=None,
        var_names=None,
        obs_extra=None,
    ):
        """Build and return minimal protein-level proteodata."""
        shape = np.asarray(X).shape
        if obs_names is None:
            obs_names = [f"s{i}" for i in range(shape[0])]
        if var_names is None:
            var_names = [f"p{i}" for i in range(shape[1])]

        obs = pd.DataFrame({"sample_id": obs_names}, index=obs_names)
        if obs_extra:
            for key, values in obs_extra.items():
                obs[key] = values

        var = pd.DataFrame({"protein_id": var_names}, index=var_names)
        return AnnData(
            X=np.asarray(X, dtype=float),
            obs=obs,
            var=var,
        )

    @staticmethod
    def normalize_and_collect(adata, inplace, **kwargs):
        """Run normalization and return its AnnData and reference."""
        returned = normalize_quantile(
            adata,
            inplace=inplace,
            **kwargs,
        )
        key_added = kwargs.get("key_added", "quantile_reference")
        if inplace:
            assert returned is None
            return adata, adata.uns[key_added]

        adata_out, reference = returned
        return adata_out, reference

    # ── A. Core normalization ────────────

    @pytest.mark.parametrize("inplace", [True, False])
    def test_complete_matrix_maps_each_rank_to_shared_reference(
        self,
        inplace,
    ):
        adata = self.make_protein_adata(
            [
                [8.0, 2.0, 6.0, 4.0],
                [5.0, 1.0, 7.0, 3.0],
                [7.0, 9.0, 3.0, 5.0],
            ]
        )

        result, _ = self.normalize_and_collect(adata, inplace)

        expected = np.array(
            [
                [8.0, 2.0, 6.0, 4.0],
                [6.0, 2.0, 8.0, 4.0],
                [6.0, 8.0, 2.0, 4.0],
            ]
        )
        assert result.shape == adata.shape
        np.testing.assert_allclose(result.X, expected)

    def test_complete_samples_share_the_reference_distribution(self):
        adata = self.make_protein_adata(
            [
                [8.0, 2.0, 6.0, 4.0],
                [5.0, 1.0, 7.0, 3.0],
                [7.0, 9.0, 3.0, 5.0],
            ]
        )

        result, _ = normalize_quantile(adata, inplace=False)

        expected_distribution = np.array([2.0, 4.0, 6.0, 8.0])
        for row in result.X:
            np.testing.assert_allclose(
                np.sort(row),
                expected_distribution,
            )

    def test_normalization_is_idempotent(self):
        adata = self.make_protein_adata(
            [
                [8.0, 2.0, 6.0, 4.0],
                [5.0, 1.0, 7.0, 3.0],
                [7.0, 9.0, 3.0, 5.0],
            ]
        )

        once, _ = normalize_quantile(adata, inplace=False)
        twice, _ = normalize_quantile(once, inplace=False)

        np.testing.assert_allclose(twice.X, once.X)

    def test_already_normalized_samples_remain_unchanged(self):
        X = np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [4.0, 1.0, 3.0, 2.0],
                [2.0, 4.0, 1.0, 3.0],
            ]
        )
        adata = self.make_protein_adata(X)

        result, _ = normalize_quantile(adata, inplace=False)

        np.testing.assert_allclose(result.X, X)

    # ── B. Storage and return contract ───────────

    @pytest.mark.parametrize("inplace", [True, False])
    def test_reference_is_stored_under_default_uns_key(self, inplace):
        adata = self.make_protein_adata([[1.0, 2.0], [3.0, 4.0]])

        result, reference = self.normalize_and_collect(adata, inplace)

        assert "quantile_reference" in result.uns
        assert reference is result.uns["quantile_reference"]

    @pytest.mark.parametrize("inplace", [True, False])
    def test_custom_key_added_is_honoured(self, inplace):
        adata = self.make_protein_adata([[1.0, 2.0], [3.0, 4.0]])

        result, reference = self.normalize_and_collect(
            adata,
            inplace,
            key_added="quantile_norm",
        )

        assert "quantile_norm" in result.uns
        assert "quantile_reference" not in result.uns
        assert reference is result.uns["quantile_norm"]

    def test_reference_dataframe_has_quantile_grid_and_counts(self):
        adata = self.make_protein_adata(
            [
                [8.0, 2.0, 6.0, 4.0],
                [5.0, 1.0, 7.0, 3.0],
                [7.0, 9.0, 3.0, 5.0],
            ]
        )

        _, reference = normalize_quantile(adata, inplace=False)

        assert list(reference.columns) == [
            "quantile_index",
            "quantile",
            "reference_value",
            "n_samples",
        ]
        np.testing.assert_array_equal(
            reference["quantile_index"].to_numpy(),
            [0, 1, 2, 3],
        )
        np.testing.assert_allclose(
            reference["quantile"].to_numpy(),
            [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0],
        )
        np.testing.assert_allclose(
            reference["reference_value"].to_numpy(),
            [2.0, 4.0, 6.0, 8.0],
        )
        np.testing.assert_array_equal(
            reference["n_samples"].to_numpy(),
            [3, 3, 3, 3],
        )

    def test_inplace_false_returns_independent_copy(self):
        adata = self.make_protein_adata([[1.0, 2.0], [3.0, 4.0]])
        original = adata.X.copy()

        result, _ = normalize_quantile(adata, inplace=False)
        result.X[0, 0] = -999.0

        assert result is not adata
        assert not np.shares_memory(result.X, adata.X)
        np.testing.assert_array_equal(adata.X, original)

    def test_inplace_true_returns_none_and_modifies_original(self):
        adata = self.make_protein_adata([[1.0, 2.0], [3.0, 5.0]])
        original = adata.X.copy()

        returned = normalize_quantile(adata)

        assert returned is None
        assert not np.array_equal(adata.X, original)

    # ── C. Missingness and stretched distributions ──────

    def test_fractional_position_interpolates_between_reference_nodes(self):
        adata = self.make_protein_adata(
            [
                [90.0, 0.0, 40.0, 10.0],
                [0.0, 40.0, 90.0, 10.0],
                [15.0, np.nan, 90.0, 0.0],
            ]
        )

        result, reference = normalize_quantile(adata, inplace=False)

        np.testing.assert_allclose(
            reference["reference_value"].to_numpy(),
            [0.0, 10.0, 40.0, 90.0],
        )
        np.testing.assert_allclose(
            result.X[2],
            [25.0, np.nan, 90.0, 0.0],
            equal_nan=True,
        )
        assert 25.0 not in set(reference["reference_value"])

    def test_ragged_samples_are_stretched_before_reference_pooling(self):
        adata = self.make_protein_adata(
            [
                [1.0, 4.0, np.nan, np.nan],
                [2.0, 6.0, 8.0, np.nan],
                [3.0, 5.0, 7.0, 9.0],
            ]
        )

        result, reference = normalize_quantile(adata, inplace=False)

        expected_reference = np.array([2.0, 35.0 / 9.0, 50.0 / 9.0, 7.0])
        expected = np.array(
            [
                [2.0, 7.0, np.nan, np.nan],
                [2.0, 85.0 / 18.0, 7.0, np.nan],
                [2.0, 35.0 / 9.0, 50.0 / 9.0, 7.0],
            ]
        )
        np.testing.assert_allclose(
            reference["reference_value"].to_numpy(),
            expected_reference,
        )
        np.testing.assert_allclose(result.X, expected, equal_nan=True)

        # Integer-rank pooling would map the first sample's maximum to
        # 5.0. Full-grid stretching correctly aligns it with maxima.
        assert result.X[0, 1] == pytest.approx(7.0)
        assert result.X[0, 1] != pytest.approx(5.0)

    def test_nan_positions_are_preserved(self):
        X = np.array(
            [
                [1.0, np.nan, 4.0, np.nan],
                [2.0, 3.0, np.nan, 8.0],
                [3.0, 5.0, 7.0, 9.0],
            ]
        )
        adata = self.make_protein_adata(X)

        result, _ = normalize_quantile(adata, inplace=False)

        np.testing.assert_array_equal(np.isnan(result.X), np.isnan(X))

    def test_all_nan_sample_is_excluded_from_reference(self):
        adata = self.make_protein_adata(
            [
                [np.nan, np.nan, np.nan],
                [1.0, 2.0, 3.0],
                [3.0, 4.0, 5.0],
            ]
        )

        result, reference = normalize_quantile(adata, inplace=False)

        np.testing.assert_array_equal(
            np.isnan(result.X[0]),
            [True, True, True],
        )
        np.testing.assert_allclose(
            reference["reference_value"].to_numpy(),
            [2.0, 3.0, 4.0],
        )
        np.testing.assert_array_equal(
            reference["n_samples"].to_numpy(),
            [2, 2, 2],
        )

    def test_zero_to_na_excludes_zeros_and_preserves_missingness(self):
        adata = self.make_protein_adata(
            [
                [0.0, 1.0, 4.0, 8.0],
                [0.0, 2.0, 6.0, 10.0],
            ]
        )

        result, _ = normalize_quantile(
            adata,
            zero_to_na=True,
            inplace=False,
        )

        assert np.isnan(result.X[:, 0]).all()
        assert np.isfinite(result.X[:, 1:]).all()

    def test_fill_na_participates_in_reference_construction(self):
        adata = self.make_protein_adata(
            [
                [np.nan, 4.0],
                [2.0, 6.0],
            ]
        )

        result, reference = normalize_quantile(
            adata,
            fill_na=0.0,
            inplace=False,
        )

        np.testing.assert_allclose(result.X, [[1.0, 5.0], [1.0, 5.0]])
        np.testing.assert_allclose(
            reference["reference_value"].to_numpy(),
            [1.0, 5.0],
        )
        assert not np.isnan(result.X).any()

    # ── D. Fixed average-rank tie policy ──────────

    def test_tied_values_receive_average_rank_interpolation(self):
        adata = self.make_protein_adata(
            [
                [1.0, 1.0, 4.0, 8.0],
                [0.0, 2.0, 6.0, 10.0],
            ]
        )

        result, reference = normalize_quantile(adata, inplace=False)

        np.testing.assert_allclose(
            reference["reference_value"].to_numpy(),
            [0.5, 1.5, 5.0, 9.0],
        )
        np.testing.assert_allclose(result.X[0], [1.0, 1.0, 5.0, 9.0])

    def test_tied_values_do_not_depend_on_feature_order(self):
        X = np.array(
            [
                [1.0, 1.0, 4.0, 8.0],
                [0.0, 2.0, 6.0, 10.0],
            ]
        )
        permutation = [2, 0, 3, 1]
        original = self.make_protein_adata(X)
        reordered = self.make_protein_adata(X[:, permutation])

        original_out, _ = normalize_quantile(original, inplace=False)
        reordered_out, _ = normalize_quantile(reordered, inplace=False)

        np.testing.assert_allclose(
            reordered_out.X,
            original_out.X[:, permutation],
        )

    def test_ties_with_nan_average_only_finite_rank_positions(self):
        adata = self.make_protein_adata(
            [
                [1.0, 1.0, np.nan, 8.0],
                [0.0, 2.0, 6.0, 10.0],
            ]
        )

        result, _ = normalize_quantile(adata, inplace=False)

        np.testing.assert_allclose(
            result.X[0],
            [1.25, 1.25, np.nan, 9.0],
            equal_nan=True,
        )

    # ── E. Grouped normalization ─────────────

    def test_group_by_builds_independent_reference_distributions(self):
        adata = self.make_protein_adata(
            [
                [1.0, 2.0, 3.0],
                [3.0, 4.0, 5.0],
                [100.0, 200.0, 300.0],
                [300.0, 400.0, 500.0],
            ],
            obs_extra={"batch": ["a", "a", "b", "b"]},
        )

        result, _ = normalize_quantile(
            adata,
            group_by="batch",
            inplace=False,
        )

        expected = np.array(
            [
                [2.0, 3.0, 4.0],
                [2.0, 3.0, 4.0],
                [200.0, 300.0, 400.0],
                [200.0, 300.0, 400.0],
            ]
        )
        np.testing.assert_allclose(result.X, expected)

    def test_grouped_reference_schema_contains_group_labels(self):
        adata = self.make_protein_adata(
            [
                [1.0, 2.0, 3.0],
                [3.0, 4.0, 5.0],
                [10.0, 20.0, 30.0],
            ],
            obs_extra={"batch": ["a", "a", "b"]},
        )

        _, reference = normalize_quantile(
            adata,
            group_by="batch",
            inplace=False,
        )

        assert list(reference.columns) == [
            "batch",
            "quantile_index",
            "quantile",
            "reference_value",
            "n_samples",
        ]
        np.testing.assert_array_equal(
            reference["batch"].to_numpy(),
            ["a", "a", "a", "b", "b", "b"],
        )
        np.testing.assert_array_equal(
            reference["n_samples"].to_numpy(),
            [2, 2, 2, 1, 1, 1],
        )

    def test_single_group_matches_ungrouped_normalization(self):
        X = np.array(
            [
                [1.0, 4.0, np.nan],
                [2.0, 6.0, 8.0],
            ]
        )
        ungrouped = self.make_protein_adata(X)
        grouped = self.make_protein_adata(
            X,
            obs_extra={"batch": ["a", "a"]},
        )

        ungrouped_out, _ = normalize_quantile(
            ungrouped,
            inplace=False,
        )
        grouped_out, _ = normalize_quantile(
            grouped,
            group_by="batch",
            inplace=False,
        )

        np.testing.assert_allclose(
            grouped_out.X,
            ungrouped_out.X,
            equal_nan=True,
        )

    def test_empty_categorical_groups_are_ignored(self):
        batches = pd.Categorical(
            ["a", "a"],
            categories=["a", "unused"],
        )
        adata = self.make_protein_adata(
            [[1.0, 2.0], [3.0, 4.0]],
            obs_extra={"batch": batches},
        )

        _, reference = normalize_quantile(
            adata,
            group_by="batch",
            inplace=False,
        )

        assert set(reference["batch"]) == {"a"}

    def test_all_nan_group_is_omitted_without_affecting_other_groups(self):
        adata = self.make_protein_adata(
            [
                [1.0, 2.0, 3.0],
                [3.0, 4.0, 5.0],
                [np.nan, np.nan, np.nan],
            ],
            obs_extra={"batch": ["a", "a", "b"]},
        )

        result, reference = normalize_quantile(
            adata,
            group_by="batch",
            inplace=False,
        )

        np.testing.assert_allclose(
            result.X[:2],
            [[2.0, 3.0, 4.0], [2.0, 3.0, 4.0]],
        )
        assert np.isnan(result.X[2]).all()
        assert set(reference["batch"]) == {"a"}

    # ── F. Numerical and structural edge cases ───────

    def test_zero_reference_value_is_valid(self):
        adata = self.make_protein_adata(
            [
                [-1.0, 0.0, 1.0],
                [-3.0, 0.0, 3.0],
            ]
        )

        result, reference = normalize_quantile(adata, inplace=False)

        np.testing.assert_allclose(
            reference["reference_value"].to_numpy(),
            [-2.0, 0.0, 2.0],
        )
        np.testing.assert_allclose(result.X[:, 1], [0.0, 0.0])

    def test_single_feature_uses_mean_reference_without_warning(self):
        adata = self.make_protein_adata([[-1.0], [0.0], [1.0]])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result, reference = normalize_quantile(
                adata,
                inplace=False,
            )

        np.testing.assert_allclose(result.X, [[0.0], [0.0], [0.0]])
        np.testing.assert_allclose(
            reference["reference_value"].to_numpy(),
            [0.0],
        )
        assert not [
            item
            for item in caught
            if issubclass(item.category, RuntimeWarning)
        ]

    def test_single_observed_value_maps_to_reference_midpoint(self):
        adata = self.make_protein_adata(
            [
                [0.0, 10.0, 40.0, 90.0],
                [np.nan, 15.0, np.nan, np.nan],
            ]
        )

        result, reference = normalize_quantile(adata, inplace=False)

        np.testing.assert_allclose(
            reference["reference_value"].to_numpy(),
            [7.5, 12.5, 27.5, 52.5],
        )
        np.testing.assert_allclose(
            result.X[1],
            [np.nan, 20.0, np.nan, np.nan],
            equal_nan=True,
        )

    def test_single_sample_preserves_its_observed_distribution(self):
        X = np.array([[1.0, np.nan, 4.0, 8.0]])
        adata = self.make_protein_adata(X)

        result, _ = normalize_quantile(adata, inplace=False)

        np.testing.assert_allclose(result.X, X, equal_nan=True)

    def test_anndata_view_is_normalized_without_changing_parent(self):
        parent = self.make_protein_adata(
            [
                [1.0, 2.0, 3.0],
                [3.0, 4.0, 5.0],
                [5.0, 6.0, 7.0],
            ]
        )
        original = parent.X.copy()
        view = parent[1:, :]

        result, _ = normalize_quantile(view, inplace=False)

        assert not result.is_view
        np.testing.assert_array_equal(parent.X, original)

    def test_special_character_identifiers_are_preserved(self):
        obs_names = ["sample/A", "sample B"]
        var_names = ["P:1", "P/2"]
        adata = self.make_protein_adata(
            [[1.0, 2.0], [3.0, 4.0]],
            obs_names=obs_names,
            var_names=var_names,
        )

        result, _ = normalize_quantile(adata, inplace=False)

        assert list(result.obs_names) == obs_names
        assert list(result.var_names) == var_names

    # ── G. Input validation ─────────────

    def test_non_anndata_input_raises_type_error(self):
        with pytest.raises(TypeError, match="must be an AnnData"):
            normalize_quantile(np.ones((2, 2)))

    def test_sparse_input_raises_with_densify_guidance(self):
        adata = self.make_protein_adata([[1.0, 2.0], [3.0, 4.0]])
        adata.X = sparse.csr_matrix(adata.X)

        with pytest.raises(
            TypeError,
            match=r"Sparse `\.X` is not supported.*Densify",
        ):
            normalize_quantile(adata)

    def test_invalid_proteodata_raises_before_normalization(self):
        adata = AnnData(
            X=np.array([[1.0, 2.0]]),
            var=pd.DataFrame(
                {"protein_id": ["p0", "p1"]},
                index=["p0", "p1"],
            ),
        )

        with pytest.raises(ValueError, match="sample_id"):
            normalize_quantile(adata)

    def test_duplicate_feature_identifiers_fail_proteodata_validation(self):
        adata = self.make_protein_adata(
            [[1.0, 2.0]],
            var_names=["p0", "p0"],
        )

        with pytest.raises(ValueError, match="unique"):
            normalize_quantile(adata)

    @pytest.mark.parametrize(
        ("kwargs", "error", "message"),
        [
            ({"fill_na": True}, TypeError, "fill_na.*numeric"),
            ({"fill_na": "zero"}, TypeError, "fill_na.*numeric"),
            ({"fill_na": np.nan}, ValueError, "fill_na.*finite"),
            ({"fill_na": np.inf}, ValueError, "fill_na.*finite"),
            ({"zero_to_na": 1}, TypeError, "zero_to_na.*bool"),
            ({"group_by": 1}, TypeError, "group_by.*string or None"),
            ({"key_added": 1}, TypeError, "key_added.*string"),
            ({"key_added": ""}, ValueError, "key_added.*non-empty"),
            ({"inplace": 1}, TypeError, "inplace.*bool"),
            ({"verbose": 1}, TypeError, "verbose.*bool"),
        ],
    )
    def test_invalid_argument_values_raise_matching_errors(
        self,
        kwargs,
        error,
        message,
    ):
        adata = self.make_protein_adata([[1.0, 2.0], [3.0, 4.0]])

        with pytest.raises(error, match=message):
            normalize_quantile(adata, **kwargs)

    def test_fill_na_and_zero_to_na_are_mutually_exclusive(self):
        adata = self.make_protein_adata([[np.nan, 2.0], [3.0, 4.0]])

        with pytest.raises(ValueError, match="mutually exclusive"):
            normalize_quantile(
                adata,
                fill_na=0.0,
                zero_to_na=True,
            )

    def test_ties_is_not_a_public_parameter(self):
        adata = self.make_protein_adata([[1.0, 1.0], [2.0, 3.0]])

        with pytest.raises(TypeError, match="unexpected keyword.*ties"):
            normalize_quantile(adata, ties="min")

    def test_missing_group_by_column_raises_key_error(self):
        adata = self.make_protein_adata([[1.0, 2.0], [3.0, 4.0]])

        with pytest.raises(KeyError, match="batch.*not found"):
            normalize_quantile(adata, group_by="batch")

    def test_nan_group_by_value_raises_value_error(self):
        adata = self.make_protein_adata(
            [[1.0, 2.0], [3.0, 4.0]],
            obs_extra={"batch": ["a", np.nan]},
        )

        with pytest.raises(ValueError, match="batch.*contains NaN"):
            normalize_quantile(adata, group_by="batch")

    @pytest.mark.parametrize(
        "X",
        [
            [[np.nan, np.nan], [np.nan, np.nan]],
            np.empty((0, 2)),
            np.empty((2, 0)),
        ],
    )
    def test_matrix_without_finite_values_raises_value_error(self, X):
        adata = self.make_protein_adata(X)

        with pytest.raises(ValueError, match="No finite values found"):
            normalize_quantile(adata)

    def test_all_zero_matrix_becomes_empty_with_zero_to_na(self):
        adata = self.make_protein_adata([[0.0, 0.0], [0.0, 0.0]])

        with pytest.raises(ValueError, match="No finite values found"):
            normalize_quantile(adata, zero_to_na=True)

    @pytest.mark.parametrize("bad_value", [np.inf, -np.inf])
    def test_infinite_input_fails_proteodata_validation(self, bad_value):
        adata = self.make_protein_adata([[1.0, bad_value], [3.0, 4.0]])

        with pytest.raises(ValueError, match="infinite values"):
            normalize_quantile(adata)

    def test_infinite_computed_result_is_rejected_before_assignment(self):
        largest = np.finfo(float).max
        adata = self.make_protein_adata(
            [[largest, largest], [largest, largest]]
        )
        original = adata.X.copy()

        with pytest.raises(ValueError, match="produced infinite values"):
            normalize_quantile(adata)

        np.testing.assert_array_equal(adata.X, original)
        assert "quantile_reference" not in adata.uns

    # ── H. Verbose reporting ────────────

    def test_verbose_reports_operation_storage_and_counts(self, capsys):
        adata = self.make_protein_adata(
            [
                [1.0, 2.0, 3.0],
                [3.0, 4.0, 5.0],
                [5.0, 6.0, 7.0],
            ]
        )

        normalize_quantile(adata, verbose=True)
        output = capsys.readouterr().out.lower()

        assert "quantile normalization" in output
        assert "adata.uns['quantile_reference']" in output
        assert "3 sample(s)" in output
        assert "1 group(s)" in output

    def test_verbose_reports_all_nan_samples(self, capsys):
        adata = self.make_protein_adata(
            [
                [np.nan, np.nan, np.nan],
                [1.0, 2.0, 3.0],
            ],
            obs_names=["missing", "observed"],
        )

        normalize_quantile(adata, verbose=True)
        output = capsys.readouterr().out.lower()

        assert "all-nan" in output
        assert "missing" in output

    def test_verbose_reports_groups_without_finite_values(self, capsys):
        adata = self.make_protein_adata(
            [
                [1.0, 2.0, 3.0],
                [np.nan, np.nan, np.nan],
            ],
            obs_extra={"batch": ["a", "empty"]},
        )

        normalize_quantile(
            adata,
            group_by="batch",
            verbose=True,
        )
        output = capsys.readouterr().out.lower()

        assert "empty" in output
        assert "no finite values" in output
