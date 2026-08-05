"""Contract tests for feature-wise KNN imputation.

The suite is organized by contract area:

1. Public API, output layers, copy semantics, and metadata.
2. Completeness calculation, simultaneous gating, and warnings.
3. Groupwise incompleteness failures and transactional behavior.
4. Distance, neighbor-selection, and kernel correctness.
5. Group-restricted imputation and reporting.
6. Input validation, log-scale enforcement, and edge cases.

Small protein-level AnnData matrices make the expected neighbor and
imputed values explicit. Private implementation helpers are never
called directly. The contract suite is skipped until ``impute_knn`` is
implemented.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse

import proteopy as pr
from proteopy.pp import imputation


@pytest.mark.skip(reason="impute_knn is not implemented yet")
class TestImputeKnn:
    """Public-contract tests for ``impute_knn``."""

    # ------------------------------------------------------------------
    # Helper constructors
    # ------------------------------------------------------------------

    @staticmethod
    def make_protein_adata(
        X,
        obs_names=None,
        var_names=None,
        obs_extra=None,
        layers=None,
    ):
        """Build a minimal valid protein-level ProteoData object."""
        X = np.asarray(X, dtype=float)
        if obs_names is None:
            obs_names = [f"s{i}" for i in range(X.shape[0])]
        if var_names is None:
            var_names = [f"p{i}" for i in range(X.shape[1])]

        obs = pd.DataFrame({"sample_id": obs_names}, index=obs_names)
        if obs_extra is not None:
            for key, values in obs_extra.items():
                obs[key] = values
        var = pd.DataFrame({"protein_id": var_names}, index=var_names)
        adata = AnnData(X=X.copy(), obs=obs, var=var)
        if layers is not None:
            for key, values in layers.items():
                adata.layers[key] = values
        return adata

    @staticmethod
    def knn_function():
        """Return the future public function with a focused red failure."""
        function = getattr(imputation, "impute_knn", None)
        if function is None:
            pytest.fail("proteopy.pp.imputation.impute_knn is not implemented")
        return function

    @classmethod
    def impute_and_collect(cls, adata, inplace=False, **kwargs):
        """Run imputation and return the object containing its outputs."""
        returned = cls.knn_function()(
            adata,
            inplace=inplace,
            **kwargs,
        )
        if inplace:
            assert returned is None
            return adata
        assert returned is not adata
        return returned

    @classmethod
    def simple_adata(cls):
        """Build a small log-scale matrix with one predictable gap."""
        return cls.make_protein_adata(
            [
                [10.0, 10.0, 20.0],
                [11.0, 11.0, 20.0],
                [12.0, 12.0, 20.0],
                [np.nan, 13.0, 20.0],
            ],
            var_names=["target", "near", "far"],
        )

    # ── A. Public API and output contract ────────────────────────

    def test_public_api_exports_impute_knn(self):
        assert hasattr(imputation, "impute_knn")
        assert hasattr(pr.pp, "impute_knn")
        assert pr.pp.impute_knn is imputation.impute_knn

    @pytest.mark.parametrize("inplace", [True, False])
    def test_feature_wise_knn_imputes_hand_calculated_neighbor(
        self,
        inplace,
    ):
        adata = self.simple_adata()
        original = adata.X.copy()

        result = self.impute_and_collect(
            adata,
            inplace=inplace,
            k=1,
        )

        expected = original.copy()
        expected[3, 0] = 13.0
        assert result.layers["X_knn"].shape == adata.shape
        np.testing.assert_allclose(result.layers["X_knn"], expected)
        np.testing.assert_array_equal(adata.X, original)

    def test_mask_marks_exactly_input_nan_positions(self):
        adata = self.simple_adata()
        expected_mask = np.isnan(adata.X)

        result = self.impute_and_collect(adata, k=1)
        mask = result.layers["imputation_mask_knn"]

        assert mask.dtype == bool
        assert mask.shape == adata.shape
        np.testing.assert_array_equal(mask, expected_mask)

    def test_observed_values_are_preserved_exactly(self):
        adata = self.simple_adata()
        original = adata.X.copy()
        observed = ~np.isnan(original)

        result = self.impute_and_collect(adata, k=1)

        np.testing.assert_array_equal(
            result.layers["X_knn"][observed],
            original[observed],
        )

    def test_successful_output_is_dense_and_nan_free(self):
        result = self.impute_and_collect(self.simple_adata(), k=1)

        assert isinstance(result.layers["X_knn"], np.ndarray)
        assert not sparse.issparse(result.layers["X_knn"])
        assert np.isfinite(result.layers["X_knn"]).all()

    def test_inplace_false_preserves_original_layers_and_uns(self):
        adata = self.simple_adata()
        original = adata.X.copy()

        result = self.impute_and_collect(adata, k=1)

        assert result is not adata
        assert "X_knn" not in adata.layers
        assert "imputation_mask_knn" not in adata.layers
        assert "knn_impute" not in adata.uns
        assert np.array_equal(adata.X, original, equal_nan=True)

    def test_custom_input_and_output_layers_preserve_sources(self):
        X = np.full((4, 3), 99.0)
        logged = np.array(
            [
                [10.0, 10.0, 20.0],
                [11.0, 11.0, 20.0],
                [12.0, 12.0, 20.0],
                [np.nan, 13.0, 20.0],
            ]
        )
        adata = self.make_protein_adata(X, layers={"logged": logged})
        original_X = adata.X.copy()
        original_layer = adata.layers["logged"].copy()

        result = self.impute_and_collect(
            adata,
            layer="logged",
            out_layer="knn_result",
            mask_layer="knn_mask",
            k=1,
        )

        np.testing.assert_array_equal(result.X, original_X)
        assert np.array_equal(
            result.layers["logged"],
            original_layer,
            equal_nan=True,
        )
        assert result.layers["knn_result"][3, 0] == pytest.approx(13.0)
        assert result.layers["knn_mask"][3, 0]

    def test_metadata_records_resolved_parameters_and_reference(self):
        result = self.impute_and_collect(
            self.simple_adata(),
            k=1,
            metric="cosine",
            kernel="gaussian",
            min_var_completeness=0.4,
            min_sample_completeness=0.3,
            min_overlap=2,
            force=True,
        )

        params = result.uns["knn_impute"]["params"]
        expected = {
            "k": 1,
            "metric": "cosine",
            "kernel": "gaussian",
            "min_var_completeness": 0.4,
            "min_sample_completeness": 0.3,
            "min_overlap": 2,
            "force": True,
            "group_by": None,
            "layer": None,
            "out_layer": "X_knn",
            "mask_layer": "imputation_mask_knn",
            "reference": "Troyanskaya2001",
        }
        for key, value in expected.items():
            assert params[key] == value
        assert isinstance(params["proteopy_version"], str)

    def test_statistics_account_for_every_missing_value(self):
        result = self.impute_and_collect(self.simple_adata(), k=1)
        stats = result.uns["knn_impute"]["stats"]
        mask = result.layers["imputation_mask_knn"]

        assert stats["n_total"] == 12
        assert stats["n_missing"] == 1
        assert stats["frac_missing"] == pytest.approx(1 / 12)
        assert stats["n_imputed"] == 1
        assert stats["n_knn_imputed"] + stats["n_meanfallback"] == 1
        assert int(mask.sum()) == stats["n_imputed"]

    # ── B. Completeness masks and warnings ──────────────────

    def test_completeness_masks_are_computed_once_and_applied_together(
        self,
    ):
        adata = self.make_protein_adata(
            [
                [10.0, 10.0, np.nan, np.nan],
                [11.0, 11.0, np.nan, np.nan],
                [12.0, np.nan, 12.0, np.nan],
                [np.nan, 13.0, 13.0, 14.0],
            ],
            var_names=["p0", "p1", "p2", "p3"],
        )

        with pytest.warns(UserWarning):
            result = self.impute_and_collect(
                adata,
                k=1,
                min_sample_completeness=0.75,
                min_var_completeness=0.5,
            )

        stats = result.uns["knn_impute"]["stats"]
        assert stats["n_samples_below_completeness"] == 3
        assert stats["n_vars_below_completeness"] == 1

    def test_feature_completeness_uses_all_original_group_samples(self):
        adata = self.make_protein_adata(
            [
                [10.0, 10.0, np.nan],
                [11.0, 11.0, np.nan],
                [12.0, np.nan, 12.0],
                [np.nan, 13.0, 13.0],
            ],
        )

        with pytest.warns(UserWarning):
            result = self.impute_and_collect(
                adata,
                k=1,
                min_sample_completeness=0.8,
                min_var_completeness=0.75,
            )

        stats = result.uns["knn_impute"]["stats"]
        assert stats["n_samples_below_completeness"] == 2
        assert stats["n_vars_below_completeness"] == 1

    def test_values_exactly_at_completeness_threshold_are_retained(self):
        adata = self.make_protein_adata(
            [
                [10.0, 10.0, np.nan, np.nan],
                [11.0, 11.0, 11.0, 11.0],
                [12.0, 12.0, 12.0, 12.0],
                [13.0, 13.0, np.nan, np.nan],
            ]
        )

        result = self.impute_and_collect(
            adata,
            k=1,
            min_sample_completeness=0.5,
            min_var_completeness=0.5,
        )

        stats = result.uns["knn_impute"]["stats"]
        assert stats["n_samples_below_completeness"] == 0
        assert stats["n_vars_below_completeness"] == 0

    def test_zero_thresholds_disable_both_completeness_gates(self):
        adata = self.make_protein_adata(
            [
                [10.0, np.nan, np.nan],
                [11.0, 20.0, np.nan],
                [12.0, np.nan, 30.0],
            ]
        )

        result = self.impute_and_collect(
            adata,
            k=1,
            min_sample_completeness=0,
            min_var_completeness=0,
        )

        stats = result.uns["knn_impute"]["stats"]
        assert stats["n_samples_below_completeness"] == 0
        assert stats["n_vars_below_completeness"] == 0

    def test_low_completeness_warning_names_sample_and_fraction(self):
        adata = self.make_protein_adata(
            [
                [10.0, np.nan, np.nan, np.nan],
                [11.0, 11.0, 11.0, 11.0],
                [12.0, 12.0, 12.0, 12.0],
            ],
            obs_names=["sparse-run", "complete-a", "complete-b"],
        )

        with pytest.warns(
            UserWarning,
            match="sparse-run.*0.25|0.25.*sparse-run",
        ):
            result = self.impute_and_collect(
                adata,
                k=1,
                min_sample_completeness=0.5,
                min_var_completeness=0,
            )

        recorded = result.uns["knn_impute"]["low_completeness_samples"]
        assert recorded["sparse-run"] == pytest.approx(0.25)

    def test_sample_completeness_uses_original_group_detected_features(
        self,
    ):
        adata = self.make_protein_adata(
            [
                [10.0, np.nan, np.nan, 13.0],
                [11.0, 11.0, 11.0, 11.0],
                [20.0, 20.0, 20.0, 20.0],
                [21.0, 21.0, 21.0, 21.0],
            ],
            obs_names=["a-sparse", "a-complete", "b0", "b1"],
            obs_extra={"batch": ["A", "A", "B", "B"]},
        )

        with pytest.warns(UserWarning, match="a-sparse.*0.5|0.5.*a-sparse"):
            result = self.impute_and_collect(
                adata,
                group_by="batch",
                k=1,
                min_sample_completeness=0.75,
                min_var_completeness=0,
            )

        recorded = result.uns["knn_impute"]["low_completeness_samples"]
        assert recorded["a-sparse"] == pytest.approx(0.5)

    def test_global_missingness_above_twenty_percent_does_not_warn(self):
        adata = self.make_protein_adata(
            [
                [10.0, 10.0, np.nan, np.nan, 10.0],
                [11.0, np.nan, 11.0, 11.0, 11.0],
                [np.nan, 12.0, 12.0, 12.0, 12.0],
                [13.0, 13.0, 13.0, np.nan, 13.0],
                [14.0, 14.0, 14.0, 14.0, np.nan],
            ]
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.impute_and_collect(
                adata,
                k=1,
                min_sample_completeness=0,
                min_var_completeness=0,
            )

        messages = [str(item.message).lower() for item in caught]
        assert not any("20%" in message for message in messages)
        assert not any("global missingness" in message for message in messages)

    # ── C. Groupwise incompleteness errors ──────────────────

    @pytest.mark.parametrize(
        ("X", "names", "entity", "affected"),
        [
            (
                [[np.nan, np.nan], [10.0, 11.0]],
                (["missing-sample", "s1"], ["p0", "p1"]),
                "sample",
                "missing-sample",
            ),
            (
                [[np.nan, 10.0], [np.nan, 11.0]],
                (["s0", "s1"], ["missing-feature", "p1"]),
                "feature",
                "missing-feature",
            ),
        ],
    )
    def test_entirely_missing_global_axis_raises_groupwise_error(
        self,
        X,
        names,
        entity,
        affected,
    ):
        adata = self.make_protein_adata(
            X,
            obs_names=names[0],
            var_names=names[1],
        )

        with pytest.raises(
            ValueError,
            match=f"groupwise incompleteness.*global.*{entity}.*{affected}",
        ):
            self.knn_function()(adata)

        assert not adata.layers
        assert "knn_impute" not in adata.uns

    @pytest.mark.parametrize("missing_axis", ["sample", "feature"])
    def test_entirely_missing_axis_within_one_group_names_group(
        self,
        missing_axis,
    ):
        X = np.array(
            [
                [10.0, 20.0, 30.0],
                [11.0, 21.0, 31.0],
                [12.0, 22.0, 32.0],
                [13.0, 23.0, 33.0],
            ]
        )
        if missing_axis == "sample":
            X[2, :] = np.nan
            affected = "b0"
        else:
            X[2:, 1] = np.nan
            affected = "protein-b"
        adata = self.make_protein_adata(
            X,
            obs_names=["a0", "a1", "b0", "b1"],
            var_names=["protein-a", "protein-b", "protein-c"],
            obs_extra={"batch": ["A", "A", "B", "B"]},
        )

        with pytest.raises(
            ValueError,
            match=f"groupwise incompleteness.*B.*{missing_axis}.*{affected}",
        ):
            self.knn_function()(adata, group_by="batch")

        assert not adata.layers
        assert "knn_impute" not in adata.uns

    def test_missing_per_sample_mean_raises_groupwise_error(self):
        adata = self.make_protein_adata(
            [
                [10.0, np.nan, np.nan],
                [np.nan, 20.0, 30.0],
                [np.nan, 21.0, 31.0],
            ],
            obs_names=["isolated", "s1", "s2"],
            var_names=["excluded", "eligible-a", "eligible-b"],
        )

        with pytest.raises(
            ValueError,
            match=(
                "groupwise incompleteness.*global.*isolated.*"
                "eligible observations"
            ),
        ):
            self.knn_function()(
                adata,
                min_var_completeness=0.5,
                min_sample_completeness=0,
            )

        assert not adata.layers
        assert "knn_impute" not in adata.uns

    def test_no_eligible_features_for_required_imputation_raises(self):
        adata = self.make_protein_adata(
            [
                [10.0, np.nan, np.nan],
                [np.nan, 20.0, np.nan],
                [np.nan, np.nan, 30.0],
            ]
        )

        with pytest.raises(
            ValueError,
            match="groupwise incompleteness.*global.*eligible features",
        ):
            self.knn_function()(
                adata,
                min_var_completeness=0.5,
                min_sample_completeness=0,
            )

        assert not adata.layers
        assert "knn_impute" not in adata.uns

    def test_groupwise_failure_preserves_preexisting_outputs(self):
        adata = self.make_protein_adata(
            [[np.nan, np.nan], [10.0, 11.0]],
            obs_names=["missing", "observed"],
        )
        existing_layer = np.full(adata.shape, 7.0)
        existing_mask = np.zeros(adata.shape, dtype=bool)
        adata.layers["X_knn"] = existing_layer.copy()
        adata.layers["imputation_mask_knn"] = existing_mask.copy()
        adata.uns["knn_impute"] = {"existing": True}

        with pytest.raises(ValueError, match="groupwise incompleteness"):
            self.knn_function()(adata)

        np.testing.assert_array_equal(adata.layers["X_knn"], existing_layer)
        np.testing.assert_array_equal(
            adata.layers["imputation_mask_knn"],
            existing_mask,
        )
        assert adata.uns["knn_impute"] == {"existing": True}

    # ── D. Distance and neighbor correctness ────────────────

    def test_euclidean_uses_mean_squared_overlap_not_raw_sum(self):
        adata = self.make_protein_adata(
            [
                [10.0, 11.0, 10.0 + np.sqrt(2.0)],
                [10.0, 11.0, np.nan],
                [10.0, 11.0, np.nan],
                [np.nan, 100.0, 20.0],
            ],
            var_names=["target", "three-overlap", "one-overlap"],
        )

        result = self.impute_and_collect(
            adata,
            k=1,
            metric="euclidean",
            min_var_completeness=0,
            min_sample_completeness=0,
        )

        # Mean squared distances are 1 and 2, so the first donor wins.
        # Raw sums would be 3 and 2 and would incorrectly choose 20.
        assert result.layers["X_knn"][3, 0] == pytest.approx(100.0)

    def test_pearson_uses_pairwise_overlap_and_selects_best_donor(self):
        adata = self.make_protein_adata(
            [
                [10.0, 20.0, 10.0],
                [11.0, 22.0, 12.0],
                [12.0, 24.0, 11.0],
                [np.nan, 30.0, 40.0],
            ],
            var_names=["target", "correlated", "other"],
        )

        result = self.impute_and_collect(
            adata,
            k=1,
            metric="pearson",
            min_var_completeness=0,
            min_sample_completeness=0,
        )

        assert result.layers["X_knn"][3, 0] == pytest.approx(30.0)

    def test_cosine_uses_pairwise_overlap_and_selects_best_donor(self):
        adata = self.make_protein_adata(
            [
                [10.0, 20.0, 30.0],
                [20.0, 40.0, 20.0],
                [30.0, 60.0, 10.0],
                [np.nan, 70.0, 80.0],
            ],
            var_names=["target", "same-direction", "other"],
        )

        result = self.impute_and_collect(
            adata,
            k=1,
            metric="cosine",
            min_var_completeness=0,
            min_sample_completeness=0,
        )

        assert result.layers["X_knn"][3, 0] == pytest.approx(70.0)

    def test_zero_overlap_donor_is_never_selected(self):
        adata = self.make_protein_adata(
            [
                [10.0, np.nan, 10.0],
                [11.0, np.nan, 11.0],
                [np.nan, 30.0, 12.0],
                [np.nan, 99.0, 13.0],
            ],
            var_names=["target", "zero-overlap", "valid"],
        )

        result = self.impute_and_collect(
            adata,
            k=1,
            min_var_completeness=0,
            min_sample_completeness=0,
        )

        assert result.layers["X_knn"][2, 0] == pytest.approx(12.0)
        assert result.layers["X_knn"][3, 0] == pytest.approx(13.0)

    def test_min_overlap_disqualifies_low_overlap_donor(self):
        adata = self.make_protein_adata(
            [
                [10.0, 10.1, 12.0],
                [10.0, np.nan, 12.0],
                [12.0, np.nan, 12.0],
                [np.nan, 99.0, 20.0],
            ],
            var_names=["target", "one-overlap", "three-overlap"],
        )

        result = self.impute_and_collect(
            adata,
            k=1,
            min_overlap=2,
            min_var_completeness=0,
            min_sample_completeness=0,
        )

        assert result.layers["X_knn"][3, 0] == pytest.approx(20.0)

    def test_equal_distance_ties_follow_original_variable_order(self):
        adata = self.make_protein_adata(
            [
                [10.0, 9.0, 11.0],
                [11.0, 10.0, 12.0],
                [np.nan, 20.0, 30.0],
            ],
            var_names=["target", "first", "second"],
        )

        result = self.impute_and_collect(
            adata,
            k=1,
            min_var_completeness=0,
            min_sample_completeness=0,
        )

        assert result.layers["X_knn"][2, 0] == pytest.approx(20.0)

    def test_self_is_excluded_from_neighbor_candidates(self):
        adata = self.make_protein_adata(
            [
                [10.0, 20.0],
                [11.0, 21.0],
                [np.nan, 22.0],
            ],
            var_names=["target", "donor"],
        )

        result = self.impute_and_collect(
            adata,
            k=1,
            min_var_completeness=0,
            min_sample_completeness=0,
        )

        assert result.layers["X_knn"][2, 0] == pytest.approx(22.0)

    def test_fewer_than_k_donors_updates_diagnostic(self):
        result = self.impute_and_collect(
            self.simple_adata(),
            k=10,
            min_var_completeness=0,
            min_sample_completeness=0,
        )

        stats = result.uns["knn_impute"]["stats"]
        assert stats["n_targets_below_k_neighbors"] >= 1

    # ── E. Identical profiles and undefined metrics ─────────────

    @pytest.mark.parametrize("metric", ["euclidean", "cosine", "pearson"])
    def test_identical_nonconstant_profiles_have_zero_distance(self, metric):
        adata = self.make_protein_adata(
            [
                [10.0, 10.0, 20.0],
                [11.0, 11.0, 20.0],
                [12.0, 12.0, 20.0],
                [np.nan, 25.0, 30.0],
            ],
            var_names=["target", "identical", "other"],
        )

        result = self.impute_and_collect(
            adata,
            k=1,
            metric=metric,
            min_var_completeness=0,
            min_sample_completeness=0,
        )

        assert result.layers["X_knn"][3, 0] == pytest.approx(25.0)

    def test_zero_norm_cosine_donor_is_ineligible(self):
        adata = self.make_protein_adata(
            [
                [10.0, 0.0, 10.0],
                [11.0, 0.0, 11.0],
                [12.0, 0.0, 12.0],
                [np.nan, 100.0, 20.0],
            ],
            var_names=["target", "zero-norm", "valid"],
        )

        result = self.impute_and_collect(
            adata,
            k=1,
            metric="cosine",
            min_var_completeness=0,
            min_sample_completeness=0,
        )

        assert result.layers["X_knn"][3, 0] == pytest.approx(20.0)

    def test_zero_variance_pearson_donor_is_ineligible(self):
        adata = self.make_protein_adata(
            [
                [10.0, 5.0, 20.0],
                [11.0, 5.0, 22.0],
                [12.0, 5.0, 24.0],
                [np.nan, 100.0, 30.0],
            ],
            var_names=["target", "constant", "valid"],
        )

        result = self.impute_and_collect(
            adata,
            k=1,
            metric="pearson",
            min_var_completeness=0,
            min_sample_completeness=0,
        )

        assert result.layers["X_knn"][3, 0] == pytest.approx(30.0)

    def test_zero_variance_pearson_target_uses_sample_mean(self):
        adata = self.make_protein_adata(
            [
                [10.0, 20.0, 30.0],
                [10.0, 21.0, 31.0],
                [10.0, 22.0, 32.0],
                [np.nan, 24.0, 34.0],
            ],
            var_names=["constant-target", "donor-a", "donor-b"],
        )

        result = self.impute_and_collect(
            adata,
            k=2,
            metric="pearson",
            min_var_completeness=0,
            min_sample_completeness=0,
        )

        assert result.layers["X_knn"][3, 0] == pytest.approx(29.0)
        stats = result.uns["knn_impute"]["stats"]
        assert stats["n_meanfallback"] == 1

    # ── F. Kernel calculations ─────────────────────────

    def test_uniform_kernel_is_arithmetic_mean(self):
        adata = self.make_protein_adata(
            [
                [10.0, 10.0, 12.0],
                [11.0, 12.0, 11.0],
                [np.nan, 20.0, 30.0],
            ]
        )

        result = self.impute_and_collect(
            adata,
            k=2,
            kernel="uniform",
            min_var_completeness=0,
            min_sample_completeness=0,
        )

        assert result.layers["X_knn"][2, 0] == pytest.approx(25.0)

    def test_distance_kernel_matches_inverse_distance_reference(self):
        adata = self.make_protein_adata(
            [
                [10.0, 10.0, 12.0],
                [11.0, 12.0, 11.0],
                [np.nan, 20.0, 30.0],
            ]
        )
        distance_a = 0.5
        distance_b = 2.0
        epsilon = np.finfo(float).eps
        expected = (
            20.0 / (distance_a + epsilon) + 30.0 / (distance_b + epsilon)
        ) / (1.0 / (distance_a + epsilon) + 1.0 / (distance_b + epsilon))

        result = self.impute_and_collect(
            adata,
            k=2,
            kernel="distance",
            min_var_completeness=0,
            min_sample_completeness=0,
        )

        assert result.layers["X_knn"][2, 0] == pytest.approx(expected)

    @pytest.mark.parametrize("kernel", ["distance", "gaussian"])
    def test_multiple_zero_distance_neighbors_receive_equal_weight(
        self,
        kernel,
    ):
        adata = self.make_protein_adata(
            [
                [10.0, 10.0, 10.0],
                [11.0, 11.0, 11.0],
                [np.nan, 20.0, 30.0],
            ],
            var_names=["target", "exact-a", "exact-b"],
        )

        result = self.impute_and_collect(
            adata,
            k=2,
            kernel=kernel,
            min_var_completeness=0,
            min_sample_completeness=0,
        )

        assert result.layers["X_knn"][2, 0] == pytest.approx(25.0)

    def test_gaussian_kernel_matches_target_median_bandwidth(self):
        adata = self.make_protein_adata(
            [
                [10.0, 10.0, 12.0],
                [11.0, 12.0, 11.0],
                [np.nan, 20.0, 30.0],
            ]
        )
        distances = np.array([0.5, 2.0])
        sigma = float(np.median(distances))
        log_weights = -(distances**2) / (2.0 * sigma**2)
        log_weights -= log_weights.max()
        weights = np.exp(log_weights)
        expected = np.average([20.0, 30.0], weights=weights)

        result = self.impute_and_collect(
            adata,
            k=2,
            kernel="gaussian",
            min_var_completeness=0,
            min_sample_completeness=0,
        )

        assert result.layers["X_knn"][2, 0] == pytest.approx(expected)

    def test_gaussian_shifted_exponents_avoid_zero_weight_sum(self):
        adata = self.make_protein_adata(
            [
                [10.0, 10.0, 10.0, 10.0, 1010.0],
                [11.0, 11.0, 11.0, 11.0, 1011.0],
                [np.nan, np.nan, np.nan, np.nan, 50.0],
            ],
            var_names=["target", "zero-a", "zero-b", "zero-c", "far"],
        )

        result = self.impute_and_collect(
            adata,
            k=4,
            kernel="gaussian",
            min_var_completeness=0,
            min_sample_completeness=0,
        )

        assert result.layers["X_knn"][2, 0] == pytest.approx(50.0)

    def test_neighbors_missing_at_target_sample_do_not_contribute(self):
        adata = self.make_protein_adata(
            [
                [10.0, 10.0, 12.0],
                [11.0, 11.0, 13.0],
                [np.nan, np.nan, 25.0],
            ]
        )

        result = self.impute_and_collect(
            adata,
            k=2,
            kernel="uniform",
            min_var_completeness=0,
            min_sample_completeness=0,
        )

        assert result.layers["X_knn"][2, 0] == pytest.approx(25.0)

    # ── G. Group-restricted behavior ────────────────────

    @classmethod
    def grouped_neighbor_adata(cls):
        """Build two batches with different nearest feature profiles."""
        return cls.make_protein_adata(
            [
                [10.0, 10.0, 12.0],
                [11.0, 11.0, 11.0],
                [12.0, 12.0, 10.0],
                [np.nan, 20.0, 30.0],
                [10.0, 12.0, 10.0],
                [11.0, 11.0, 11.0],
                [12.0, 10.0, 12.0],
                [np.nan, 30.0, 40.0],
            ],
            obs_names=["a0", "a1", "a2", "a3", "b0", "b1", "b2", "b3"],
            var_names=["target", "donor-a", "donor-b"],
            obs_extra={"batch": ["A"] * 4 + ["B"] * 4},
        )

    def test_group_by_selects_neighbors_independently(self):
        result = self.impute_and_collect(
            self.grouped_neighbor_adata(),
            group_by="batch",
            k=1,
            min_var_completeness=0,
            min_sample_completeness=0,
        )

        assert result.layers["X_knn"][3, 0] == pytest.approx(20.0)
        assert result.layers["X_knn"][7, 0] == pytest.approx(40.0)

    def test_changing_other_group_does_not_change_group_imputation(self):
        first = self.grouped_neighbor_adata()
        second = self.grouped_neighbor_adata()
        second.X[4:, 1:] += 500.0

        result_a = self.impute_and_collect(
            first,
            group_by="batch",
            k=1,
            min_var_completeness=0,
            min_sample_completeness=0,
        )
        result_b = self.impute_and_collect(
            second,
            group_by="batch",
            k=1,
            min_var_completeness=0,
            min_sample_completeness=0,
        )

        np.testing.assert_allclose(
            result_a.layers["X_knn"][:4],
            result_b.layers["X_knn"][:4],
        )

    def test_group_reassembly_preserves_original_observation_order(self):
        adata = self.grouped_neighbor_adata()
        order = [4, 0, 5, 1, 6, 2, 7, 3]
        shuffled = adata[order].copy()

        result = self.impute_and_collect(
            shuffled,
            group_by="batch",
            k=1,
            min_var_completeness=0,
            min_sample_completeness=0,
        )

        assert list(result.obs_names) == list(shuffled.obs_names)
        by_name = dict(
            zip(
                result.obs_names,
                result.layers["X_knn"][:, 0],
            )
        )
        assert by_name["a3"] == pytest.approx(20.0)
        assert by_name["b3"] == pytest.approx(40.0)

    def test_per_group_stats_sum_to_global_stats(self):
        result = self.impute_and_collect(
            self.grouped_neighbor_adata(),
            group_by="batch",
            k=1,
            min_var_completeness=0,
            min_sample_completeness=0,
        )

        stats = result.uns["knn_impute"]["stats"]
        per_group = stats["per_group"]
        assert set(per_group) == {"A", "B"}
        for key in (
            "n_total",
            "n_missing",
            "n_imputed",
            "n_knn_imputed",
            "n_meanfallback",
        ):
            assert stats[key] == sum(
                group[key] for group in per_group.values()
            )

    def test_verbose_prints_global_and_per_group_counts(self, capsys):
        self.impute_and_collect(
            self.grouped_neighbor_adata(),
            group_by="batch",
            k=1,
            min_var_completeness=0,
            min_sample_completeness=0,
            verbose=True,
        )

        output = capsys.readouterr().out
        assert "[impute_knn]" in output
        assert "[impute_knn:A]" in output
        assert "[impute_knn:B]" in output
        assert "KNN" in output
        assert "fallback" in output

    # ── H. Log checks, validation, and edge cases ──────────────

    def test_non_log_input_raises_with_user_guidance(self):
        adata = self.make_protein_adata(
            [
                [1_000.0, 2_000.0, 3_000.0],
                [10_000.0, 20_000.0, 30_000.0],
                [100_000.0, 200_000.0, np.nan],
            ]
        )

        with pytest.raises(
            ValueError,
            match="log-transform.*force=True|force=True.*log-transform",
        ):
            self.knn_function()(adata, force=False)

    def test_force_true_skips_log_detection(self):
        adata = self.make_protein_adata(
            [
                [1_000.0, 1_000.0, 10_000.0],
                [2_000.0, 2_000.0, 20_000.0],
                [np.nan, 3_000.0, 30_000.0],
            ]
        )

        result = self.impute_and_collect(adata, force=True, k=1)

        assert result.layers["X_knn"][2, 0] == pytest.approx(3_000.0)

    def test_force_does_not_change_logged_calculation(self):
        normal = self.impute_and_collect(self.simple_adata(), k=1)
        forced = self.impute_and_collect(
            self.simple_adata(),
            k=1,
            force=True,
        )

        np.testing.assert_allclose(
            normal.layers["X_knn"],
            forced.layers["X_knn"],
        )

    @pytest.mark.parametrize("source", ["X", "layer"])
    def test_sparse_selected_matrix_raises_with_densify_guidance(
        self,
        source,
    ):
        adata = self.simple_adata()
        dense = np.nan_to_num(adata.X, nan=0.0)
        if source == "X":
            adata.X = sparse.csr_matrix(dense)
            kwargs = {}
        else:
            adata.layers["sparse"] = sparse.csr_matrix(dense)
            kwargs = {"layer": "sparse"}

        with pytest.raises(
            TypeError,
            match="Sparse.*not supported.*[Dd]ensify",
        ):
            self.knn_function()(adata, **kwargs)

        assert "X_knn" not in adata.layers
        assert "knn_impute" not in adata.uns

    @pytest.mark.parametrize(
        ("kwargs", "exception", "message"),
        [
            ({"k": True}, TypeError, "`k` must be an int"),
            ({"k": 1.5}, TypeError, "`k` must be an int"),
            ({"k": 0}, ValueError, "`k` must be positive"),
            ({"metric": "manhattan"}, ValueError, "`metric`"),
            ({"metric": 1}, TypeError, "`metric` must be a string"),
            ({"kernel": "triangular"}, ValueError, "`kernel`"),
            ({"kernel": 1}, TypeError, "`kernel` must be a string"),
            (
                {"min_var_completeness": True},
                TypeError,
                "`min_var_completeness` must be numeric",
            ),
            (
                {"min_var_completeness": -0.1},
                ValueError,
                r"min_var_completeness.*\[0, 1\]",
            ),
            (
                {"min_sample_completeness": "0.5"},
                TypeError,
                "`min_sample_completeness` must be numeric",
            ),
            (
                {"min_sample_completeness": 1.1},
                ValueError,
                r"min_sample_completeness.*\[0, 1\]",
            ),
            ({"min_overlap": True}, TypeError, "`min_overlap` must be an int"),
            ({"min_overlap": 1.5}, TypeError, "`min_overlap` must be an int"),
            ({"min_overlap": 0}, ValueError, "`min_overlap` must be positive"),
            ({"force": 1}, TypeError, "`force` must be a bool"),
            ({"group_by": 1}, TypeError, "`group_by` must be a string"),
            ({"inplace": 1}, TypeError, "`inplace` must be a bool"),
            ({"verbose": 1}, TypeError, "`verbose` must be a bool"),
            ({"layer": 1}, TypeError, "`layer` must be a string"),
            ({"out_layer": 1}, TypeError, "`out_layer` must be a string"),
            ({"out_layer": ""}, ValueError, "`out_layer`.*non-empty"),
            ({"mask_layer": 1}, TypeError, "`mask_layer` must be a string"),
            ({"mask_layer": ""}, ValueError, "`mask_layer`.*non-empty"),
        ],
    )
    def test_invalid_arguments_raise_without_mutation(
        self,
        kwargs,
        exception,
        message,
    ):
        adata = self.simple_adata()

        with pytest.raises(exception, match=message):
            self.knn_function()(adata, **kwargs)

        assert "X_knn" not in adata.layers
        assert "knn_impute" not in adata.uns

    def test_non_anndata_input_raises_type_error(self):
        with pytest.raises(TypeError, match="AnnData"):
            self.knn_function()(np.array([[10.0, np.nan]]))

    def test_missing_input_layer_raises_key_error(self):
        adata = self.simple_adata()

        with pytest.raises(KeyError, match="missing.*adata.layers"):
            self.knn_function()(adata, layer="missing")

    def test_missing_group_column_raises_key_error(self):
        adata = self.simple_adata()

        with pytest.raises(KeyError, match="batch.*adata.obs"):
            self.knn_function()(adata, group_by="batch")

    def test_nan_group_label_raises_value_error(self):
        adata = self.make_protein_adata(
            [[10.0, 11.0], [12.0, 13.0]],
            obs_extra={"batch": ["A", np.nan]},
        )

        with pytest.raises(ValueError, match="group_by.*contains NaN"):
            self.knn_function()(adata, group_by="batch")

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            (
                {"out_layer": "same", "mask_layer": "same"},
                "out_layer.*mask_layer.*different",
            ),
            (
                {"layer": "logged", "out_layer": "logged"},
                "out_layer.*input layer",
            ),
        ],
    )
    def test_conflicting_layer_keys_raise(self, kwargs, message):
        adata = self.simple_adata()
        if kwargs.get("layer") == "logged":
            adata.layers["logged"] = adata.X.copy()

        with pytest.raises(ValueError, match=message):
            self.knn_function()(adata, **kwargs)

    def test_proteodata_validation_failure_propagates(self):
        adata = AnnData(
            X=np.array([[10.0, 11.0]]),
            obs=pd.DataFrame(index=["s0"]),
            var=pd.DataFrame(
                {"protein_id": ["p0", "p1"]},
                index=["p0", "p1"],
            ),
        )

        with pytest.raises(ValueError, match="sample_id"):
            self.knn_function()(adata)

    def test_infinite_selected_layer_fails_proteodata_validation(self):
        adata = self.simple_adata()
        invalid = adata.X.copy()
        invalid[0, 0] = np.inf
        adata.layers["invalid"] = invalid

        with pytest.raises(ValueError, match="contains infinite values"):
            self.knn_function()(adata, layer="invalid")

        assert "X_knn" not in adata.layers
        assert "knn_impute" not in adata.uns

    def test_no_missing_values_still_writes_empty_mask_and_stats(self):
        adata = self.make_protein_adata([[10.0, 11.0], [12.0, 13.0]])

        result = self.impute_and_collect(adata)

        np.testing.assert_array_equal(result.layers["X_knn"], adata.X)
        assert not result.layers["imputation_mask_knn"].any()
        assert result.uns["knn_impute"]["stats"]["n_imputed"] == 0

    @pytest.mark.parametrize("shape", [(1, 3), (3, 1)])
    def test_complete_single_axis_input_is_supported(self, shape):
        X = np.arange(np.prod(shape), dtype=float).reshape(shape) + 10.0
        result = self.impute_and_collect(self.make_protein_adata(X))

        np.testing.assert_array_equal(result.layers["X_knn"], X)

    @pytest.mark.parametrize("shape", [(0, 2), (2, 0)])
    def test_empty_axis_raises_clear_value_error(self, shape):
        adata = self.make_protein_adata(np.empty(shape))

        with pytest.raises(ValueError, match="empty|zero|groupwise"):
            self.knn_function()(adata)

    @pytest.mark.parametrize("inplace", [True, False])
    def test_anndata_view_is_supported(self, inplace):
        parent = self.make_protein_adata(
            [
                [10.0, 10.0, 20.0],
                [11.0, 11.0, 20.0],
                [12.0, 12.0, 20.0],
                [np.nan, 13.0, 20.0],
                [14.0, 14.0, 20.0],
            ]
        )
        view = parent[:4, :]
        parent_before = parent.X.copy()

        result = self.impute_and_collect(view, inplace=inplace, k=1)

        assert result.layers["X_knn"][3, 0] == pytest.approx(13.0)
        assert np.array_equal(parent.X, parent_before, equal_nan=True)

    def test_special_names_are_preserved(self):
        adata = self.make_protein_adata(
            [
                [10.0, 10.0],
                [11.0, 11.0],
                [np.nan, 12.0],
            ],
            obs_names=["sample/0", "sample 1", "sample:β"],
            var_names=["P|target", "P donor"],
        )

        result = self.impute_and_collect(adata, k=1)

        assert list(result.obs_names) == ["sample/0", "sample 1", "sample:β"]
        assert list(result.var_names) == ["P|target", "P donor"]

    def test_duplicate_names_fail_proteodata_validation(self):
        adata = self.simple_adata()
        adata.obs_names = ["dup", "dup", "s2", "s3"]
        adata.obs["sample_id"] = adata.obs_names

        with pytest.raises(ValueError, match="Duplicate names"):
            self.knn_function()(adata)

    def test_repeated_imputation_can_use_first_output_as_input(self):
        first = self.impute_and_collect(self.simple_adata(), k=1)

        second = self.impute_and_collect(
            first,
            layer="X_knn",
            out_layer="X_knn_second",
            mask_layer="imputation_mask_knn_second",
            k=1,
        )

        np.testing.assert_allclose(
            second.layers["X_knn_second"],
            first.layers["X_knn"],
        )
        assert not second.layers["imputation_mask_knn_second"].any()
