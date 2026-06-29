"""Tests for median normalization preprocessing."""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse

from proteopy.pp.normalize_median import normalize_median


class TestNormalizeMedian:
    """Contract tests for normalize_median."""

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
        """Build a minimal protein-level proteodata AnnData.

        Returns an AnnData with valid sample and protein identifiers.
        """
        if obs_names is None:
            obs_names = [f"s{i}" for i in range(np.asarray(X).shape[0])]
        if var_names is None:
            var_names = [f"p{i}" for i in range(np.asarray(X).shape[1])]

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
        """Run normalize_median with a consistent test return shape.

        Returns the normalized AnnData and factors DataFrame.
        """
        returned = normalize_median(
            adata,
            inplace=inplace,
            **kwargs,
        )
        key_added = kwargs.get("key_added", "normalization_factors")
        if inplace:
            assert returned is None
            return adata, adata.uns[key_added]

        adata_out, factors = returned
        return adata_out, factors

    # ── A. Log-space normalization ───────────────────────────────────

    @pytest.mark.parametrize("inplace", [True, False])
    def test_log_space_median_target_normalizes_samples(self, inplace):
        adata = self.make_protein_adata(
            [
                [18.0, 20.0, 25.0],
                [19.0, 21.0, 22.0],
                [16.0, 19.0, 28.0],
            ]
        )
        original = adata.X.copy()

        result, factors = self.normalize_and_collect(
            adata,
            inplace,
            log_space=True,
        )

        expected = np.array(
            [
                [18.0, 20.0, 25.0],
                [18.0, 20.0, 21.0],
                [17.0, 20.0, 29.0],
            ]
        )
        assert result.shape == adata.shape
        np.testing.assert_allclose(result.X, expected)
        assert list(factors.columns) == [
            "sample_index",
            "sample_id",
            "shift_log",
        ]
        np.testing.assert_array_equal(
            factors["sample_index"].to_numpy(),
            [0, 1, 2],
        )
        np.testing.assert_array_equal(
            factors["sample_id"].to_numpy(),
            ["s0", "s1", "s2"],
        )
        np.testing.assert_allclose(
            factors["shift_log"].to_numpy(),
            [0.0, -1.0, 1.0],
        )

        if inplace:
            np.testing.assert_allclose(adata.X, expected)
        else:
            assert result is not adata
            np.testing.assert_allclose(adata.X, original)

    @pytest.mark.parametrize("inplace", [True, False])
    def test_target_is_case_insensitive(self, inplace):
        adata = self.make_protein_adata(
            [
                [18.0, 20.0, 25.0],
                [19.0, 21.0, 22.0],
                [16.0, 19.0, 28.0],
            ]
        )

        result, factors = self.normalize_and_collect(
            adata,
            inplace,
            log_space=True,
            target="MAX",
        )

        expected = np.array(
            [
                [19.0, 21.0, 26.0],
                [19.0, 21.0, 22.0],
                [18.0, 21.0, 30.0],
            ]
        )
        np.testing.assert_allclose(result.X, expected)
        np.testing.assert_allclose(
            factors["shift_log"].to_numpy(),
            [1.0, 0.0, 2.0],
        )

    # ── B. Storage and factor schema ─────────────────────────────────

    @pytest.mark.parametrize("inplace", [True, False])
    def test_key_added_is_honoured(self, inplace):
        adata = self.make_protein_adata(
            [
                [18.0, 20.0, 22.0],
                [20.0, 22.0, 24.0],
            ]
        )

        result, factors = self.normalize_and_collect(
            adata,
            inplace,
            log_space=True,
            key_added="median_norm",
        )

        assert "median_norm" in result.uns
        assert "normalization_factors" not in result.uns
        assert factors is result.uns["median_norm"]

    @pytest.mark.parametrize("inplace", [True, False])
    def test_factors_stored_under_default_uns_key(self, inplace):
        adata = self.make_protein_adata(
            [
                [18.0, 20.0, 22.0],
                [20.0, 22.0, 24.0],
            ]
        )

        result, factors = self.normalize_and_collect(
            adata,
            inplace,
            log_space=True,
        )

        assert "normalization_factors" in result.uns
        assert factors is result.uns["normalization_factors"]

    # ── C. Linear-space normalization ────────────────────────────────

    @pytest.mark.parametrize("inplace", [True, False])
    def test_linear_space_scales_samples_to_common_median(
        self,
        inplace,
    ):
        adata = self.make_protein_adata(
            [
                [100.0, 200.0, 300.0],
                [200.0, 400.0, 600.0],
                [300.0, 600.0, 900.0],
            ]
        )

        result, factors = self.normalize_and_collect(
            adata,
            inplace,
            log_space=False,
        )

        expected = np.array(
            [
                [200.0, 400.0, 600.0],
                [200.0, 400.0, 600.0],
                [200.0, 400.0, 600.0],
            ]
        )
        np.testing.assert_allclose(result.X, expected)
        np.testing.assert_allclose(
            factors["scale_linear"].to_numpy(),
            [2.0, 1.0, 2.0 / 3.0],
        )

    def test_factors_dataframe_linear_space_column_name(self):
        adata = self.make_protein_adata(
            [
                [100.0, 200.0, 300.0],
                [200.0, 400.0, 600.0],
            ]
        )

        _, factors = normalize_median(
            adata,
            log_space=False,
            inplace=False,
        )

        assert list(factors.columns) == [
            "sample_index",
            "sample_id",
            "scale_linear",
        ]
        assert "shift_log" not in factors.columns

    # ── D. Grouped normalization ─────────────────────────────────────

    @pytest.mark.parametrize("inplace", [True, False])
    def test_group_by_normalizes_within_each_group(
        self,
        inplace,
    ):
        adata = self.make_protein_adata(
            [
                [18.0, 20.0, 22.0],
                [20.0, 22.0, 24.0],
                [30.0, 32.0, 34.0],
                [34.0, 36.0, 38.0],
            ],
            obs_extra={"batch": ["a", "a", "b", "b"]},
        )

        result, factors = self.normalize_and_collect(
            adata,
            inplace,
            log_space=True,
            group_by="batch",
        )

        expected = np.array(
            [
                [19.0, 21.0, 23.0],
                [19.0, 21.0, 23.0],
                [32.0, 34.0, 36.0],
                [32.0, 34.0, 36.0],
            ]
        )
        np.testing.assert_allclose(result.X, expected)
        np.testing.assert_array_equal(
            factors["batch"].to_numpy(),
            ["a", "a", "b", "b"],
        )
        np.testing.assert_allclose(
            factors["shift_log"].to_numpy(),
            [1.0, -1.0, 2.0, -2.0],
        )

    # ── E. Edge cases ────────────────────────────────────────────────

    @pytest.mark.parametrize("inplace", [True, False])
    def test_single_sample_has_no_log_shift(self, inplace):
        adata = self.make_protein_adata([[18.0, 20.0, 22.0]])

        result, factors = self.normalize_and_collect(
            adata,
            inplace,
            log_space=True,
        )

        np.testing.assert_allclose(result.X, [[18.0, 20.0, 22.0]])
        np.testing.assert_allclose(factors["shift_log"].to_numpy(), [0.0])

    @pytest.mark.parametrize("inplace", [True, False])
    def test_same_median_has_no_shift(self, inplace):
        adata = self.make_protein_adata(
            [
                [18.0, 20.0, 22.0],
                [10.0, 20.0, 30.0],
                [19.0, 20.0, 21.0],
            ]
        )
        original = adata.X.copy()

        result, factors = self.normalize_and_collect(
            adata,
            inplace,
            log_space=True,
        )

        np.testing.assert_allclose(result.X, original)
        np.testing.assert_allclose(
            factors["shift_log"].to_numpy(),
            [0.0, 0.0, 0.0],
        )

    @pytest.mark.parametrize("inplace", [True, False])
    def test_even_number_of_values_uses_interpolated_median(
        self,
        inplace,
    ):
        adata = self.make_protein_adata(
            [
                [18.0, 20.0, 22.0, 24.0],
                [22.0, 24.0, 26.0, 28.0],
            ]
        )

        result, factors = self.normalize_and_collect(
            adata,
            inplace,
            log_space=True,
        )

        expected = np.array(
            [
                [20.0, 22.0, 24.0, 26.0],
                [20.0, 22.0, 24.0, 26.0],
            ]
        )
        assert result.shape == adata.shape
        np.testing.assert_allclose(result.X, expected)
        np.testing.assert_allclose(
            factors["shift_log"].to_numpy(),
            [2.0, -2.0],
        )

    @pytest.mark.parametrize("inplace", [True, False])
    def test_zero_to_na_persists_and_ignores_zero_in_median(
        self,
        inplace,
    ):
        adata = self.make_protein_adata(
            [
                [0.0, 20.0, 22.0],
                [18.0, 20.0, 22.0],
            ]
        )

        result, factors = self.normalize_and_collect(
            adata,
            inplace,
            log_space=True,
            zero_to_na=True,
        )

        expected = np.array(
            [
                [np.nan, 19.5, 21.5],
                [18.5, 20.5, 22.5],
            ]
        )
        np.testing.assert_allclose(result.X, expected)
        np.testing.assert_allclose(
            factors["shift_log"].to_numpy(),
            [-0.5, 0.5],
        )

    @pytest.mark.parametrize("inplace", [True, False])
    def test_fill_na_persists_before_normalization(self, inplace):
        adata = self.make_protein_adata(
            [
                [np.nan, 20.0, 22.0],
                [18.0, 20.0, 22.0],
            ]
        )

        result, factors = self.normalize_and_collect(
            adata,
            inplace,
            log_space=True,
            fill_na=10.0,
        )

        expected = np.array(
            [
                [10.0, 20.0, 22.0],
                [18.0, 20.0, 22.0],
            ]
        )
        np.testing.assert_allclose(result.X, expected)
        np.testing.assert_allclose(
            factors["shift_log"].to_numpy(),
            [0.0, 0.0],
        )

    @pytest.mark.parametrize("inplace", [True, False])
    def test_all_nan_sample_keeps_nan_row_and_factor(self, inplace):
        adata = self.make_protein_adata(
            [
                [np.nan, np.nan, np.nan],
                [18.0, 20.0, 22.0],
                [20.0, 22.0, 24.0],
            ]
        )

        result, factors = self.normalize_and_collect(
            adata,
            inplace,
            log_space=True,
        )

        expected = np.array(
            [
                [np.nan, np.nan, np.nan],
                [19.0, 21.0, 23.0],
                [19.0, 21.0, 23.0],
            ]
        )
        np.testing.assert_allclose(result.X, expected)
        assert np.isnan(factors.loc[0, "shift_log"])
        np.testing.assert_allclose(
            factors.loc[1:, "shift_log"].to_numpy(),
            [1.0, -1.0],
        )

    def test_inplace_false_returns_independent_copy_and_factors(self):
        adata = self.make_protein_adata(
            [
                [18.0, 20.0, 22.0],
                [20.0, 22.0, 24.0],
            ]
        )

        adata_out, factors = normalize_median(
            adata,
            log_space=True,
            inplace=False,
            key_added="median_norm",
        )

        assert adata_out is not adata
        assert factors is adata_out.uns["median_norm"]
        assert "median_norm" not in adata.uns
        adata_out.X[0, 0] = -100.0
        assert adata.X[0, 0] == 18.0

    # ── F. Log-space detection ───────────────────────────────────────

    def test_logspace_detected_allows_log_space_without_force(self):
        adata = self.make_protein_adata(
            [
                [18.0, 20.0, 22.0],
                [20.0, 22.0, 24.0],
            ]
        )

        normalize_median(
            adata,
            log_space=True,
        )

        assert "normalization_factors" in adata.uns

    def test_raises_when_log_space_true_for_linear_like_data(self):
        adata = self.make_protein_adata(
            [
                [100.0, 200.0, 300.0],
                [200.0, 400.0, 600.0],
            ]
        )

        with pytest.raises(ValueError, match="do not look log-transformed"):
            normalize_median(
                adata,
                log_space=True,
            )

    # ── G. Verbose output ────────────────────────────────────────────

    def test_verbose_reports_all_nan_sample(self, capsys):
        adata = self.make_protein_adata(
            [
                [np.nan, np.nan, np.nan],
                [18.0, 20.0, 22.0],
            ]
        )

        normalize_median(
            adata,
            log_space=True,
            verbose=True,
        )

        captured = capsys.readouterr()
        assert "1 sample(s) had an all-NaN median" in captured.out
        assert "s0" in captured.out

    @pytest.mark.parametrize(
        ("log_space", "expected_space"),
        [
            (True, "log"),
            (False, "linear"),
        ],
    )
    def test_verbose_reports_space_and_storage(
        self,
        capsys,
        log_space,
        expected_space,
    ):
        adata = self.make_protein_adata(
            [
                [100.0, 200.0, 300.0],
                [200.0, 400.0, 600.0],
            ]
        )

        normalize_median(
            adata,
            log_space=log_space,
            key_added="median_norm",
            force=log_space,
            verbose=True,
        )

        captured = capsys.readouterr()
        assert f"Normalizing in {expected_space} space" in captured.out
        assert "adata.uns['median_norm']" in captured.out

    def test_verbose_reports_group_count(self, capsys):
        adata = self.make_protein_adata(
            [
                [18.0, 20.0, 22.0],
                [20.0, 22.0, 24.0],
                [30.0, 32.0, 34.0],
            ],
            obs_extra={"batch": ["a", "a", "b"]},
        )

        normalize_median(
            adata,
            log_space=True,
            group_by="batch",
            verbose=True,
        )

        captured = capsys.readouterr()
        assert "Normalizing in log space" in captured.out
        assert "Summary: normalized 3 sample(s) across 2 group(s)." in (
            captured.out
        )

    # ── H. Validation errors ─────────────────────────────────────────

    def test_raises_for_non_anndata_input(self):
        with pytest.raises(TypeError, match="AnnData"):
            normalize_median(
                np.array([[1.0, 2.0]]),
                log_space=True,
            )

    def test_raises_for_sparse_input(self):
        adata = self.make_protein_adata(
            [
                [18.0, 20.0],
                [20.0, 22.0],
            ]
        )
        adata.X = sparse.csr_matrix(adata.X)

        with pytest.raises(TypeError, match="Sparse `.X` is not supported"):
            normalize_median(adata)

    def test_raises_when_proteodata_validation_fails(self):
        adata = AnnData(
            X=np.array([[18.0, 20.0]]),
            obs=pd.DataFrame(index=["s0"]),
            var=pd.DataFrame(
                {"protein_id": ["p0", "p1"]},
                index=["p0", "p1"],
            ),
        )

        with pytest.raises(ValueError, match="sample_id"):
            normalize_median(adata)

    def test_raises_for_invalid_target(self):
        adata = self.make_protein_adata(
            [
                [18.0, 20.0],
                [20.0, 22.0],
            ]
        )

        with pytest.raises(ValueError, match="`target` must be one of"):
            normalize_median(
                adata,
                target="mean",
            )

    @pytest.mark.parametrize(
        ("kwargs", "exception_type", "message"),
        [
            ({"log_space": "yes"}, TypeError, "`log_space` must be a bool"),
            ({"target": 1}, TypeError, "`target` must be a string"),
            ({"fill_na": True}, TypeError, "`fill_na` must be a numeric"),
            ({"fill_na": "0"}, TypeError, "`fill_na` must be a numeric"),
            ({"fill_na": np.nan}, ValueError, "`fill_na` must be a finite"),
            ({"fill_na": np.inf}, ValueError, "`fill_na` must be a finite"),
            ({"zero_to_na": 1}, TypeError, "`zero_to_na` must be a bool"),
            ({"key_added": 1}, TypeError, "`key_added` must be a string"),
            ({"key_added": ""}, ValueError, "`key_added` must be a non-empty"),
            ({"group_by": 1}, TypeError, "`group_by` must be a string"),
            ({"inplace": 1}, TypeError, "`inplace` must be a bool"),
            ({"force": 1}, TypeError, "`force` must be a bool"),
            ({"verbose": 1}, TypeError, "`verbose` must be a bool"),
        ],
    )
    def test_raises_for_invalid_argument_types_and_values(
        self,
        kwargs,
        exception_type,
        message,
    ):
        adata = self.make_protein_adata(
            [
                [18.0, 20.0],
                [20.0, 22.0],
            ]
        )

        with pytest.raises(exception_type, match=message):
            normalize_median(
                adata,
                **kwargs,
            )

    def test_raises_when_fill_na_and_zero_to_na_are_both_set(self):
        adata = self.make_protein_adata(
            [
                [18.0, 20.0],
                [20.0, 22.0],
            ]
        )

        with pytest.raises(ValueError, match="mutually exclusive"):
            normalize_median(
                adata,
                fill_na=0.0,
                zero_to_na=True,
            )

    def test_raises_for_missing_group_by_column(self):
        adata = self.make_protein_adata(
            [
                [18.0, 20.0],
                [20.0, 22.0],
            ]
        )

        with pytest.raises(KeyError, match="not found in adata.obs"):
            normalize_median(
                adata,
                group_by="batch",
            )

    def test_raises_when_group_by_contains_nan(self):
        adata = self.make_protein_adata(
            [
                [18.0, 20.0],
                [20.0, 22.0],
            ],
            obs_extra={"batch": ["a", np.nan]},
        )

        with pytest.raises(ValueError, match="contains NaN"):
            normalize_median(
                adata,
                group_by="batch",
            )

    def test_raises_when_log_space_disagrees_with_detection(self):
        adata = self.make_protein_adata(
            [
                [18.0, 20.0],
                [20.0, 22.0],
            ]
        )

        with pytest.raises(ValueError, match="look log-transformed"):
            normalize_median(
                adata,
                log_space=False,
            )

    def test_force_allows_log_space_detection_mismatch(self):
        adata = self.make_protein_adata(
            [
                [18.0, 20.0],
                [20.0, 22.0],
            ]
        )

        normalize_median(
            adata,
            log_space=False,
            force=True,
        )

        np.testing.assert_allclose(
            adata.uns["normalization_factors"]["scale_linear"],
            [20.0 / 19.0, 20.0 / 21.0],
        )

    def test_raises_for_zero_median_in_linear_space(self):
        adata = self.make_protein_adata(
            [
                [0.0, 0.0, 0.0],
                [100.0, 200.0, 300.0],
            ]
        )

        with pytest.raises(ValueError, match="sample\\(s\\): s0"):
            normalize_median(
                adata,
                log_space=False,
            )

    def test_raises_when_no_finite_values_are_available(self):
        adata = self.make_protein_adata(
            [
                [np.nan, np.nan],
                [np.nan, np.nan],
            ]
        )

        with pytest.raises(ValueError, match="No finite values found"):
            normalize_median(
                adata,
                log_space=True,
            )
