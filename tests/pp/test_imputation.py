import pytest
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse
from scipy.stats import ks_2samp, kstest, norm, chi2

from proteopy.pp.imputation import (
    impute_downshift,
    _impute_rows,
    _impute_by_group,
)


# ── Fixture builders ────────────────────────────────────────────────


def _make_log_adata_with_missing(
    n_obs: int = 80,
    n_vars: int = 200,
    miss_frac: float = 0.25,
    seed: int = 0,
) -> AnnData:
    """Log2 intensities mimicking real proteomics data.

    Raw intensities are drawn from a lognormal (the empirical shape of
    MS1 quantitative intensities) then log2-transformed, yielding a
    Gaussian log-intensity distribution with mean ≈ 23 and sd ≈ 2.5.
    NaN missingness is injected MCAR.
    Sized for KS/LRT statistical power.
    """
    rng = np.random.default_rng(seed)
    # ln-space params chosen so log2(raw) ~ N(23, 2.5)
    mu_ln = 23.0 * np.log(2)
    sigma_ln = 2.5 * np.log(2)
    raw = rng.lognormal(mean=mu_ln, sigma=sigma_ln, size=(n_obs, n_vars))
    X = np.log2(raw)
    # inject MCAR missingness
    miss = rng.random(size=X.shape) < miss_frac
    X[miss] = np.nan

    obs_names = [f"s{i}" for i in range(n_obs)]
    var_names = [f"p{i}" for i in range(n_vars)]
    obs = pd.DataFrame({"sample_id": obs_names}, index=obs_names)
    var = pd.DataFrame({"protein_id": var_names}, index=var_names)
    return AnnData(X=X, obs=obs, var=var)


def _make_small_log_adata() -> AnnData:
    """Tiny 4×3 log-scale matrix with hand-picked NaN positions."""
    n = np.nan
    X = np.array(
        [
            [10.0, 12.0, n],
            [11.0, n, 14.0],
            [n, 13.0, 15.0],
            [12.0, 14.0, n],
        ],
        dtype=float,
    )
    obs_names = ["s0", "s1", "s2", "s3"]
    var_names = ["p0", "p1", "p2"]
    obs = pd.DataFrame({"sample_id": obs_names}, index=obs_names)
    var = pd.DataFrame({"protein_id": var_names}, index=var_names)
    return AnnData(X=X, obs=obs, var=var)


def _make_grouped_log_adata(
    n_per_group: int = 20,
    n_vars: int = 50,
    miss_frac: float = 0.25,
    seed: int = 1,
) -> AnnData:
    """Two-group log2 proteomics intensities with distinct medians.

    Both groups draw raw intensities from a lognormal then log2-transform,
    matching the Gaussian shape of real proteomics log-intensities. The
    medians differ by ~6 log2 units.
    """
    rng = np.random.default_rng(seed)
    sigma_ln = 1.5 * np.log(2)  # log2-sd ≈ 1.5
    raw_a = rng.lognormal(
        mean=18.0 * np.log(2), sigma=sigma_ln,
        size=(n_per_group, n_vars),
    )
    raw_b = rng.lognormal(
        mean=24.0 * np.log(2), sigma=sigma_ln,
        size=(n_per_group, n_vars),
    )
    X = np.vstack([np.log2(raw_a), np.log2(raw_b)])
    miss = rng.random(size=X.shape) < miss_frac
    X[miss] = np.nan

    n_obs = 2 * n_per_group
    obs_names = [f"s{i}" for i in range(n_obs)]
    groups = ["A"] * n_per_group + ["B"] * n_per_group
    var_names = [f"p{i}" for i in range(n_vars)]
    obs = pd.DataFrame(
        {"sample_id": obs_names, "group": groups},
        index=obs_names,
    )
    var = pd.DataFrame({"protein_id": var_names}, index=var_names)
    return AnnData(X=X, obs=obs, var=var)


def _make_non_log_adata() -> AnnData:
    """Raw-intensity-scale proteomics AnnData.

    Fails the log-transform heuristic. Lognormal raw intensities,
    equivalent to the sister log2 fixture but NOT log-transformed;
    used to exercise the ``force=False`` log check.
    """
    rng = np.random.default_rng(2)
    X = rng.lognormal(mean=23.0 * np.log(2), sigma=2.5 * np.log(2),
                      size=(20, 30))
    miss = rng.random(size=X.shape) < 0.2
    X[miss] = np.nan

    obs_names = [f"s{i}" for i in range(20)]
    var_names = [f"p{i}" for i in range(30)]
    obs = pd.DataFrame({"sample_id": obs_names}, index=obs_names)
    var = pd.DataFrame({"protein_id": var_names}, index=var_names)
    return AnnData(X=X, obs=obs, var=var)


# ────────────────────────────────────────────────────────────────────


class TestImputeDownshift:
    """Tests for ``impute_downshift``."""

    # ── A. Existing values & metadata invariants ────────────────────

    @pytest.mark.parametrize("inplace", [True, False])
    def test_observed_values_preserved(self, inplace):
        """Non-missing entries are bit-identical after imputation."""
        adata = _make_small_log_adata()
        X_in = adata.X.copy()
        finite_mask_in = np.isfinite(X_in) & (X_in != 0)

        result = impute_downshift(adata, inplace=inplace)
        target = adata if inplace else result
        if inplace:
            assert result is None
        else:
            assert result is not None

        np.testing.assert_array_equal(
            np.asarray(target.X)[finite_mask_in],
            X_in[finite_mask_in],
        )

    def test_imputation_mask_layer_present_and_correct(self):
        adata = _make_small_log_adata()
        X_in = adata.X.copy()
        expected_mask = ~np.isfinite(X_in) | (X_in == 0)

        result = impute_downshift(adata, inplace=False)
        mask = np.asarray(result.layers["imputation_mask_X"])

        assert mask.dtype == bool
        assert mask.shape == X_in.shape
        np.testing.assert_array_equal(mask, expected_mask)

    def test_no_nan_or_zero_in_output(self):
        adata = _make_small_log_adata()
        result = impute_downshift(adata, inplace=False)
        X_out = np.asarray(result.X)
        assert np.isfinite(X_out).all()
        assert (X_out != 0).all()

    def test_uns_metadata_keys_and_values(self):
        adata = _make_small_log_adata()
        n_missing_in = int((~np.isfinite(adata.X) | (adata.X == 0)).sum())

        result = impute_downshift(
            adata,
            downshift=1.8,
            width=0.3,
            random_state=42,
            inplace=False,
        )

        meta = result.uns["imputation"]
        assert meta["method"] == "downshift_normal"
        assert meta["downshift"] == pytest.approx(1.8)
        assert meta["width"] == pytest.approx(0.3)
        assert meta["group_by"] is None
        assert meta["random_state"] == 42
        assert meta["n_imputed"] == n_missing_in
        assert meta["pct_imputed"] == pytest.approx(
            100.0 * n_missing_in / adata.X.size,
        )

    def test_no_missing_values_returns_input_unchanged(self):
        """When there is nothing to impute, output equals input and the
        mask is all False."""
        rng = np.random.default_rng(0)
        raw = rng.lognormal(
            mean=23.0 * np.log(2), sigma=2.5 * np.log(2),
            size=(20, 30),
        )
        X = np.log2(raw)
        obs_names = [f"s{i}" for i in range(20)]
        var_names = [f"p{i}" for i in range(30)]
        obs = pd.DataFrame({"sample_id": obs_names}, index=obs_names)
        var = pd.DataFrame({"protein_id": var_names}, index=var_names)
        adata = AnnData(X=X, obs=obs, var=var)
        X_in = adata.X.copy()

        result = impute_downshift(
            adata, zero_to_na=False, inplace=False,
        )

        np.testing.assert_array_equal(np.asarray(result.X), X_in)
        mask = np.asarray(result.layers["imputation_mask_X"])
        assert not mask.any()
        assert result.uns["imputation"]["n_imputed"] == 0
        assert result.uns["imputation"]["pct_imputed"] == 0.0

    def test_zero_to_na_false_keeps_zeros(self):
        adata = _make_small_log_adata()
        # Inject a couple of zeros that should NOT be imputed.
        adata.X[0, 0] = 0.0
        adata.X[3, 0] = 0.0

        # force=True decouples this test from the log-transform heuristic
        # — its purpose is zero handling, not log detection.
        result = impute_downshift(
            adata, zero_to_na=False, force=True, inplace=False,
        )

        X_out = np.asarray(result.X)
        assert X_out[0, 0] == 0.0
        assert X_out[3, 0] == 0.0
        mask = np.asarray(result.layers["imputation_mask_X"])
        assert not mask[0, 0]
        assert not mask[3, 0]

    # ── B. Statistical shape of imputed values ──────────────────────
    #
    # Tests in this section verify that the imputed values follow the
    # documented Perseus-style downshifted Gaussian:
    #     N(median(observed) - downshift*sd(observed),
    #       (width*sd(observed))^2).
    # All draws use random_state=42 and a fixed data seed (0). Tolerances
    # are expressed relative to the theoretical sigma so they survive
    # plausible drift in numpy's RNG. Every threshold has >=10× margin
    # over the value seen at these seeds.

    @staticmethod
    def _split_observed_imputed(adata_in, result):
        """Return (observed, imputed) value arrays from input + result."""
        X_in = np.asarray(adata_in.X)
        X_out = np.asarray(result.X)
        mask = np.asarray(result.layers["imputation_mask_X"])
        observed = X_out[~mask]
        imputed = X_out[mask]
        # sanity: observed values match input at observed positions
        np.testing.assert_array_equal(observed, X_in[~mask])
        return observed, imputed

    @staticmethod
    def _theoretical_params(observed, downshift, width):
        med = float(np.median(observed))
        sd = float(np.std(observed))
        return med - downshift * sd, width * sd, med, sd

    def test_imputed_mean_is_smaller_than_observed_mean(self):
        adata_in = _make_log_adata_with_missing()
        result = impute_downshift(
            adata_in.copy(),
            downshift=1.8, width=0.3,
            random_state=42, inplace=False,
        )
        observed, imputed = self._split_observed_imputed(adata_in, result)
        # downshift=1.8 SDs leftward → imputed mean clearly below observed
        _, _, _, sd = self._theoretical_params(observed, 1.8, 0.3)
        assert imputed.mean() < observed.mean() - sd

    def test_imputed_mean_matches_theoretical(self):
        adata_in = _make_log_adata_with_missing()
        result = impute_downshift(
            adata_in.copy(),
            downshift=1.8, width=0.3,
            random_state=42, inplace=False,
        )
        observed, imputed = self._split_observed_imputed(adata_in, result)
        mu_th, sigma_th, _, _ = self._theoretical_params(observed, 1.8, 0.3)
        np.testing.assert_allclose(
            imputed.mean(), mu_th, atol=0.1 * sigma_th,
        )

    @pytest.mark.parametrize("q", [0.25, 0.5, 0.75])
    def test_imputed_quantile_below_observed_quantile(self, q):
        adata_in = _make_log_adata_with_missing()
        result = impute_downshift(
            adata_in.copy(),
            downshift=1.8, width=0.3,
            random_state=42, inplace=False,
        )
        observed, imputed = self._split_observed_imputed(adata_in, result)
        assert np.quantile(imputed, q) < np.quantile(observed, q)

    @pytest.mark.parametrize("p", [5, 25, 50, 75, 95])
    def test_imputed_percentiles_match_theoretical_normal(self, p):
        adata_in = _make_log_adata_with_missing()
        result = impute_downshift(
            adata_in.copy(),
            downshift=1.8, width=0.3,
            random_state=42, inplace=False,
        )
        observed, imputed = self._split_observed_imputed(adata_in, result)
        mu_th, sigma_th, _, _ = self._theoretical_params(observed, 1.8, 0.3)
        expected = norm.ppf(p / 100.0, loc=mu_th, scale=sigma_th)
        np.testing.assert_allclose(
            np.percentile(imputed, p), expected, atol=0.15 * sigma_th,
        )

    @pytest.mark.parametrize(
        "width,direction",
        [(0.3, "smaller"), (2.0, "bigger")],
    )
    def test_imputed_variance_vs_observed(self, width, direction):
        """Imputed variance is `(width*sd)^2`; smaller or bigger
        by user choice."""
        adata_in = _make_log_adata_with_missing()
        result = impute_downshift(
            adata_in.copy(),
            downshift=1.8, width=width,
            random_state=42, inplace=False,
        )
        observed, imputed = self._split_observed_imputed(adata_in, result)
        _, sigma_th, _, _ = self._theoretical_params(observed, 1.8, width)

        if direction == "smaller":
            assert imputed.var() < observed.var()
        else:
            assert imputed.var() > observed.var()

        # In both directions, the empirical std matches width*sd.
        np.testing.assert_allclose(
            imputed.std(), sigma_th, rtol=0.1,
        )

    def test_kolmogorov_smirnov_two_sample_observed_vs_imputed_rejects(self):
        """Imputed and observed distributions are clearly different."""
        adata_in = _make_log_adata_with_missing()
        result = impute_downshift(
            adata_in.copy(),
            downshift=1.8, width=0.3,
            random_state=42, inplace=False,
        )
        observed, imputed = self._split_observed_imputed(adata_in, result)
        stat, pvalue = ks_2samp(observed, imputed)
        # Massive shift + narrow imputed → KS pvalue effectively 0.
        assert pvalue < 1e-10
        assert stat > 0.5

    def test_kolmogorov_smirnov_one_sample_imputed_matches_theoretical(self):
        """Imputed values are consistent with the theoretical normal."""
        adata_in = _make_log_adata_with_missing()
        result = impute_downshift(
            adata_in.copy(),
            downshift=1.8, width=0.3,
            random_state=42, inplace=False,
        )
        observed, imputed = self._split_observed_imputed(adata_in, result)
        mu_th, sigma_th, _, _ = self._theoretical_params(observed, 1.8, 0.3)
        # Fail-to-reject: imputed values look like theoretical normal draws.
        result_ks = kstest(imputed, "norm", args=(mu_th, sigma_th))
        assert result_ks.pvalue > 0.01

    def test_likelihood_ratio_test_favors_downshifted_model(self):
        """LRT-style log-likelihood comparison strongly prefers the
        downshifted-normal model over the observed-normal model."""
        adata_in = _make_log_adata_with_missing()
        result = impute_downshift(
            adata_in.copy(),
            downshift=1.8, width=0.3,
            random_state=42, inplace=False,
        )
        observed, imputed = self._split_observed_imputed(adata_in, result)
        mu_th, sigma_th, mu_obs, sd_obs = self._theoretical_params(
            observed, 1.8, 0.3,
        )

        # Direct log-likelihood comparison (non-nested models).
        ll_obs = norm.logpdf(imputed, loc=mu_obs, scale=sd_obs).sum()
        ll_th = norm.logpdf(imputed, loc=mu_th, scale=sigma_th).sum()
        D = 2.0 * (ll_th - ll_obs)
        # 1.8-SD shift on ~4000 values → D in the thousands. Threshold is
        # ~30× lower than the actual value to absorb RNG drift.
        assert D > 100

        # Nested LRT: H0 = N(mu_obs, sd_obs) vs H1 = N(mu_hat, sigma_hat).
        mu_hat = float(imputed.mean())
        sigma_hat = float(imputed.std(ddof=0))
        ll_full = norm.logpdf(imputed, loc=mu_hat, scale=sigma_hat).sum()
        lrt = 2.0 * (ll_full - ll_obs)
        # df=2 (mu, sigma both freed). Under H0, lrt ~ chi^2(2). The true
        # generating distribution differs strongly → reject H0.
        p = 1.0 - chi2.cdf(lrt, df=2)
        assert p < 1e-10

    # ── C. Sparse/dense + inplace/copy semantics ────────────────────

    def test_inplace_true_returns_none_and_mutates(self):
        adata = _make_small_log_adata()
        X_in = adata.X.copy()

        returned = impute_downshift(adata, inplace=True)

        assert returned is None
        X_out = np.asarray(adata.X)
        assert np.isfinite(X_out).all()
        assert "imputation_mask_X" in adata.layers
        # Values at originally observed positions are still the same.
        finite_mask = np.isfinite(X_in) & (X_in != 0)
        np.testing.assert_array_equal(
            X_out[finite_mask], X_in[finite_mask],
        )

    def test_inplace_false_returns_copy_and_preserves_original(self):
        adata = _make_small_log_adata()
        X_in_snapshot = adata.X.copy()

        result = impute_downshift(adata, inplace=False)

        assert result is not adata
        # Original .X bit-identical (NaNs included).
        assert np.array_equal(
            adata.X, X_in_snapshot, equal_nan=True,
        )
        assert "imputation_mask_X" not in adata.layers
        assert "imputation_mask_X" in result.layers

    def test_sparse_input_yields_sparse_output(self):
        adata = _make_small_log_adata()
        # csr_matrix can't store NaN reliably across versions for this
        # contract; build sparse from a matrix where the missingness is
        # only zeros (impute_downshift treats zeros as missing by default).
        X_dense = np.array(
            [
                [10.0, 12.0, 0.0],
                [11.0, 0.0, 14.0],
                [0.0, 13.0, 15.0],
                [12.0, 14.0, 0.0],
            ],
        )
        adata.X = sparse.csr_matrix(X_dense)

        result = impute_downshift(adata, inplace=False)

        assert sparse.issparse(result.X)
        assert isinstance(result.X, sparse.csr_matrix)

    def test_dense_input_yields_dense_output(self):
        adata = _make_small_log_adata()
        result = impute_downshift(adata, inplace=False)
        assert not sparse.issparse(result.X)

    def test_random_state_reproducibility(self):
        adata1 = _make_log_adata_with_missing()
        adata2 = _make_log_adata_with_missing()

        r1 = impute_downshift(adata1, random_state=42, inplace=False)
        r2 = impute_downshift(adata2, random_state=42, inplace=False)

        np.testing.assert_array_equal(np.asarray(r1.X), np.asarray(r2.X))

    def test_random_state_none_is_nondeterministic(self):
        adata1 = _make_log_adata_with_missing()
        adata2 = _make_log_adata_with_missing()

        r1 = impute_downshift(adata1, random_state=None, inplace=False)
        r2 = impute_downshift(adata2, random_state=None, inplace=False)

        mask = np.asarray(r1.layers["imputation_mask_X"])
        # Imputed positions differ across runs.
        assert not np.array_equal(
            np.asarray(r1.X)[mask], np.asarray(r2.X)[mask],
        )

    # ── D. group_by behavior ────────────────────────────────────────

    def test_group_by_uses_group_specific_stats(self):
        adata = _make_grouped_log_adata()
        X_in = adata.X.copy()

        result = impute_downshift(
            adata, group_by="group",
            downshift=1.8, width=0.3,
            random_state=42, inplace=False,
        )

        mask = np.asarray(result.layers["imputation_mask_X"])
        X_out = np.asarray(result.X)
        groups = result.obs["group"].to_numpy()

        # Each group's imputed mean should track its own (downshifted) median.
        for label, expected_loc in (("A", 18.0), ("B", 24.0)):
            row_idx = np.where(groups == label)[0]
            grp_mask = mask[row_idx, :]
            grp_imputed = X_out[row_idx, :][grp_mask]
            # Imputed values for group B are clearly above those for group A
            # iff group-specific stats were used.
            assert grp_imputed.mean() < expected_loc
            assert grp_imputed.mean() > expected_loc - 5.0

        # Group A's imputed mean must be lower than group B's.
        idx_a = np.where(groups == "A")[0]
        idx_b = np.where(groups == "B")[0]
        mean_a = X_out[idx_a, :][mask[idx_a, :]].mean()
        mean_b = X_out[idx_b, :][mask[idx_b, :]].mean()
        assert mean_a < mean_b - 3.0  # groups separated by ~6 in log-space

        # Sanity: input not mutated.
        assert np.array_equal(adata.X, X_in, equal_nan=True)

    def test_group_by_fallback_to_global_when_group_too_small(self):
        adata = _make_grouped_log_adata(
            n_per_group=20, n_vars=50, miss_frac=0.25, seed=1,
        )
        # Build a third group "C" with only 1 observation AND fewer than
        # 3 finite values across that row — this is what triggers the
        # fallback path (`grp_vals.size >= 3` is False).
        groups = list(adata.obs["group"].astype(object).to_numpy())
        groups[0] = "C"
        adata.obs["group"] = groups
        # Keep only 2 finite values in the C-group row; rest become NaN.
        adata.X[0, 2:] = np.nan

        result = impute_downshift(
            adata, group_by="group",
            downshift=1.8, width=0.3,
            random_state=42, inplace=False,
        )
        mask = np.asarray(result.layers["imputation_mask_X"])
        X_out = np.asarray(result.X)

        # With group A median ≈ 18 (sd≈1.5) and group B median ≈ 24
        # (sd≈1.5), the global pool has median ≈ 21 and sd ≈ 3.3, so
        # the fallback imputes around 21 - 1.8*3.3 ≈ 15.0. Group-C's
        # own 2 finite values (drawn from group A's distribution) would
        # have given mean ≈ 18 - small shift → around 17. The interval
        # below distinguishes the two paths.
        c_idx = np.where(adata.obs["group"].to_numpy() == "C")[0]
        c_imputed = X_out[c_idx, :][mask[c_idx, :]]
        assert c_imputed.size > 40  # most of the row was imputed
        assert c_imputed.mean() < 16.5  # fallback (global), not group-A
        assert c_imputed.mean() > 12.0

    def test_group_by_invalid_column_raises_keyerror(self):
        adata = _make_small_log_adata()
        with pytest.raises(KeyError, match=r"not found"):
            impute_downshift(
                adata, group_by="not_a_col", inplace=False,
            )

    def test_group_by_records_in_uns(self):
        adata = _make_grouped_log_adata()
        result = impute_downshift(
            adata, group_by="group", inplace=False,
        )
        assert result.uns["imputation"]["group_by"] == "group"

    # ── E. Validation / errors ──────────────────────────────────────

    @pytest.mark.parametrize("bad_adata", ["x", 42, None])
    def test_invalid_adata_type(self, bad_adata):
        with pytest.raises(TypeError, match=r"AnnData"):
            impute_downshift(bad_adata)

    @pytest.mark.parametrize("bad", ["1.8", True, [1.8]])
    def test_invalid_downshift_type(self, bad):
        adata = _make_small_log_adata()
        with pytest.raises(TypeError, match=r"downshift"):
            impute_downshift(adata, downshift=bad)

    @pytest.mark.parametrize("bad", ["0.3", True, [0.3]])
    def test_invalid_width_type(self, bad):
        adata = _make_small_log_adata()
        with pytest.raises(TypeError, match=r"width"):
            impute_downshift(adata, width=bad)

    @pytest.mark.parametrize("bad", [0, 0.0, -1, -0.5])
    def test_invalid_width_value(self, bad):
        adata = _make_small_log_adata()
        with pytest.raises(ValueError, match=r"positive"):
            impute_downshift(adata, width=bad)

    @pytest.mark.parametrize(
        "param,bad",
        [
            ("zero_to_na", "yes"),
            ("zero_to_na", 1),
            ("inplace", "true"),
            ("inplace", 0),
            ("force", "no"),
            ("force", 1),
            ("verbose", "yes"),
            ("verbose", 1),
        ],
    )
    def test_invalid_bool_params(self, param, bad):
        adata = _make_small_log_adata()
        with pytest.raises(TypeError, match=param):
            impute_downshift(adata, **{param: bad})

    @pytest.mark.parametrize("bad", [1.5, "42", [42]])
    def test_invalid_random_state_type(self, bad):
        adata = _make_small_log_adata()
        with pytest.raises(TypeError, match=r"random_state"):
            impute_downshift(adata, random_state=bad)

    @pytest.mark.parametrize("bad", [42, [1, 2], 3.14])
    def test_invalid_group_by_type(self, bad):
        adata = _make_small_log_adata()
        with pytest.raises(TypeError, match=r"group_by"):
            impute_downshift(adata, group_by=bad)

    def test_non_log_data_without_force_raises(self):
        adata = _make_non_log_adata()
        with pytest.raises(ValueError, match=r"log-transformed"):
            impute_downshift(adata, force=False, inplace=False)

    def test_non_log_data_with_force_succeeds(self):
        adata = _make_non_log_adata()
        result = impute_downshift(adata, force=True, inplace=False)
        assert result is not None
        assert np.isfinite(np.asarray(result.X)).all()

    def test_too_few_finite_values_raises(self):
        n = np.nan
        # 2 finite values total, the rest NaN.
        X = np.array(
            [
                [10.0, n, n],
                [n, 11.0, n],
                [n, n, n],
            ],
            dtype=float,
        )
        obs_names = ["s0", "s1", "s2"]
        var_names = ["p0", "p1", "p2"]
        obs = pd.DataFrame(
            {"sample_id": obs_names}, index=obs_names,
        )
        var = pd.DataFrame(
            {"protein_id": var_names}, index=var_names,
        )
        adata = AnnData(X=X, obs=obs, var=var)
        with pytest.raises(
            ValueError, match=r"Not enough finite values",
        ):
            impute_downshift(adata, force=True, inplace=False)

    def test_zero_variance_raises(self):
        X = np.full((4, 3), 12.0)
        obs_names = [f"s{i}" for i in range(4)]
        var_names = [f"p{i}" for i in range(3)]
        obs = pd.DataFrame(
            {"sample_id": obs_names}, index=obs_names,
        )
        var = pd.DataFrame(
            {"protein_id": var_names}, index=var_names,
        )
        adata = AnnData(X=X, obs=obs, var=var)
        # Inject a single NaN so there is at least something to impute.
        adata.X[0, 0] = np.nan
        with pytest.raises(
            ValueError, match=r"standard deviation",
        ):
            impute_downshift(adata, force=True, inplace=False)

    def test_verbose_prints_stats(self, capsys):
        adata = _make_small_log_adata()
        impute_downshift(adata, verbose=True, inplace=True)
        out = capsys.readouterr().out
        assert "Measured" in out
        assert "Imputed" in out

    def test_verbose_false_prints_nothing(self, capsys):
        adata = _make_small_log_adata()
        impute_downshift(adata, verbose=False, inplace=True)
        assert capsys.readouterr().out == ""


# ────────────────────────────────────────────────────────────────────


class TestImputeRowsHelper:
    """Tests for the private helper ``_impute_rows``."""

    def test_writes_only_at_mask_true(self):
        Y_imp = np.full((3, 4), -999.0)
        miss_mask = np.array(
            [
                [True, False, True, False],
                [False, False, False, False],
                [False, True, False, True],
            ],
        )
        rng = np.random.default_rng(0)

        _impute_rows(
            Y_imp, miss_mask, range(3),
            median=10.0, sd=1.0,
            downshift=1.8, width=0.3, rng=rng,
        )

        # Untouched cells remain at the sentinel value.
        assert (Y_imp[~miss_mask] == -999.0).all()
        # Touched cells are finite (sampled from a normal).
        assert np.isfinite(Y_imp[miss_mask]).all()
        assert (Y_imp[miss_mask] != -999.0).all()

    def test_uses_provided_rng(self):
        median, sd = 10.0, 1.0
        downshift, width = 1.8, 0.3
        # Two row × three col matrix; all cells missing.
        miss_mask = np.ones((2, 3), dtype=bool)
        Y_imp_a = np.zeros((2, 3))
        Y_imp_b = np.zeros((2, 3))

        _impute_rows(
            Y_imp_a, miss_mask, range(2),
            median=median, sd=sd,
            downshift=downshift, width=width,
            rng=np.random.default_rng(7),
        )
        _impute_rows(
            Y_imp_b, miss_mask, range(2),
            median=median, sd=sd,
            downshift=downshift, width=width,
            rng=np.random.default_rng(7),
        )

        np.testing.assert_array_equal(Y_imp_a, Y_imp_b)
        # And different seed → different output.
        Y_imp_c = np.zeros((2, 3))
        _impute_rows(
            Y_imp_c, miss_mask, range(2),
            median=median, sd=sd,
            downshift=downshift, width=width,
            rng=np.random.default_rng(8),
        )
        assert not np.array_equal(Y_imp_a, Y_imp_c)

    def test_skips_rows_with_no_missing(self):
        Y_imp = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        )
        miss_mask = np.zeros_like(Y_imp, dtype=bool)
        original = Y_imp.copy()

        _impute_rows(
            Y_imp, miss_mask, range(2),
            median=10.0, sd=1.0,
            downshift=1.8, width=0.3,
            rng=np.random.default_rng(0),
        )

        np.testing.assert_array_equal(Y_imp, original)

    def test_clamps_scale_minimum(self):
        """`width=0`/`sd=0` is clamped internally to avoid scale=0 errors."""
        Y_imp = np.zeros((2, 2))
        miss_mask = np.ones_like(Y_imp, dtype=bool)
        _impute_rows(
            Y_imp, miss_mask, range(2),
            median=5.0, sd=0.0,
            downshift=1.8, width=0.0,
            rng=np.random.default_rng(0),
        )
        assert np.isfinite(Y_imp).all()


class TestImputeByGroupHelper:
    """Tests for the private helper ``_impute_by_group``."""

    def test_uses_per_group_stats(self):
        rng_data = np.random.default_rng(0)
        # 12 obs split into two groups with very different medians.
        n_per = 6
        n_vars = 30
        Y = np.vstack(
            [
                rng_data.normal(0.0, 1.0, size=(n_per, n_vars)),
                rng_data.normal(10.0, 1.0, size=(n_per, n_vars)),
            ],
        )
        # In the production flow, miss_mask is derived from Y itself
        # (`miss_mask = ~np.isfinite(Y)`), so Y has NaN at miss positions
        # before _impute_by_group is called. Mirror that here.
        miss_mask = rng_data.random(size=Y.shape) < 0.5
        Y[miss_mask] = np.nan
        Y_imp = Y.copy()
        groups = pd.Series(["A"] * n_per + ["B"] * n_per)

        _impute_by_group(
            Y, Y_imp, miss_mask, groups,
            g_median=5.0, g_sd=5.5,
            downshift=1.8, width=0.3,
            rng=np.random.default_rng(0),
        )

        a_imp = Y_imp[:n_per][miss_mask[:n_per]]
        b_imp = Y_imp[n_per:][miss_mask[n_per:]]
        # Group A's imputed values cluster well below group B's.
        assert a_imp.mean() < b_imp.mean() - 3.0

    def test_falls_back_when_group_too_small(self):
        """Group with <3 finite values uses the supplied global stats."""
        # 4 obs: groups A (3 obs, plenty of data) and C (1 obs, only 1 finite).
        n_vars = 50
        rng_data = np.random.default_rng(1)
        Y_A = rng_data.normal(0.0, 1.0, size=(3, n_vars))
        Y_C = np.full((1, n_vars), np.nan)
        Y_C[0, 0] = 50.0  # one finite value far from group A
        Y = np.vstack([Y_A, Y_C])

        miss_mask = ~np.isfinite(Y)
        Y_imp = Y.copy()
        groups = pd.Series(["A", "A", "A", "C"])

        # Global stats provided (g_median=100, g_sd=10) deliberately far
        # from any group's data so we can detect which one was used.
        _impute_by_group(
            Y, Y_imp, miss_mask, groups,
            g_median=100.0, g_sd=10.0,
            downshift=1.8, width=0.3,
            rng=np.random.default_rng(0),
        )

        c_imp = Y_imp[3, :][miss_mask[3, :]]
        # Fallback → centred near 100 - 1.8*10 = 82, not near group A's mean.
        assert c_imp.mean() > 70.0
        assert c_imp.mean() < 95.0

    def test_falls_back_when_group_sd_zero(self):
        """A group whose finite values are all identical (sd=0) falls back."""
        n_vars = 40
        rng_data = np.random.default_rng(2)
        Y_A = rng_data.normal(0.0, 1.0, size=(3, n_vars))
        # Group B: many finite values but all identical → sd=0.
        Y_B = np.full((4, n_vars), 5.0)
        Y_B[0, 0] = np.nan  # at least one missing to impute
        Y = np.vstack([Y_A, Y_B])

        miss_mask = ~np.isfinite(Y)
        Y_imp = Y.copy()
        groups = pd.Series(["A"] * 3 + ["B"] * 4)

        _impute_by_group(
            Y, Y_imp, miss_mask, groups,
            g_median=100.0, g_sd=10.0,
            downshift=1.8, width=0.3,
            rng=np.random.default_rng(0),
        )

        b_imp = Y_imp[3:, :][miss_mask[3:, :]]
        # Should fall back to global stats: mean ~ 100 - 1.8*10 = 82.
        assert b_imp.mean() > 70.0
        assert b_imp.mean() < 95.0
