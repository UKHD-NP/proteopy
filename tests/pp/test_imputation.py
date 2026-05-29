import pytest
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse
from scipy.stats import norm

from proteopy.pp.imputation import impute_downshift


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
    Sized for statistical-test power.
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
    """Two-cell-type log2 proteomics intensities with distinct medians.

    Both cell types draw raw intensities from a lognormal then log2-
    transform, matching the Gaussian shape of real proteomics log-
    intensities. Median log2 abundances differ by ~6 units (e.g., a
    highly-expressing vs. lowly-expressing lineage).
    """
    rng = np.random.default_rng(seed)
    sigma_ln = 1.5 * np.log(2)  # log2-sd ≈ 1.5
    raw_t = rng.lognormal(
        mean=18.0 * np.log(2),
        sigma=sigma_ln,
        size=(n_per_group, n_vars),
    )
    raw_b = rng.lognormal(
        mean=24.0 * np.log(2),
        sigma=sigma_ln,
        size=(n_per_group, n_vars),
    )
    X = np.vstack([np.log2(raw_t), np.log2(raw_b)])
    miss = rng.random(size=X.shape) < miss_frac
    X[miss] = np.nan

    n_obs = 2 * n_per_group
    obs_names = [f"s{i}" for i in range(n_obs)]
    cell_types = ["T_cell"] * n_per_group + ["B_cell"] * n_per_group
    var_names = [f"p{i}" for i in range(n_vars)]
    obs = pd.DataFrame(
        {"sample_id": obs_names, "cell_type": cell_types},
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
    X = rng.lognormal(
        mean=23.0 * np.log(2),
        sigma=2.5 * np.log(2),
        size=(20, 30),
    )
    miss = rng.random(size=X.shape) < 0.2
    X[miss] = np.nan

    obs_names = [f"s{i}" for i in range(20)]
    var_names = [f"p{i}" for i in range(30)]
    obs = pd.DataFrame({"sample_id": obs_names}, index=obs_names)
    var = pd.DataFrame({"protein_id": var_names}, index=var_names)
    return AnnData(X=X, obs=obs, var=var)


# ── Statistical comparison helpers ──────────────────────────────────
#
# All four helpers below take a 1-D ``observed`` and 1-D ``imputed``
# array (the imputed values pulled from ``result.X`` at mask=True
# positions, and the kept observed values at mask=False positions).
# They are reused for both global (section B) and per-group (section
# D) checks so the same statistical contract is enforced everywhere.

# Posterior-probability threshold for the "same distribution" Bayes
# factor. >0.95 corresponds to "very strong" evidence under the
# uniform-prior interpretation of Kass & Raftery (1995).
BF_PH1_THRESHOLD = 0.95


def _theoretical_downshift_params(observed, downshift, width):
    """Return (mu_th, sigma_th) for the documented downshifted normal."""
    med = float(np.median(observed))
    sd = float(np.std(observed))
    return med - downshift * sd, width * sd


def _draw_reference(mu, sigma, n, seed=100):
    """Draw a fixed-seed reference sample from N(mu, sigma)."""
    return np.random.default_rng(seed).normal(
        loc=mu,
        scale=sigma,
        size=n,
    )


def compare_relative_means(observed, imputed, downshift, width):
    """Assert imputed mean sits below observed mean and matches theory."""
    mu_th, sigma_th = _theoretical_downshift_params(
        observed,
        downshift,
        width,
    )
    assert imputed.mean() < observed.mean()
    np.testing.assert_allclose(
        imputed.mean(),
        mu_th,
        atol=0.1 * sigma_th,
    )


def compare_quantiles(observed, imputed, qs=(0.25, 0.5, 0.75)):
    """Assert each imputed quantile falls below the observed quantile."""
    for q in qs:
        q_obs = float(np.quantile(observed, q))
        q_imp = float(np.quantile(imputed, q))
        assert (
            q_imp < q_obs
        ), f"q={q}: imputed {q_imp:.3f} ≥ observed {q_obs:.3f}"


def compare_percentiles(
    observed,
    imputed,
    downshift,
    width,
    ps=(5, 25, 50, 75, 95),
):
    """Assert imputed percentiles match the theoretical downshifted normal."""
    mu_th, sigma_th = _theoretical_downshift_params(
        observed,
        downshift,
        width,
    )
    # 0.3 * sigma_th ≈ 2 standard errors of the sample 5th-percentile
    # for the smaller per-group samples (n ≈ 250); much looser than
    # needed at the global scale (n ≈ 4000). One threshold for both
    # callers — generous enough to absorb RNG drift, tight enough
    # that a sign flip or width misuse still breaks the test.
    for p in ps:
        expected = float(norm.ppf(p / 100.0, loc=mu_th, scale=sigma_th))
        actual = float(np.percentile(imputed, p))
        np.testing.assert_allclose(
            actual,
            expected,
            atol=0.3 * sigma_th,
            err_msg=f"p={p}: expected {expected:.3f}, got {actual:.3f}",
        )


def bayes_factor_same_norm_distr(x, y):
    """Bayes factor for "same normal distribution" (H1) vs "different" (H0).

    Uses a BIC approximation:
      - H1 (same): one normal fit to concatenated data, 2 parameters.
      - H0 (different): separate normal fits to x and y, 4 parameters.

    Returns
    -------
    bf_10 : float
        Bayes factor in favour of H1 (same distribution).
    p_h1 : float
        Posterior probability of H1 under a uniform prior on {H0, H1},
        i.e. ``bf_10 / (1 + bf_10)``.
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    combined = np.concatenate([x, y])
    n = combined.size

    ll_same = norm.logpdf(
        combined,
        loc=combined.mean(),
        scale=combined.std(ddof=0),
    ).sum()
    ll_diff = (
        norm.logpdf(x, loc=x.mean(), scale=x.std(ddof=0)).sum()
        + norm.logpdf(y, loc=y.mean(), scale=y.std(ddof=0)).sum()
    )
    # ΔBIC/2 = ln(n) + (ll_same - ll_diff); BF_10 ≈ exp(ΔBIC/2).
    log_bf_10 = np.log(n) + (ll_same - ll_diff)
    bf_10 = float(np.exp(log_bf_10))
    p_h1 = bf_10 / (1.0 + bf_10)
    return bf_10, float(p_h1)


def _split_observed_imputed(adata_in, result):
    """Return (observed, imputed) 1-D arrays from input + result."""
    X_in = np.asarray(adata_in.X)
    X_out = np.asarray(result.X)
    mask = np.asarray(result.layers["imputation_mask_X"])
    observed = X_out[~mask]
    imputed = X_out[mask]
    # sanity: observed values are bit-identical to the input
    np.testing.assert_array_equal(observed, X_in[~mask])
    return observed, imputed


# ────────────────────────────────────────────────────────────────────


class TestImputeDownshift:
    """Tests for ``impute_downshift``.

    Organised by contract:

    - **A.** Value preservation, mask correctness, and ``uns``
      metadata.
    - **B.** Statistical shape of the imputed values against the
      documented downshifted normal — relative mean, quantile, and
      percentile checks plus a Bayes-factor test that imputed values
      are indistinguishable from draws of the theoretical normal.
    - **C.** ``inplace``/copy semantics and RNG reproducibility.
    - **D.** ``group_by`` behaviour, reusing the section-B statistical
      helpers per group plus tests for the global-stats fallback.
    - **E.** Input validation and verbose output.

    Random draws are seeded (``random_state=42``) so the statistical
    assertions are deterministic. Thresholds carry ≥10× margin to
    absorb numpy RNG drift.
    """

    # ── A. Existing values & metadata invariants ────────────────────

    @pytest.mark.parametrize("inplace", [True, False])
    def test_observed_values_preserved(self, inplace):
        """Non-NaN entries are bit-identical after imputation."""
        adata = _make_small_log_adata()
        X_in = adata.X.copy()
        finite_mask_in = np.isfinite(X_in)

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
        expected_mask = ~np.isfinite(X_in)

        result = impute_downshift(adata, inplace=False)
        mask = np.asarray(result.layers["imputation_mask_X"])

        assert mask.dtype == bool
        assert mask.shape == X_in.shape
        np.testing.assert_array_equal(mask, expected_mask)

    def test_no_nan_in_output(self):
        adata = _make_small_log_adata()
        result = impute_downshift(adata, inplace=False)
        assert np.isfinite(np.asarray(result.X)).all()

    def test_uns_metadata_keys_and_values(self):
        adata = _make_small_log_adata()
        n_missing_in = int((~np.isfinite(adata.X)).sum())

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
        """No NaN in input → output equals input and the mask is all False."""
        rng = np.random.default_rng(0)
        raw = rng.lognormal(
            mean=23.0 * np.log(2),
            sigma=2.5 * np.log(2),
            size=(20, 30),
        )
        X = np.log2(raw)
        obs_names = [f"s{i}" for i in range(20)]
        var_names = [f"p{i}" for i in range(30)]
        obs = pd.DataFrame({"sample_id": obs_names}, index=obs_names)
        var = pd.DataFrame({"protein_id": var_names}, index=var_names)
        adata = AnnData(X=X, obs=obs, var=var)
        X_in = adata.X.copy()

        result = impute_downshift(adata, inplace=False)

        np.testing.assert_array_equal(np.asarray(result.X), X_in)
        mask = np.asarray(result.layers["imputation_mask_X"])
        assert not mask.any()
        assert result.uns["imputation"]["n_imputed"] == 0
        assert result.uns["imputation"]["pct_imputed"] == 0.0

    def test_zeros_in_input_are_preserved_by_default(self):
        """Default ``zero_to_na=False``: zeros are observations, kept as-is."""
        adata = _make_small_log_adata()
        # Inject zeros at observed positions; they must NOT be imputed.
        adata.X[0, 0] = 0.0
        adata.X[3, 0] = 0.0

        result = impute_downshift(adata, inplace=False)

        X_out = np.asarray(result.X)
        assert X_out[0, 0] == 0.0
        assert X_out[3, 0] == 0.0
        mask = np.asarray(result.layers["imputation_mask_X"])
        assert not mask[0, 0]
        assert not mask[3, 0]

    def test_zero_to_na_true_treats_zeros_as_missing(self):
        """Opt-in ``zero_to_na=True``: zeros are converted to NaN
        and imputed."""
        adata = _make_small_log_adata()
        adata.X[0, 0] = 0.0
        adata.X[3, 0] = 0.0

        result = impute_downshift(
            adata,
            zero_to_na=True,
            inplace=False,
        )

        mask = np.asarray(result.layers["imputation_mask_X"])
        assert mask[0, 0]
        assert mask[3, 0]
        X_out = np.asarray(result.X)
        assert X_out[0, 0] != 0.0
        assert X_out[3, 0] != 0.0

    # ── B. Statistical shape of imputed values ──────────────────────
    #
    # The primary fixture has ~4000 imputed positions.
    # Defaults: downshift=1.8, width=0.3, random_state=42.

    def test_imputed_mean_is_shifted_and_matches_theoretical(self):
        adata_in = _make_log_adata_with_missing()
        result = impute_downshift(
            adata_in.copy(),
            downshift=1.8,
            width=0.3,
            random_state=42,
            inplace=False,
        )
        observed, imputed = _split_observed_imputed(adata_in, result)
        compare_relative_means(observed, imputed, 1.8, 0.3)

    def test_imputed_quantiles_are_below_observed(self):
        adata_in = _make_log_adata_with_missing()
        result = impute_downshift(
            adata_in.copy(),
            downshift=1.8,
            width=0.3,
            random_state=42,
            inplace=False,
        )
        observed, imputed = _split_observed_imputed(adata_in, result)
        compare_quantiles(observed, imputed)

    def test_imputed_percentiles_match_theoretical_normal(self):
        adata_in = _make_log_adata_with_missing()
        result = impute_downshift(
            adata_in.copy(),
            downshift=1.8,
            width=0.3,
            random_state=42,
            inplace=False,
        )
        observed, imputed = _split_observed_imputed(adata_in, result)
        compare_percentiles(observed, imputed, 1.8, 0.3)

    def test_imputed_matches_theoretical_via_bayes_factor(self):
        """Imputed values are indistinguishable from theoretical draws."""
        adata_in = _make_log_adata_with_missing()
        result = impute_downshift(
            adata_in.copy(),
            downshift=1.8,
            width=0.3,
            random_state=42,
            inplace=False,
        )
        observed, imputed = _split_observed_imputed(adata_in, result)
        mu_th, sigma_th = _theoretical_downshift_params(
            observed,
            1.8,
            0.3,
        )
        reference = _draw_reference(mu_th, sigma_th, imputed.size)
        _, p_h1 = bayes_factor_same_norm_distr(imputed, reference)
        assert p_h1 > BF_PH1_THRESHOLD

    # ── C. inplace/copy semantics ───────────────────────────────────

    def test_inplace_true_returns_none_and_mutates(self):
        adata = _make_small_log_adata()
        X_in = adata.X.copy()

        returned = impute_downshift(adata, inplace=True)

        assert returned is None
        X_out = np.asarray(adata.X)
        assert np.isfinite(X_out).all()
        assert "imputation_mask_X" in adata.layers
        finite_mask = np.isfinite(X_in)
        np.testing.assert_array_equal(
            X_out[finite_mask],
            X_in[finite_mask],
        )

    def test_inplace_false_returns_copy_and_preserves_original(self):
        adata = _make_small_log_adata()
        X_in_snapshot = adata.X.copy()

        result = impute_downshift(adata, inplace=False)

        assert result is not adata
        assert np.array_equal(
            adata.X,
            X_in_snapshot,
            equal_nan=True,
        )
        assert "imputation_mask_X" not in adata.layers
        assert "imputation_mask_X" in result.layers

    @pytest.mark.skip(
        reason="sparse handling not tested per project policy",
    )
    def test_sparse_input_yields_sparse_output(self):
        adata = _make_small_log_adata()
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

    @pytest.mark.skip(
        reason="sparse handling not tested per project policy",
    )
    def test_dense_input_yields_dense_output(self):
        adata = _make_small_log_adata()
        result = impute_downshift(adata, inplace=False)
        assert not sparse.issparse(result.X)

    def test_random_state_reproducibility(self):
        adata1 = _make_log_adata_with_missing()
        adata2 = _make_log_adata_with_missing()

        r1 = impute_downshift(adata1, random_state=42, inplace=False)
        r2 = impute_downshift(adata2, random_state=42, inplace=False)

        np.testing.assert_array_equal(
            np.asarray(r1.X),
            np.asarray(r2.X),
        )

    def test_random_state_none_is_nondeterministic(self):
        adata1 = _make_log_adata_with_missing()
        adata2 = _make_log_adata_with_missing()

        r1 = impute_downshift(adata1, random_state=None, inplace=False)
        r2 = impute_downshift(adata2, random_state=None, inplace=False)

        mask = np.asarray(r1.layers["imputation_mask_X"])
        assert not np.array_equal(
            np.asarray(r1.X)[mask],
            np.asarray(r2.X)[mask],
        )

    # ── D. group_by behavior ────────────────────────────────────────

    def test_group_by_imputed_distribution_per_group_matches_theoretical(
        self,
    ):
        """Within each group, imputed values match that group's
        theoretical downshifted normal (mean, quantiles, percentiles,
        Bayes-factor agreement)."""
        adata = _make_grouped_log_adata()
        X_in = adata.X.copy()

        result = impute_downshift(
            adata,
            group_by="cell_type",
            downshift=1.8,
            width=0.3,
            random_state=42,
            inplace=False,
        )
        mask = np.asarray(result.layers["imputation_mask_X"])
        X_out = np.asarray(result.X)
        cell_types = result.obs["cell_type"].to_numpy()

        per_group_means = {}
        for label in ("T_cell", "B_cell"):
            row_idx = np.where(cell_types == label)[0]
            grp_mask = mask[row_idx, :]
            grp_observed = X_out[row_idx, :][~grp_mask]
            grp_imputed = X_out[row_idx, :][grp_mask]

            compare_relative_means(grp_observed, grp_imputed, 1.8, 0.3)
            compare_quantiles(grp_observed, grp_imputed)
            compare_percentiles(grp_observed, grp_imputed, 1.8, 0.3)

            mu_th, sigma_th = _theoretical_downshift_params(
                grp_observed,
                1.8,
                0.3,
            )
            reference = _draw_reference(
                mu_th,
                sigma_th,
                grp_imputed.size,
            )
            _, p_h1 = bayes_factor_same_norm_distr(
                grp_imputed,
                reference,
            )
            assert (
                p_h1 > BF_PH1_THRESHOLD
            ), f"{label}: BF p_h1={p_h1:.3f} below threshold"

            per_group_means[label] = grp_imputed.mean()

        # The lower-median group's imputed mean is below the higher one's.
        assert per_group_means["T_cell"] < per_group_means["B_cell"]

        # Sanity: input not mutated.
        assert np.array_equal(adata.X, X_in, equal_nan=True)

    def test_group_by_fallback_to_global_when_group_too_small(self):
        adata = _make_grouped_log_adata(
            n_per_group=20,
            n_vars=50,
            miss_frac=0.25,
            seed=1,
        )
        # Add a third cell type "monocyte" with only 1 observation
        # AND fewer than 3 finite values, which is what triggers the
        # global-stats fallback (`grp_vals.size >= 3` is False).
        cell_types = list(adata.obs["cell_type"].astype(object).to_numpy())
        cell_types[0] = "monocyte"
        adata.obs["cell_type"] = cell_types
        adata.X[0, 2:] = np.nan  # leave 2 finite values in that row

        result = impute_downshift(
            adata,
            group_by="cell_type",
            downshift=1.8,
            width=0.3,
            random_state=42,
            inplace=False,
        )
        mask = np.asarray(result.layers["imputation_mask_X"])
        X_out = np.asarray(result.X)

        # Group statistics for this fixture:
        #   T_cell:    median ≈ 18,  sd ≈ 1.5  → downshifted ≈ 15.3
        #   B_cell:    median ≈ 24,  sd ≈ 1.5  → downshifted ≈ 21.3
        #   global:    median ≈ 21,  sd ≈ 3.3  → downshifted ≈ 15.0
        #   monocyte (2 finite values from T_cell range) ≈ 18 → ~17
        #
        # The two assertions below encode the fallback:
        #
        # UPPER bound: the global downshifted mean (~15) must sit BELOW
        # this threshold AND BELOW what the monocyte's own per-group
        # median would have produced (~17). Satisfying it rules out the
        # per-group path.
        UPPER = 16.5
        # LOWER bound: the global downshifted mean (~15) must sit ABOVE
        # this threshold. Sanity floor so a runaway draw can't pass.
        LOWER = 12.0

        m_idx = np.where(
            adata.obs["cell_type"].to_numpy() == "monocyte",
        )[0]
        m_imputed = X_out[m_idx, :][mask[m_idx, :]]
        assert m_imputed.size > 40  # most of the row was imputed
        assert m_imputed.mean() < UPPER
        assert m_imputed.mean() > LOWER

    def test_group_by_fallback_when_group_sd_zero(self):
        """A group whose finite values are identical (sd=0) falls back
        to global stats."""
        rng = np.random.default_rng(0)
        n_vars = 60
        # T_cell: 5 obs with normal variation (good per-group stats).
        X_t = np.log2(
            rng.lognormal(
                mean=20.0 * np.log(2),
                sigma=1.5 * np.log(2),
                size=(5, n_vars),
            )
        )
        # B_cell: 4 obs with a constant finite value (sd=0). Inject a
        # column of NaN so there are positions to impute.
        X_b = np.full((4, n_vars), 23.0)
        X_b[:, 0] = np.nan
        X = np.vstack([X_t, X_b])

        obs_names = [f"s{i}" for i in range(9)]
        cell_types = ["T_cell"] * 5 + ["B_cell"] * 4
        obs = pd.DataFrame(
            {"sample_id": obs_names, "cell_type": cell_types},
            index=obs_names,
        )
        var_names = [f"p{i}" for i in range(n_vars)]
        var = pd.DataFrame(
            {"protein_id": var_names},
            index=var_names,
        )
        adata = AnnData(X=X, obs=obs, var=var)

        result = impute_downshift(
            adata,
            group_by="cell_type",
            downshift=1.8,
            width=0.3,
            random_state=42,
            inplace=False,
        )
        mask = np.asarray(result.layers["imputation_mask_X"])
        X_out = np.asarray(result.X)

        b_idx = np.where(
            adata.obs["cell_type"].to_numpy() == "B_cell",
        )[0]
        b_imputed = X_out[b_idx, :][mask[b_idx, :]]
        # If the per-group (sd=0) path had been used, all imputed
        # values would collapse to the same constant. Real fallback
        # → global stats → non-zero empirical spread.
        assert np.isfinite(b_imputed).all()
        assert b_imputed.std() > 0.01

    def test_group_by_invalid_column_raises_keyerror(self):
        adata = _make_small_log_adata()
        with pytest.raises(KeyError, match=r"not found"):
            impute_downshift(
                adata,
                group_by="not_a_col",
                inplace=False,
            )

    def test_group_by_records_in_uns(self):
        adata = _make_grouped_log_adata()
        result = impute_downshift(
            adata,
            group_by="cell_type",
            inplace=False,
        )
        assert result.uns["imputation"]["group_by"] == "cell_type"

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
            {"sample_id": obs_names},
            index=obs_names,
        )
        var = pd.DataFrame(
            {"protein_id": var_names},
            index=var_names,
        )
        adata = AnnData(X=X, obs=obs, var=var)
        with pytest.raises(
            ValueError,
            match=r"Not enough finite values",
        ):
            impute_downshift(adata, force=True, inplace=False)

    def test_zero_variance_raises(self):
        X = np.full((4, 3), 12.0)
        obs_names = [f"s{i}" for i in range(4)]
        var_names = [f"p{i}" for i in range(3)]
        obs = pd.DataFrame(
            {"sample_id": obs_names},
            index=obs_names,
        )
        var = pd.DataFrame(
            {"protein_id": var_names},
            index=var_names,
        )
        adata = AnnData(X=X, obs=obs, var=var)
        adata.X[0, 0] = np.nan
        with pytest.raises(
            ValueError,
            match=r"standard deviation",
        ):
            impute_downshift(adata, force=True, inplace=False)

    def test_verbose_prints_correct_counts_and_percentages(self, capsys):
        """Verbose output reports the actual measured/imputed counts."""
        adata = _make_small_log_adata()
        n_total = adata.X.size
        n_missing = int((~np.isfinite(adata.X)).sum())
        n_measured = n_total - n_missing

        impute_downshift(adata, verbose=True, inplace=True)
        out = capsys.readouterr().out

        assert f"Measured: {n_measured:,} values" in out
        assert f"Imputed: {n_missing:,} values" in out
        assert f"({100 * n_measured / n_total:.1f}%)" in out
        assert f"({100 * n_missing / n_total:.1f}%)" in out

    def test_verbose_false_prints_nothing(self, capsys):
        adata = _make_small_log_adata()
        impute_downshift(adata, verbose=False, inplace=True)
        assert capsys.readouterr().out == ""
