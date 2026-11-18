"""Behavioral tests for ``proteopy.datasets.toy_peptides``.

1. Basic output and reproducibility.
2. Log2 model, sampling levels, and group effects.
3. Missingness, raw-scale aggregation, and statistical sanity.
4. A few representative invalid inputs.
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from proteopy.datasets import toy_peptides
from proteopy.utils.anndata import check_proteodata


class TestToyPeptides:
    """Check that the generated data follows the intended model."""

    # ── Shared fixtures ────────────────────────────────────────────────

    @pytest.fixture
    def deterministic_kwargs(self):
        """Return a small log2-20 model with no variation or missingness."""
        return {
            "n_proteins": 2,
            "n_peps_per_protein": [2, 1],
            "groups": ["z", "a", "b", "z", "a", "b"],
            "effect_size": 0,
            "protein_abundance": (20, 0),
            "peptide_response": (0, 0),
            "biological_variation": (0, 0),
            "loading_factor": (0, 0),
            "measurement_noise": (0, 0),
            "missing_rate": 0,
            "log2_detection_limit": None,
            "random_state": 42,
        }

    # ── A. Basic output ────────────────────────────────────────────────

    def test_defaults_produce_valid_peptide_intensities(self):
        data = toy_peptides()
        assert isinstance(data, ad.AnnData)
        assert data.shape == (10, 100)
        assert isinstance(data.X, np.ndarray)
        assert data.X.dtype == np.float64
        observed = data.X[~np.isnan(data.X)]
        assert observed.size > 0
        assert np.isfinite(observed).all()
        assert (observed > 0).all()
        assert data.obs["group"].tolist() == (
            ["control"] * 5 + ["treatment"] * 5
        )
        assert check_proteodata(data) == (True, "peptide")

    def test_generated_peptides_are_unique_canonical_sequences(self):
        data = toy_peptides()
        sequences = data.var["peptide_id"].tolist()
        canonical = set("ACDEFGHIKLMNPQRSTVWY")
        assert len(set(sequences)) == data.n_vars
        for sequence in sequences:
            assert 8 <= len(sequence) <= 20
            assert set(sequence) <= canonical

    def test_unequal_peptide_counts_follow_protein_blocks(self):
        data = toy_peptides(n_proteins=3, n_peps_per_protein=[3, 1, 2])
        assert data.shape == (10, 6)
        assert data.var["protein_id"].tolist() == (
            ["MAP001"] * 3 + ["MAP002"] + ["MAP003"] * 2
        )

    # ── B. Reproducibility ─────────────────────────────────────────────

    def test_same_seed_reproduces_data_and_identifiers(self):
        first = toy_peptides(random_state=42)
        second = toy_peptides(random_state=42)
        np.testing.assert_array_equal(first.X, second.X)
        pd.testing.assert_frame_equal(first.obs, second.obs)
        pd.testing.assert_frame_equal(first.var, second.var)

    def test_changed_seed_changes_numeric_draws(self):
        first = toy_peptides(
            random_state=41, missing_rate=0, log2_detection_limit=None
        )
        second = toy_peptides(
            random_state=42, missing_rate=0, log2_detection_limit=None
        )
        assert not np.array_equal(first.X, second.X)

    def test_does_not_change_global_numpy_rng_state(self):
        state = np.random.get_state()
        try:
            np.random.seed(123)
            before = np.random.get_state()
            toy_peptides(random_state=42)
            after = np.random.get_state()
            np.testing.assert_equal(before, after)
        finally:
            np.random.set_state(state)

    # ── C. Log2 model and sampling levels ──────────────────────────────

    def test_zero_variance_model_adds_all_component_means(
        self, deterministic_kwargs
    ):
        deterministic_kwargs.update(
            protein_abundance=(10, 0),
            peptide_response=(2, 0),
            biological_variation=(3, 0),
            loading_factor=(4, 0),
            measurement_noise=(5, 0),
        )
        data = toy_peptides(**deterministic_kwargs)
        assert data.shape == (6, 3)
        np.testing.assert_array_equal(data.X, np.full((6, 3), 2.0**24))

    def test_protein_abundance_is_shared_by_peptides_and_samples(
        self, deterministic_kwargs
    ):
        deterministic_kwargs["protein_abundance"] = (20, 2)
        data = toy_peptides(**deterministic_kwargs)
        log2_values = np.log2(data.X)
        np.testing.assert_array_equal(log2_values[:, 0], log2_values[:, 1])
        np.testing.assert_array_equal(
            log2_values, np.tile(log2_values[0], (6, 1))
        )
        assert log2_values[0, 0] != log2_values[0, 2]

    def test_peptide_responses_are_fixed_across_samples(
        self, deterministic_kwargs
    ):
        deterministic_kwargs["peptide_response"] = (0, 1)
        data = toy_peptides(**deterministic_kwargs)
        np.testing.assert_array_equal(data.X, np.tile(data.X[0], (6, 1)))
        assert np.unique(data.X[0]).size == 3

    def test_biological_variation_is_shared_only_within_protein_sample(
        self, deterministic_kwargs
    ):
        deterministic_kwargs["biological_variation"] = (0, 1)
        data = toy_peptides(**deterministic_kwargs)
        np.testing.assert_array_equal(data.X[:, 0], data.X[:, 1])
        assert np.unique(data.X[:, 0]).size == 6
        assert not np.array_equal(data.X[:, 0], data.X[:, 2])

    def test_loading_factor_is_shared_by_all_peptides_in_each_sample(
        self, deterministic_kwargs
    ):
        deterministic_kwargs["loading_factor"] = (0, 1)
        data = toy_peptides(**deterministic_kwargs)
        np.testing.assert_array_equal(data.X[:, 0], data.X[:, 1])
        np.testing.assert_array_equal(data.X[:, 0], data.X[:, 2])
        assert np.unique(data.X[:, 0]).size == 6

    def test_measurement_noise_is_drawn_per_observation(
        self, deterministic_kwargs
    ):
        deterministic_kwargs["measurement_noise"] = (0, 1)
        data = toy_peptides(**deterministic_kwargs)
        assert np.unique(data.X).size == data.X.size

    @pytest.mark.parametrize(
        "component",
        [
            "protein_abundance",
            "peptide_response",
            "biological_variation",
            "loading_factor",
            "measurement_noise",
        ],
    )
    def test_distribution_pairs_use_standard_deviation_not_variance(
        self, deterministic_kwargs, component
    ):
        deterministic_kwargs.update(
            n_proteins=100,
            n_peps_per_protein=2,
            groups=["only"] * 100,
            effect_size=None,
        )
        mean = 20 if component == "protein_abundance" else 3
        deterministic_kwargs[component] = (mean, 2)
        data = toy_peptides(**deterministic_kwargs)
        draws = np.unique(np.log2(data.X))
        expected_mean = 20 if component == "protein_abundance" else 23
        assert abs(draws.mean() - expected_mean) < 0.6
        assert abs(draws.std(ddof=1) - 2) < 0.5

    # ── D. Group effects ───────────────────────────────────────────────

    def test_scalar_effect_applies_to_every_target_and_contrast(
        self, deterministic_kwargs
    ):
        cases = {-2: 18, 0: 20, 2: 22}
        for effect, treatment_log2 in cases.items():
            deterministic_kwargs["effect_size"] = effect
            data = toy_peptides(**deterministic_kwargs)
            expected = np.full((6, 3), treatment_log2, dtype=float)
            expected[[0, 3]] = 20
            np.testing.assert_allclose(np.log2(data.X), expected)

    def test_protein_effect_vector_broadcasts_across_contrasts(
        self, deterministic_kwargs
    ):
        deterministic_kwargs.update(
            effect_size=[1, -2], differential_fraction=0
        )
        data = toy_peptides(**deterministic_kwargs)
        expected = np.array(
            [
                [20, 20, 20],
                [21, 21, 18],
                [21, 21, 18],
                [20, 20, 20],
                [21, 21, 18],
                [21, 21, 18],
            ]
        )
        assert data.shape == expected.shape
        np.testing.assert_allclose(np.log2(data.X), expected)

    def test_effect_matrix_rows_follow_encountered_contrast_order(
        self, deterministic_kwargs
    ):
        deterministic_kwargs["effect_size"] = [[1, -2], [-3, 4]]
        data = toy_peptides(**deterministic_kwargs)
        expected = np.array(
            [
                [20, 20, 20],
                [21, 21, 18],
                [17, 17, 24],
                [20, 20, 20],
                [21, 21, 18],
                [17, 17, 24],
            ]
        )
        assert data.shape == expected.shape
        np.testing.assert_allclose(np.log2(data.X), expected)

    def test_peptide_effect_matrix_can_describe_multiple_profiles(
        self, deterministic_kwargs
    ):
        deterministic_kwargs.update(
            effect_level="peptide", effect_size=[[1, -1, 0], [-2, 2, 3]]
        )
        data = toy_peptides(**deterministic_kwargs)
        expected = np.array(
            [
                [20, 20, 20],
                [21, 19, 20],
                [18, 22, 23],
                [20, 20, 20],
                [21, 19, 20],
                [18, 22, 23],
            ]
        )
        assert data.shape == expected.shape
        np.testing.assert_allclose(np.log2(data.X), expected)

    def test_single_group_has_no_contrasts_or_effects(
        self, deterministic_kwargs
    ):
        deterministic_kwargs.update(groups=["only"] * 3, effect_size=None)
        data = toy_peptides(**deterministic_kwargs)
        truth = data.uns["toy_peptides"]
        assert truth["reference_group"] == "only"
        assert len(truth["contrast_groups"]) == 0
        assert truth["effects"].shape == (0, 2)
        np.testing.assert_array_equal(data.X, np.full((3, 3), 2.0**20))

    def test_automatic_effects_approximate_default_mixture(self):
        data = toy_peptides(
            n_proteins=1500,
            n_peps_per_protein=1,
            groups=["z", "a", "b", "c"],
            missing_rate=0,
            log2_detection_limit=None,
            random_state=42,
        )
        effects = data.uns["toy_peptides"]["effects"]
        assert effects.shape == (3, 1500)
        assert set(np.unique(effects)) == {-1.5, 0, 1.5}
        for effect, probability in {0: 0.8, 1.5: 0.1, -1.5: 0.1}.items():
            assert abs(np.mean(effects == effect) - probability) < 0.03
        assert not np.array_equal(effects[0], effects[1])

    # ── E. Missingness and aggregation ─────────────────────────────────

    def test_complete_random_deletion_retains_all_axes(self):
        data = toy_peptides(missing_rate=1, log2_detection_limit=None)
        assert data.shape == (10, 100)
        assert np.isnan(data.X).all()
        assert check_proteodata(data) == (True, "peptide")

    def test_threshold_censoring_is_strictly_below_limit(
        self, deterministic_kwargs
    ):
        cases = {19: False, 20: False, 21: True}
        for threshold, all_missing in cases.items():
            deterministic_kwargs["log2_detection_limit"] = threshold
            data = toy_peptides(**deterministic_kwargs)
            np.testing.assert_array_equal(
                np.isnan(data.X), np.full((6, 3), all_missing)
            )
            if not all_missing:
                np.testing.assert_array_equal(data.X, np.full((6, 3), 2.0**20))

    def test_combined_missingness_is_union_of_independent_masks(self):
        kwargs = {"random_state": 42}
        complete = toy_peptides(
            **kwargs, missing_rate=0, log2_detection_limit=None
        )
        deleted = toy_peptides(
            **kwargs, missing_rate=0.3, log2_detection_limit=None
        )
        censored = toy_peptides(
            **kwargs, missing_rate=0, log2_detection_limit=20
        )
        combined = toy_peptides(
            **kwargs, missing_rate=0.3, log2_detection_limit=20
        )
        expected = np.isnan(deleted.X) | np.isnan(censored.X)
        assert np.isfinite(complete.X).all()
        assert (np.isnan(deleted.X) & ~np.isnan(censored.X)).any()
        assert (np.isnan(censored.X) & ~np.isnan(deleted.X)).any()
        np.testing.assert_array_equal(np.isnan(combined.X), expected)
        np.testing.assert_array_equal(
            combined.X[~expected], complete.X[~expected]
        )
        np.testing.assert_array_equal(
            np.isnan(censored.X), np.log2(complete.X) < 20
        )

    def test_random_missing_fraction_is_approximately_requested(self):
        data = toy_peptides(
            n_proteins=1000,
            n_peps_per_protein=2,
            missing_rate=0.3,
            log2_detection_limit=None,
            random_state=42,
        )
        assert abs(np.isnan(data.X).mean() - 0.3) < 0.03

    def test_protein_output_is_median_of_observed_raw_peptides(self):
        kwargs = {
            "n_proteins": 3,
            "n_peps_per_protein": [4, 3, 2],
            "protein_ids": ["Z", "A", "M"],
            "gene_ids": ["GZ", "GA", "GM"],
            "missing_rate": 0.25,
            "log2_detection_limit": 19,
            "random_state": 42,
        }
        peptides = toy_peptides(**kwargs)
        proteins = toy_peptides(**kwargs, level="protein")
        expected = np.full((peptides.n_obs, 3), np.nan)
        for column, protein_id in enumerate(kwargs["protein_ids"]):
            block = peptides.X[:, peptides.var["protein_id"] == protein_id]
            for row, values in enumerate(block):
                observed = values[~np.isnan(values)]
                if observed.size:
                    expected[row, column] = np.median(observed)
        assert proteins.shape == expected.shape
        assert isinstance(proteins.X, np.ndarray)
        assert proteins.X.dtype == np.float64
        np.testing.assert_allclose(proteins.X, expected, equal_nan=True)
        assert check_proteodata(proteins) == (True, "protein")

    def test_even_peptide_median_is_computed_on_raw_scale(
        self, deterministic_kwargs
    ):
        deterministic_kwargs.update(
            n_proteins=1,
            n_peps_per_protein=2,
            groups=["z", "a"],
            effect_level="peptide",
            effect_size=[0, 2],
        )
        data = toy_peptides(**deterministic_kwargs, level="protein")
        assert data.shape == (2, 1)
        np.testing.assert_allclose(data.X[:, 0], [2.0**20, 2.5 * 2.0**20])

    def test_entirely_censored_protein_remains_nan(self, deterministic_kwargs):
        deterministic_kwargs.update(
            groups=["z", "a"], effect_size=[-2, 2], log2_detection_limit=21
        )
        data = toy_peptides(**deterministic_kwargs, level="protein")
        assert data.shape == (2, 2)
        assert np.isnan(data.X[0]).all()
        assert np.isnan(data.X[:, 0]).all()
        assert data.X[1, 1] == 2.0**22

    # ── F. Model sanity checks ─────────────────────────────────────────

    def test_raw_mean_variance_association_disappears_on_log2_scale(self):
        data = toy_peptides(
            n_proteins=200,
            n_peps_per_protein=3,
            groups=["z"] * 40 + ["a"] * 40,
            effect_size=0,
            missing_rate=0,
            log2_detection_limit=None,
            random_state=42,
        )
        for group in ("z", "a"):
            raw = data.X[data.obs["group"] == group]
            log2_values = np.log2(raw)
            raw_correlation = spearmanr(
                raw.mean(axis=0), raw.var(axis=0, ddof=1)
            ).statistic
            log2_correlation = spearmanr(
                log2_values.mean(axis=0), log2_values.var(axis=0, ddof=1)
            ).statistic
            assert raw_correlation > 0.8
            assert abs(log2_correlation) < 0.2

    # ── G. Representative invalid inputs ───────────────────────────────

    @pytest.mark.parametrize(
        "key, value",
        [
            ("n_proteins", 0),
            ("n_peps_per_protein", [1]),
            ("measurement_noise", (0, -1)),
            ("missing_rate", 1.1),
            ("effect_size", [1, 2, 3]),
        ],
    )
    def test_rejects_invalid_model_parameters(self, key, value):
        kwargs = {"n_proteins": 2, key: value}
        with pytest.raises((TypeError, ValueError), match=key):
            toy_peptides(**kwargs)
