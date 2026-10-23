"""Tests for ``pr.tl.proteoform_group_inference``.

The function is a thin dispatcher over the currently recommended
proteoform inference method. These tests check that it:

- validates its input
- selects the right method
- gives exactly the same result as running that method's individual
  ``pr.tl`` steps by hand (for ``"copf"``: correlations, dendrograms,
  clusters, scores)
- honours the shared ``inplace`` / ``verbose`` conventions
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import proteopy as pr

N_OBS = 12


# --------------------------------------------------------------------
# Fixtures and helpers
# --------------------------------------------------------------------


@pytest.fixture
def peptide_adata():
    """Peptide-level data with one proteoform and one uniform protein.

    PROT_A carries two groups of three peptides with opposite profiles;
    PROT_B has six peptides sharing one profile. Peptide IDs are
    sequence-like because `pairwise_peptide_correlations` splits peptide
    pairs on underscores.
    """
    rng = np.random.default_rng(0)
    base = np.linspace(1.0, 10.0, N_OBS)
    profiles = {
        "PROT_A": [base] * 3 + [base[::-1]] * 3,
        "PROT_B": [base] * 6,
    }

    columns, var_rows = [], []
    for protein, protein_profiles in profiles.items():
        for i, profile in enumerate(protein_profiles):
            columns.append(profile + rng.normal(0, 0.3, N_OBS) + 20)
            var_rows.append(
                {
                    "peptide_id": f"{protein[-1]}PEPTIDE{'ACDEFG'[i]}K",
                    "protein_id": protein,
                }
            )

    var = pd.DataFrame(var_rows)
    var.index = var["peptide_id"].to_numpy()
    obs = pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(N_OBS)],
            "batch": ["b1"] * (N_OBS // 2) + ["b2"] * (N_OBS // 2),
        }
    )
    obs.index = obs["sample_id"].to_numpy()
    return ad.AnnData(X=np.column_stack(columns), obs=obs, var=var)


def run_copf_steps(adata, batch_key=None, min_score=0.1, min_adj_pval=0.1):
    pr.tl.pairwise_peptide_correlations(adata, batch_key=batch_key)
    pr.tl.peptide_dendograms_by_correlation(
        adata, method="agglomerative-hierarchical-clustering"
    )
    pr.tl.peptide_clusters_from_dendograms(
        adata, n_clusters=2, min_peptides_per_cluster=2
    )
    pr.tl.proteoform_scores(
        adata, min_score=min_score, min_pval_adj=min_adj_pval
    )


# --------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------


def test_protein_level_raises(peptide_adata):
    protein = peptide_adata[:, :1].copy()
    protein.var = pd.DataFrame({"protein_id": ["PROT_A"]}, index=["PROT_A"])
    with pytest.raises(ValueError, match="peptide-level"):
        pr.tl.proteoform_group_inference(protein)


def test_unknown_method_raises(peptide_adata):
    with pytest.raises(ValueError, match="Unknown method"):
        pr.tl.proteoform_group_inference(peptide_adata, method="foo")


def test_missing_batch_key_raises(peptide_adata):
    with pytest.raises(KeyError, match="batch_key"):
        pr.tl.proteoform_group_inference(peptide_adata, batch_key="missing")


@pytest.mark.parametrize("arg", ["min_score", "min_adj_pval"])
def test_non_numeric_threshold_raises(peptide_adata, arg):
    with pytest.raises(TypeError, match=arg):
        pr.tl.proteoform_group_inference(peptide_adata, **{arg: "0.1"})


@pytest.mark.parametrize("min_adj_pval", [0, -0.1, 1.5])
def test_min_adj_pval_out_of_range_raises(peptide_adata, min_adj_pval):
    with pytest.raises(ValueError, match="min_adj_pval"):
        pr.tl.proteoform_group_inference(
            peptide_adata,
            min_adj_pval=min_adj_pval,
        )


# --------------------------------------------------------------------
# Method dispatch
# --------------------------------------------------------------------


def test_default_method_is_copf(peptide_adata):
    expected = peptide_adata.copy()
    pr.tl.proteoform_group_inference(expected, method="copf")

    pr.tl.proteoform_group_inference(peptide_adata)

    pd.testing.assert_frame_equal(peptide_adata.var, expected.var)


# --------------------------------------------------------------------
# COPF integration: parity with the individual pr.tl steps
# --------------------------------------------------------------------


@pytest.mark.parametrize("batch_key", [None, "batch"])
def test_copf_matches_individual_steps(peptide_adata, batch_key):
    expected = peptide_adata.copy()
    run_copf_steps(expected, batch_key=batch_key)

    pr.tl.proteoform_group_inference(
        peptide_adata, method="copf", batch_key=batch_key
    )

    pd.testing.assert_frame_equal(peptide_adata.var, expected.var)
    pd.testing.assert_frame_equal(
        peptide_adata.uns["pairwise_peptide_correlations"],
        expected.uns["pairwise_peptide_correlations"],
    )
    assert ("pairwise_peptide_correlations_by_batch" in peptide_adata.uns) == (
        batch_key is not None
    )


def test_thresholds_are_forwarded(peptide_adata):
    expected = peptide_adata.copy()
    run_copf_steps(expected, min_score=0.5, min_adj_pval=0.01)

    pr.tl.proteoform_group_inference(
        peptide_adata, method="copf", min_score=0.5, min_adj_pval=0.01
    )

    pd.testing.assert_series_equal(
        peptide_adata.var["is_proteoform"], expected.var["is_proteoform"]
    )


def test_detects_proteoform(peptide_adata):
    pr.tl.proteoform_group_inference(peptide_adata, method="copf")

    is_pf = peptide_adata.var.groupby("protein_id")["is_proteoform"].first()
    assert is_pf["PROT_A"] == 1
    assert is_pf["PROT_B"] != 1


# --------------------------------------------------------------------
# Output: return value, in-place modification and verbosity
# --------------------------------------------------------------------


def test_inplace_true_returns_none(peptide_adata):
    assert pr.tl.proteoform_group_inference(peptide_adata) is None
    assert "is_proteoform" in peptide_adata.var.columns


def test_inplace_false_returns_copy(peptide_adata):
    original_var = peptide_adata.var.copy()

    out = pr.tl.proteoform_group_inference(
        peptide_adata, method="copf", inplace=False
    )

    assert isinstance(out, ad.AnnData)
    assert "is_proteoform" in out.var.columns
    pd.testing.assert_frame_equal(peptide_adata.var, original_var)
    assert "pairwise_peptide_correlations" not in peptide_adata.uns


def test_verbose_prints(peptide_adata, capsys):
    pr.tl.proteoform_group_inference(peptide_adata, verbose=True)
    out = capsys.readouterr().out
    assert "copf" in out
    assert "called as having proteoforms" in out
