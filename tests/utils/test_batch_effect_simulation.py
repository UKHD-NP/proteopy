"""Tests for ``pr.utils.apply_peptide_batch_effects``."""

import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import proteopy as pr

N_PER_BATCH = 6
BATCHES = ["b1", "b2", "b3"]


# --------------------------------------------------------------------
# Fixtures and helpers
# --------------------------------------------------------------------


def make_peptide_adata(peptides_per_protein, seed=0):
    """Raw-scale peptide-level data with three batches."""
    rng = np.random.default_rng(seed)
    var_rows = []
    for p, n_peptides in enumerate(peptides_per_protein):
        for i in range(n_peptides):
            var_rows.append(
                {
                    "peptide_id": f"PROT{p:03d}_PEP{i}",
                    "protein_id": f"PROT{p:03d}",
                }
            )
    var = pd.DataFrame(var_rows)
    var.index = var["peptide_id"].to_numpy()
    n_obs = N_PER_BATCH * len(BATCHES)
    obs = pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(n_obs)],
            "batch": np.repeat(BATCHES, N_PER_BATCH),
        }
    )
    obs.index = obs["sample_id"].to_numpy()
    X = 2.0 ** rng.normal(20, 1, (n_obs, len(var)))
    return ad.AnnData(X=X, obs=obs, var=var)


@pytest.fixture
def adata():
    # 10 proteins with 3-5 peptides, 2 with only 2 peptides
    return make_peptide_adata([3, 4, 5, 3, 4, 5, 3, 4, 5, 3, 2, 2])


def apply(adata, **kwargs):
    kwargs = {"batch_key": "batch", "batches": ["b2"], **kwargs}
    return pr.utils.apply_peptide_batch_effects(adata, inplace=False, **kwargs)


def ratio(out, adata):
    return out.X / adata.X


# --------------------------------------------------------------------
# Shift
# --------------------------------------------------------------------


def test_shifts_only_flagged_peptides_in_affected_batch(adata):
    out = apply(adata, log2fc=-1.0, random_state=0)
    flagged = out.var["batch_effect"].to_numpy()
    in_batch = (out.obs["batch"] == "b2").to_numpy()
    r = ratio(out, adata)

    np.testing.assert_allclose(r[np.ix_(in_batch, flagged)], 0.5)
    np.testing.assert_allclose(r[np.ix_(~in_batch, flagged)], 1.0)
    np.testing.assert_allclose(r[:, ~flagged], 1.0)


def test_log2fc_values(adata):
    expected = {-1.0: 0.5, 1.0: 2.0, 2.0: 4.0, 0.0: 1.0}
    for log2fc, factor in expected.items():
        out = apply(adata, log2fc=log2fc, random_state=0)
        flagged = out.var["batch_effect"].to_numpy()
        in_batch = (out.obs["batch"] == "b2").to_numpy()
        np.testing.assert_allclose(
            ratio(out, adata)[np.ix_(in_batch, flagged)], factor
        )


def test_log_space_adds_log2fc(adata):
    logged = adata.copy()
    logged.X = np.log2(adata.X)
    for log2fc in (-1.0, 0.5, 2.0):
        out = apply(logged, log2fc=log2fc, log_space=True, random_state=0)
        flagged = out.var["batch_effect"].to_numpy()
        in_batch = (out.obs["batch"] == "b2").to_numpy()
        diff = out.X - logged.X
        np.testing.assert_allclose(diff[np.ix_(in_batch, flagged)], log2fc)
        np.testing.assert_allclose(diff[np.ix_(~in_batch, flagged)], 0.0)
        np.testing.assert_allclose(diff[:, ~flagged], 0.0)


def test_log_space_matches_linear_space(adata):
    linear = apply(adata, log2fc=-1.5, random_state=0)
    logged = adata.copy()
    logged.X = np.log2(adata.X)
    out = apply(logged, log2fc=-1.5, log_space=True, random_state=0)
    np.testing.assert_allclose(out.X, np.log2(linear.X))


def test_n_peptides_per_affected_protein(adata):
    out = apply(adata, n_peptides=2, random_state=0)
    per_protein = out.var.groupby("protein_id")["batch_effect"].sum()
    # Proteins with exactly 2 peptides are not eligible
    assert (per_protein[["PROT010", "PROT011"]] == 0).all()
    assert (per_protein.drop(["PROT010", "PROT011"]) == 2).all()


def test_fraction_limits_affected_proteins(adata):
    out = apply(adata, fraction=0.5, random_state=0)
    per_protein = out.var.groupby("protein_id")["batch_effect"].sum()
    assert (per_protein > 0).sum() == 5
    assert out.uns["batch_effect_simulation"]["n_proteins_affected"] == 5


def test_preserves_within_batch_correlation(adata):
    out = apply(adata, log2fc=-2.0, random_state=0)
    for b in BATCHES:
        mask = (out.obs["batch"] == b).to_numpy()
        np.testing.assert_allclose(
            np.corrcoef(out.X[mask].T),
            np.corrcoef(adata.X[mask].T),
        )


def test_nan_and_zero_stay(adata):
    adata.X[0, 0] = np.nan
    adata.X[N_PER_BATCH, :] = 0.0
    out = apply(adata, random_state=0)
    assert np.isnan(out.X[0, 0])
    assert (out.X[N_PER_BATCH, :] == 0).all()


def test_missing_batch_label_not_affected(adata):
    adata.obs["batch"] = adata.obs["batch"].astype(object)
    adata.obs.iloc[N_PER_BATCH, adata.obs.columns.get_loc("batch")] = None
    out = apply(adata, random_state=0)
    np.testing.assert_allclose(out.X[N_PER_BATCH], adata.X[N_PER_BATCH])


# --------------------------------------------------------------------
# Batch selection
# --------------------------------------------------------------------


def test_int_batches_draws_that_many(adata):
    out = apply(adata, batches=2, random_state=0)
    affected = out.uns["batch_effect_simulation"]["batches"]
    assert len(affected) == 2
    assert set(affected) <= set(BATCHES)
    changed = ~np.isclose(ratio(out, adata), 1.0).all(axis=1)
    assert set(out.obs.loc[changed, "batch"]) == set(affected)


def test_numeric_batch_labels(adata):
    adata.obs["batch"] = np.repeat([1, 2, 3], N_PER_BATCH)
    out = apply(adata, batches=[3], random_state=0)
    assert out.uns["batch_effect_simulation"]["batches"] == [3]
    assert isinstance(out.uns["batch_effect_simulation"]["batches"][0], int)


def test_categorical_batch_key(adata):
    adata.obs["batch"] = pd.Categorical(adata.obs["batch"])
    out = apply(adata, batches=["b3"], random_state=0)
    in_batch = (out.obs["batch"] == "b3").to_numpy()
    flagged = out.var["batch_effect"].to_numpy()
    np.testing.assert_allclose(
        ratio(out, adata)[np.ix_(in_batch, flagged)], 0.5
    )


# --------------------------------------------------------------------
# Reproducibility, annotations and modes
# --------------------------------------------------------------------


def test_same_seed_same_result(adata):
    a = apply(adata, batches=1, random_state=3)
    b = apply(adata, batches=1, random_state=3)
    np.testing.assert_array_equal(a.X, b.X)
    assert a.uns["batch_effect_simulation"] == b.uns["batch_effect_simulation"]


def test_none_seed_is_stored_and_reproduces(adata):
    a = apply(adata, batches=1)
    seed = a.uns["batch_effect_simulation"]["random_state"]
    b = apply(adata, batches=1, random_state=seed)
    np.testing.assert_array_equal(a.X, b.X)


def test_uns_records_parameters(adata):
    out = apply(adata, n_peptides=1, log2fc=0.5, random_state=1)
    params = out.uns["batch_effect_simulation"]
    assert params["flavor"] == "subset_shift"
    assert params["batch_key"] == "batch"
    assert params["batches"] == ["b2"]
    assert params["n_peptides"] == 1
    assert params["log2fc"] == 0.5
    assert params["n_peptides_affected"] == out.var["batch_effect"].sum()


def test_inplace_modifies_and_returns_none(adata):
    X0 = adata.X.copy()
    assert (
        pr.utils.apply_peptide_batch_effects(
            adata, batch_key="batch", batches=["b1"], random_state=0
        )
        is None
    )
    assert "batch_effect" in adata.var
    assert not np.allclose(adata.X, X0)


def test_not_inplace_leaves_input(adata):
    X0 = adata.X.copy()
    apply(adata, random_state=0)
    np.testing.assert_array_equal(adata.X, X0)
    assert "batch_effect" not in adata.var


def test_sparse_input_warns_and_matches_dense(adata):
    dense = apply(adata, random_state=0)
    sp = adata.copy()
    sp.X = sparse.csr_matrix(sp.X)
    with pytest.warns(UserWarning, match="densified"):
        out = apply(sp, random_state=0)
    assert not sparse.issparse(out.X)
    np.testing.assert_allclose(out.X, dense.X)


def test_warns_when_log_data_treated_as_linear(adata):
    adata.X = np.log2(adata.X)
    with pytest.warns(UserWarning, match="looks log-transformed"):
        apply(adata, random_state=0)


def test_warns_when_linear_data_treated_as_log(adata):
    with pytest.warns(UserWarning, match="looks linear"):
        apply(adata, log_space=True, random_state=0)


def test_no_warning_for_raw_data(adata):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        apply(adata, random_state=0)


def test_works_on_simulated_data(adata):
    sim = next(
        pr.utils.proteoform_group_simulator(
            adata, n_mixed=3, n_unmixed=3, random_state=0
        )
    )
    pr.utils.apply_peptide_batch_effects(
        sim, batch_key="batch", batches=1, random_state=0
    )
    assert sim.var["batch_effect"].any()


# --------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"flavor": "peptide_mix"}, ValueError),
        ({"batch_key": "missing"}, KeyError),
        ({"batches": 0}, ValueError),
        ({"batches": 3}, ValueError),
        ({"batches": BATCHES}, ValueError),
        ({"batches": []}, ValueError),
        ({"batches": ["b1", "b1"]}, ValueError),
        ({"batches": ["b9"]}, ValueError),
        ({"batches": "b1"}, TypeError),
        ({"n_peptides": 0}, ValueError),
        ({"log2fc": np.inf}, ValueError),
        ({"fraction": 0}, ValueError),
        ({"fraction": 1.5}, ValueError),
        ({"log_space": 2}, TypeError),
        ({"log_space": None}, TypeError),
        ({"random_state": -1}, ValueError),
    ],
)
def test_invalid_arguments(adata, kwargs, error):
    with pytest.raises(error):
        apply(adata, **kwargs)


def test_single_batch_raises(adata):
    adata.obs["batch"] = "b1"
    with pytest.raises(ValueError, match="at least 2 batches"):
        apply(adata, batches=["b1"])


def test_protein_level_raises():
    var = pd.DataFrame({"protein_id": ["P1", "P2"]}, index=["P1", "P2"])
    obs = pd.DataFrame(
        {"sample_id": ["s1", "s2"], "batch": ["b1", "b2"]},
        index=["s1", "s2"],
    )
    prot = ad.AnnData(X=np.ones((2, 2)) * 1e6, obs=obs, var=var)
    with pytest.raises(ValueError, match="peptide-level"):
        pr.utils.apply_peptide_batch_effects(
            prot, batch_key="batch", batches=["b1"]
        )
