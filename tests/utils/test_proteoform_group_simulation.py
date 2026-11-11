"""Tests for ``pr.utils.proteoform_group_simulator`` and
``pr.utils.proteoform_simulation_metrics``."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import proteopy as pr
from proteopy.utils.anndata import is_proteodata

N_OBS = 8
AA = "ACDEFGHIKLMNPQRSTVWY"


# --------------------------------------------------------------------
# Fixtures and helpers
# --------------------------------------------------------------------


def make_peptide_adata(peptides_per_protein, seed=0):
    """Peptide-level data, one protein per entry of the given counts."""
    rng = np.random.default_rng(seed)
    var_rows = []
    for p, n_peptides in enumerate(peptides_per_protein):
        for i in range(n_peptides):
            var_rows.append(
                {
                    "peptide_id": f"{AA[p // 20]}{AA[p % 20]}PEP{AA[i]}K",
                    "protein_id": f"PROT{p:03d}",
                }
            )
    var = pd.DataFrame(var_rows)
    var.index = var["peptide_id"].to_numpy()
    obs = pd.DataFrame({"sample_id": [f"s{i}" for i in range(N_OBS)]})
    obs.index = obs["sample_id"].to_numpy()
    X = rng.normal(20, 1, (N_OBS, len(var)))
    return ad.AnnData(X=X, obs=obs, var=var)


@pytest.fixture
def adata():
    # 30 proteins with 2-5 peptides, 10 with a single peptide
    return make_peptide_adata([2, 3, 4, 5] * 7 + [3, 4] + [1] * 10)


def protein_types(sim):
    return sim.var.groupby("protein_id")["simulation_type"].first()


def simulate(adata, **kwargs):
    return list(pr.utils.proteoform_group_simulator(adata, **kwargs))


def assert_sims_equal(a, b):
    pd.testing.assert_frame_equal(a.var, b.var)
    np.testing.assert_array_equal(a.X, b.X)
    assert a.uns["proteoform_simulation"] == b.uns["proteoform_simulation"]


# --------------------------------------------------------------------
# Simulator: iteration and reproducibility
# --------------------------------------------------------------------


def test_default_yields_single_item(adata):
    sims = simulate(adata, n_mixed=3, n_unmixed=3, random_state=0)
    assert len(sims) == 1


def test_yields_n_iterations(adata):
    sims = simulate(
        adata, n_iterations=4, n_mixed=3, n_unmixed=3, random_state=0
    )
    assert len(sims) == 4
    assert [s.uns["proteoform_simulation"]["iteration"] for s in sims] == [
        0,
        1,
        2,
        3,
    ]


def test_returns_single_pass_generator(adata):
    gen = pr.utils.proteoform_group_simulator(
        adata, n_iterations=2, n_mixed=3, n_unmixed=3, random_state=0
    )
    assert len(list(gen)) == 2
    assert list(gen) == []


def test_same_random_state_is_reproducible(adata):
    kwargs = dict(n_iterations=3, n_mixed=3, n_unmixed=3, random_state=7)
    for a, b in zip(simulate(adata, **kwargs), simulate(adata, **kwargs)):
        assert_sims_equal(a, b)


def test_item_independent_of_n_iterations(adata):
    kwargs = dict(n_mixed=3, n_unmixed=3, random_state=7)
    short = simulate(adata, n_iterations=2, **kwargs)
    long = simulate(adata, n_iterations=5, **kwargs)
    for a, b in zip(short, long):
        assert_sims_equal(a, b)


def test_iterations_differ(adata):
    a, b = simulate(
        adata, n_iterations=2, n_mixed=3, n_unmixed=3, random_state=7
    )
    assert list(a.var["protein_id"]) != list(b.var["protein_id"])


def test_random_state_none_is_recorded_and_reproducible(adata):
    (sim,) = simulate(adata, n_mixed=3, n_unmixed=3)
    seed = sim.uns["proteoform_simulation"]["random_state"]
    (again,) = simulate(adata, n_mixed=3, n_unmixed=3, random_state=seed)
    assert_sims_equal(sim, again)


# --------------------------------------------------------------------
# Simulator: content of the simulated objects
# --------------------------------------------------------------------


def test_output_is_peptide_level_proteodata(adata):
    (sim,) = simulate(adata, n_mixed=3, n_unmixed=3, random_state=0)
    assert is_proteodata(sim) == (True, "peptide")


def test_protein_counts_per_type(adata):
    (sim,) = simulate(adata, n_mixed=4, n_unmixed=6, random_state=0)
    counts = protein_types(sim).value_counts()
    assert counts["mixed"] == 4
    assert counts["unmixed"] == 6


@pytest.mark.parametrize("n_mixed_peptides", [1, 2, 3])
def test_mixed_proteins_are_background_plus_donor_peptides(
    adata, n_mixed_peptides
):
    (sim,) = simulate(
        adata,
        n_mixed=5,
        n_unmixed=5,
        n_mixed_peptides=n_mixed_peptides,
        random_state=0,
    )
    orig_counts = adata.var["protein_id"].value_counts()
    mixed = sim.var[sim.var["simulation_type"] == "mixed"]
    for protein_id, df in mixed.groupby("protein_id"):
        background, donor = protein_id.split("_")
        src = df["source_protein_id"].value_counts()
        assert set(src.index) == {background, donor}
        assert src[background] == orig_counts[background]
        assert src[donor] == n_mixed_peptides


def test_unmixed_proteins_are_unchanged(adata):
    (sim,) = simulate(adata, n_mixed=3, n_unmixed=5, random_state=0)
    unmixed = sim.var[sim.var["simulation_type"] == "unmixed"]
    assert (unmixed["protein_id"] == unmixed["source_protein_id"]).all()
    for protein_id, df in unmixed.groupby("protein_id"):
        orig = adata.var.index[adata.var["protein_id"] == protein_id]
        assert set(df.index) == set(orig)


def test_background_and_unmixed_have_two_peptides(adata):
    sims = simulate(
        adata,
        n_iterations=5,
        n_mixed=10,
        n_unmixed=20,
        n_mixed_peptides=1,
        random_state=0,
    )
    for sim in sims:
        backgrounds = sim.var["protein_id"].str.split("_").str[0]
        n_bg_peptides = (
            sim.var[sim.var["source_protein_id"] == backgrounds]
            .groupby("protein_id")
            .size()
        )
        assert (n_bg_peptides >= 2).all()


def test_values_are_copied_from_adata(adata):
    (sim,) = simulate(adata, n_mixed=3, n_unmixed=3, random_state=0)
    np.testing.assert_array_equal(sim.X, adata[:, sim.var_names].X)


def test_uns_records_iteration_parameters(adata):
    (sim,) = simulate(
        adata, n_mixed=3, n_unmixed=4, n_mixed_peptides=2, random_state=5
    )
    assert sim.uns["proteoform_simulation"] == {
        "iteration": 0,
        "random_state": 5,
        "n_mixed": 3,
        "n_unmixed": 4,
        "n_mixed_peptides": 2,
    }


def test_inference_results_warn_and_are_kept(adata):
    adata.var["is_proteoform"] = 0
    adata.uns["dendograms"] = {}
    with pytest.warns(UserWarning, match="inference results"):
        gen = pr.utils.proteoform_group_simulator(
            adata, n_mixed=3, n_unmixed=3, random_state=0
        )
    sim = next(gen)
    assert "is_proteoform" in sim.var.columns
    assert "dendograms" in sim.uns


def test_sparse_input_warns_and_matches_dense(adata):
    (dense,) = simulate(adata, n_mixed=3, n_unmixed=3, random_state=0)
    adata.X = sparse.csr_matrix(adata.X)
    with pytest.warns(UserWarning, match="Sparse"):
        (sim,) = simulate(adata, n_mixed=3, n_unmixed=3, random_state=0)
    assert not sparse.issparse(sim.X)
    assert_sims_equal(sim, dense)


# --------------------------------------------------------------------
# Simulator: list parameters
# --------------------------------------------------------------------


LIST_PARAM_CASES = {
    "all_ints": (
        dict(n_mixed=2, n_unmixed=3, n_mixed_peptides=2),
        [(2, 3, 2)] * 3,
    ),
    "all_lists": (
        dict(
            n_mixed=[1, 2, 3],
            n_unmixed=[6, 5, 4],
            n_mixed_peptides=[1, 2, 3],
        ),
        [(1, 6, 1), (2, 5, 2), (3, 4, 3)],
    ),
    "mixed_lists_and_int": (
        dict(n_mixed=[1, 2, 3], n_unmixed=[6, 5, 4], n_mixed_peptides=2),
        [(1, 6, 2), (2, 5, 2), (3, 4, 2)],
    ),
}


@pytest.mark.parametrize("case", LIST_PARAM_CASES)
def test_list_parameters_per_iteration(adata, case):
    kwargs, expected = LIST_PARAM_CASES[case]
    sims = simulate(adata, n_iterations=3, random_state=0, **kwargs)
    for sim, (n_mixed, n_unmixed, n_mixed_peptides) in zip(sims, expected):
        params = sim.uns["proteoform_simulation"]
        assert params["n_mixed"] == n_mixed
        assert params["n_unmixed"] == n_unmixed
        assert params["n_mixed_peptides"] == n_mixed_peptides
        counts = protein_types(sim).value_counts()
        assert counts.get("mixed", 0) == n_mixed
        assert counts.get("unmixed", 0) == n_unmixed


def test_int_equals_repeated_list(adata):
    from_int = simulate(
        adata, n_iterations=2, n_mixed=3, n_unmixed=3, random_state=0
    )
    from_list = simulate(
        adata,
        n_iterations=2,
        n_mixed=[3, 3],
        n_unmixed=(3, 3),
        random_state=0,
    )
    for a, b in zip(from_int, from_list):
        assert_sims_equal(a, b)


# --------------------------------------------------------------------
# Simulator: validation (raised at call time, before iterating)
# --------------------------------------------------------------------


INVALID_ARGS = {
    "list_too_short": (dict(n_iterations=3, n_mixed=[1, 2]), ValueError),
    "list_too_long": (dict(n_iterations=1, n_unmixed=[1, 2]), ValueError),
    "zero_iterations": (dict(n_iterations=0), ValueError),
    "negative_n_mixed": (dict(n_mixed=-1), ValueError),
    "zero_n_mixed_peptides": (dict(n_mixed_peptides=0), ValueError),
    "both_zero": (dict(n_mixed=0, n_unmixed=0), ValueError),
    "bool_n_mixed": (dict(n_mixed=True), TypeError),
    "float_n_unmixed": (dict(n_unmixed=2.5), TypeError),
    "float_in_list": (dict(n_iterations=2, n_mixed=[1, 2.0]), TypeError),
    "negative_random_state": (dict(random_state=-1), ValueError),
    "too_many_proteins": (dict(n_mixed=20, n_unmixed=20), ValueError),
    "too_many_donors": (dict(n_mixed=8, n_mixed_peptides=5), ValueError),
}


@pytest.mark.parametrize("case", INVALID_ARGS)
def test_invalid_arguments_raise_at_call(adata, case):
    kwargs, error = INVALID_ARGS[case]
    kwargs = {"n_mixed": 3, "n_unmixed": 3, **kwargs}
    with pytest.raises(error):
        pr.utils.proteoform_group_simulator(adata, **kwargs)


def test_infeasible_iteration_is_named(adata):
    with pytest.raises(ValueError, match="Iteration 1"):
        pr.utils.proteoform_group_simulator(
            adata, n_iterations=2, n_mixed=[3, 30], n_unmixed=3
        )


def test_protein_level_input_raises():
    protein_adata = make_peptide_adata([1] * 10)
    protein_adata.var = protein_adata.var.drop(columns="peptide_id")
    protein_adata.var.index = protein_adata.var["protein_id"].to_numpy()
    with pytest.raises(ValueError, match="peptide-level"):
        pr.utils.proteoform_group_simulator(protein_adata)


# --------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------


def make_result_adata(proteins, uns=None):
    """Simulated-and-inferred AnnData from per-protein specifications.

    Each protein is ``(protein_id, simulation_type, called, score,
    pval_adj, sources, clusters)`` with one entry per peptide in
    ``sources`` and ``clusters``.
    """
    var_rows = []
    for p, protein in enumerate(proteins):
        protein_id, sim_type, called, score, pval, sources, clusters = protein
        for i, (source, cluster) in enumerate(zip(sources, clusters)):
            var_rows.append(
                {
                    "peptide_id": f"{AA[p]}PEP{AA[i]}K",
                    "protein_id": protein_id,
                    "source_protein_id": source,
                    "simulation_type": sim_type,
                    "cluster_id": cluster,
                    "proteoform_score": score,
                    "proteoform_score_pval_adj": pval,
                    "is_proteoform": called,
                }
            )
    var = pd.DataFrame(var_rows)
    var.index = var["peptide_id"].to_numpy()
    obs = pd.DataFrame({"sample_id": ["s0", "s1"]})
    obs.index = obs["sample_id"].to_numpy()
    adata = ad.AnnData(X=np.ones((2, len(var))), obs=obs, var=var)
    if uns is not None:
        adata.uns["proteoform_simulation"] = uns
    return adata


# Called, exact clustering / called, wrong clustering / not called,
# exact clustering; one called and one quiet unmixed protein.
# Scores and p-values rank the proteins differently:
#   score (higher first): G 0.95, A_B 0.9, C_D 0.8, E_F 0.1, H 0.05
#   p-value (lower first): A_B 0.01, C_D 0.02, G 0.03, E_F 0.5, H 0.9
MIXED_RECOVERED = ("A_B", "mixed", 1, 0.9, 0.01, "AABB", [0, 0, 1, 1])
MIXED_MISCLUSTERED = ("C_D", "mixed", 1, 0.8, 0.02, "CCDD", [0, 0, 0, 1])
MIXED_MISSED = ("E_F", "mixed", 0, 0.1, 0.5, "EEFF", [0, 0, 1, 1])
UNMIXED_CALLED = ("G", "unmixed", 1, 0.95, 0.03, "GGGG", [0, 0, 1, 1])
UNMIXED_QUIET = ("H", "unmixed", 0, 0.05, 0.9, "HHHH", [0, 0, 0, 0])

RANKING_KEYS = [
    f"{ranking}_{area}_{level}"
    for ranking in ["score", "pval"]
    for area in ["auroc", "auprc"]
    for level in ["call", "cluster"]
]


@pytest.fixture
def all_cases_adata():
    return make_result_adata(
        [
            MIXED_RECOVERED,
            MIXED_MISCLUSTERED,
            MIXED_MISSED,
            UNMIXED_CALLED,
            UNMIXED_QUIET,
        ]
    )


def test_metrics_keys():
    adata = make_result_adata([MIXED_RECOVERED, UNMIXED_QUIET])
    metrics, _ = pr.utils.proteoform_simulation_metrics(adata)
    assert list(metrics) == [
        "detection_rate_call",
        "background_detection_rate_call",
        "score_auroc_call",
        "score_auprc_call",
        "pval_auroc_call",
        "pval_auprc_call",
        "detection_rate_cluster",
        "background_detection_rate_cluster",
        "score_auroc_cluster",
        "score_auprc_cluster",
        "pval_auroc_cluster",
        "pval_auprc_cluster",
        "mean_ari",
        "cluster_match_rate",
    ]


def test_metrics_perfect_recovery():
    adata = make_result_adata([MIXED_RECOVERED, UNMIXED_QUIET])
    metrics, _ = pr.utils.proteoform_simulation_metrics(adata)
    expected = dict.fromkeys(metrics, 1.0)
    zero_keys = [
        "background_detection_rate_call",
        "background_detection_rate_cluster",
    ]
    expected.update(dict.fromkeys(zero_keys, 0.0))
    assert metrics == expected


def test_metrics_all_wrong():
    adata = make_result_adata([MIXED_MISSED, UNMIXED_CALLED])
    metrics, _ = pr.utils.proteoform_simulation_metrics(adata)
    expected = dict.fromkeys(metrics, 0.0)
    expected.update(
        {
            "background_detection_rate_call": 1.0,
            "background_detection_rate_cluster": 1.0,
            # Only two proteins: the mixed one sits at recall 1 with
            # precision 1/2
            "score_auprc_call": 0.5,
            "score_auprc_cluster": 0.5,
            "pval_auprc_call": 0.5,
            "pval_auprc_cluster": 0.5,
        }
    )
    # No called mixed proteins
    for key in ["mean_ari", "cluster_match_rate"]:
        assert np.isnan(metrics.pop(key)), key
        expected.pop(key)
    assert metrics == expected


CALL_LEVEL_EXPECTED = {
    "detection_rate_call": 2 / 3,
    "background_detection_rate_call": 1 / 2,
    # Each mixed protein outranks H but not G
    "score_auroc_call": 3 / 6,
    # Hits at G, A_B, C_D, E_F, H: recall 1/3, 2/3, 1 at precision
    # 1/2, 2/3, 3/4
    "score_auprc_call": (1 / 2 + 2 / 3 + 3 / 4) / 3,
    # A_B and C_D outrank G and H, E_F outranks only H
    "pval_auroc_call": 5 / 6,
    "pval_auprc_call": (1 + 1 + 3 / 4) / 3,
}


def test_metrics_call_level(all_cases_adata):
    metrics, _ = pr.utils.proteoform_simulation_metrics(all_cases_adata)
    for key, value in CALL_LEVEL_EXPECTED.items():
        assert metrics[key] == pytest.approx(value), key


# Cluster-level positives: A_B and E_F (mixed, ARI 1).
# Background: C_D (mixed, ARI 0), G and H (unmixed).
CLUSTER_LEVEL_EXPECTED = {
    # A_B called, E_F not
    "detection_rate_cluster": 1 / 2,
    # C_D and G called, H not
    "background_detection_rate_cluster": 2 / 3,
    # A_B outranks C_D and H, E_F only H
    "score_auroc_cluster": 3 / 6,
    # Hits at G, A_B, C_D, E_F, H: recall 1/2, 1 at precision 1/2, 2/4
    "score_auprc_cluster": (1 / 2 + 2 / 4) / 2,
    # A_B outranks C_D, G and H, E_F only H
    "pval_auroc_cluster": 4 / 6,
    # Hits at A_B, C_D, G, E_F, H: recall 1/2, 1 at precision 1, 2/4
    "pval_auprc_cluster": (1 + 2 / 4) / 2,
    # Called mixed proteins: A_B (ARI 1), C_D (ARI 0)
    "mean_ari": 1 / 2,
    "cluster_match_rate": 1 / 2,
}


def test_metrics_cluster_level(all_cases_adata):
    metrics, _ = pr.utils.proteoform_simulation_metrics(all_cases_adata)
    for key, value in CLUSTER_LEVEL_EXPECTED.items():
        assert metrics[key] == pytest.approx(value), key


def test_metrics_misclustered_call_is_cluster_background_detection():
    adata = make_result_adata([MIXED_MISCLUSTERED, UNMIXED_QUIET])
    metrics, _ = pr.utils.proteoform_simulation_metrics(adata)
    assert metrics["detection_rate_call"] == 1.0
    assert metrics["background_detection_rate_call"] == 0.0
    assert metrics["background_detection_rate_cluster"] == 1 / 2
    # No cluster-level positives
    for key in [
        "detection_rate_cluster",
        "score_auroc_cluster",
        "pval_auprc_cluster",
    ]:
        assert np.isnan(metrics[key]), key


def test_metrics_pval_ranks_lower_first():
    # Same order as the score: a correct direction gives AUROC 1
    adata = make_result_adata([MIXED_RECOVERED, UNMIXED_QUIET])
    adata.var["proteoform_score_pval_adj"] = np.where(
        adata.var["simulation_type"] == "mixed", 0.001, 0.999
    )
    metrics, _ = pr.utils.proteoform_simulation_metrics(adata)
    assert metrics["pval_auroc_call"] == 1.0
    assert metrics["pval_auprc_call"] == 1.0


def test_metrics_nan_scores_rank_lowest():
    mixed_unscored = (
        "A_B",
        "mixed",
        np.nan,
        np.nan,
        np.nan,
        "AABB",
        [0, 0, 1, 1],
    )
    adata = make_result_adata([mixed_unscored, UNMIXED_QUIET])
    metrics, _ = pr.utils.proteoform_simulation_metrics(adata)
    assert metrics["detection_rate_call"] == 0.0
    for ranking in ["score", "pval"]:
        assert metrics[f"{ranking}_auroc_call"] == 0.0
        assert metrics[f"{ranking}_auroc_cluster"] == 0.0


def test_metrics_single_class_gives_nan():
    adata = make_result_adata([UNMIXED_CALLED, UNMIXED_QUIET])
    metrics, _ = pr.utils.proteoform_simulation_metrics(adata)
    nan_keys = [
        "detection_rate_call",
        "detection_rate_cluster",
        "mean_ari",
        "cluster_match_rate",
    ]
    for key in [*nan_keys, *RANKING_KEYS]:
        assert np.isnan(metrics[key]), key


def test_metrics_exclude_simulation_parameters():
    params = {"iteration": 2, "random_state": 1, "n_mixed": 1}
    adata = make_result_adata([MIXED_RECOVERED, UNMIXED_QUIET], uns=params)
    metrics, _ = pr.utils.proteoform_simulation_metrics(adata)
    assert not set(params) & set(metrics)


def test_metrics_custom_keys():
    adata = make_result_adata([MIXED_RECOVERED, UNMIXED_QUIET])
    adata.var = adata.var.rename(
        columns={
            "proteoform_score": "score",
            "proteoform_score_pval_adj": "pval",
            "is_proteoform": "call",
            "cluster_id": "cluster",
        }
    )
    metrics, _ = pr.utils.proteoform_simulation_metrics(
        adata,
        score_key="score",
        pval_key="pval",
        call_key="call",
        cluster_key="cluster",
    )
    assert metrics["detection_rate_cluster"] == 1.0
    assert metrics["pval_auroc_call"] == 1.0


@pytest.mark.parametrize(
    "column", ["is_proteoform", "proteoform_score_pval_adj"]
)
def test_metrics_missing_column_raises(column):
    adata = make_result_adata([MIXED_RECOVERED, UNMIXED_QUIET])
    adata.var = adata.var.drop(columns=column)
    with pytest.raises(KeyError, match=column):
        pr.utils.proteoform_simulation_metrics(adata)


def test_metrics_returns_per_protein_results():
    adata = make_result_adata([MIXED_MISCLUSTERED, UNMIXED_CALLED])
    _, results = pr.utils.proteoform_simulation_metrics(adata)
    assert list(results.index) == ["C_D", "G"]
    assert results.index.name == "protein_id"
    assert list(results.columns) == [
        "simulation_type",
        "called",
        "score",
        "pval",
        "ari",
        "cluster_match",
    ]
    assert results["simulation_type"].tolist() == ["mixed", "unmixed"]
    assert results["called"].tolist() == [True, True]
    assert results["score"].tolist() == [0.8, 0.95]
    assert results["pval"].tolist() == [0.02, 0.03]
    assert results.loc["C_D", "ari"] == pytest.approx(0.0)
    assert np.isnan(results.loc["G", "ari"])
    assert results["cluster_match"].tolist() == [False, False]
