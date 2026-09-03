"""Tests for ``pr.tl.peptide_proximity``.

The reference-parity tests compare against CCprofiler's own output for
the Bludau et al. 2021 mouse tissue dataset, shipped under
``tests/data/mouse_tissue/`` -- see the README there for provenance.
Every input the function needs is reference output too, so these tests
pin the whole chain: FASTA -> peptide positions -> ranks -> per-cluster
permutation p-values -> the counts the publication reports.

Exact p-value equality is not achievable and is not asserted. The
reference draws its permutations with R's ``sample()`` under
``set.seed(123)``, a stream NumPy cannot generate, so two independent
1,000-draw estimates of the same p-value differ by sampling noise of
order ``1 / 1001``. What is asserted exactly is every discrete
outcome: which groups are tested, and which proteins a threshold
selects.
"""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.multitest import multipletests

import proteopy as pr
from proteopy.tl.peptide_proximity import (
    CCPROFILER_SEED,
    NOISE,
    _dispersion_exact,
    normalized_sd,
    peptide_position_ranks,
)

TEST_DIR = Path(__file__).parent.parent
DATA_DIR = TEST_DIR / "data" / "mouse_tissue"

FASTA = DATA_DIR / "uniprot_mouse-copf-proteins_subset.fasta"

# CCprofiler's outlier marker, which the shipped cluster assignment
# uses. `pr.tl.peptide_clusters_from_dendograms` writes 1e6 instead, so
# the fixtures translate; the two are not interchangeable.
RCOPF_NOISE = 100

# What the reference run produces on this dataset, and what the
# publication reports. Sources: Bludau et al. 2021 Fig. 6C and its
# Results text; `traces_proteoform-location_rcopf.tsv` for the run.
N_TESTS = 3161
N_SINGLE_PEPTIDE_CLUSTERS = 283
N_PROTEINS_TESTED = 1272
N_PROTEINS = 2885
N_SIGNIFICANT = 63
N_CLASSICAL = 7
N_PSEUDO_ONLY = 19
N_NEITHER = 37
N_NOISE_CLUSTERS_RCOPF = 900
THRESHOLD = 0.1

# Multi-peptide noise clusters that are tested (the reference's 900
# noise clusters less the 283 single-peptide ones it scores 1 as a
# placeholder).
N_NOISE_CLUSTERS_TESTED = N_NOISE_CLUSTERS_RCOPF - N_SINGLE_PEPTIDE_CLUSTERS

# One of the 19 the publication counts, and the reason the unified
# statistics skip missing clusters instead of propagating them: its
# evidence lives in a two-peptide cluster while its noise cluster holds
# a single peptide that no test can score.
UNIFIED_KEEPS = "P10852"

# The one peptide of the 24,534 whose sequence does not occur in its
# protein's canonical sequence. The reference keeps it, with an
# unresolved position; see `test_parity_needs_the_keep_policy`.
UNLOCATED = ("P35235", "VGQALLQGNTER")


# --------------------------------------------------------------------
# Fixtures: the reference dataset
# --------------------------------------------------------------------


@pytest.fixture(scope="module")
def cluster_assignment():
    """R's per-peptide cluster assignment, noise marker translated."""
    path = DATA_DIR / "traces_annotation_cluster-assignment_rcopf.tsv"
    frame = pd.read_csv(path, sep="\t")
    frame = frame.rename(columns={"id": "peptide_id"})
    frame = frame[
        [
            "protein_id",
            "peptide_id",
            "cluster",
            "PeptidePositionStart",
        ]
    ].copy()
    frame["cluster_id"] = (
        frame["cluster"].replace({RCOPF_NOISE: NOISE}).astype(float)
    )
    return frame


@pytest.fixture(scope="module")
def adata_rcopf(cluster_assignment):
    """The reference peptide set as an AnnData, in R's row order.

    Row order matters: peptides tied on start position keep it, and
    the shipped tables are still in the descending-identifier order
    the reference workflow leaves behind.
    """
    traces = pd.read_csv(DATA_DIR / "traces_pre-processed_rcopf.tsv", sep="\t")
    fractions = pd.read_csv(DATA_DIR / "fraction_annotation.tsv", sep="\t")

    var = cluster_assignment.set_index("peptide_id", drop=False)
    var.index.name = None
    traces = traces.set_index("id").loc[var["peptide_id"]]
    samples = [str(i) for i in fractions["id"]]

    obs = fractions.set_index("filename", drop=False)
    obs.index.name = None
    obs = obs.rename(columns={"filename": "sample_id"})

    return ad.AnnData(
        X=traces[samples].to_numpy(dtype=float).T,
        obs=obs[["sample_id"]],
        var=var[["peptide_id", "protein_id", "cluster_id"]],
    )


@pytest.fixture(scope="module")
def result(adata_rcopf):
    """One run at the defaults, shared by the read-only tests."""
    return pr.tl.peptide_proximity(
        adata_rcopf,
        FASTA,
        on_unlocated_peptide="keep",
        inplace=False,
    )


@pytest.fixture(scope="module")
def tests_table(result):
    return result.uns["peptide_proximity"]["tests"]


@pytest.fixture(scope="module")
def proteoform_to_cluster(cluster_assignment):
    """Map each reference ``proteoform_id`` to a ``cluster_id``.

    Derived from the shipped per-peptide assignment rather than from
    the identifier suffix, so the reference p-values are joined by
    cluster membership and no label convention is assumed.
    """
    path = DATA_DIR / "traces_proteoform-assignment_rcopf.tsv"
    assignment = pd.read_csv(path, sep="\t")
    merged = assignment.merge(
        cluster_assignment[["protein_id", "peptide_id", "cluster_id"]],
        on=["protein_id", "peptide_id"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    assert (merged["_merge"] == "both").all()
    grouped = merged.groupby(["protein_id", "proteoform_id"])["cluster_id"]
    return grouped.agg(["nunique", "first"])


@pytest.fixture(scope="module")
def location_ref(proteoform_to_cluster):
    """Reference proximity p-values, keyed by (protein, cluster)."""
    path = DATA_DIR / "traces_proteoform-location_rcopf.tsv"
    ref = pd.read_csv(path, sep="\t")
    ref = ref[ref["genomLocation_pval"].notna()].copy()
    ref = ref.join(
        proteoform_to_cluster["first"].rename("cluster_id"),
        on=["protein_id", "proteoform_id"],
    )
    assert ref["cluster_id"].notna().all()
    return ref


@pytest.fixture(scope="module")
def significant_proteins():
    """The 63 proteins COPF calls significant on this dataset.

    The canonical inclusive predicate: ``proteoform_score >= 0.1`` and
    ``proteoform_score_pval_adj <= 0.1``.
    """
    path = DATA_DIR / "trace_annotation_proteoform-scores_rcopf.tsv"
    scores = pd.read_csv(path, sep="\t")
    per_protein = scores.groupby("protein_id")[
        ["proteoform_score", "proteoform_score_pval_adj"]
    ].first()
    keep = (per_protein["proteoform_score"] >= 0.1) & (
        per_protein["proteoform_score_pval_adj"] <= 0.1
    )
    proteins = set(per_protein.index[keep])
    assert len(proteins) == N_SIGNIFICANT
    return proteins


# --------------------------------------------------------------------
# Fixtures: a small synthetic protein, for behaviour that does not
# need the reference dataset
# --------------------------------------------------------------------


@pytest.fixture
def toy():
    """Two proteins: one testable, one too small to be eligible.

    ``P1`` has six peptides -- a tight cluster of three at the N
    terminus, a spread cluster of two, and one noise peptide. ``P2``
    has two, so it fails ``min_peptides_per_protein``.
    """
    sequence = "AAAK" * 40
    peptides = {
        "AAAKAAAKAAAK": ("P1", 1.0),
        "AAAKAAAKAAAKA": ("P1", 1.0),
        "AAAKAAAKAAAKAA": ("P1", 1.0),
        "KAAAKAAAKAAAK": ("P1", 2.0),
        "AAAKAAAKAAAKAAAKAAAKAAAKAAAKAAAK": ("P1", 2.0),
        "AAAKAAAKAAAKAAAKAAAKAAAKAAAK": ("P1", NOISE),
        "AAAKAAAKAAAKAAAKA": ("P2", 1.0),
        "AAAKAAAKAAAKAAAKAA": ("P2", 2.0),
    }
    var = pd.DataFrame(
        {
            "peptide_id": list(peptides),
            "protein_id": [v[0] for v in peptides.values()],
            "cluster_id": [v[1] for v in peptides.values()],
        },
        index=list(peptides),
    )
    obs = pd.DataFrame({"sample_id": ["s1", "s2"]}, index=["s1", "s2"])
    rng = np.random.default_rng(0)
    adata = ad.AnnData(
        X=rng.random((2, len(var))),
        obs=obs,
        var=var,
    )
    return adata, {"P1": sequence, "P2": sequence}


# --------------------------------------------------------------------
# The score
# --------------------------------------------------------------------


def test_normalized_sd_of_consecutive_ranks_is_one():
    """A cluster whose peptides are adjacent scores the minimum, 1.

    The denominator is the standard deviation of ``k`` consecutive
    integers, so consecutive ranks divide out whatever the offset.
    The quotient is 1 only to within a last bit: R's arithmetic order
    is reproduced so that two *scores* can be compared with each
    other, not to make either exact, which is why ties are decided by
    `_dispersion_exact` instead.
    """
    cases = {
        (1, 2): 1.0,
        (1, 2, 3): 1.0,
        (1, 2, 3, 4): 1.0,
        (7, 8, 9): 1.0,
        (12, 13, 14, 15, 16): 1.0,
    }
    for ranks, expected in cases.items():
        assert normalized_sd(np.array(ranks)) == pytest.approx(expected)


def test_normalized_sd_grows_with_spread():
    """Spreading a cluster over the protein raises the score above 1."""
    cases = {
        (1, 2): 1.0,
        (1, 3): 2.0,
        (1, 5): 4.0,
        (1, 9): 8.0,
    }
    for ranks, expected in cases.items():
        assert normalized_sd(np.array(ranks)) == pytest.approx(expected)


def test_normalized_sd_of_one_peptide_is_nan():
    """R's ``sd`` returns NA for a single value; so does this."""
    assert np.isnan(normalized_sd(np.array([3])))


def test_normalized_sd_accepts_a_matrix_of_row_vectors():
    matrix = np.array([[1, 2, 3], [1, 2, 5], [4, 5, 6]])
    out = normalized_sd(matrix)
    assert out.shape == (3,)
    for i in range(3):
        assert out[i] == normalized_sd(matrix[i])


def test_exact_dispersion_orders_like_normalized_sd():
    """The integer stand-in must not reorder any pair of clusters.

    That is what lets it decide ties exactly without changing which
    permutations count as smaller.
    """
    rng = np.random.default_rng(0)
    for k in (2, 3, 5, 8):
        ranks = rng.integers(1, 60, size=(200, k))
        scores = normalized_sd(ranks)
        exact = _dispersion_exact(ranks)
        order_float = np.argsort(scores, kind="stable")
        order_exact = np.argsort(exact, kind="stable")
        assert np.array_equal(
            scores[order_float], np.sort(scores, kind="stable")
        )
        assert np.allclose(scores[order_exact], scores[order_float])


# --------------------------------------------------------------------
# Contract and argument handling
# --------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"n_permutations": 0}, ValueError),
        ({"n_permutations": 10.0}, TypeError),
        ({"min_peptides_per_protein": 0}, ValueError),
        ({"min_peptides_per_proteoform": 0}, ValueError),
        ({"min_clusters_per_protein": 0}, ValueError),
        ({"tie_arithmetic": "float"}, ValueError),
        ({"random_state": "123"}, TypeError),
    ],
)
def test_rejects_invalid_arguments(toy, kwargs, error):
    adata, sequences = toy
    with pytest.raises(error):
        pr.tl.peptide_proximity(adata, sequences, **kwargs)


def test_raises_when_the_cluster_column_is_missing(toy):
    adata, sequences = toy
    del adata.var["cluster_id"]
    with pytest.raises(KeyError, match="peptide_clusters_from_dendograms"):
        pr.tl.peptide_proximity(adata, sequences)


def test_unresolvable_peptide_raises_by_default(toy):
    """Silence about an unlocated peptide has to be opted into."""
    adata, sequences = toy
    sequences = dict(sequences)
    sequences["P1"] = "MMMM" * 10
    with pytest.raises(ValueError, match="not found in their protein"):
        pr.tl.peptide_proximity(adata, sequences)


def test_inplace_false_leaves_the_input_untouched(toy):
    adata, sequences = toy
    before = list(adata.var.columns)
    out = pr.tl.peptide_proximity(adata, sequences, inplace=False)
    assert list(adata.var.columns) == before
    assert "peptide_proximity" not in adata.uns
    assert "peptide_proximity_pval" in out.var.columns
    assert out.uns["peptide_proximity"]["n_tests"] > 0


def test_inplace_true_returns_none_and_writes_six_columns(toy):
    adata, sequences = toy
    assert pr.tl.peptide_proximity(adata, sequences) is None
    for suffix in ("pval", "pval_adj", "pseudo_pval"):
        assert f"peptide_proximity_{suffix}" in adata.var.columns
        assert f"peptide_proximity_unified_{suffix}" in adata.var.columns


def test_key_added_renames_the_outputs(toy):
    adata, sequences = toy
    pr.tl.peptide_proximity(adata, sequences, key_added="prox")
    assert "prox_pval" in adata.var.columns
    assert "peptide_proximity_pval" not in adata.var.columns
    assert "prox" in adata.uns


def test_repeated_runs_are_identical(toy):
    """No RNG leaks in: the generator is re-seeded per peptide count."""
    adata, sequences = toy
    first = pr.tl.peptide_proximity(adata, sequences, inplace=False)
    second = pr.tl.peptide_proximity(adata, sequences, inplace=False)
    assert np.array_equal(
        first.var["peptide_proximity_pval"].to_numpy(),
        second.var["peptide_proximity_pval"].to_numpy(),
        equal_nan=True,
    )


def test_random_state_none_means_the_ccprofiler_seed(toy):
    """The default run is the one comparable with the publication."""
    adata, sequences = toy
    default = pr.tl.peptide_proximity(adata, sequences, inplace=False)
    pinned = pr.tl.peptide_proximity(
        adata, sequences, random_state=CCPROFILER_SEED, inplace=False
    )
    other = pr.tl.peptide_proximity(
        adata, sequences, random_state=7, inplace=False
    )
    column = "peptide_proximity_pval"
    assert np.array_equal(
        default.var[column].to_numpy(),
        pinned.var[column].to_numpy(),
        equal_nan=True,
    )
    assert default.uns["peptide_proximity"]["params"]["random_state"] == (
        CCPROFILER_SEED
    )
    assert other.uns["peptide_proximity"]["params"]["random_state"] == 7


def test_single_peptide_cluster_is_not_tested(toy):
    """One peptide gives no standard deviation, so there is no test.

    The peptide keeps its row in ``.var`` and gets ``NaN``. Its
    protein's other clusters are unaffected, and so is the
    protein-level summary, because the minimum skips a cluster with
    no value.
    """
    adata, sequences = toy
    pr.tl.peptide_proximity(adata, sequences)
    tests = adata.uns["peptide_proximity"]["tests"]
    assert (tests["n_peptides_per_cluster"] >= 2).all()

    var = adata.var
    lonely = var["cluster_id"] >= NOISE
    assert lonely.sum() == 1
    assert var.loc[lonely, "peptide_proximity_pval"].isna().all()

    in_p1 = var["protein_id"] == "P1"
    assert var.loc[in_p1 & ~lonely, "peptide_proximity_pval"].notna().all()

    # P1 still has its two real clusters, so the protein-level summary
    # stands: the minimum skips the cluster it could not score.
    unified = var.loc[in_p1, "peptide_proximity_unified_pval"]
    assert unified.notna().all()
    assert unified.iloc[0] == pytest.approx(
        var.loc[in_p1, "peptide_proximity_pval"].min()
    )


def test_single_cluster_protein_is_skipped_by_default(toy):
    """A lone cluster covering every peptide cannot be informative.

    Its ranks are ``1..n`` and so is every permutation of them, so the
    score is exactly 1 and the pseudo p-value hits its floor whatever
    the data say. The reference scores it; ``min_clusters_per_protein``
    defaults to 2 so that it is skipped.
    """
    adata, sequences = toy
    adata.var["cluster_id"] = 1.0  # one cluster per protein
    pr.tl.peptide_proximity(adata, sequences)
    assert adata.uns["peptide_proximity"]["n_tests"] == 0
    assert adata.var["peptide_proximity_pval"].isna().all()

    out = pr.tl.peptide_proximity(
        adata, sequences, min_clusters_per_protein=1, inplace=False
    )
    degenerate = out.uns["peptide_proximity"]["tests"]
    assert len(degenerate) == 1
    assert degenerate["normalized_sd"].iloc[0] == pytest.approx(1.0)
    assert degenerate["pval"].iloc[0] == 1.0
    assert degenerate["pseudo_pval"].iloc[0] == pytest.approx(1 / 1001)


def test_ineligible_protein_is_left_nan(toy):
    """``P2`` has two peptides, below the reference's minimum of four.

    Its peptides get ``NaN`` in every column. ``P1`` is eligible, so
    its two multi-peptide clusters are tested -- only its lone noise
    peptide is left out, having no standard deviation.
    """
    adata, sequences = toy
    pr.tl.peptide_proximity(adata, sequences)
    var = adata.var
    small = var["protein_id"] == "P2"
    for suffix in ("pval", "unified_pval"):
        column = f"peptide_proximity_{suffix}"
        assert var.loc[small, column].isna().all()

    testable = (var["protein_id"] == "P1") & (var["cluster_id"] < NOISE)
    assert testable.sum() == 5
    assert var.loc[testable, "peptide_proximity_pval"].notna().all()


def test_pvalue_cannot_go_below_the_permutation_floor(toy):
    adata, sequences = toy
    for n_permutations in (99, 1000):
        out = pr.tl.peptide_proximity(
            adata, sequences, n_permutations=n_permutations, inplace=False
        )
        floor = 1.0 / (n_permutations + 1)
        for column in ("pval", "pseudo_pval"):
            values = out.uns["peptide_proximity"]["tests"][column]
            assert values.min() >= floor


# --------------------------------------------------------------------
# Shape of the annotation
# --------------------------------------------------------------------


def test_statistics_repeat_across_the_peptides_of_a_group(result):
    """Each statistic is a property of (protein_id, cluster_id).

    Every peptide of a group carries the same value, and the raw
    p-values are not constant overall -- otherwise the check would be
    vacuous.
    """
    var = result.var
    for suffix in ("pval", "pval_adj", "pseudo_pval"):
        column = f"peptide_proximity_{suffix}"
        per_group = var.groupby(["protein_id", "cluster_id"])[column].nunique(
            dropna=False
        )
        assert (per_group <= 1).all()

        # The unified variant is a property of the protein, so it is
        # constant one level up.
        unified = f"peptide_proximity_unified_{suffix}"
        per_protein = var.groupby("protein_id")[unified].nunique(dropna=False)
        assert (per_protein <= 1).all()

    # Not vacuous: the two raw statistics do vary between groups.
    # `pval_adj` deliberately does not -- see
    # `test_bh_adjustment_saturates_on_this_dataset`.
    for suffix in ("pval", "pseudo_pval"):
        assert var[f"peptide_proximity_{suffix}"].nunique() > 1
        assert var[f"peptide_proximity_unified_{suffix}"].nunique() > 1


def test_pseudo_pval_never_exceeds_pval(tests_table):
    """``#{random < real} <= #{random <= real}``, by construction.

    This is what forces the pseudo criterion to select the larger set
    of proteins, and it is the reason the publication's 19 and 7
    cannot belong to the criteria its text assigns them to.
    """
    assert (tests_table["pseudo_pval"] <= tests_table["pval"]).all()


def test_pval_adj_is_bh_over_the_tests_and_nothing_else(tests_table):
    """BH's ``n`` is the number of tests, not the number of peptides.

    Only the classical p-values enter the correction; the pseudo
    p-values are left raw, as the publication reports them.
    """
    expected = multipletests(tests_table["pval"].to_numpy(), method="fdr_bh")[
        1
    ]
    assert np.allclose(
        tests_table["pval_adj"].to_numpy(), expected, rtol=0, atol=0
    )
    assert len(tests_table) == N_TESTS


def test_multi_peptide_noise_clusters_are_tested(tests_table):
    """The noise cluster is a proteoform group, not a leftover.

    CCprofiler labels it ``<protein>_0`` and loops over it, and it
    takes part in the per-protein minimum, so dropping it could change
    a published count. Every noise cluster with enough peptides to
    have a standard deviation is tested here.
    """
    noise = tests_table[tests_table["is_noise_cluster"]]
    assert len(noise) == N_NOISE_CLUSTERS_TESTED
    assert noise["pval"].notna().all()
    assert (noise["n_peptides_per_cluster"] >= 2).all()


def test_untested_proteins_are_nan_in_var(result, tests_table):
    """1,272 of 2,885 proteins clear the reference's eligibility rule.

    A value on every row would mean the ``min_peptides_per_protein``
    filter had been dropped.
    """
    tested = set(tests_table["protein_id"])
    assert len(tested) == N_PROTEINS_TESTED
    assert result.var["protein_id"].nunique() == N_PROTEINS

    var = result.var
    is_tested = var["protein_id"].isin(tested)
    column = var["peptide_proximity_pval"]
    assert column[~is_tested].isna().all()

    # Within a tested protein, a peptide has a value exactly when its
    # own cluster was big enough to test.
    cluster_size = var.groupby(["protein_id", "cluster_id"])[
        "peptide_id"
    ].transform("size")
    in_tested_cluster = is_tested & (cluster_size >= 2)
    assert column[in_tested_cluster].notna().all()
    assert column[is_tested & ~in_tested_cluster].isna().all()
    assert int((is_tested & ~in_tested_cluster).sum()) == (
        N_SINGLE_PEPTIDE_CLUSTERS
    )


# --------------------------------------------------------------------
# Parity with CCprofiler
# --------------------------------------------------------------------


def test_peptide_positions_from_fasta_vs_rcopf(
    adata_rcopf, cluster_assignment
):
    """The FASTA reproduces the reference's peptide positions exactly.

    All 24,534 of them, including the single peptide the reference
    could not locate -- which it keeps rather than drops.
    """
    ranks = peptide_position_ranks(
        adata_rcopf.var,
        FASTA,
        on_unlocated_peptide="keep",
    )
    assert len(ranks) == 24534
    reference = (
        cluster_assignment.set_index("peptide_id")["PeptidePositionStart"]
        .reindex(adata_rcopf.var.index)
        .to_numpy(dtype=float)
    )
    ours = ranks["position"].reindex(adata_rcopf.var.index).to_numpy()
    assert np.array_equal(ours, reference, equal_nan=True)
    unresolved = ranks[ranks["position"].isna()]
    assert len(unresolved) == 1
    assert unresolved["protein_id"].iloc[0] == UNLOCATED[0]
    assert unresolved["peptide_id"].iloc[0] == UNLOCATED[1]


def test_reference_proteoform_partition_equals_the_cluster_partition(
    proteoform_to_cluster,
):
    """Justifies joining the reference p-values by cluster membership.

    Every reference proteoform group maps onto exactly one cluster of
    the shipped assignment, so the two tables describe the same
    partition and the join needs no assumption about identifier
    suffixes.
    """
    assert (proteoform_to_cluster["nunique"] == 1).all()


def test_tested_groups_vs_rcopf(tests_table, location_ref):
    """The same proteins, and every group matched to a reference one.

    Eligibility is a discrete outcome, so the protein set is asserted
    exactly. At the cluster level the assertion is one-directional on
    purpose: every group tested here is a group CCprofiler tested,
    with the same peptides. The reference additionally emits a row for
    each single-peptide cluster, scored 1 because its standard
    deviation is undefined; those are placeholders rather than
    results, so they get no row here.
    """
    ours = set(zip(tests_table["protein_id"], tests_table["cluster_id"]))
    theirs = set(zip(location_ref["protein_id"], location_ref["cluster_id"]))

    assert len(ours) == N_TESTS
    assert ours <= theirs
    assert tests_table["protein_id"].nunique() == N_PROTEINS_TESTED
    assert set(tests_table["protein_id"]) == set(location_ref["protein_id"])


def test_pvalues_vs_rcopf_within_sampling_noise(tests_table, location_ref):
    """Agreement to the granularity of two 1,000-draw estimates.

    Exact equality is impossible -- see the module docstring. The
    bounds below are the measured agreement with ~50 % headroom, so
    they fail on a genuine algorithmic divergence while tolerating
    the permutation stream.
    """
    merged = tests_table.merge(
        location_ref, on=["protein_id", "cluster_id"], validate="one_to_one"
    )
    assert len(merged) == N_TESTS
    pairs = [
        ("pval", "genomLocation_pval"),
        ("pseudo_pval", "genomLocation_pval_lim"),
    ]
    for ours, theirs in pairs:
        delta = (merged[ours] - merged[theirs]).abs()
        assert delta.median() <= 0.02
        assert delta.quantile(0.95) <= 0.06
        assert delta.max() <= 0.10


def test_paper_sequence_proximity_result_vs_rcopf(
    tests_table, location_ref, significant_proteins
):
    """The publication's headline proximity numbers, as sets.

    ``getProteoformStats`` takes each protein's minimum over its
    proteoform groups, independently per p-value, and splits the
    significant proteins three ways at ``0.1``. Recovering the counts
    while recovering a *different* set of proteins would not be a
    reproduction, so both are asserted.

    Note which count belongs to which criterion: **7** proteins meet
    the classical criterion and **19** the pseudo-only one. The paper
    states these the other way round. The direction is forced by
    ``pseudo_pval <= pval`` -- see
    `test_pseudo_pval_never_exceeds_pval`.
    """

    def split(table, pval_col, pseudo_col):
        per_protein = table.groupby("protein_id")[[pval_col, pseudo_col]].min()
        per_protein = per_protein.loc[
            per_protein.index.isin(significant_proteins)
        ]
        classical = per_protein[pval_col] <= THRESHOLD
        pseudo_only = (per_protein[pseudo_col] <= THRESHOLD) & ~classical
        return (
            set(per_protein.index[classical]),
            set(per_protein.index[pseudo_only]),
        )

    ref_classical, ref_pseudo = split(
        location_ref, "genomLocation_pval", "genomLocation_pval_lim"
    )
    assert len(ref_classical) == N_CLASSICAL
    assert len(ref_pseudo) == N_PSEUDO_ONLY

    classical, pseudo = split(tests_table, "pval", "pseudo_pval")
    assert classical == ref_classical
    assert pseudo == ref_pseudo
    assert len(significant_proteins - (classical | pseudo)) == N_NEITHER


def test_unified_columns_reproduce_the_paper(result, significant_proteins):
    """The same split, read straight off the ``unified_`` columns.

    They are the per-protein minimum, which is what
    ``getProteoformStats`` thresholds, so no aggregation is needed --
    and they reach the published numbers because the minimum **skips**
    a cluster with no value rather than propagating it. `P10852` is
    the case that makes the difference: its evidence is a two-peptide
    cluster whose pseudo p-value hits the floor, while its noise
    cluster holds one peptide that no test can score. Propagating
    would have dropped it and reported 18.
    """
    per_protein = result.var.groupby("protein_id")[
        [
            "peptide_proximity_unified_pval",
            "peptide_proximity_unified_pseudo_pval",
        ]
    ].first()
    per_protein = per_protein.loc[per_protein.index.isin(significant_proteins)]
    assert per_protein["peptide_proximity_unified_pval"].notna().all()

    classical = per_protein["peptide_proximity_unified_pval"] <= THRESHOLD
    pseudo = (
        per_protein["peptide_proximity_unified_pseudo_pval"] <= THRESHOLD
    ) & ~classical

    assert int(classical.sum()) == N_CLASSICAL
    assert int(pseudo.sum()) == N_PSEUDO_ONLY
    assert pseudo.loc[UNIFIED_KEEPS]
    assert int((~classical & ~pseudo).sum()) == N_NEITHER


def test_unified_is_the_na_skipping_minimum_over_the_clusters(result):
    """For a qualifying protein, exactly that: min, skipping NaN."""
    var = result.var
    qualifies = var["peptide_proximity_unified_pval"].notna()
    by_protein = var[qualifies].groupby("protein_id")
    for stat in ("pval", "pseudo_pval"):
        expected = by_protein[f"peptide_proximity_{stat}"].min()
        actual = by_protein[f"peptide_proximity_unified_{stat}"].first()
        assert np.array_equal(actual.to_numpy(), expected.to_numpy())


def test_unified_needs_two_non_noise_clusters(toy):
    """One real cluster means no proteoform split to localise.

    The cluster is still tested and still gets a per-cluster p-value;
    what is withheld is the protein-level summary, because a minimum
    over a single group says nothing about where a split sits.
    """
    adata, sequences = toy
    var = adata.var
    var.loc[var["cluster_id"] == 2.0, "cluster_id"] = 1.0  # merge the two
    pr.tl.peptide_proximity(adata, sequences)

    in_p1 = adata.var["protein_id"] == "P1"
    real = in_p1 & (adata.var["cluster_id"] < NOISE)
    assert adata.var.loc[real, "peptide_proximity_pval"].notna().all()
    assert adata.var.loc[in_p1, "peptide_proximity_unified_pval"].isna().all()


def test_unified_pval_adj_is_bh_over_the_per_protein_pvalues(result):
    """The protein-level family is corrected as a protein-level family.

    ``unified_pval_adj`` is Benjamini-Hochberg applied to
    ``unified_pval``, with ``n`` the number of proteins carrying one
    -- not the minimum of the per-cluster ``pval_adj``, which would
    import the cluster-level ``n``. The two are easy to tell apart
    here: the per-cluster correction is saturated at 1.0, the
    per-protein one is not.
    """
    per_protein = (
        result.var.groupby("protein_id")[
            [
                "peptide_proximity_unified_pval",
                "peptide_proximity_unified_pval_adj",
            ]
        ]
        .first()
        .dropna()
    )
    assert len(per_protein) == N_PROTEINS_TESTED

    raw = per_protein["peptide_proximity_unified_pval"].to_numpy()
    expected = multipletests(raw, method="fdr_bh")[1]
    actual = per_protein["peptide_proximity_unified_pval_adj"].to_numpy()
    assert np.allclose(actual, expected, rtol=0, atol=0)

    # `n` is the protein count, not the test count.
    assert len(per_protein) < len(result.uns["peptide_proximity"]["tests"])

    # And it is not the min of the per-cluster adjusted values, which
    # are all 1.0 on this dataset.
    tests = result.uns["peptide_proximity"]["tests"]
    assert (tests.groupby("protein_id")["pval_adj"].min() == 1.0).all()
    assert actual.min() < 1.0


def test_unlocated_peptide_policies(adata_rcopf):
    """``raise`` blocks, ``keep`` is the reference, ``skip`` drops.

    Exactly one of the 24,534 peptides -- ``P35235``'s
    ``VGQALLQGNTER`` -- does not occur in its protein's canonical
    sequence. CCprofiler keeps it, with an unresolved position that
    ``setkey`` ranks first; this function refuses to do that silently,
    so reproducing the reference is an explicit choice.

    On this dataset the two permissive policies happen to give the
    same p-values, because ``P35235``'s peptides are all noise, which
    makes the protein ineligible either way. The policies are still
    not interchangeable: they disagree on how many peptides enter the
    ranking, and on another dataset that would move a tested
    protein's ranks.
    """
    with pytest.raises(ValueError, match="not found in their protein"):
        pr.tl.peptide_proximity(adata_rcopf, FASTA, inplace=False)

    n_ranked = {
        policy: len(
            peptide_position_ranks(
                adata_rcopf.var, FASTA, on_unlocated_peptide=policy
            )
        )
        for policy in ("keep", "skip")
    }
    assert n_ranked == {"keep": 24534, "skip": 24533}

    runs = {
        policy: pr.tl.peptide_proximity(
            adata_rcopf, FASTA, on_unlocated_peptide=policy, inplace=False
        )
        for policy in ("keep", "skip")
    }
    column = "peptide_proximity_pval"
    peptide = adata_rcopf.var["peptide_id"] == UNLOCATED[1]
    for run in runs.values():
        # All-noise protein: ineligible, so NaN under either policy.
        assert run.var.loc[peptide, column].isna().all()
    assert (
        runs["keep"]
        .uns["peptide_proximity"]["tests"]
        .equals(runs["skip"].uns["peptide_proximity"]["tests"])
    )


def test_bh_adjustment_saturates_on_this_dataset(tests_table):
    """Every adjusted p-value is 1.0 -- a fact about these data.

    Benjamini-Hochberg's smallest adjusted value is
    ``min_j (n / j) * p_(j)`` over the ascending p-values, **not**
    ``n * p_(1)``: the ``j = 1`` term is the largest of the family,
    not a bound on it, so a floor on the raw p-value saturates
    nothing by itself.

    What is asserted here is the sharp statement. The whole curve
    sits strictly above 1 for every ``j < n`` and touches 1 exactly
    at ``j = n``, where ``p_(n) = 1``. Equivalently
    ``p_(j) > j / n`` throughout: the empirical distribution of the
    p-values lies entirely below the uniform diagonal, so there is no
    ``j`` at which an excess of small p-values could be declared, at
    any FDR level. Raising ``n_permutations`` cannot create one --
    only two tests reach the raw floor.

    The publication reports raw p-values and CCprofiler computes no
    adjusted ones at all, so this column is an addition; it must not
    be used when comparing against the paper.
    """
    n = len(tests_table)
    ascending = np.sort(tests_table["pval"].to_numpy())
    ranks = np.arange(1, n + 1)
    bh_curve = (n / ranks) * ascending

    assert (bh_curve[:-1] > 1.0).all()
    assert bh_curve[-1] == pytest.approx(1.0)
    assert (ascending[:-1] > ranks[:-1] / n).all()
    assert (tests_table["pval_adj"] == 1.0).all()

    # Not the floor's doing: almost nothing is near it.
    assert int((ascending <= 2 / 1001).sum()) < 5
