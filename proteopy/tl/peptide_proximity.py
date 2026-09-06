"""Sequence proximity analysis for COPF proteoform peptide clusters.

Reimplements CCprofiler's ``evaluateProteoformLocation`` [1]_ -- the
test asking whether the peptides assigned to one proteoform cluster
sit closer together in the protein sequence than a random grouping of
the same size would.

Reference source, read by git ref rather than by working-tree path::

    git -C CCprofiler show 31a3043:R/proteoformLocationMapping.R

``31a3043`` is the head of branch ``proteoformLocationMapping``, which
is what the publication's analysis script installs and what produced
the published numbers. The release tag ``v1.0.1-copf`` does not even
contain this function.

The algorithm, per protein
--------------------------
Peptides are ranked ``1..n`` by their start position in the protein
sequence. For each proteoform cluster, with ``v`` the cluster's rank
vector and ``k = len(v)``::

    NormalizedSD(v) = population_sd(v) / sqrt((k**2 - 1) / 12)

The denominator is the standard deviation of ``k`` consecutive
integers, so a cluster whose peptides are adjacent in the ranking
scores exactly ``1`` -- the smallest value attainable -- and one
spread across the protein scores above it. Low means tightly
grouped, which is what the test is looking for. The cluster's ranks
are then compared against ``n_permutations`` random permutations of
``1..n`` read at the same positions::

    pval        = (#{random <= real} + 1) / (n_permutations + 1)
    pseudo_pval = (#{random <  real} + 1) / (n_permutations + 1)

``pval`` is the classical empirical p-value. ``pseudo_pval`` is the
reference's ``genomLocation_pval_lim``, which counts only strictly
smaller permutations, and is the paper's "lowest 10 % of possible
p-values" criterion: when no permutation can beat the observation it
reaches the floor ``1 / (n_permutations + 1)``. Because
``pseudo_pval <= pval`` always, the pseudo criterion is the more
permissive of the two.

Five properties reproduced deliberately, not accidents to be fixed
------------------------------------------------------------------
1. ``k`` in the normalisation is the **cluster** size while the ranks
   themselves run over the **protein's** peptide count. The asymmetry
   is the reference's.
2. Population standard deviation, not sample: R computes
   ``sd(v) * sqrt((k - 1) / k)``.
3. Add-one empirical p-values, denominator ``n_permutations + 1``.
   The paper's methods text describes a different division; the code
   is authoritative.
4. Eligibility is ``n_clusters_excluding_noise != 0`` and
   ``n_peptides >= min_peptides_per_protein`` and
   ``median(peptides per cluster) >= min_peptides_per_proteoform``.
   The default ``4`` differs from the ``>= 2`` the main COPF pipeline
   filters on, so the proximity result generally covers a strict
   subset of the proteins COPF scored. Ineligible proteins get
   ``NaN``, matching the reference's ``NA``.
5. The noise cluster is tested like any other cluster and, being one
   of the protein's clusters, participates in a per-protein minimum
   taken downstream. CCprofiler labels it ``<protein>_0`` and loops
   over it; dropping it would change published results.

What this implementation does *not* reproduce
---------------------------------------------
**The permutation stream.** The reference calls ``set.seed(123)`` and
draws with R's ``sample()``; NumPy cannot generate that stream. Two
independent 1,000-draw estimates of the same p-value therefore differ
by sampling noise of order ``1 / (n_permutations + 1)``. Read a small
p-value difference against CCprofiler as expected rather than as a
defect; what does match exactly is every discrete outcome --
eligibility, cluster membership, and the counts a threshold produces.
Two runs of *this* function are identical, because the generator is
re-seeded per peptide count exactly as R re-seeds per protein.

**The placeholder for a single-peptide cluster.** One peptide has no
standard deviation, so there is no observation to compare against the
permutations. The reference tests such a cluster anyway and records
``pval = pseudo_pval = 1``; here it is left untested and ``NaN``,
because 1 is a placeholder rather than a result. Nothing downstream
changes: a value of 1 can never be a protein's minimum unless every
sibling is also 1, and the ``unified_`` statistics skip a missing
cluster rather than propagating it. On the mouse tissue dataset the
reference emits 283 such rows -- every one of them a noise cluster,
since COPF never leaves a real cluster with a single peptide -- and
the published counts are identical either way.

**R's floating-point tie detection.** Whether a permutation counts as
*equal* to the observation is what separates the two p-values, and
comparing float scores misjudges it. ``tie_arithmetic='exact'`` (the
default) compares an exact integer dispersion with the same ordering
instead. Measured against exactly enumerated null distributions this
tracks CCprofiler more closely than a float imitation of R's own
arithmetic order does.

On the publication's two numbers
--------------------------------
Bludau et al. report for the mouse tissue dataset that "the
proteoforms for 19 proteins (30 %) were significantly closer in
sequence proximity than expected by chance" and that "an additional 7
proteins (11 %)" scored among the lowest possible p-values. Running
the authors' own workflow gives **7** proteins on the classical
criterion and **19** on the pseudo-only criterion: the two numbers are
attached to the opposite criteria. The direction is forced by
``pseudo_pval <= pval``, so the pseudo criterion cannot select the
smaller set. A run reporting 7 classical and 19 pseudo-only is
correct and must not be "fixed".

References
----------
.. [1] Bludau, I. et al. Systematic detection of functional
   proteoform groups from bottom-up proteomic datasets. Nat Commun
   12, 3810 (2021). https://doi.org/10.1038/s41467-021-24030-x
"""

import re
from pathlib import Path
from collections.abc import Iterable

import anndata as ad
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from proteopy.pp.summarize_peptides_by_neighbourhood_union import (
    CCPROFILER_MOD_REGEX,
    IUPAC_AMINO_ACIDS,
    _locate,
    _resolve_annotator,
    _strip_and_validate,
)
from proteopy.utils.anndata import check_proteodata

# What `pr.tl.peptide_clusters_from_dendograms` writes for peptides that
# ended up in no multi-peptide cluster. CCprofiler's equivalent marker is
# `cluster == 100`; the two conventions are not interchangeable, which is
# why the marker is a parameter rather than a guess.
NOISE = 1e6

# `calculateScoresPerProtein(p, data, n_random = 1000, seed = 123)`. The
# seed is hard-coded upstream and neither the paper's script nor
# `evaluateProteoformLocation` overrides it, so 123 is what produced the
# published numbers.
CCPROFILER_SEED = 123
N_PERMUTATIONS = 1000

# `evaluateProteoformLocation.traces` defaults.
MIN_PEPTIDES_PER_PROTEIN = 4
MIN_PEPTIDES_PER_PROTEOFORM = 2

# Not a policy but an arithmetic floor: one peptide has no standard
# deviation, so there is no observation to compare against the
# permutations. The reference tests such a cluster anyway and assigns
# a p-value of 1; that is a placeholder rather than a result, so here
# the cluster is left NaN. Deliberately not configurable.
MIN_PEPTIDES_PER_CLUSTER = 2

# A guard the reference does not apply, on because the result it
# excludes carries no information, and inert on the published mouse
# tissue analysis -- see the function docstring.
MIN_CLUSTERS_PER_PROTEIN = 2

_TIE_ARITHMETIC_MODES = ("exact", "reference")

_PVAL = "pval"
_PVAL_ADJ = "pval_adj"
_PSEUDO_PVAL = "pseudo_pval"

STAT_COLUMNS = (_PVAL, _PVAL_ADJ, _PSEUDO_PVAL)

# Per-protein variants: the minimum over the protein's clusters. A
# protein needs at least this many distinct non-noise clusters for the
# minimum to mean anything -- with fewer there is no proteoform split
# to localise, which is what COPF detected in the first place.
MIN_NON_NOISE_CLUSTERS_FOR_UNIFIED = 2
UNIFIED = "unified"
UNIFIED_COLUMNS = tuple(f"{UNIFIED}_{stat}" for stat in STAT_COLUMNS)

TEST_COLUMNS = (
    "protein_id",
    "cluster_id",
    "is_noise_cluster",
    "n_peptides",
    "n_peptides_per_cluster",
    "normalized_sd",
    "n_random_le",
    "n_random_lt",
    _PVAL,
    _PSEUDO_PVAL,
)


def normalized_sd(ranks) -> np.ndarray | float:
    """CCprofiler's ``getNormalizedSD``, in R's arithmetic order.

    Accepts a 1-D rank vector or an ``(m, k)`` matrix of row vectors
    and returns a scalar or a length-``m`` array, through one code
    path: the real score and the permuted scores are compared with
    ``==``, so they must come out of bitwise identical arithmetic or
    ties are miscounted.

    R evaluates ``sd(v) * sqrt((k - 1) / k)`` -- a sample standard
    deviation rescaled to a population one. The algebraically
    identical ``sqrt(mean(centred ** 2))`` differs in the last bit,
    and that is enough to change how many permutations are judged
    equal to the observation.

    Returns ``nan`` for a vector of length one, matching R's ``sd``.
    The caller then assigns that cluster a p-value of 1.

    Parameters
    ----------
    ranks : array-like
        Integer peptide ranks, as a vector or a matrix of row
        vectors.

    Returns
    -------
    float | numpy.ndarray
        The normalised standard deviation.
    """
    matrix = np.atleast_2d(np.asarray(ranks, dtype=float))
    k = matrix.shape[1]
    if k < 2:
        out = np.full(matrix.shape[0], np.nan)
    else:
        centred = matrix - matrix.mean(axis=1, keepdims=True)
        sample_sd = np.sqrt((centred**2).sum(axis=1) / (k - 1))
        out = sample_sd * np.sqrt((k - 1) / k)
        out = out / np.sqrt((k**2 - 1) / 12.0)
    return float(out[0]) if np.ndim(ranks) == 1 else out


def _dispersion_exact(ranks) -> np.ndarray | np.int64:
    """An exact, order-preserving stand-in for `normalized_sd`.

    For integer ranks, ``k * sum(v ** 2) - sum(v) ** 2`` is an exact
    integer proportional to the sum of squared deviations. Within one
    test ``k`` is fixed, so ordering by this integer orders clusters
    exactly as `normalized_sd` does mathematically, with no rounding
    -- so ties are decided correctly by construction.
    """
    matrix = np.atleast_2d(np.asarray(ranks, dtype=np.int64))
    k = matrix.shape[1]
    out = k * (matrix**2).sum(axis=1) - matrix.sum(axis=1) ** 2
    return out[0] if np.ndim(ranks) == 1 else out


class _PermutationBank:
    """Permutations of ``1..n``, one set per peptide count.

    ``calculateScoresPerProtein`` re-seeds inside itself, so every
    protein with the same peptide count sees the *same* permutations.
    Caching by ``n`` reproduces that and is also what makes the run
    cheap.
    """

    def __init__(self, n_permutations: int, seed: int) -> None:
        self.n_permutations = n_permutations
        self.seed = seed
        self._cache: dict[int, np.ndarray] = {}

    def get(self, n: int) -> np.ndarray:
        if n not in self._cache:
            rng = np.random.default_rng(self.seed)
            base = np.tile(np.arange(1, n + 1), (self.n_permutations, 1))
            self._cache[n] = rng.permuted(base, axis=1)
        return self._cache[n]


def _validate_arguments(
    n_permutations: int,
    min_peptides_per_protein: int,
    min_peptides_per_proteoform: int,
    min_clusters_per_protein: int,
    tie_arithmetic: str,
    random_state: int | None,
) -> None:
    """Group every contract check ahead of the algorithm."""
    if not isinstance(n_permutations, (int, np.integer)):
        raise TypeError(
            f"n_permutations must be an int, got "
            f"{type(n_permutations).__name__}."
        )
    if n_permutations < 1:
        raise ValueError(f"n_permutations must be >= 1, got {n_permutations}.")
    if min_peptides_per_protein < 1:
        raise ValueError(
            "min_peptides_per_protein must be >= 1, got "
            f"{min_peptides_per_protein}."
        )
    if min_peptides_per_proteoform < 1:
        raise ValueError(
            "min_peptides_per_proteoform must be >= 1, got "
            f"{min_peptides_per_proteoform}."
        )
    if min_clusters_per_protein < 1:
        raise ValueError(
            "min_clusters_per_protein must be >= 1, got "
            f"{min_clusters_per_protein}."
        )
    if tie_arithmetic not in _TIE_ARITHMETIC_MODES:
        raise ValueError(
            f"tie_arithmetic must be one of {_TIE_ARITHMETIC_MODES}, "
            f"got {tie_arithmetic!r}."
        )
    if random_state is not None and not isinstance(
        random_state, (int, np.integer)
    ):
        raise TypeError(
            "random_state must be an int or None, got "
            f"{type(random_state).__name__}."
        )


def _validate_var(
    var: pd.DataFrame,
    protein_col: str,
    peptide_col: str,
    cluster_col: str,
) -> None:
    """Fail with a pointer to the missing step, not a KeyError."""
    for column in (protein_col, peptide_col):
        if column not in var.columns:
            raise KeyError(
                f"'{column}' not found in adata.var. Columns present: "
                f"{sorted(var.columns)[:20]}"
            )
    if cluster_col not in var.columns:
        raise KeyError(
            f"'{cluster_col}' not found in adata.var. Peptide clusters "
            "come from pr.tl.peptide_clusters_from_dendograms, which "
            "writes 'cluster_id'; run the COPF steps "
            "(pairwise_peptide_correlations -> "
            "peptide_dendograms_by_correlation -> "
            "peptide_clusters_from_dendograms) first, or pass "
            "cluster_col."
        )


def peptide_position_ranks(
    var: pd.DataFrame,
    annotator: str | Path | dict[str, str],
    *,
    protein_col: str = "protein_id",
    peptide_col: str = "peptide_id",
    mod_regex: str = CCPROFILER_MOD_REGEX,
    alphabet: Iterable[str] = IUPAC_AMINO_ACIDS,
    on_unknown_protein: str = "raise",
    on_unlocated_peptide: str = "raise",
) -> pd.DataFrame:
    """Resolve peptide start positions and rank them within a protein.

    Positions come from ``annotator`` exactly as
    ``pr.pp.summarize_peptides_by_neighbourhood_union`` resolves them:
    the modification-stripped identifier is located in the protein
    sequence and the 1-based index of its first occurrence is taken.
    The same policy arguments therefore apply, and mean the same
    things.

    Ranking follows ``setkeyv(data_sub, c("protein_id",
    "PeptidePositionStart"))``: ascending position, with unresolved
    positions **first** because that is where ``data.table`` puts
    ``NA``. Peptides tied on position keep their ``var`` row order, so
    a caller who needs the reference's tie-breaking must leave the
    rows in the reference's order (descending peptide identifier).

    Parameters
    ----------
    var : pandas.DataFrame
        Peptide annotation, one row per peptide.
    annotator : str | pathlib.Path | dict
        Path to a FASTA file, or a pre-parsed
        ``{accession: sequence}`` mapping.
    protein_col, peptide_col : str, optional
        Columns holding the protein and peptide identifiers.
    mod_regex : str, optional
        Pattern for annotations to disregard when locating a peptide.
        Defaults to CCprofiler's ``\\(UniMod:[0-9]+\\)``.
    alphabet : Iterable[str], optional
        Characters allowed to remain after ``mod_regex`` is applied.
        Its purpose is to make ``mod_regex`` self-checking.
    on_unknown_protein : {'raise', 'keep', 'skip'}, optional
        What to do about a protein absent from ``annotator``.
        ``'keep'`` gives its peptides unresolved positions, which is
        the CCprofiler behaviour; ``'skip'`` excludes them.
    on_unlocated_peptide : {'raise', 'keep', 'skip'}, optional
        The same, for a peptide whose protein is present but whose
        sequence is not found in it.

    Returns
    -------
    pandas.DataFrame
        Columns ``peptide_id``, ``protein_id``, ``position`` and
        ``rank``, one row per peptide that took part, indexed by the
        ``var`` index.
    """
    peptides = var[peptide_col].astype(str).tolist()
    proteins = var[protein_col].astype(str).tolist()
    stripped = _strip_and_validate(
        peptides, re.compile(mod_regex), set(alphabet)
    )
    starts, _, keep = _locate(
        peptides,
        stripped,
        proteins,
        _resolve_annotator(annotator),
        on_unknown_protein,
        on_unlocated_peptide,
    )

    frame = pd.DataFrame(
        {
            "peptide_id": peptides,
            "protein_id": proteins,
            "position": starts,
            "_row": np.arange(len(var)),
        },
        index=var.index,
    )
    frame = frame.loc[keep]

    # -- `setkeyv` order: NA first, then ascending position; ties keep
    #    the caller's row order.
    frame["_unresolved"] = frame["position"].isna()
    frame = frame.sort_values(
        ["protein_id", "_unresolved", "position", "_row"],
        ascending=[True, False, True, True],
        kind="stable",
    )
    frame["rank"] = frame.groupby("protein_id", sort=False).cumcount() + 1
    return frame.drop(columns=["_unresolved", "_row"])


def _eligible_proteins(
    peptides: pd.DataFrame,
    min_peptides_per_protein: int,
    min_peptides_per_proteoform: int,
    min_clusters_per_protein: int,
) -> set:
    """Which proteins are tested at all.

    The reference's three conditions are ``n_proteoforms != 0``,
    ``n_peptides >= minPepPerProtein`` and
    ``median(n_peptides_per_proteoform) >= minPepPerProteoform``.
    ``n_proteoforms`` is built in ``annotateTracesWithProteoforms``
    as the number of distinct proteoform ids minus one where a
    ``_0`` group exists, i.e. the count of **non-noise** clusters, so
    a protein whose peptides are all noise is not tested -- exactly
    the set COPF could not score. The median, by contrast, runs over
    *all* clusters including the noise one, as ``medianPerProt`` does.

    ``min_clusters_per_protein`` is the one addition. A protein with a
    single cluster covering every one of its peptides has rank vector
    ``1..n``, and so does every permutation of it, so the test is
    degenerate: the score is exactly 1 and the pseudo p-value hits its
    floor whatever the data say. The reference would score it; here it
    is skipped by default. On the published analysis the two rules
    select the identical 1,272 proteins, because a single-cluster
    protein there is always an all-noise one.
    """
    per_cluster = peptides.groupby(
        ["protein_id", "cluster_id"], sort=False, dropna=False
    ).agg(n=("peptide_id", "size"), is_noise=("is_noise", "all"))

    n_peptides = peptides.groupby("protein_id", sort=False).size()
    n_real = (
        per_cluster[~per_cluster["is_noise"]]
        .groupby("protein_id", sort=False)
        .size()
        .reindex(n_peptides.index)
        .fillna(0)
    )
    median_n = (
        per_cluster.groupby("protein_id", sort=False)["n"]
        .median()
        .reindex(n_peptides.index)
    )

    n_clusters = per_cluster.groupby("protein_id", sort=False).size()
    n_clusters = n_clusters.reindex(n_peptides.index).fillna(0)

    eligible = (
        (n_peptides >= min_peptides_per_protein)
        & (n_real > 0)
        & (median_n >= min_peptides_per_proteoform)
        & (n_clusters >= min_clusters_per_protein)
    )
    return set(eligible.index[eligible.to_numpy()])


def _test_one_cluster(
    ranks: np.ndarray,
    permutations: np.ndarray,
    positions: np.ndarray,
    n_permutations: int,
    tie_arithmetic: str,
) -> tuple:
    """One cluster's scores and p-values.

    ``positions`` are 0-based column indices into a permutation, i.e.
    the cluster's ranks minus one -- which is what R's
    ``x[idx]`` amounts to once the protein is sorted by position.
    """
    real = normalized_sd(ranks)
    if np.isnan(real):
        return real, None, None, 1.0, 1.0

    if tie_arithmetic == "exact":
        real_cmp = _dispersion_exact(ranks)
        random_cmp = _dispersion_exact(permutations[:, positions])
    else:
        real_cmp = real
        random_cmp = normalized_sd(permutations[:, positions])

    n_le = int(np.count_nonzero(random_cmp <= real_cmp))
    n_lt = int(np.count_nonzero(random_cmp < real_cmp))
    return (
        real,
        n_le,
        n_lt,
        (n_le + 1) / (n_permutations + 1),
        (n_lt + 1) / (n_permutations + 1),
    )


def peptide_proximity_(
    peptides: pd.DataFrame,
    *,
    n_permutations: int = N_PERMUTATIONS,
    random_state: int | None = None,
    min_peptides_per_protein: int = MIN_PEPTIDES_PER_PROTEIN,
    min_peptides_per_proteoform: int = MIN_PEPTIDES_PER_PROTEOFORM,
    min_clusters_per_protein: int = MIN_CLUSTERS_PER_PROTEIN,
    tie_arithmetic: str = "exact",
) -> pd.DataFrame:
    """Run one proximity test per eligible (protein, cluster) group.

    The low-level entry point: it takes a ranked peptide table rather
    than an ``AnnData``, and returns the per-test table that
    CCprofiler computes, with raw p-values only. Multiple-testing
    correction happens in `peptide_proximity`, so that its ``n`` is
    the number of tests actually performed.

    Parameters
    ----------
    peptides : pandas.DataFrame
        One row per peptide, with columns ``peptide_id``,
        ``protein_id``, ``cluster_id``, ``is_noise`` and ``rank``.
    n_permutations : int, optional
        Random groupings drawn per protein.
    random_state : int | None, optional
        Seed for the permutations. ``None`` uses CCprofiler's
        hard-coded 123, which is what produced the published numbers.
    min_peptides_per_protein : int, optional
        Proteins with fewer peptides are not tested.
    min_peptides_per_proteoform : int, optional
        Proteins whose median cluster size is below this are not
        tested.
    min_clusters_per_protein : int, optional
        Proteins with fewer clusters are not tested. ``1`` reproduces
        the reference.
    tie_arithmetic : {'exact', 'reference'}, optional
        How a permutation scoring equal to the observation is
        detected. See the module docstring.

    Returns
    -------
    pandas.DataFrame
        One row per test, sorted by ``protein_id`` then
        ``cluster_id``.
    """
    seed = CCPROFILER_SEED if random_state is None else int(random_state)
    tested = _eligible_proteins(
        peptides,
        min_peptides_per_protein,
        min_peptides_per_proteoform,
        min_clusters_per_protein,
    )
    bank = _PermutationBank(n_permutations, seed)

    rows: list[dict] = []
    for protein, block in peptides.groupby("protein_id", sort=False):
        if protein not in tested:
            continue
        permutations = bank.get(len(block))
        ranks_all = block["rank"].to_numpy()
        for cluster, group in block.groupby(
            "cluster_id", sort=False, dropna=False
        ):
            if len(group) < MIN_PEPTIDES_PER_CLUSTER:
                continue
            positions = group["rank"].to_numpy() - 1
            real, n_le, n_lt, pval, pseudo = _test_one_cluster(
                ranks_all[positions],
                permutations,
                positions,
                n_permutations,
                tie_arithmetic,
            )
            rows.append(
                {
                    "protein_id": protein,
                    "cluster_id": cluster,
                    "is_noise_cluster": bool(group["is_noise"].all()),
                    "n_peptides": len(block),
                    "n_peptides_per_cluster": len(group),
                    "normalized_sd": real,
                    "n_random_le": n_le,
                    "n_random_lt": n_lt,
                    _PVAL: pval,
                    _PSEUDO_PVAL: pseudo,
                }
            )

    tests = pd.DataFrame(rows, columns=list(TEST_COLUMNS))
    tests = tests.sort_values(["protein_id", "cluster_id"])
    return tests.reset_index(drop=True)


def _peptide_table(
    var: pd.DataFrame,
    annotator: str | Path | dict[str, str],
    protein_col: str,
    peptide_col: str,
    cluster_col: str,
    noise: float,
    locate_kwargs: dict,
) -> pd.DataFrame:
    """Ranked peptides with their cluster, ready for the tests."""
    ranked = peptide_position_ranks(
        var,
        annotator,
        protein_col=protein_col,
        peptide_col=peptide_col,
        **locate_kwargs,
    )
    cluster = pd.to_numeric(var[cluster_col], errors="coerce")
    ranked["cluster_id"] = cluster.reindex(ranked.index).to_numpy()
    ranked["is_noise"] = ranked["cluster_id"].to_numpy() >= noise
    # A peptide without a cluster belongs to no proteoform group and so
    # cannot be tested; it is dropped here and left NaN in `.var`.
    return ranked.loc[ranked["cluster_id"].notna()]


def _adjust(pvalues: pd.Series) -> pd.Series:
    """Benjamini-Hochberg over the non-missing entries only.

    ``n`` is the number of entries that carry a p-value, so a family
    is never inflated by the members that were not tested. Missing
    entries stay missing.
    """
    out = pd.Series(np.nan, index=pvalues.index, dtype=float)
    present = pvalues.notna().to_numpy()
    if present.any():
        out.loc[present] = multipletests(
            pvalues.to_numpy(dtype=float)[present], method="fdr_bh"
        )[1]
    return out


def _write_var(
    var: pd.DataFrame,
    tests: pd.DataFrame,
    protein_col: str,
    cluster_col: str,
    noise: float,
    key_added: str,
) -> None:
    """Broadcast the per-test values onto every peptide of the group.

    Two families are written. The per-cluster statistics repeat the
    ``(protein_id, cluster_id)`` group's value across its peptides.
    The ``unified_`` statistics repeat the protein's **minimum over
    its clusters**, skipping clusters that have no value.

    What guards the minimum is the protein's structure, not the
    presence of every cluster: a protein qualifies when it has at
    least ``MIN_NON_NOISE_CLUSTERS_FOR_UNIFIED`` distinct non-noise
    clusters, and gets ``NaN`` otherwise. Below two there is no
    proteoform split to localise, so a minimum would be meaningless;
    at two or more, every non-noise cluster carries a value anyway,
    since COPF never leaves a real cluster with fewer than two
    peptides. Skipping rather than propagating is what keeps an
    untested *noise* cluster -- a placeholder the reference scores 1
    -- from suppressing a real finding in a sibling cluster.

    The two families are corrected **separately**, because they answer
    different questions. ``pval_adj`` corrects over the tests, one
    entry per cluster. ``unified_pval_adj`` is computed from
    ``unified_pval`` directly, one entry per protein -- not as the
    minimum of the per-cluster adjusted values, which would carry the
    cluster-level ``n`` into a protein-level statement.
    """
    keys = pd.DataFrame(
        {
            "protein_id": var[protein_col].astype(str).to_numpy(),
            "cluster_id": pd.to_numeric(
                var[cluster_col], errors="coerce"
            ).to_numpy(),
        },
        index=var.index,
    )
    indexed = tests.set_index(["protein_id", "cluster_id"])
    lookup = pd.MultiIndex.from_frame(keys)

    # -- per cluster
    for stat in STAT_COLUMNS:
        var[f"{key_added}_{stat}"] = (
            indexed[stat].reindex(lookup).to_numpy(dtype=float)
        )

    # -- per protein, over the clusters `.var` actually holds, so a
    #    cluster that got no row still counts as missing
    groups = keys.drop_duplicates()
    groups_lookup = pd.MultiIndex.from_frame(groups)
    proteins = keys["protein_id"].to_numpy()

    # A protein qualifies on its number of distinct non-noise clusters.
    real = groups[
        groups["cluster_id"].notna() & (groups["cluster_id"] < noise)
    ]
    n_real = (
        real.groupby("protein_id")["cluster_id"]
        .nunique()
        .reindex(pd.unique(groups["protein_id"]))
        .fillna(0)
    )
    qualifies = n_real >= MIN_NON_NOISE_CLUSTERS_FOR_UNIFIED

    unified: dict[str, pd.Series] = {}
    for stat in (_PVAL, _PSEUDO_PVAL):
        values = pd.Series(
            indexed[stat].reindex(groups_lookup).to_numpy(dtype=float),
            index=groups["protein_id"].to_numpy(),
        )
        # `min` skips NaN, so an untested cluster simply does not vote.
        minimum = values.groupby(level=0).min()
        unified[stat] = minimum.where(qualifies.reindex(minimum.index))

    # One correction per protein, over the per-protein p-values.
    unified[_PVAL_ADJ] = _adjust(unified[_PVAL])

    for stat in STAT_COLUMNS:
        var[f"{key_added}_{UNIFIED}_{stat}"] = (
            unified[stat].reindex(proteins).to_numpy(dtype=float)
        )


def peptide_proximity(
    adata: ad.AnnData,
    annotator: str | Path | dict[str, str],
    *,
    protein_col: str = "protein_id",
    peptide_col: str = "peptide_id",
    cluster_col: str = "cluster_id",
    noise: float = NOISE,
    n_permutations: int = N_PERMUTATIONS,
    min_peptides_per_protein: int = MIN_PEPTIDES_PER_PROTEIN,
    min_peptides_per_proteoform: int = MIN_PEPTIDES_PER_PROTEOFORM,
    min_clusters_per_protein: int = MIN_CLUSTERS_PER_PROTEIN,
    mod_regex: str = CCPROFILER_MOD_REGEX,
    alphabet: Iterable[str] = IUPAC_AMINO_ACIDS,
    on_unknown_protein: str = "raise",
    on_unlocated_peptide: str = "raise",
    tie_arithmetic: str = "exact",
    random_state: int | None = None,
    key_added: str = "peptide_proximity",
    inplace: bool = True,
    verbose: bool = False,
) -> ad.AnnData | None:
    """
    Test whether a proteoform's peptides cluster in the sequence.

    Reimplements CCprofiler's ``evaluateProteoformLocation`` [1]_, the
    downstream characterisation COPF applies to the proteoform groups
    it detects. Peptide positions are resolved from ``annotator``,
    peptides are ranked by position within their protein, and each
    proteoform cluster's rank dispersion is compared against
    ``n_permutations`` random groupings of the same size. See the
    module docstring for the formula and for the six reference
    behaviours reproduced on purpose.

    Six columns are written to ``.var``, in two families of three.
    The first is a property of the ``(protein_id, cluster_id)`` group
    and is therefore **repeated across every peptide of that group**:

    ============================= =================================
    ``{key_added}_pval``          classical empirical p-value
    ``{key_added}_pval_adj``      the same, Benjamini-Hochberg
                                  adjusted over the tests performed
    ``{key_added}_pseudo_pval``   pseudo p-value, counting only
                                  strictly smaller permutations
    ============================= =================================

    The second is a property of the **protein**: the minimum over its
    clusters, repeated across all its peptides, and named
    ``{key_added}_unified_pval``, ``{key_added}_unified_pval_adj`` and
    ``{key_added}_unified_pseudo_pval``. This is the quantity
    CCprofiler's ``getProteoformStats`` thresholds, so it reproduces
    the published split directly.

    The minimum **skips** clusters with no value, and what decides
    whether a protein gets one at all is its structure: it needs at
    least two distinct non-noise clusters, and is ``NaN`` otherwise.
    Below two there is no proteoform split to localise, so a minimum
    would say nothing; at two or more, every non-noise cluster carries
    a value anyway, because COPF never leaves a real cluster with
    fewer than two peptides. Skipping rather than propagating is what
    keeps an untested *noise* cluster -- one peptide, no standard
    deviation, a placeholder the reference scores 1 -- from
    suppressing a real finding in a sibling cluster.

    The two families are corrected **separately**, because they answer
    different questions. ``pval_adj`` corrects over the tests, one
    entry per cluster. ``unified_pval_adj`` is Benjamini-Hochberg
    applied to ``unified_pval`` itself, one entry per protein -- *not*
    the minimum of the per-cluster adjusted values, which would drag
    the cluster-level ``n`` into a protein-level statement. In both
    families ``n`` counts only the entries that carry a p-value, so a
    family is never inflated by members that were not tested.

    ⚠️ ``unified_pval`` is a **minimum over several dependent tests**,
    so it is not a calibrated p-value: its null is stochastically
    smaller than uniform, and on the reference dataset its median is
    0.30 rather than 0.50. Benjamini-Hochberg over it is therefore
    anti-conservative. Read ``unified_pval_adj`` as a ranking
    statistic, and if a calibrated per-protein p-value is needed,
    correct within the protein first -- Šidák, ``1 - (1 - p) ** k``
    over its ``k`` clusters -- rather than treating the minimum as
    though it were one test.

    ``P10852`` is the case that fixes the rule. Its evidence is a
    two-peptide cluster whose pseudo p-value reaches the floor, while
    its noise cluster holds a single peptide no test can score. It is
    one of the 19 the publication counts, and propagating the missing
    cluster instead of skipping it would have dropped it.

    🔴 On the reference dataset **every** adjusted p-value comes out
    at ``1.0``, and that is a property of these data rather than of
    the arithmetic. Benjamini-Hochberg's smallest adjusted value is
    ``min_j (n / j) * p_(j)`` over the ascending p-values -- the
    ``j = 1`` term is the largest member of that family, not a bound
    on it, so a floor on the raw p-value saturates nothing by itself.
    Here the curve sits strictly above 1 for every ``j < n`` and
    touches 1 at ``j = n``; equivalently ``p_(j) > j / n``
    throughout, so the empirical p-value distribution lies entirely
    below the uniform diagonal and no FDR level could declare an
    excess of small p-values.

    Two reasons, and only the second is decisive. The classical
    p-value is *conservative by construction*, because it counts
    permutations scoring **exactly** the observed value as evidence
    against, and rank dispersion is discrete enough that ties are
    abundant -- a mean of 89 tied permutations per 1,000, and 160 for
    the two-peptide clusters that are 41 % of the family. But the
    calibrated mid-p variant removes that bias exactly, its median
    landing on 0.500, and BH still returns nothing below 0.95. So the
    real reason is that proximity is a *sparse* effect: the
    publication flags 26 proteins, which is 26 of some three thousand
    clusters -- far too few to move a procedure controlling the false
    discovery rate over the whole family. Raising ``n_permutations``
    does not help, and neither does restricting the family: over just
    the 63 significant proteins' clusters the smallest adjusted value
    is 0.28.

    CCprofiler computes no adjusted p-value at all --
    ``evaluateProteoformLocation`` accepts an ``adj.method`` argument
    and never uses it -- and the published counts are raw, so compare
    against the publication on ``{key_added}_pval``.

    ⚠️ Do not reach for ``{key_added}_pseudo_pval`` as a way round
    this. It is the opposite extreme -- it discards the ties rather
    than counting them -- and is anti-conservative to the point of
    being unusable under FDR control: 383 tests reach its floor, 284
    of them two-peptide clusters whose *classical* p-value has a
    median of 0.23. Applying BH to it would "reject" 390 tests on the
    strength of that discreteness alone. The publication uses it as a
    secondary "lowest possible p-value" criterion, never as a p-value
    to correct.

    Peptides of a protein that was not eligible, or that carry no
    cluster, get ``NaN`` -- as does every peptide when
    ``on_unknown_protein`` or ``on_unlocated_peptide`` excluded it.
    **Do not expect a value on every row**: with the reference
    defaults a protein needs at least four peptides and a median
    cluster size of two, so on the mouse tissue dataset 1,272 of
    2,885 proteins are tested.

    ``.X`` is neither read nor written, so ``layer``, ``zero_to_na``
    and ``fill_na`` have no meaning here and are absent.

    Parameters
    ----------
    adata : AnnData
        Peptide-level data. Only ``.var`` is read.
    annotator : str | pathlib.Path | dict
        Path to a FASTA file, or a pre-parsed
        ``{accession: sequence}`` mapping, supplying the protein
        sequences that peptide positions are resolved against. This
        is the load-bearing sequence input: the ranks, and hence
        every p-value, come from it.
    protein_col, peptide_col : str, optional
        Columns in ``.var`` holding the protein and peptide
        identifiers.
    cluster_col : str, optional
        Column in ``.var`` holding the peptide cluster, as written by
        :func:`~proteopy.tl.peptide_clusters_from_dendograms`.
    noise : float, optional
        Cluster label marking peptides that joined no multi-peptide
        cluster. The noise cluster is tested like any other -- it is
        one of the protein's proteoform groups in the reference --
        but is flagged in the per-test table.
    n_permutations : int, optional
        Random groupings drawn per protein. The empirical p-value
        cannot go below ``1 / (n_permutations + 1)``.
    min_peptides_per_protein : int, optional
        Proteins with fewer peptides are not tested. The reference's
        ``4``, which is deliberately stricter than the ``2`` the main
        COPF pipeline filters on.
    min_peptides_per_proteoform : int, optional
        Proteins whose median cluster size is below this are not
        tested.
    min_clusters_per_protein : int, optional
        Proteins with fewer clusters are not tested. A single cluster
        holding every peptide of a protein has rank vector ``1..n``,
        and so does every permutation of it, so its score is exactly
        1 and its pseudo p-value hits the floor whatever the data
        say. Pass ``1`` for the reference's behaviour; on the
        published analysis both select the identical 1,272 proteins.
    mod_regex : str, optional
        Pattern for annotations to disregard when locating a peptide
        in its protein sequence. Defaults to CCprofiler's
        ``\\(UniMod:[0-9]+\\)``.
    alphabet : Iterable[str], optional
        Characters allowed to remain after ``mod_regex`` is applied,
        which is what makes ``mod_regex`` self-checking.
    on_unknown_protein : {'raise', 'keep', 'skip'}, optional
        What to do about a protein absent from ``annotator``.
        ``'keep'`` gives its peptides unresolved positions, which
        ranks them first and is the CCprofiler behaviour; ``'skip'``
        excludes them from the analysis. ``'raise'`` is the default
        because silently proceeding is the reference's blind spot,
        and reproducing it should be an explicit choice.
    on_unlocated_peptide : {'raise', 'keep', 'skip'}, optional
        The same, for a peptide whose protein is present but whose
        sequence is not found in it.
    tie_arithmetic : {'exact', 'reference'}, optional
        How a permutation scoring equal to the observation is
        detected -- the comparison that separates the two p-values.
        ``'exact'`` compares an exact integer dispersion with the
        same ordering as the score; ``'reference'`` compares float
        scores in R's arithmetic order, which misjudges ties.
    random_state : int | None, optional
        Seed for the permutations. ``None`` uses CCprofiler's
        hard-coded ``123``, so the default run is the one comparable
        with the publication.
    key_added : str, optional
        Prefix of the three ``.var`` columns, and the ``.uns`` key
        holding the per-test table.
    inplace : bool, optional
        If False, return a modified copy instead of writing to
        ``adata``.
    verbose : bool, optional
        Print what was read, how many tests ran, and where results
        were stored.

    Returns
    -------
    AnnData | None
        ``None`` when ``inplace`` is True, otherwise the modified
        copy.

    Raises
    ------
    KeyError
        If ``protein_col``, ``peptide_col`` or ``cluster_col`` is
        absent from ``.var``.
    ValueError
        On an out-of-range or unknown argument, or -- under the
        default policies -- if a protein or peptide sequence cannot
        be resolved from ``annotator``.

    See Also
    --------
    proteopy.tl.peptide_clusters_from_dendograms : produces
        ``cluster_id``.
    proteopy.pp.summarize_peptides_by_neighbourhood_union : resolves
        peptide positions the same way.

    Examples
    --------
    >>> import proteopy as pr
    >>> adata = pr.datasets.example_peptide_data()
    >>> pr.tl.peptide_proximity(adata, "mouse.fasta")
    >>> pr.tl.peptide_proximity(
    ...     adata,
    ...     "mouse.fasta",
    ...     on_unlocated_peptide="keep",
    ...     verbose=True,
    ... )
    >>> adata.var["peptide_proximity_pval"].head(2)

    References
    ----------
    .. [1] Bludau, I. et al. Systematic detection of functional
       proteoform groups from bottom-up proteomic datasets. Nat
       Commun 12, 3810 (2021).
       https://doi.org/10.1038/s41467-021-24030-x
    """
    check_proteodata(adata)
    _validate_arguments(
        n_permutations,
        min_peptides_per_protein,
        min_peptides_per_proteoform,
        min_clusters_per_protein,
        tie_arithmetic,
        random_state,
    )
    _validate_var(adata.var, protein_col, peptide_col, cluster_col)

    target = adata if inplace else adata.copy()

    peptides = _peptide_table(
        target.var,
        annotator,
        protein_col,
        peptide_col,
        cluster_col,
        noise,
        {
            "mod_regex": mod_regex,
            "alphabet": alphabet,
            "on_unknown_protein": on_unknown_protein,
            "on_unlocated_peptide": on_unlocated_peptide,
        },
    )
    tests = peptide_proximity_(
        peptides,
        n_permutations=n_permutations,
        random_state=random_state,
        min_peptides_per_protein=min_peptides_per_protein,
        min_peptides_per_proteoform=min_peptides_per_proteoform,
        min_clusters_per_protein=min_clusters_per_protein,
        tie_arithmetic=tie_arithmetic,
    )

    # -- Benjamini-Hochberg over the classical p-values only, one entry
    #    per test rather than per peptide, so `n` is the number of
    #    tests performed. The reference computes no adjusted p-value at
    #    all -- `evaluateProteoformLocation` takes an `adj.method`
    #    argument and never uses it, and the published counts are raw.
    tests[_PVAL_ADJ] = (
        _adjust(tests[_PVAL]) if len(tests) else pd.Series(dtype=float)
    )
    tests = tests[list(TEST_COLUMNS[:-2]) + list(STAT_COLUMNS)]

    _write_var(target.var, tests, protein_col, cluster_col, noise, key_added)
    target.uns[key_added] = {
        "tests": tests,
        "n_tests": int(len(tests)),
        "n_proteins_tested": int(tests["protein_id"].nunique()),
        "params": {
            "n_permutations": int(n_permutations),
            "random_state": (
                CCPROFILER_SEED if random_state is None else int(random_state)
            ),
            "min_peptides_per_protein": int(min_peptides_per_protein),
            "min_peptides_per_proteoform": int(min_peptides_per_proteoform),
            "min_clusters_per_protein": int(min_clusters_per_protein),
            "tie_arithmetic": tie_arithmetic,
            "noise": float(noise),
            "rng": (
                "numpy.random.default_rng, re-seeded per peptide count "
                "-- not R's sample() stream"
            ),
        },
    }

    if verbose:
        n_peptides = len(peptides)
        print(
            f"peptide_proximity: ranked {n_peptides} peptides from "
            f"adata.var['{peptide_col}'] against sequences in "
            f"{annotator if isinstance(annotator, (str, Path)) else 'mapping'}"
        )
        print(
            f"peptide_proximity: {len(tests)} tests over "
            f"{tests['protein_id'].nunique()} proteins "
            f"({n_permutations} permutations, seed "
            f"{CCPROFILER_SEED if random_state is None else random_state})"
        )
        written = [f"{key_added}_{stat}" for stat in STAT_COLUMNS]
        written += [f"{key_added}_{col}" for col in UNIFIED_COLUMNS]
        print(
            "peptide_proximity: wrote .var["
            + ", ".join(f"'{name}'" for name in written)
            + f"] and .uns['{key_added}']"
        )

    if inplace:
        check_proteodata(target)
        return None
    check_proteodata(target)
    return target
