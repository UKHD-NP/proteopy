"""Simulated proteoform groups for evaluating proteoform inference.

`proteoform_group_simulator` yields peptide-level AnnData objects in
which known proteoform groups have been planted by merging peptides of
two real proteins under one protein ID. After running a proteoform
inference method on each object (e.g.
`pr.tl.infer_proteoform_groups`), `proteoform_simulation_metrics`
measures how well the planted groups were recovered. Together they
estimate whether a dataset is suited for proteoform inference.
"""

import warnings
from collections.abc import Iterator, Sequence
from numbers import Integral

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import (
    adjusted_rand_score,
    average_precision_score,
    roc_auc_score,
)

from proteopy.utils.anndata import check_proteodata

SIMULATION_UNS_KEY = "proteoform_simulation"
MIXED = "mixed"
UNMIXED = "unmixed"
# COPF needs at least two peptides per cluster
MIN_GROUP_PEPTIDES = 2
# Metrics are computed under their classical classification names and
# renamed on return. Unmixed proteins may carry real proteoforms, so
# they are background rather than certain negatives.
METRIC_RENAMES = {
    "tpr": "detection_rate",
    "fpr": "background_detection_rate",
}

INFERENCE_VAR_COLUMNS = (
    "cluster_id",
    "proteoform_id",
    "proteoform_score",
    "proteoform_score_z",
    "proteoform_score_dz",
    "proteoform_score_pval",
    "proteoform_score_pval_adj",
    "is_proteoform",
)
INFERENCE_UNS_KEYS = (
    "pairwise_peptide_correlations",
    "pairwise_peptide_correlations_by_batch",
    "dendograms",
    "clusters",
)


def _is_int(value) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def _per_iteration(
    name: str,
    value,
    n_iterations: int,
    minimum: int,
) -> list[int]:
    """Coerce an int or a sequence of ints to one value per iteration."""
    if _is_int(value):
        values = [int(value)] * n_iterations
    elif isinstance(value, (Sequence, np.ndarray)) and not isinstance(
        value, str
    ):
        values = list(value)
        if len(values) != n_iterations:
            raise ValueError(
                f"`{name}` has {len(values)} elements but "
                f"`n_iterations` is {n_iterations}; a list must have "
                "exactly one element per iteration."
            )
        if not all(_is_int(v) for v in values):
            raise TypeError(f"All elements of `{name}` must be integers.")
        values = [int(v) for v in values]
    else:
        raise TypeError(
            f"`{name}` must be an int or a list of ints, "
            f"got {type(value).__name__}."
        )

    if any(v < minimum for v in values):
        raise ValueError(f"`{name}` values must be >= {minimum}.")
    return values


def _check_feasible(
    iteration: int,
    n_mixed: int,
    n_unmixed: int,
    n_mixed_peptides: int,
    n_group_eligible: int,
    n_donor_eligible: int,
) -> None:
    """Raise if the eligible proteins cannot fill one iteration."""
    # Pools are nested, so checking each pool and the union suffices
    n_cluster = n_mixed + n_unmixed
    n_union = max(n_group_eligible, n_donor_eligible)
    if (
        n_cluster > n_group_eligible
        or n_mixed > n_donor_eligible
        or n_cluster + n_mixed > n_union
    ):
        raise ValueError(
            f"Iteration {iteration} needs {n_cluster} proteins with >= "
            f"{MIN_GROUP_PEPTIDES} peptides (backgrounds and unmixed) "
            f"and {n_mixed} further donor proteins with >= "
            f"{n_mixed_peptides} peptides, but only {n_group_eligible} "
            f"proteins have >= {MIN_GROUP_PEPTIDES} peptides and "
            f"{n_donor_eligible} have >= {n_mixed_peptides}."
        )


def _validate_simulator_args(
    adata,
    n_iterations,
    n_mixed,
    n_unmixed,
    n_mixed_peptides,
    random_state,
):
    """Validate arguments and return the per-iteration parameters."""
    _, level = check_proteodata(adata)
    if level != "peptide":
        raise ValueError(
            "proteoform_group_simulator requires peptide-level data, "
            f"got {level}-level data."
        )
    if not _is_int(n_iterations) or n_iterations < 1:
        raise ValueError(
            f"`n_iterations` must be a positive int, got {n_iterations!r}."
        )
    if random_state is not None and (
        not _is_int(random_state) or random_state < 0
    ):
        raise ValueError(
            "`random_state` must be None or a non-negative int, "
            f"got {random_state!r}."
        )

    params = pd.DataFrame(
        {
            "n_mixed": _per_iteration("n_mixed", n_mixed, n_iterations, 0),
            "n_unmixed": _per_iteration(
                "n_unmixed", n_unmixed, n_iterations, 0
            ),
            "n_mixed_peptides": _per_iteration(
                "n_mixed_peptides", n_mixed_peptides, n_iterations, 1
            ),
        }
    )
    if ((params["n_mixed"] + params["n_unmixed"]) == 0).any():
        raise ValueError(
            "Each iteration needs at least one protein: `n_mixed` and "
            "`n_unmixed` must not both be 0."
        )
    return params


def _warn_inference_results(adata) -> None:
    present = [c for c in INFERENCE_VAR_COLUMNS if c in adata.var.columns]
    present += [f".uns['{k}']" for k in INFERENCE_UNS_KEYS if k in adata.uns]
    if present:
        warnings.warn(
            "`adata` contains proteoform inference results "
            f"({', '.join(present)}). They are copied into the simulated "
            "objects but do not describe the simulated proteins; rerun "
            "the inference on each simulated object before computing "
            "metrics.",
            UserWarning,
            stacklevel=3,
        )


def _draw_proteins(
    rng,
    group_pool,
    donor_pool,
    n_mixed,
    n_unmixed,
    n_mixed_peptides,
):
    """Draw disjoint background, donor and unmixed proteins."""
    n_cluster = n_mixed + n_unmixed
    # Draw from the smaller (nested) pool first so the draw never
    # fails when the counts are feasible
    if n_mixed_peptides >= MIN_GROUP_PEPTIDES:
        donors = rng.choice(donor_pool, n_mixed, replace=False)
        group = rng.choice(
            np.setdiff1d(group_pool, donors),
            n_cluster,
            replace=False,
        )
    else:
        group = rng.choice(group_pool, n_cluster, replace=False)
        donors = rng.choice(
            np.setdiff1d(donor_pool, group),
            n_mixed,
            replace=False,
        )
    return group[:n_mixed], donors, group[n_mixed:]


def _simulate_one(
    adata,
    rng,
    protein_indices,
    group_pool,
    donor_pool,
    n_mixed,
    n_unmixed,
    n_mixed_peptides,
):
    """Build one simulated AnnData."""
    backgrounds, donors, unmixed = _draw_proteins(
        rng,
        group_pool,
        donor_pool,
        n_mixed,
        n_unmixed,
        n_mixed_peptides,
    )

    indices, protein_ids, sim_types = [], [], []
    for background, donor in zip(backgrounds, donors):
        idx = np.concatenate(
            [
                protein_indices[background],
                rng.choice(
                    protein_indices[donor],
                    n_mixed_peptides,
                    replace=False,
                ),
            ]
        )
        indices.append(idx)
        protein_ids += [f"{background}_{donor}"] * len(idx)
        sim_types += [MIXED] * len(idx)
    for protein in unmixed:
        idx = protein_indices[protein]
        indices.append(idx)
        protein_ids += [protein] * len(idx)
        sim_types += [UNMIXED] * len(idx)

    sim = adata[:, np.concatenate(indices)].copy()
    if sparse.issparse(sim.X):
        sim.X = sim.X.toarray()
    sim.var["source_protein_id"] = sim.var["protein_id"].astype(str)
    sim.var["protein_id"] = protein_ids
    sim.var["simulation_type"] = sim_types
    return sim


def _simulate(
    adata,
    params,
    random_state,
    verbose,
):
    proteins = adata.var["protein_id"].astype(str)
    protein_indices = proteins.groupby(proteins).indices
    counts = pd.Series(
        {p: len(idx) for p, idx in protein_indices.items()}
    ).sort_index()
    group_pool = counts.index[counts >= MIN_GROUP_PEPTIDES].to_numpy()

    seeds = np.random.SeedSequence(random_state).spawn(len(params))
    for iteration, row in params.iterrows():
        donor_pool = counts.index[counts >= row["n_mixed_peptides"]].to_numpy()

        if verbose:
            print(
                f"Simulating iteration {iteration + 1}/{len(params)}: "
                f"n_mixed={row['n_mixed']}, "
                f"n_unmixed={row['n_unmixed']}, "
                f"n_mixed_peptides={row['n_mixed_peptides']}."
            )

        sim = _simulate_one(
            adata,
            rng=np.random.default_rng(seeds[iteration]),
            protein_indices=protein_indices,
            group_pool=group_pool,
            donor_pool=donor_pool,
            n_mixed=int(row["n_mixed"]),
            n_unmixed=int(row["n_unmixed"]),
            n_mixed_peptides=int(row["n_mixed_peptides"]),
        )
        sim.uns[SIMULATION_UNS_KEY] = {
            "iteration": int(iteration),
            "random_state": int(random_state),
            "n_mixed": int(row["n_mixed"]),
            "n_unmixed": int(row["n_unmixed"]),
            "n_mixed_peptides": int(row["n_mixed_peptides"]),
        }
        check_proteodata(sim)
        yield sim


def proteoform_group_simulator(
    adata: ad.AnnData,
    *,
    n_iterations: int = 1,
    n_mixed: int | Sequence[int] = 1000,
    n_unmixed: int | Sequence[int] = 1000,
    n_mixed_peptides: int | Sequence[int] = 2,
    random_state: int | None = None,
    verbose: bool = False,
) -> Iterator[ad.AnnData]:
    """
    Iterate over peptide-level datasets with planted proteoform groups.

    Each iteration draws proteins from ``adata`` and builds a new
    AnnData from their peptides:

    - **mixed** proteins: all peptides of a *background* protein plus
      ``n_mixed_peptides`` peptides of a *donor* protein, relabelled
      with the protein ID ``"{background}_{donor}"``. The two source
      proteins form two known proteoform groups.
    - **unmixed** proteins: all peptides of a protein, unchanged.

    Background and unmixed proteins have at least two peptides; donor
    proteins at least ``n_mixed_peptides``. All drawn proteins within
    an iteration are distinct.

    Unmixed proteins are real proteins and may carry real proteoform
    groups.

    Running a proteoform inference method on each yielded object and
    passing it to :func:`~proteopy.utils.proteoform_simulation_metrics`
    estimates how well the method recovers proteoform groups in this
    dataset.

    Parameters
    ----------
    adata : AnnData
        Peptide-level :class:`~anndata.AnnData` with ``.var`` columns
        ``peptide_id`` and ``protein_id``.
    n_iterations : int
        Number of simulated datasets to yield.
    n_mixed : int | Sequence[int]
        Number of mixed proteins per iteration.
    n_unmixed : int | Sequence[int]
        Number of unmixed proteins per iteration.
    n_mixed_peptides : int | Sequence[int]
        Number of donor peptides added to each mixed protein.
    random_state : int | None
        Seed for the simulation. Iteration *i* depends only on
        ``adata``, its parameters, ``random_state`` and *i*, not on
        ``n_iterations``, so it is identical whether 5 or 50 iterations
        are requested. If ``None``, a random seed is drawn and stored
        in ``.uns['proteoform_simulation']`` so the run can be
        repeated.
    verbose : bool
        If True, print the parameters of each iteration.

    ``n_mixed``, ``n_unmixed`` and ``n_mixed_peptides`` accept a
    single int, used in every iteration, or a sequence with exactly
    ``n_iterations`` elements whose *i*-th element is used in iteration
    *i*. Ints and sequences can be mixed, e.g. to sweep the
    mixed:unmixed ratio at a fixed ``n_mixed_peptides``.

    Returns
    -------
    Iterator[AnnData]
        A single-pass generator of ``n_iterations`` peptide-level
        AnnData objects. ``.X`` is dense. Each object has, in addition
        to the columns of ``adata``:

        - ``.var['protein_id']``: simulated protein ID
        - ``.var['source_protein_id']``: the peptide's original protein
        - ``.var['simulation_type']``: ``"mixed"`` or ``"unmixed"``
        - ``.uns['proteoform_simulation']``: ``iteration``,
          ``random_state``, ``n_mixed``, ``n_unmixed`` and
          ``n_mixed_peptides`` of this iteration

    Raises
    ------
    ValueError
        If ``adata`` is not peptide-level, a sequence parameter does
        not have ``n_iterations`` elements, or ``adata`` has too few
        eligible proteins for any iteration.

    Warns
    -----
    UserWarning
        If ``.X`` is sparse (the simulated objects are dense), or if
        ``adata`` already holds proteoform inference results. These
        are copied into the simulated objects unchanged and must be
        recomputed.

    Examples
    --------
    >>> import pandas as pd
    >>> import proteopy as pr
    >>> records = []
    >>> for sim in pr.utils.proteoform_group_simulator(
    ...     adata,
    ...     n_iterations=10,
    ...     n_mixed=500,
    ...     n_unmixed=500,
    ...     n_mixed_peptides=2,
    ...     random_state=42,
    ... ):
    ...     pr.tl.infer_proteoform_groups(sim, method="copf")
    ...     metrics, _ = pr.utils.proteoform_simulation_metrics(sim)
    ...     records.append({**sim.uns["proteoform_simulation"], **metrics})
    >>> results = pd.DataFrame(records)

    Sweep the mixed:unmixed ratio in one call:

    >>> sims = pr.utils.proteoform_group_simulator(
    ...     adata,
    ...     n_iterations=3,
    ...     n_mixed=[100, 250, 500],
    ...     n_unmixed=[900, 750, 500],
    ...     random_state=42,
    ... )

    A single simulated dataset:

    >>> sim = next(pr.utils.proteoform_group_simulator(adata))
    """
    params = _validate_simulator_args(
        adata,
        n_iterations,
        n_mixed,
        n_unmixed,
        n_mixed_peptides,
        random_state,
    )

    counts = adata.var["protein_id"].astype(str).value_counts()
    n_group_eligible = int((counts >= MIN_GROUP_PEPTIDES).sum())
    for iteration, row in params.iterrows():
        _check_feasible(
            iteration,
            n_mixed=row["n_mixed"],
            n_unmixed=row["n_unmixed"],
            n_mixed_peptides=row["n_mixed_peptides"],
            n_group_eligible=n_group_eligible,
            n_donor_eligible=int((counts >= row["n_mixed_peptides"]).sum()),
        )

    if sparse.issparse(adata.X):
        warnings.warn(
            "Sparse input is densified in each simulated AnnData.",
            UserWarning,
            stacklevel=2,
        )
    _warn_inference_results(adata)

    if random_state is None:
        random_state = int(
            np.random.SeedSequence().generate_state(1, np.uint32)[0]
        )
    if verbose:
        print(
            f"Simulating {n_iterations} dataset(s) from "
            f"{adata.n_vars} peptides with random_state={random_state}."
        )

    return _simulate(adata, params, random_state, verbose)


def _per_protein_results(
    var,
    score_key,
    pval_key,
    call_key,
    cluster_key,
):
    """One row per simulated protein: type, call, score, p and ARI."""
    rows = []
    for protein_id, df in var.groupby("protein_id", observed=True):
        sim_type = df["simulation_type"].iloc[0]
        ari = np.nan
        if sim_type == MIXED:
            ari = adjusted_rand_score(
                df["source_protein_id"].astype(str),
                df[cluster_key].astype(str),
            )
        rows.append(
            {
                "protein_id": protein_id,
                "simulation_type": sim_type,
                "called": bool(df[call_key].fillna(0).astype(bool).any()),
                "score": df[score_key].iloc[0],
                "pval": df[pval_key].iloc[0],
                "ari": ari,
                "cluster_match": bool(ari == 1.0),
            }
        )
    return pd.DataFrame(rows).set_index("protein_id")


def _ranking_metrics(is_mixed, scores):
    """AUROC and AUPRC of scores where higher ranks first.

    NaN scores rank lowest.
    """
    if is_mixed.all() or not is_mixed.any():
        return np.nan, np.nan
    scores = scores.astype(float)
    finite = scores[np.isfinite(scores)]
    floor = finite.min() - 1 if len(finite) else 0.0
    scores = np.where(np.isfinite(scores), scores, floor)
    return (
        float(roc_auc_score(is_mixed, scores)),
        float(average_precision_score(is_mixed, scores)),
    )


def _mean_or_nan(values) -> float:
    return float(np.mean(values)) if len(values) else np.nan


def _level_metrics(is_positive, called, rankings, level):
    """Confusion-matrix rates and ranking areas for one positive set.

    Keys use the classical names (tpr, fpr, ...); see
    `_rename_metrics`.
    """
    tpr = _mean_or_nan(called[is_positive])
    fpr = _mean_or_nan(called[~is_positive])
    metrics = {
        f"tpr_{level}": tpr,
        f"fpr_{level}": fpr,
        # Not reported: unmixed proteins are not certain negatives, so
        # these are hard to interpret. Uncomment to restore.
        # f"fnr_{level}": 1 - tpr,
        # f"tnr_{level}": 1 - fpr,
        # f"precision_{level}": _mean_or_nan(is_positive[called]),
    }
    for name, ranking in rankings.items():
        auroc, auprc = _ranking_metrics(is_positive, ranking)
        metrics[f"{name}_auroc_{level}"] = auroc
        metrics[f"{name}_auprc_{level}"] = auprc
    return metrics


def _rename_metrics(metrics):
    """Rename classical metric keys with `METRIC_RENAMES`."""
    renamed = {}
    for key, value in metrics.items():
        name, _, level = key.partition("_")
        if name in METRIC_RENAMES:
            key = f"{METRIC_RENAMES[name]}_{level}"
        renamed[key] = value
    return renamed


def proteoform_simulation_metrics(
    adata: ad.AnnData,
    *,
    score_key: str = "proteoform_score",
    pval_key: str = "proteoform_score_pval_adj",
    call_key: str = "is_proteoform",
    cluster_key: str = "cluster_id",
    verbose: bool = False,
) -> tuple[dict, pd.DataFrame]:
    """
    Measure how well planted proteoform groups were recovered.

    Takes one object yielded by
    :func:`~proteopy.utils.proteoform_group_simulator` after a
    proteoform inference method has been run on it, and returns one
    record of summary metrics plus the per-protein results they are
    computed from.

    Parameters
    ----------
    adata : AnnData
        Simulated peptide-level :class:`~anndata.AnnData` with
        ``.var`` columns ``source_protein_id``, ``simulation_type``,
        ``score_key``, ``pval_key``, ``call_key`` and ``cluster_key``.
    score_key : str
        ``.var`` column with the per-protein proteoform score. Higher
        values rank a protein as more likely a proteoform.
    pval_key : str
        ``.var`` column with the per-protein (adjusted) p-value. Lower
        values rank a protein as more likely a proteoform.
    call_key : str
        ``.var`` column flagging proteins called as having
        proteoforms (truthy values; NaN counts as not called).
    cluster_key : str
        ``.var`` column with the inferred peptide cluster.
    verbose : bool
        If True, print the metrics.

    Returns
    -------
    metrics : dict
        Summary metrics. The simulation parameters are not included;
        they are in ``.uns['proteoform_simulation']`` of ``adata``.

        Metrics come at two levels, which differ in which proteins
        are positives (planted proteoform groups). All other proteins
        are background (pseudo-negatives), and a called protein is a
        predicted positive at both levels.

        - ``_call``: mixed proteins are positives.
        - ``_cluster``: mixed proteins whose inferred clusters match
          their source proteins exactly (ARI = 1) are positives.
          Mixed proteins with a wrong clustering are background, so
          calling them counts toward
          ``background_detection_rate_cluster``. Which mixed proteins
          are positives depends on the clustering of the inference
          method.

        Background proteins are not certain negatives: unmixed
        proteins are real proteins and may carry real proteoforms.
        Metrics are therefore named after what they measure in the
        simulation rather than their classical classification terms.

        For each level (``*`` = ``call`` or ``cluster``):

        - ``detection_rate_*``: fraction of positives called (true
          positive rate, recall)
        - ``background_detection_rate_*``: fraction of background
          proteins called (false positive rate). An upper bound of the
          false positive rate, as some calls may be real proteoforms
        - ``score_auroc_*``, ``score_auprc_*``: area under the ROC
          and precision-recall curves, ranking proteins by
          ``score_key`` (higher first)
        - ``pval_auroc_*``, ``pval_auprc_*``: the same, ranking
          proteins by ``pval_key`` (lower first)

        and, independent of the level:

        - ``mean_ari``: mean adjusted Rand index between the inferred
          clusters and the source proteins over called mixed
          proteins. Each cluster label, including the noise label,
          counts as its own group.
        - ``cluster_match_rate``: fraction of called mixed proteins
          whose inferred clusters match their source proteins
          exactly (ARI = 1)

        Proteins without a score or p-value rank last. The ROC and
        precision-recall curves sweep a cutoff over the proteins of
        this one dataset, so they do not depend on the calling
        thresholds of the inference method. They treat background
        proteins as negatives, so real proteoforms among them lower
        both areas.

        Rates are NaN when undefined (e.g. no positives). The
        calls of most methods depend on multiple-testing correction
        and thereby on the mixed:unmixed ratio, as does AUPRC, whose
        chance level is the fraction of positives. AUROC does not
        depend on the ratio.
    results : pandas.DataFrame
        One row per simulated protein, indexed by ``protein_id``, with
        columns ``simulation_type``, ``called`` (bool), ``score`` (from
        ``score_key``), ``pval`` (from ``pval_key``), ``ari``
        (adjusted Rand index; NaN for unmixed proteins) and
        ``cluster_match`` (ARI = 1; False for unmixed proteins).

    Raises
    ------
    KeyError
        If a required ``.var`` column is missing.

    Examples
    --------
    >>> import proteopy as pr
    >>> sim = next(pr.utils.proteoform_group_simulator(
    ...     adata, random_state=42,
    ... ))
    >>> pr.tl.infer_proteoform_groups(sim, method="copf")
    >>> metrics, results = pr.utils.proteoform_simulation_metrics(sim)
    """
    check_proteodata(adata)
    simulation_columns = ["source_protein_id", "simulation_type"]
    inference_columns = [score_key, pval_key, call_key, cluster_key]
    missing = [
        c
        for c in simulation_columns + inference_columns
        if c not in adata.var.columns
    ]
    if missing:
        raise KeyError(
            f"Missing .var columns: {missing}. Simulate with "
            "`proteoform_group_simulator` and run a proteoform "
            "inference method (e.g. `pr.tl.infer_proteoform_groups`) "
            "first."
        )

    results = _per_protein_results(
        adata.var,
        score_key=score_key,
        pval_key=pval_key,
        call_key=call_key,
        cluster_key=cluster_key,
    )
    is_mixed = (results["simulation_type"] == MIXED).to_numpy()
    called = results["called"].to_numpy()
    ari = results["ari"].to_numpy()
    cluster_match = results["cluster_match"].to_numpy()
    # Higher ranks first, so p-values are negated
    rankings = {
        "score": results["score"].to_numpy(dtype=float),
        "pval": -results["pval"].to_numpy(dtype=float),
    }

    # -- Call level: mixed proteins are positives
    metrics = _level_metrics(is_mixed, called, rankings, "call")
    # -- Cluster level: only exactly clustered mixed proteins are positives
    metrics.update(
        _level_metrics(is_mixed & cluster_match, called, rankings, "cluster")
    )
    metrics["mean_ari"] = _mean_or_nan(ari[is_mixed & called])
    metrics["cluster_match_rate"] = _mean_or_nan(
        cluster_match[is_mixed & called]
    )
    metrics = _rename_metrics(metrics)

    if verbose:
        print(
            f"Metrics from .var columns {score_key!r}, {pval_key!r}, "
            f"{call_key!r} and {cluster_key!r} over {len(results)} "
            "proteins:"
        )
        print(pd.Series(metrics).to_string())

    return metrics, results
