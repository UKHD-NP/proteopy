"""Proteoform group inference: a stable entry point over the methods.

`infer_proteoform_groups` dispatches to the currently recommended
proteoform inference approach. The building blocks remain available
individually (e.g. the COPF steps in `proteopy.tl.copf`) for users who
need finer control over each step.
"""

from numbers import Real

import anndata as ad

from proteopy.tl.copf import (
    pairwise_peptide_correlations,
    peptide_clusters_from_dendograms,
    peptide_dendograms_by_correlation,
    proteoform_scores,
)
from proteopy.utils.anndata import check_proteodata

DEFAULT_METHOD = "copf"


def _infer_copf(
    adata: ad.AnnData,
    *,
    batch_key: str | None,
    min_score: float,
    min_adj_pval: float,
    verbose: bool,
) -> None:
    """Run the COPF pipeline in place on ``adata``."""
    if verbose:
        batches = (
            f"batches from .obs['{batch_key}']" if batch_key else "no batches"
        )
        print(f"Computing pairwise peptide correlations ({batches}).")
    pairwise_peptide_correlations(adata, batch_key=batch_key)

    if verbose:
        print("Building peptide dendrograms from correlations.")
    peptide_dendograms_by_correlation(
        adata,
        method="agglomerative-hierarchical-clustering",
    )

    if verbose:
        print("Cutting dendrograms into two peptide clusters.")
    peptide_clusters_from_dendograms(
        adata,
        n_clusters=2,
        min_peptides_per_cluster=2,
    )

    if verbose:
        print(
            f"Scoring proteoforms (min_score={min_score}, "
            f"min_adj_pval={min_adj_pval})."
        )
    proteoform_scores(
        adata,
        min_score=min_score,
        min_pval_adj=min_adj_pval,
    )

    if verbose:
        is_pf = adata.var.groupby("protein_id", observed=True)[
            "is_proteoform"
        ].first()
        print(
            f"{int((is_pf == 1).sum())} of {len(is_pf)} proteins called "
            "as having proteoforms. Results stored in .var and .uns."
        )


METHODS = {
    "copf": _infer_copf,
}


def _validate_args(
    adata: ad.AnnData,
    method: str,
    batch_key: str | None,
    min_score: float,
    min_adj_pval: float,
) -> None:
    _, level = check_proteodata(adata)
    if level != "peptide":
        raise ValueError(
            "infer_proteoform_groups requires peptide-level data, "
            f"got {level}-level data."
        )

    if method not in METHODS:
        raise ValueError(
            f"Unknown method '{method}'. "
            f"Available methods: {sorted(METHODS)}."
        )

    if batch_key is not None and batch_key not in adata.obs.columns:
        raise KeyError(f"batch_key '{batch_key}' not found in .obs.")

    for name, value in (
        ("min_score", min_score),
        ("min_adj_pval", min_adj_pval),
    ):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(
                f"{name} must be a number, got {type(value).__name__}."
            )

    if not 0 < min_adj_pval <= 1:
        raise ValueError(
            f"min_adj_pval must be in (0, 1], got {min_adj_pval}."
        )


def infer_proteoform_groups(
    adata: ad.AnnData,
    method: str = DEFAULT_METHOD,
    *,
    batch_key: str | None = None,
    min_score: float = 0.1,
    min_adj_pval: float = 0.1,
    inplace: bool = True,
    verbose: bool = False,
) -> ad.AnnData | None:
    """
    Infer proteoform groups from peptide-level data.

    A wrapper around the currently recommended proteoform inference
    approach in ProteoPy.

    .. warning::

       This is a **dynamic** function. The default ``method`` points
       to the approach the ProteoPy developers currently recommend and
       may change in future releases as better approaches become
       available, which would change the results of calls that rely on
       the default. For reproducible analyses, always pass ``method``
       explicitly, e.g. ``method="copf"``.

    Available methods:

    ``"copf"`` (current default)
        COPF, COrrelation based functional ProteoForm assessment
        [1]_. Equivalent to running::

            pr.tl.pairwise_peptide_correlations(
                adata, batch_key=batch_key,
            )
            pr.tl.peptide_dendograms_by_correlation(
                adata, method="agglomerative-hierarchical-clustering",
            )
            pr.tl.peptide_clusters_from_dendograms(
                adata, n_clusters=2, min_peptides_per_cluster=2,
            )
            pr.tl.proteoform_scores(
                adata, min_score=min_score, min_pval_adj=min_adj_pval,
            )

        Call these functions individually for finer control over each
        step.

    Parameters
    ----------
    adata : AnnData
        Peptide-level :class:`~anndata.AnnData` with ``.var``
        columns ``peptide_id`` and ``protein_id``.
    method : str
        Proteoform inference method. One of ``"copf"``. The default
        tracks the currently recommended method and may change; pass
        it explicitly for reproducibility.
    batch_key : str | None
        ``.obs`` column defining batches. For ``"copf"``, peptide
        correlations are computed per batch and pooled across batches
        (see :func:`~proteopy.tl.pairwise_peptide_correlations`). If
        ``None``, all observations are treated as a single batch.
    min_score : float
        Minimum proteoform score for a protein to be called as having
        proteoforms (see :func:`~proteopy.tl.proteoform_scores`).
    min_adj_pval : float
        Maximum Benjamini-Hochberg adjusted p-value for a protein to
        be called as having proteoforms.
    inplace : bool
        If True, modify ``adata`` in place; else return a modified
        copy.
    verbose : bool
        If True, print status messages about each step.

    Returns
    -------
    AnnData | None
        ``None`` if ``inplace=True``, else a modified copy of
        ``adata``. For ``"copf"``, the following are written:

        - ``.uns['pairwise_peptide_correlations']`` (and
          ``.uns['pairwise_peptide_correlations_by_batch']`` if
          ``batch_key`` is given)
        - ``.uns['dendograms']`` and ``.uns['clusters']``
        - ``.var`` columns ``cluster_id``, ``proteoform_id``,
          ``proteoform_score``, ``proteoform_score_z``,
          ``proteoform_score_dz``, ``proteoform_score_pval``,
          ``proteoform_score_pval_adj`` and ``is_proteoform``

    Examples
    --------
    >>> import proteopy as pr
    >>> pr.tl.infer_proteoform_groups(
    ...     adata,
    ...     method="copf",
    ...     batch_key="multiplex_batch",
    ...     min_score=0.1,
    ...     min_adj_pval=0.1,
    ... )
    >>> adata.var.loc[adata.var["is_proteoform"] == 1, "protein_id"]

    References
    ----------
    .. [1] Bludau I, Frank M, Dörig C, Cai Y, Heusel M, Rosenberger G,
       Picotti P, Collins BC, Röst H, and Aebersold R. "Systematic
       Detection of Functional Proteoform Groups from Bottom-Up
       Proteomic Datasets." *Nature Communications*, 12:3810, 2021.
       DOI: 10.1038/s41467-021-24030-x.
    """
    _validate_args(
        adata,
        method=method,
        batch_key=batch_key,
        min_score=min_score,
        min_adj_pval=min_adj_pval,
    )

    if verbose:
        print(f"Inferring proteoform groups with method:\n=== {method} ===")
    target = adata if inplace else adata.copy()
    METHODS[method](
        target,
        batch_key=batch_key,
        min_score=min_score,
        min_adj_pval=min_adj_pval,
        verbose=verbose,
    )
    check_proteodata(target)

    if inplace:
        return None
    return target
