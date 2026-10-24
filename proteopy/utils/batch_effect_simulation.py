"""Simulated batch effects for evaluating batch-aware methods.

`apply_peptide_batch_effects` shifts the intensities of selected peptides in
selected batches.
"""

import warnings
from collections.abc import Sequence
from numbers import Integral, Real

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from proteopy.utils.anndata import check_proteodata
from proteopy.utils.array import is_log_transformed

BATCH_EFFECT_UNS_KEY = "batch_effect_simulation"
BATCH_EFFECT_VAR_KEY = "batch_effect"
FLAVORS = ("subset_shift",)


def _is_int(value) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def _batch_labels(adata, batch_key):
    """Distinct non-missing batch labels in lexicographic order."""
    labels = pd.unique(adata.obs[batch_key].dropna())
    return sorted(labels, key=str)


def _validate_batches(batches, labels, batch_key):
    if _is_int(batches):
        if not 1 <= batches < len(labels):
            raise ValueError(
                f"`batches` as an int must be between 1 and "
                f"{len(labels) - 1} (the number of batches in "
                f".obs['{batch_key}'] minus 1), got {batches}."
            )
        return
    if isinstance(batches, str) or not isinstance(
        batches, (Sequence, np.ndarray, pd.Index)
    ):
        raise TypeError(
            "`batches` must be an int or a list of batch labels, "
            f"got {type(batches).__name__}."
        )
    batches = list(batches)
    if len(batches) == 0:
        raise ValueError("`batches` must not be empty.")
    if len(set(batches)) != len(batches):
        raise ValueError("`batches` must not contain duplicates.")
    missing = [b for b in batches if b not in labels]
    if missing:
        raise ValueError(
            f"Batches {missing} not found in .obs['{batch_key}']. "
            f"Available: {labels}."
        )
    if len(batches) == len(labels):
        raise ValueError(
            "`batches` must leave at least one batch unaffected; "
            "shifting every batch is a global shift that does not "
            "act as a batch effect."
        )


def _validate_args(
    adata,
    batch_key,
    batches,
    flavor,
    n_peptides,
    log2fc,
    fraction,
    log_space,
    random_state,
):
    """Validate arguments and return the batch labels."""
    _, level = check_proteodata(adata)
    if level != "peptide":
        raise ValueError(
            "apply_peptide_batch_effects requires peptide-level data, "
            f"got {level}-level data."
        )
    if flavor not in FLAVORS:
        raise ValueError(f"`flavor` must be one of {FLAVORS}, got {flavor!r}.")
    if batch_key not in adata.obs.columns:
        raise KeyError(f"batch_key '{batch_key}' not found in .obs.")
    labels = _batch_labels(adata, batch_key)
    if len(labels) < 2:
        raise ValueError(
            f".obs['{batch_key}'] must contain at least 2 batches, "
            f"found {len(labels)}."
        )
    _validate_batches(batches, labels, batch_key)

    if not _is_int(n_peptides) or n_peptides < 1:
        raise ValueError(
            f"`n_peptides` must be a positive int, got {n_peptides!r}."
        )
    if (
        not isinstance(log2fc, Real)
        or isinstance(log2fc, bool)
        or not np.isfinite(log2fc)
    ):
        raise ValueError(f"`log2fc` must be a finite number, got {log2fc!r}.")
    if (
        not isinstance(fraction, Real)
        or isinstance(fraction, bool)
        or not 0 < fraction <= 1
    ):
        raise ValueError(f"`fraction` must be in (0, 1], got {fraction!r}.")
    if not isinstance(log_space, bool):
        raise TypeError(
            f"`log_space` must be a bool, got {type(log_space).__name__}."
        )
    if random_state is not None and (
        not _is_int(random_state) or random_state < 0
    ):
        raise ValueError(
            "`random_state` must be None or a non-negative int, "
            f"got {random_state!r}."
        )
    return labels


def _draw_peptides(rng, proteins, n_peptides, fraction):
    """Draw affected proteins and `n_peptides` peptides of each.

    Returns the positional var indices of the drawn peptides and the
    number of drawn proteins.
    """
    protein_indices = proteins.groupby(proteins).indices
    # A protein must keep at least one unshifted peptide, else the
    # shift moves the whole protein
    eligible = sorted(
        p for p, idx in protein_indices.items() if len(idx) > n_peptides
    )
    n_affected = int(round(fraction * len(eligible)))
    if n_affected == 0:
        return np.array([], dtype=int), 0
    affected = rng.choice(eligible, n_affected, replace=False)
    var_idx = np.concatenate(
        [
            rng.choice(protein_indices[p], n_peptides, replace=False)
            for p in affected
        ]
    )
    return np.sort(var_idx), n_affected


def apply_peptide_batch_effects(
    adata: ad.AnnData,
    batch_key: str,
    batches: int | Sequence,
    *,
    flavor: str = "subset_shift",
    n_peptides: int = 2,
    log2fc: float = -1.0,
    fraction: float = 1.0,
    log_space: bool = False,
    random_state: int | None = None,
    inplace: bool = True,
    verbose: bool = False,
) -> ad.AnnData | None:
    r"""
    Simulate peptide-specific batch effects in peptide-level data.

    With ``flavor="subset_shift"``, ``n_peptides`` peptides are drawn
    per protein and their intensities are changed by the same fold
    change ``2 ** log2fc`` in every sample of the affected batches.
    The drawn peptides then move together across batches, apart from
    the other peptides of their protein, mimicking batch effects that
    act on single peptides (ionisation, digestion, retention-time
    drift) rather than on whole proteins.

    With :math:`B` the samples of the affected batches and :math:`P`
    the shifted peptides, the ``"subset_shift"`` flavor computes, in
    linear space (``log_space=False``) and in log2 space
    (``log_space=True``) respectively:

    .. math::

        X'_{s,p} = X_{s,p} \cdot 2^{\mathrm{log2fc}}
        \qquad
        X'_{s,p} = X_{s,p} + \mathrm{log2fc}
        \qquad
        \text{for } s \in B,\ p \in P

    All other values are unchanged.

    Parameters
    ----------
    adata : AnnData
        Peptide-level :class:`~anndata.AnnData`.
    batch_key : str
        ``.obs`` column defining the batches. Observations with a
        missing batch label are never affected.
    batches : int | Sequence
        Batches receiving the effect. A list gives the batch labels
        (values of ``.obs[batch_key]``). An int draws that many batches
        at random. At least one batch must stay unaffected.
    flavor : str
        Kind of batch effect. Only ``"subset_shift"`` is available.
    n_peptides : int
        Number of peptides shifted per affected protein. Only proteins
        with more than ``n_peptides`` peptides are eligible, so each
        keeps at least one unshifted peptide. The same peptides are
        shifted in all affected batches.
    log2fc : float
        Log2 fold change applied to the drawn peptides in the affected
        batches, e.g. ``-1`` halves their intensities.
    fraction : float
        Fraction of eligible proteins, randomly selected, that receive
        the effect.
    log_space : bool
        If False, ``.X`` holds linear intensities and is multiplied by
        ``2 ** log2fc``. If True, ``.X`` is assumed to be
        log2-transformed and ``log2fc`` is added. Data in another log
        base must be converted to log2 first. Missing values stay
        missing. In linear space zeros stay zero; in log space every
        finite value is shifted.
    random_state : int | None
        Seed for drawing batches, proteins and peptides. If ``None``, a
        random seed is drawn and stored in
        ``.uns['batch_effect_simulation']``.
    inplace : bool
        If True, modify ``adata`` in place; else return a new AnnData.
    verbose : bool
        If True, print the affected batches, the number of affected
        proteins and peptides, and where the annotations are stored.

    Returns
    -------
    AnnData | None
        ``None`` if ``inplace=True``, else a new AnnData. ``.X`` is
        dense. Adds:

        - ``.var['batch_effect']``: True for the shifted peptides
        - ``.uns['batch_effect_simulation']``: ``flavor``,
          ``batch_key``, the affected ``batches``, ``n_peptides``,
          ``log2fc``, ``fraction``, ``log_space``, ``random_state``,
          ``n_proteins_affected`` and ``n_peptides_affected``

        A second call overwrites both; shifts already applied to
        ``.X`` are kept.

    Raises
    ------
    ValueError
        If ``adata`` is not peptide-level, ``.obs[batch_key]`` has
        fewer than 2 batches, ``batches`` names unknown batches or all
        of them, or a numeric argument is out of range.
    KeyError
        If ``batch_key`` is not in ``.obs``.
    TypeError
        If ``batches`` is neither an int nor a list of labels, or
        ``log_space`` is not a bool.

    Warns
    -----
    UserWarning
        If ``.X`` is sparse (it is densified), or if ``log_space``
        does not match whether ``.X`` looks log-transformed.

    Examples
    --------
    Shift specific batches of log2-transformed data:

    >>> import proteopy as pr
    >>> pr.utils.apply_peptide_batch_effects(
    ...     adata,
    ...     batch_key="batch",
    ...     batches=["batch_1"],
    ...     log2fc=-3,
    ...     log_space=True,
    ... )
    """
    labels = _validate_args(
        adata,
        batch_key,
        batches,
        flavor,
        n_peptides,
        log2fc,
        fraction,
        log_space,
        random_state,
    )

    Xsrc = adata.X
    if sparse.issparse(Xsrc):
        warnings.warn(
            "Sparse input is being densified for computation.",
            UserWarning,
            stacklevel=2,
        )
        X = Xsrc.toarray().astype(float)
    else:
        X = np.array(Xsrc, dtype=float)

    looks_log = is_log_transformed(adata)[0]
    if looks_log != log_space:
        warnings.warn(
            f"`log_space={log_space}` but `.X` looks "
            f"{'log-transformed' if looks_log else 'linear'}, so the "
            f"effect is applied "
            f"{'additively' if log_space else 'multiplicatively'}. Check "
            "that `log_space` matches the scale of `.X`.",
            UserWarning,
            stacklevel=2,
        )

    if random_state is None:
        random_state = int(
            np.random.SeedSequence().generate_state(1, np.uint32)[0]
        )
    rng = np.random.default_rng(random_state)

    # -- Draw batches, proteins and peptides
    if _is_int(batches):
        idx = rng.choice(len(labels), batches, replace=False)
        affected_batches = [labels[i] for i in sorted(idx)]
    else:
        affected_batches = list(batches)
    # Plain Python scalars keep .uns writable to h5ad
    affected_batches = [
        b.item() if isinstance(b, np.generic) else b for b in affected_batches
    ]
    var_idx, n_proteins = _draw_peptides(
        rng,
        adata.var["protein_id"].astype(str).reset_index(drop=True),
        n_peptides,
        fraction,
    )
    obs_mask = adata.obs[batch_key].isin(affected_batches).to_numpy()

    # -- Shift
    block = np.ix_(obs_mask, var_idx)
    if log_space:
        X[block] += log2fc
    else:
        X[block] *= 2.0**log2fc

    var_flag = np.zeros(adata.n_vars, dtype=bool)
    var_flag[var_idx] = True
    params = {
        "flavor": flavor,
        "batch_key": batch_key,
        "batches": affected_batches,
        "n_peptides": int(n_peptides),
        "log2fc": float(log2fc),
        "fraction": float(fraction),
        "log_space": log_space,
        "random_state": int(random_state),
        "n_proteins_affected": int(n_proteins),
        "n_peptides_affected": int(len(var_idx)),
    }

    if verbose:
        print(
            f"Shifted {len(var_idx)} peptides of {n_proteins} proteins "
            f"by log2fc={log2fc} in batches {affected_batches} of "
            f".obs['{batch_key}'] ({int(obs_mask.sum())} samples, "
            f"random_state={random_state}).\n"
            f"Flagged shifted peptides in .var['{BATCH_EFFECT_VAR_KEY}'] "
            f"and stored parameters in .uns['{BATCH_EFFECT_UNS_KEY}']."
        )

    out = adata if inplace else adata.copy()
    out.X = X
    out.var[BATCH_EFFECT_VAR_KEY] = var_flag
    out.uns[BATCH_EFFECT_UNS_KEY] = params
    check_proteodata(out)
    return None if inplace else out
