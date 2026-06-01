"""Parsers and slot discovery for COPF results stored in ``adata.uns``."""
from __future__ import annotations

import anndata as ad
import pandas as pd

from proteopy.utils.pandas import long_pairs_to_symmetric_matrix


_PAIRWISE_VAR_CORRELATIONS_PREFIX = "pairwise_correlations;"
_PAIRWISE_VAR_CORRELATIONS_SLOT_COUNT = 5


def find_pairwise_var_correlations_keys(adata: ad.AnnData) -> list[str]:
    """Return ``adata.uns`` keys matching ``pairwise_var_correlations`` output.

    A key is considered a match when it starts with the
    ``"pairwise_correlations;"`` prefix, splits into exactly five
    semicolon-separated fields, and points to a
    :class:`pandas.DataFrame` carrying both ``varA`` and ``varB``
    columns. Keys that merely share the prefix but fail any of these
    checks are skipped. Matches are returned in ``adata.uns``
    insertion order.
    """
    matches: list[str] = []
    for key in adata.uns.keys():
        if not isinstance(key, str):
            continue
        if not key.startswith(_PAIRWISE_VAR_CORRELATIONS_PREFIX):
            continue
        if len(key.split(";")) != _PAIRWISE_VAR_CORRELATIONS_SLOT_COUNT:
            continue
        value = adata.uns[key]
        if not isinstance(value, pd.DataFrame):
            continue
        if "varA" not in value.columns or "varB" not in value.columns:
            continue
        matches.append(key)
    return matches


def resolve_pairwise_var_correlations_key(
    adata: ad.AnnData,
    corrs_key: str | None,
) -> str:
    """Return ``corrs_key`` or infer it from ``adata.uns``.

    When ``corrs_key`` is a non-empty string, it is returned as-is.
    When it is ``None``, the function searches ``adata.uns`` for
    slots emitted by :func:`pairwise_var_correlations`. If exactly
    one such slot exists it is returned; if none exist or several
    exist a :class:`ValueError` is raised with a message guiding the
    caller on how to disambiguate.
    """
    if isinstance(corrs_key, str) and corrs_key:
        return corrs_key
    candidates = find_pairwise_var_correlations_keys(adata)
    if not candidates:
        raise ValueError(
            "No pairwise correlation slot found in adata.uns. "
            "Run proteopy.tl.pairwise_var_correlations() first."
        )
    if len(candidates) > 1:
        raise ValueError(
            "Multiple pairwise correlation slots found in adata.uns; "
            "pass `corrs_key` explicitly to disambiguate. "
            f"Candidates: {candidates}"
        )
    return candidates[0]


def resolve_corrs_key(
    adata: ad.AnnData,
    corrs_key: str | None,
    *,
    legacy: bool = False,
) -> str:
    """Resolve ``corrs_key`` honouring legacy fallback and auto-inference.

    In legacy mode, ``None`` falls back to the historical default
    ``'pairwise_peptide_correlations'``. In non-legacy mode the call
    delegates to :func:`resolve_pairwise_var_correlations_key`, which
    auto-infers the key when exactly one matching slot exists and
    raises otherwise.
    """
    if legacy:
        if corrs_key is None:
            return "pairwise_peptide_correlations"
        return corrs_key
    return resolve_pairwise_var_correlations_key(adata, corrs_key)


def _pairwise_var_correlations_df_to_sym(
    df: pd.DataFrame,
    *,
    group_id: str | None,
    source_label: str,
) -> tuple[str | None, pd.DataFrame]:
    """Pivot a modern-schema pairwise correlation frame into a matrix.

    Operates on an already-resolved DataFrame (as opposed to looking
    one up in ``adata.uns``) so the same logic can be shared between
    :func:`parse_pairwise_var_correlations_result` and callers that
    supply the frame directly. ``source_label`` is interpolated into
    error messages to identify the data source (e.g.
    ``"adata.uns['<key>']"`` or ``"corrs_key DataFrame"``).
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError(
            f"{source_label} must be a pandas DataFrame; "
            f"got {type(df).__name__}."
        )

    has_group = "group_id" in df.columns
    if has_group and group_id is None:
        available = sorted(df["group_id"].unique().tolist())
        raise ValueError(
            f"{source_label} is grouped; provide a group_id. "
            f"Available group_ids: {available}"
        )
    if not has_group and group_id is not None:
        raise ValueError(
            f"{source_label} is ungrouped but group_id="
            f"{group_id!r} was provided."
        )

    if has_group:
        sub = df.loc[df["group_id"] == group_id]
        if sub.empty:
            available = sorted(df["group_id"].unique().tolist())
            raise ValueError(
                f"group_id={group_id!r} not found in "
                f"{source_label}. Available: {available}"
            )
        resolved_group_id = group_id
    else:
        sub = df
        resolved_group_id = None

    sym = long_pairs_to_symmetric_matrix(
        sub,
        var_a_col="varA",
        var_b_col="varB",
        value_col="corr",
        diagonal_value=1.0,
    )
    return resolved_group_id, sym


def parse_pairwise_var_correlations_result(
    adata: ad.AnnData,
    *,
    corrs_key: str,
    group_id: str | None = None,
) -> tuple[str | None, pd.DataFrame]:
    """Parse a stored pairwise correlation frame into a matrix.

    Reads ``adata.uns[corrs_key]`` -- as produced by
    :func:`pairwise_var_correlations` -- and pivots the long-form
    ``(varA, varB, corr, ...)`` rows into a symmetric correlation
    matrix. If the stored frame contains a ``group_id`` column
    (grouped output), ``group_id`` must be provided to select the
    subset.

    Parameters
    ----------
    adata : AnnData
        AnnData carrying the stored correlations under
        ``adata.uns[corrs_key]``.
    corrs_key : str
        Key in ``adata.uns`` holding the long-form correlation frame.
    group_id : str | None
        Identifier selecting a single group when the stored frame is
        grouped (i.e. has a ``group_id`` column). Must be left as
        ``None`` for ungrouped frames.

    Returns
    -------
    tuple[str | None, pandas.DataFrame]
        ``(resolved_group_id, symmetric_matrix)``.
        ``resolved_group_id`` is ``None`` for ungrouped frames.

    Raises
    ------
    KeyError
        If ``corrs_key`` is not found in ``adata.uns``.
    ValueError
        If the frame is grouped but no ``group_id`` is supplied, if
        the requested ``group_id`` is absent, or if the frame is
        ungrouped but a ``group_id`` is provided.
    """
    if corrs_key not in adata.uns:
        raise KeyError(
            f"corrs_key '{corrs_key}' not found in adata.uns. "
            "Run pairwise_var_correlations() first."
        )
    return _pairwise_var_correlations_df_to_sym(
        adata.uns[corrs_key],
        group_id=group_id,
        source_label=f"adata.uns['{corrs_key}']",
    )


def _pairwise_peptide_correlations_legacy_df_to_sym(
    df: pd.DataFrame,
    *,
    group_id: str | None,
    source_label: str,
) -> tuple[str | None, pd.DataFrame]:
    """Pivot a legacy-schema peptide correlation frame into a matrix.

    Operates on an already-resolved DataFrame so the logic can be
    shared between
    :func:`parse_pairwise_peptide_correlations_result_legacy` and
    callers that supply the frame directly. The legacy frame is
    always grouped by ``protein_id`` stored in the *index*, so
    ``group_id`` is mandatory.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError(
            f"{source_label} must be a pandas DataFrame; "
            f"got {type(df).__name__}."
        )

    if group_id is None:
        available = sorted(pd.Index(df.index).unique().tolist())
        raise ValueError(
            f"{source_label} is grouped by protein_id; "
            f"provide a group_id. Available group_ids: {available}"
        )

    sub = df.loc[df.index == group_id]
    if sub.empty:
        available = sorted(pd.Index(df.index).unique().tolist())
        raise ValueError(
            f"group_id={group_id!r} not found in "
            f"{source_label}. Available: {available}"
        )

    sym = long_pairs_to_symmetric_matrix(
        sub,
        var_a_col="pepA",
        var_b_col="pepB",
        value_col="PCC",
        diagonal_value=1.0,
    )
    return group_id, sym


def parse_pairwise_peptide_correlations_result_legacy(
    adata: ad.AnnData,
    *,
    corrs_key: str,
    group_id: str | None = None,
) -> tuple[str | None, pd.DataFrame]:
    """Parse a legacy pairwise peptide correlation frame into a matrix.

    Reads ``adata.uns[corrs_key]`` as produced by
    :func:`pairwise_peptide_correlations_legacy` and pivots the
    long-form ``(pepA, pepB, PCC, ...)`` rows into a symmetric
    correlation matrix. The legacy frame stores ``protein_id`` as the
    index rather than as a column, so ``group_id`` is required and
    selects rows by index.

    Parameters
    ----------
    adata : AnnData
        AnnData carrying the stored correlations under
        ``adata.uns[corrs_key]``.
    corrs_key : str
        Key in ``adata.uns`` holding the legacy long-form correlation
        frame.
    group_id : str | None
        ``protein_id`` selecting a single protein's peptide pairs.
        Required because the legacy frame is always grouped by
        ``protein_id`` (stored in the index).

    Returns
    -------
    tuple[str | None, pandas.DataFrame]
        ``(resolved_group_id, symmetric_matrix)``.

    Raises
    ------
    KeyError
        If ``corrs_key`` is not found in ``adata.uns``.
    ValueError
        If the stored object is not a DataFrame, if ``group_id`` is
        not provided, or if ``group_id`` is absent from the frame's
        index.
    """
    if corrs_key not in adata.uns:
        raise KeyError(
            f"corrs_key '{corrs_key}' not found in adata.uns. "
            "Run pairwise_peptide_correlations_legacy() first."
        )
    return _pairwise_peptide_correlations_legacy_df_to_sym(
        adata.uns[corrs_key],
        group_id=group_id,
        source_label=f"adata.uns['{corrs_key}']",
    )
