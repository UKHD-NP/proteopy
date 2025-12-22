from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from anndata import AnnData

from proteopy.utils.anndata import check_proteodata


def proteoforms_df(
    adata: AnnData,
    proteins: Sequence[str] | str | None = None,
    *,
    only_proteins: bool = False,
    score_threshold: float | None = None,
    pval_threshold: float | None = None,
    pval_adj_threshold: float | None = None,
) -> pd.DataFrame:
    """
    Return proteoform peptide assignment results as a tidy dataframe.

    Parameters
    ----------
    adata : :class:`~anndata.AnnData`
        Annotated data object containing proteoform annotations in ``.var``.
    proteins : str | Sequence[str] | None
        Optional subset of protein identifiers to include.
    only_proteins : bool
        When ``True``, output unique protein-level information of identified
        proteoforms.
    score_threshold : float | None
        Minimum proteoform score to retain.
    pval_threshold : float | None
        Maximum raw p-value allowed.
    pval_adj_threshold : float | None
        Maximum adjusted p-value allowed.

    Returns
    -------
    pandas.DataFrame
        Proteoform assignments filtered according to the provided arguments.

    Raises
    ------
    TypeError
        If ``proteins`` is neither a string nor a sequence of strings.
    KeyError
        If the expected proteoform columns are not present in ``adata.var``.
    """
    check_proteodata(adata)

    proteoform_columns = [
        "protein_id",
        "peptide_id",
        "cluster_id",
        "proteoform_score",
        "proteoform_score_pval",
        "proteoform_score_pval_adj",
        "is_proteoform",
    ]

    missing_columns = [
        column for column in proteoform_columns if column not in adata.var.columns
    ]

    if missing_columns:
        missing = ", ".join(missing_columns)
        raise KeyError(
            "Missing required proteoform annotation columns in `adata.var`: "
            f"{missing}"
        )

    if proteins is None:
        selected_proteins = adata.var["protein_id"].tolist()
    elif isinstance(proteins, str):
        selected_proteins = [proteins]
    elif isinstance(proteins, Sequence):
        if not all(isinstance(protein, str) for protein in proteins):
            raise TypeError(
                "`proteins` must contain only strings; received "
                f"{proteins!r}."
            )
        selected_proteins = list(proteins)
    else:
        raise TypeError(
            "`proteins` must be a string or a sequence of strings, "
            f"received {type(proteins)!r}."
        )

    selection = adata.var["protein_id"].isin(selected_proteins)
    proteoforms = adata.var.loc[selection, proteoform_columns].copy()

    proteoforms = proteoforms[
        proteoforms["proteoform_score_pval"].notna()
    ].sort_values(
        ["proteoform_score_pval_adj", "proteoform_score", "cluster_id"]
    )

    if score_threshold is not None:
        proteoforms = proteoforms[
            proteoforms["proteoform_score"] >= score_threshold
        ]

    if pval_threshold is not None:
        proteoforms = proteoforms[
            proteoforms["proteoform_score_pval"] <= pval_threshold
        ]

    if pval_adj_threshold is not None:
        proteoforms = proteoforms[
            proteoforms["proteoform_score_pval_adj"] <= pval_adj_threshold
        ]

    if only_proteins:
        proteoforms = (
            proteoforms.drop(columns=["peptide_id", "cluster_id"])
            .drop_duplicates(ignore_index=True)
        )
        return proteoforms

    return proteoforms.reset_index(drop=True)
