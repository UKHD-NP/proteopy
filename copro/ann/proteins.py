from pathlib import Path

import pandas as pd
from anndata import AnnData

from copro.ann import base_anndata as ann_base


def proteins_from_csv(
    adata: AnnData,
    file_path: str | Path,
    *,
    sep: str = ",",
    file_protein_col: str = "protein_id",
    suffix: str = "_annotated",
    inplace: bool = True,
) -> AnnData | None:
    """Annotate protein-level metadata from a CSV file.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing proteomics data with a ``protein_id`` column
        in ``adata.var``.
    file_path : str | Path
        Path to the CSV file holding additional protein annotations.
    sep : str, optional
        Column delimiter passed to :func:`pandas.read_csv`.
    file_protein_col : str, optional
        Column in the CSV file used to match entries by protein ID.
    suffix : str, optional
        Suffix applied to colliding column names passed through to
        :func:`copro.ann.base_anndata.var`.
    inplace : bool, optional
        Forwarded to :func:`copro.ann.base_anndata.var`. If ``True``, modify
        ``adata`` in-place and return ``None``.

    Returns
    -------
    AnnData or None
        Updated AnnData when ``inplace`` is ``False``; otherwise ``None``.
    """
    annotations = pd.read_csv(file_path, sep=sep)
    return ann_base.var(
        adata,
        annotations,
        var_on="protein_id",
        df_on=file_protein_col,
        suffix=suffix,
        inplace=inplace,
    )
