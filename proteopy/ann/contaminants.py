"""
Annotation helpers for marking contaminant variables in an AnnData.
"""
import os
import warnings
from typing import Callable

from anndata import AnnData

from proteopy.utils.anndata import check_proteodata, is_proteodata
from proteopy.utils.parsers import read_protein_ids


def _print_contaminant_summary(
    adata: AnnData,
    mask_values,
    protein_key: str,
    key_added: str,
) -> None:
    """Print a level-aware summary of annotated contaminants."""
    _, level = is_proteodata(adata)
    if level == "peptide":
        n_pep = int(mask_values.sum())
        if n_pep == 0:
            n_prot = 0
        else:
            n_prot = int(
                adata.var.loc[mask_values, protein_key].nunique()
            )
        print(
            f"Annotated {n_pep} peptides from {n_prot} "
            f"contaminating proteins at adata.var['{key_added}']."
        )
    elif level == "protein":
        n_prot = int(mask_values.sum())
        print(
            f"Annotated {n_prot} contaminating proteins at "
            f"adata.var['{key_added}']."
        )
    else:
        n = int(mask_values.sum())
        print(
            f"Annotated {n} contaminating variables at "
            f"adata.var['{key_added}']."
        )


def contaminants(
    adata: AnnData,
    contaminant_path: str | os.PathLike,
    *,
    protein_key: str = "protein_id",
    key_added: str = "is_contaminant",
    header_parser: Callable[[str], str] | None = None,
    has_header: bool = True,
    inplace: bool = True,
    verbose: bool = False,
) -> AnnData | None:
    """
    Annotate contaminant variables by flagging them in ``adata.var``.

    Reads protein identifiers from a contaminant list (FASTA / CSV /
    TSV) and writes a boolean column ``adata.var[key_added]`` that is
    ``True`` where ``adata.var[protein_key]`` matches an entry in the
    list. Unlike :func:`proteopy.pp.remove_contaminants`, the
    contaminants are kept in the AnnData and only flagged, so the
    annotation can be used for downstream filtering decisions, QC
    plots, or contaminant-aware normalization.

    Parameters
    ----------
    adata : AnnData
        :class:`~anndata.AnnData` annotated data matrix.
    contaminant_path : str | os.PathLike
        Path to the contaminant list. Supported formats: FASTA
        (``.fasta`` / ``.fa`` / ``.faa``), CSV (``.csv``), TSV
        (``.tsv``). See :func:`proteopy.utils.parsers.read_protein_ids`
        for parsing details.
    protein_key : str, optional
        Column in ``adata.var`` holding protein identifiers used for
        matching. Defaults to ``"protein_id"``.
    key_added : str, optional
        Name of the boolean column written to ``adata.var``. Defaults
        to ``"is_contaminant"``.
    header_parser : callable, optional
        Function to extract protein IDs from FASTA headers. Defaults
        to splitting the header on ``"|"`` and returning the second
        element, falling back to the full header. Ignored (with a
        warning) for tabular formats.
    has_header : bool, optional
        For CSV / TSV files, whether the first row is a header line.
        Set to ``False`` for plain single-column ID lists.

    Returns
    -------
    AnnData or None
        ``None`` when ``inplace=True``; otherwise the annotated copy
        of ``adata``.

    Raises
    ------
    KeyError
        If ``protein_key`` is not a column of ``adata.var``.

    Warns
    -----
    UserWarning
        Emitted when ``key_added`` already exists in ``adata.var``
        (the column is overwritten). All warnings raised by
        :func:`~proteopy.utils.parsers.read_protein_ids` (empty /
        non-string parsed IDs, no IDs parsed, parser ignored for
        tabular files) are propagated.

    See Also
    --------
    proteopy.pp.remove_contaminants : Remove rather than annotate.
    proteopy.utils.parsers.read_protein_ids : Underlying file reader.

    Examples
    --------
    >>> import proteopy as pr
    >>> adata = pr.datasets.example_protein_data()  # doctest: +SKIP
    >>> pr.ann.contaminants(
    ...     adata,
    ...     "contaminants.fasta",
    ...     verbose=True,
    ... )  # doctest: +SKIP
    >>> adata.var["is_contaminant"].sum()  # doctest: +SKIP
    """
    check_proteodata(adata)

    if protein_key not in adata.var.columns:
        raise KeyError(
            f"`protein_key`='{protein_key}' not found in adata.var"
        )
    if key_added in adata.var.columns:
        warnings.warn(
            f"`key_added`='{key_added}' already exists in adata.var; "
            "overwriting.",
            UserWarning,
        )

    adata_target = adata if inplace else adata.copy()

    contaminant_ids = read_protein_ids(
        contaminant_path,
        header_parser=header_parser,
        has_header=has_header,
    )

    mask = adata_target.var[protein_key].isin(contaminant_ids)
    mask_values = mask.astype(bool).values
    adata_target.var[key_added] = mask_values

    if verbose:
        _print_contaminant_summary(
            adata_target, mask_values, protein_key, key_added,
        )

    check_proteodata(adata_target)
    return None if inplace else adata_target
