from pathlib import Path

import pandas as pd

from proteopy.datasets import williams_2018 as _load_williams_2018
from proteopy.utils.string import detect_separator_from_extension


_DEFAULT_INTENSITIES = (
    "williams-2018_ms-proteomics"
    "_mouse-tissue_intensities.tsv"
)
_DEFAULT_VAR = (
    "williams-2018_ms-proteomics"
    "_mouse-tissue_peptide-annotation.tsv"
)
_DEFAULT_SAMPLE = (
    "williams-2018_ms-proteomics"
    "_mouse-tissue_sample-annotation.tsv"
)


def _check_williams_2018_types(
    intensities_path,
    var_annotation_path,
    sample_annotation_path,
    sep,
    fill_na,
    force,
):
    """Type-check parameters for :func:`williams_2018`."""
    for name, value in (
        ("intensities_path", intensities_path),
        ("var_annotation_path", var_annotation_path),
        ("sample_annotation_path", sample_annotation_path),
    ):
        if not isinstance(value, (str, Path)):
            raise TypeError(
                f"{name} must be str or Path, "
                f"got {type(value).__name__}"
            )
    if sep is not None and not isinstance(sep, str):
        raise TypeError(
            f"sep must be str or None, "
            f"got {type(sep).__name__}"
        )
    if fill_na is not None and (
        isinstance(fill_na, bool)
        or not isinstance(fill_na, (int, float))
    ):
        raise TypeError(
            f"fill_na must be float, int, or None, "
            f"got {type(fill_na).__name__}"
        )
    if not isinstance(force, bool):
        raise TypeError(
            f"force must be bool, "
            f"got {type(force).__name__}"
        )


def _check_williams_2018_paths(
    intensities_path,
    var_annotation_path,
    sample_annotation_path,
    force,
):
    """Resolve paths, check for overlaps and existing files."""
    intensities_path = Path(intensities_path)
    var_annotation_path = Path(var_annotation_path)
    sample_annotation_path = Path(sample_annotation_path)

    paths = {
        "intensities_path": intensities_path.resolve(),
        "var_annotation_path": var_annotation_path.resolve(),
        "sample_annotation_path": sample_annotation_path.resolve(),
    }
    seen: dict[Path, str] = {}
    for name, resolved in paths.items():
        if resolved in seen:
            raise ValueError(
                f"{name} and {seen[resolved]} resolve to "
                f"the same path: {resolved}"
            )
        seen[resolved] = name

    if not force:
        for name, resolved in paths.items():
            if resolved.exists():
                raise FileExistsError(
                    f"{name} already exists: {resolved}. "
                    "Use force=True to overwrite."
                )

    return intensities_path, var_annotation_path, sample_annotation_path


def williams_2018(
    intensities_path: str | Path = _DEFAULT_INTENSITIES,
    var_annotation_path: str | Path = _DEFAULT_VAR,
    sample_annotation_path: str | Path = _DEFAULT_SAMPLE,
    *,
    sep: str | None = None,
    fill_na: float | int | None = None,
    force: bool = False,
) -> None:
    """Save Williams 2018 SWATH-MS mouse tissue dataset to disk.

    Download and process the peptide-level SWATH-MS dataset from
    Williams et al. (2018) [1]_ and save it as three tabular files:
    intensities in long format, peptide annotations, and sample
    annotations.

    The dataset consists of the protein expression of eight
    genetically diverse BXD mouse strains across five tissues. Only
    the whole cell fraction is included; peptide intensities from
    different charge states are summed per peptide sequence.

    Data are sourced from the Elsevier supplementary archive
    (DOI: 10.1074/mcp.RA118.000554).

    Parameters
    ----------
    intensities_path : str | Path, optional
        Destination path for the intensities file. Columns:
        ``sample_id``, ``peptide_id``, ``intensity``.
    var_annotation_path : str | Path, optional
        Destination path for the peptide annotation file. Columns:
        ``peptide_id``, ``protein_id``, ``gene_id``.
    sample_annotation_path : str | Path, optional
        Destination path for the sample annotation file. Columns:
        ``sample_id``, ``tissue``, ``mouse_id``.
    sep : str | None, optional
        Column separator for all output files. When ``None``, the
        separator is inferred from each file extension via
        ``detect_separator_from_extension()``
        (``.tsv`` → tab, ``.csv`` → comma).
    fill_na : float | int | None, optional
        If not ``None``, replace NaN values in the long-format
        intensities DataFrame with this value before saving.
    force : bool, optional
        If ``True``, overwrite existing files at the output
        paths. Otherwise, raise ``FileExistsError`` when a
        destination file already exists.

    Returns
    -------
    None
        Writes files to disk; does not return a value.

    Examples
    --------
    >>> import proteopy as pr
    >>> pr.download.williams_2018(
    ...     intensities_path="intensities.tsv",
    ...     var_annotation_path="peptide_annotations.tsv",
    ...     sample_annotation_path="sample_annotations.tsv",
    ... )

    References
    ----------
    .. [1] Williams EG, Wu Y, Wolski W, Kim JY, Lan J, Hasan M,
       Halter C, Jha P, Ryu D, Auwerx J, and Aebersold R.
       "Quantifying and Localizing the Mitochondrial Proteome Across
       Five Tissues in A Mouse Population." *Molecular & Cellular
       Proteomics*, 2018, 17(9):1766-1777.
       DOI: 10.1074/mcp.RA118.000554.
    """
    _check_williams_2018_types(
        intensities_path,
        var_annotation_path,
        sample_annotation_path,
        sep,
        fill_na,
        force,
    )
    intensities_path, var_annotation_path, sample_annotation_path = (
        _check_williams_2018_paths(
            intensities_path,
            var_annotation_path,
            sample_annotation_path,
            force,
        )
    )

    adata = _load_williams_2018()

    # Auto-detect separator from file extension if not provided
    if sep is None:
        sep_intensities = detect_separator_from_extension(
            intensities_path,
        )
        sep_var = detect_separator_from_extension(
            var_annotation_path,
        )
        sep_sample = detect_separator_from_extension(
            sample_annotation_path,
        )
    else:
        sep_intensities = sep
        sep_var = sep
        sep_sample = sep

    # Melt .X to long format: sample_id, peptide_id, intensity
    df_x = pd.DataFrame(
        adata.X,
        index=adata.obs_names,
        columns=adata.var_names,
    )
    df_x.index.name = "sample_id"
    df_long = df_x.reset_index().melt(
        id_vars="sample_id",
        var_name="peptide_id",
        value_name="intensity",
    )
    if fill_na is not None:
        df_long["intensity"] = df_long["intensity"].fillna(
            fill_na,
        )
    intensities_path.parent.mkdir(parents=True, exist_ok=True)
    df_long.to_csv(
        intensities_path,
        sep=sep_intensities,
        index=False,
    )

    # Save .var annotation
    df_var = adata.var[
        ["peptide_id", "protein_id", "gene_id"]
    ].copy()
    var_annotation_path.parent.mkdir(
        parents=True, exist_ok=True,
    )
    df_var.to_csv(
        var_annotation_path,
        sep=sep_var,
        index=False,
    )

    # Save .obs annotation
    df_obs = adata.obs[
        ["sample_id", "tissue", "mouse_id"]
    ].copy()
    sample_annotation_path.parent.mkdir(
        parents=True, exist_ok=True,
    )
    df_obs.to_csv(
        sample_annotation_path,
        sep=sep_sample,
        index=False,
    )
