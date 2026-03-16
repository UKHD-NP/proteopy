from __future__ import annotations

import warnings
from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd
from pathlib import Path

from proteopy.utils.anndata import check_proteodata
from proteopy.utils.pandas import load_dataframe


def _validate_column_map(
    column_map: dict[str, str] | None,
    valid_keys: set[str],
) -> dict[str, str]:
    """Build column alias dict, validating *column_map* if given."""
    aliases = {k: k for k in valid_keys}
    if not column_map:
        return aliases
    unexpected = set(column_map).difference(valid_keys)
    if unexpected:
        raise ValueError(
            "column_map contains unsupported keys: "
            f"{', '.join(sorted(unexpected))}"
        )
    if len(set(column_map.values())) != len(column_map):
        raise ValueError(
            "column_map must map each canonical key "
            "to a unique source column."
        )
    aliases.update(column_map)
    return aliases


def _validate_intensities_df(
    df: pd.DataFrame,
    *,
    column_aliases: dict[str, str],
    required_keys: list[str],
    id_columns: list[str],
    duplicate_subset: list[str],
    fill_na: float | None,
    zero_to_na: bool,
) -> pd.DataFrame:
    """Copy, validate, and rename the intensities DataFrame."""
    if fill_na is not None and zero_to_na:
        raise ValueError(
            "fill_na and zero_to_na are mutually exclusive."
        )
    df = df.copy()
    if df.empty:
        raise ValueError(
            "Intensities DataFrame is empty."
        )
    required = {column_aliases[k] for k in required_keys}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            "Intensities DataFrame is missing required "
            f"columns: {', '.join(sorted(missing))}"
        )
    rename_map = {
        actual: canonical
        for canonical, actual in column_aliases.items()
    }
    df = df.rename(columns=rename_map)
    for col in id_columns:
        n_na = df[col].isna().sum()
        if n_na:
            raise ValueError(
                f"Column '{col}' contains {n_na} missing "
                "value(s). All ID columns must be "
                "non-null."
            )
    if not pd.api.types.is_numeric_dtype(df["intensity"]):
        raise TypeError(
            "Column 'intensity' must be numeric, got "
            f"dtype '{df['intensity'].dtype}'."
        )
    dup_mask = df.duplicated(subset=duplicate_subset)
    if dup_mask.any():
        duplicated = df.loc[dup_mask, duplicate_subset]
        n_duplicates = len(duplicated)
        examples = duplicated.head(5).to_dict(
            orient="records"
        )
        extra = (
            f" (showing first 5 of {n_duplicates})"
            if n_duplicates > 5
            else ""
        )
        entity = " and ".join(duplicate_subset)
        raise ValueError(
            "Intensities contain duplicate entries for "
            f"the same {entity} combination"
            f"{extra}: {examples}"
        )
    return df


def _resolve_protein_id(
    df: pd.DataFrame,
    peptide_annotation_df: pd.DataFrame | None,
    column_aliases: dict[str, str],
    protein_id_in_intensities: bool,
    verbose: bool,
) -> pd.DataFrame:
    """Resolve protein_id, merging from annotation if needed.

    Also validates that each peptide maps to exactly one
    protein.
    """
    protein_id_col = column_aliases["protein_id"]

    if not protein_id_in_intensities:
        if peptide_annotation_df is None:
            raise ValueError(
                f"Column '{protein_id_col}' (protein_id) "
                "is missing from the intensities DataFrame"
                " and no peptide_annotation_df was "
                "provided."
            )
        if protein_id_col not in peptide_annotation_df.columns:
            raise ValueError(
                f"Column '{protein_id_col}' (protein_id) "
                "is missing from both the intensities "
                "DataFrame and the peptide annotation "
                "DataFrame."
            )
        ann_df = peptide_annotation_df.copy()
        rename_map = {
            column_aliases[key]: key
            for key in ("peptide_id", "protein_id")
            if column_aliases[key] in ann_df.columns
            and column_aliases[key] != key
        }
        ann_df = ann_df.rename(columns=rename_map)
        protein_map = (
            ann_df[["peptide_id", "protein_id"]]
            .drop_duplicates(
                subset=["peptide_id"], keep="first"
            )
        )
        df = df.merge(
            protein_map, on="peptide_id", how="left"
        )
        n_unresolved = df["protein_id"].isna().sum()
        if n_unresolved:
            raise ValueError(
                f"{n_unresolved} peptide(s) in the "
                "intensities DataFrame could not be "
                "mapped to a protein_id using the "
                "peptide annotation DataFrame."
            )
    elif (
        verbose
        and peptide_annotation_df is not None
        and protein_id_col in peptide_annotation_df.columns
    ):
        print(
            "protein_id found in both intensities and "
            "peptide annotation DataFrames; using "
            "intensities DataFrame."
        )

    protein_counts = (
        df.groupby("peptide_id")["protein_id"].nunique()
    )
    inconsistent = protein_counts[protein_counts > 1]
    if not inconsistent.empty:
        raise ValueError(
            "Each peptide_id must map to exactly one "
            "protein_id; conflicts for: "
            f"{', '.join(map(str, inconsistent.index))}"
        )
    return df


def _merge_sample_annotations(
    obs: pd.DataFrame,
    sample_annotation_df: pd.DataFrame,
    column_aliases: dict[str, str],
    verbose: bool,
) -> tuple[pd.DataFrame, list[str]]:
    """Merge sample annotations into obs DataFrame.

    Returns the merged obs and the annotation sample order.
    """
    annotation_df = sample_annotation_df.copy()
    actual_sample_id = column_aliases["sample_id"]
    if (
        actual_sample_id in annotation_df.columns
        and actual_sample_id != "sample_id"
    ):
        annotation_df = annotation_df.rename(
            columns={actual_sample_id: "sample_id"}
        )

    if "sample_id" not in annotation_df.columns:
        raise ValueError(
            "Annotation file is missing the required "
            "`sample_id` column."
        )

    dup_mask = annotation_df.duplicated(
        subset=["sample_id"], keep=False
    )
    if dup_mask.any():
        dup_count = (
            annotation_df
            .loc[dup_mask, "sample_id"]
            .nunique()
        )
        warnings.warn(
            "Duplicate sample entries found in "
            "annotation file; keeping the first "
            f"occurrence for {dup_count} sample IDs.",
            UserWarning,
        )

    annotation_unique = annotation_df.drop_duplicates(
        subset=["sample_id"], keep="first"
    )

    obs_samples = set(obs["sample_id"])
    ann_samples = set(annotation_unique["sample_id"])

    ignored = len(ann_samples.difference(obs_samples))
    if verbose and ignored:
        print(
            f"{ignored} sample_id entries in the "
            "annotation file were not present in the "
            "intensity table and were ignored."
        )

    missing = len(obs_samples.difference(ann_samples))
    if verbose and missing:
        print(
            f"{missing} sample_id entries in the "
            "intensity table did not have a matching "
            "annotation."
        )

    annotation_order = [
        name
        for name in annotation_unique["sample_id"]
        if name in obs_samples
    ]

    # preserve original index through merge
    obs_reset = obs.reset_index(names="_obs_index")
    merged = obs_reset.merge(
        annotation_unique,
        how="left",
        on="sample_id",
        suffixes=("", "_annotation"),
    )
    merged.set_index("_obs_index", inplace=True)
    merged.index.name = None
    return merged, annotation_order


def _merge_var_annotations(
    var: pd.DataFrame,
    annotation_df: pd.DataFrame,
    id_column: str,
    column_aliases: dict[str, str],
    rename_keys: list[str],
    entity_name: str,
    verbose: bool,
) -> pd.DataFrame:
    """Merge variable annotations into var DataFrame."""
    annotation_df = annotation_df.copy()
    rename_map = {
        column_aliases[key]: key
        for key in rename_keys
        if column_aliases[key] in annotation_df.columns
        and column_aliases[key] != key
    }
    annotation_df = annotation_df.rename(columns=rename_map)

    if id_column not in annotation_df.columns:
        raise ValueError(
            f"{entity_name.capitalize()} annotation file "
            f"is missing the required `{id_column}` "
            "column."
        )

    dup_mask = annotation_df.duplicated(
        subset=[id_column], keep=False
    )
    if dup_mask.any():
        dup_count = (
            annotation_df
            .loc[dup_mask, id_column]
            .nunique()
        )
        warnings.warn(
            f"Duplicate {entity_name} entries found in "
            f"{entity_name} annotation file; keeping "
            f"the first occurrence for {dup_count} "
            f"{entity_name}s.",
            UserWarning,
        )

    annotation_unique = annotation_df.drop_duplicates(
        subset=[id_column], keep="first"
    )

    var_ids = set(var[id_column])
    ann_ids = set(annotation_unique[id_column])

    ignored = len(ann_ids.difference(var_ids))
    if verbose and ignored:
        print(
            f"{ignored} {entity_name} entries in the "
            "annotation file were not present in the "
            "intensity matrix and were ignored."
        )

    missing = len(var_ids.difference(ann_ids))
    if verbose and missing:
        print(
            f"{missing} {entity_name} entries in the "
            "intensity matrix did not have a matching "
            f"{entity_name} annotation."
        )

    var_reset = var.reset_index(names="_var_index")
    merged = var_reset.merge(
        annotation_unique,
        how="left",
        on=id_column,
        suffixes=("", "_annotation"),
    )
    merged.set_index("_var_index", inplace=True)
    merged.index.name = None
    return merged


def _reorder_observations(
    intensity_matrix: pd.DataFrame,
    obs: pd.DataFrame,
    annotation_order: list[str] | None,
    default_obs_order: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reorder intensity matrix and obs by desired order."""
    desired_order = annotation_order or default_obs_order
    seen: set[str] = set()
    final_order: list[str] = []
    for name in desired_order:
        if name in intensity_matrix.index and name not in seen:
            final_order.append(name)
            seen.add(name)
    for name in intensity_matrix.index:
        if name not in seen:
            final_order.append(name)
            seen.add(name)
    return (
        intensity_matrix.reindex(final_order),
        obs.loc[final_order],
    )


def _finalize_adata(
    intensity_matrix: pd.DataFrame,
    obs: pd.DataFrame,
    var: pd.DataFrame,
    zero_to_na: bool,
    expected_level: str,
) -> ad.AnnData:
    """Create AnnData from components and validate."""
    adata = ad.AnnData(
        X=intensity_matrix.to_numpy(copy=True),
        obs=obs,
        var=var,
    )
    if zero_to_na:
        X = adata.X
        X[X == 0] = np.nan
        adata.X = X
    adata.strings_to_categoricals()
    _, detected_level = check_proteodata(adata)
    if detected_level != expected_level:
        raise ValueError(
            f"Expected {expected_level}-level proteodata "
            f"but detected '{detected_level}'."
        )
    return adata


def _peptides_long_from_df(
    intensities_df: pd.DataFrame,
    *,
    sample_annotation_df: pd.DataFrame | None = None,
    peptide_annotation_df: pd.DataFrame | None = None,
    column_map: dict[str, str] | None = None,
    fill_na: float | None = None,
    zero_to_na: bool = False,
    sort_obs_by_annotation: bool = False,
    verbose: bool = False,
) -> ad.AnnData:
    """Convert peptide-level pandas DataFrame tables into an AnnData
    container.

    Requires ``intensities_df`` to contain columns for ``sample_id``,
    ``intensity``, and ``peptide_id``. The ``protein_id`` column is
    resolved with the following priority:

    1. If ``protein_id`` is present in ``intensities_df``, it is used
       directly. When ``peptide_annotation_df`` also contains
       ``protein_id``, the intensities copy takes precedence.
    2. If ``protein_id`` is absent from ``intensities_df``, it is
       looked up from ``peptide_annotation_df``, which must then be
       supplied and contain both ``peptide_id`` and ``protein_id``
       columns. Any peptide present in ``intensities_df`` but absent
       from the annotation raises a ``ValueError``.

    ``sample_annotation_df``, when provided, must contain a
    ``sample_id`` column. Its remaining columns are merged into
    ``adata.obs``.

    ``peptide_annotation_df``, when provided, must contain a
    ``peptide_id`` column. Its remaining columns (excluding
    ``protein_id`` if already resolved from ``intensities_df``) are
    merged into ``adata.var``. Duplicate peptides in either annotation are
    deduplicated by keeping the first occurrence.

    Column names may deviate from these defaults by supplying
    ``column_map``.
    """
    # -- Validate inputs and normalize column names
    column_aliases = _validate_column_map(
        column_map,
        {"peptide_id", "protein_id", "sample_id", "intensity"},
    )

    protein_id_col = column_aliases["protein_id"]
    protein_id_in_intensities = (
        protein_id_col in intensities_df.columns
    )

    required_keys = ["sample_id", "intensity", "peptide_id"]
    id_columns = ["sample_id", "peptide_id"]
    if protein_id_in_intensities:
        required_keys.append("protein_id")
        id_columns.append("protein_id")

    df = _validate_intensities_df(
        intensities_df,
        column_aliases=column_aliases,
        required_keys=required_keys,
        id_columns=id_columns,
        duplicate_subset=["sample_id", "peptide_id"],
        fill_na=fill_na,
        zero_to_na=zero_to_na,
    )

    # -- Resolve protein_id
    df = _resolve_protein_id(
        df, peptide_annotation_df, column_aliases,
        protein_id_in_intensities, verbose,
    )

    default_obs_order = (
        df["sample_id"].drop_duplicates().tolist()
    )
    annotation_order = None

    # -- Build .X
    intensity_matrix = df.pivot(
        index="sample_id",
        columns="peptide_id",
        values="intensity",
    )
    intensity_matrix = intensity_matrix.astype(float)
    intensity_matrix = (
        intensity_matrix.sort_index().sort_index(axis=1)
    )
    if fill_na is not None:
        intensity_matrix = intensity_matrix.fillna(
            float(fill_na)
        )
    intensity_matrix.index.name = None
    intensity_matrix.columns.name = None

    peptide_to_protein = (
        df.groupby("peptide_id", sort=False)["protein_id"]
        .first()
        .reindex(intensity_matrix.columns)
    )

    # -- Build .obs
    obs = pd.DataFrame(index=intensity_matrix.index)
    obs["sample_id"] = obs.index

    if sample_annotation_df is not None:
        obs, annotation_order = _merge_sample_annotations(
            obs, sample_annotation_df,
            column_aliases, verbose,
        )

    # -- Build .var
    var = pd.DataFrame(index=intensity_matrix.columns)
    var.index.name = None
    var["peptide_id"] = var.index
    var["protein_id"] = (
        peptide_to_protein.loc[var.index].values
    )

    if peptide_annotation_df is not None:
        var = _merge_var_annotations(
            var, peptide_annotation_df,
            id_column="peptide_id",
            column_aliases=column_aliases,
            rename_keys=["peptide_id", "protein_id"],
            entity_name="peptide",
            verbose=verbose,
        )

    # -- Reorder observations
    if sort_obs_by_annotation:
        intensity_matrix, obs = _reorder_observations(
            intensity_matrix, obs,
            annotation_order, default_obs_order,
        )

    # -- Build AnnData
    return _finalize_adata(
        intensity_matrix, obs, var,
        zero_to_na, "peptide",
    )


def _proteins_long_from_df(
    intensities_df: pd.DataFrame,
    *,
    sample_annotation_df: pd.DataFrame | None = None,
    protein_annotation_df: pd.DataFrame | None = None,
    column_map: dict[str, str] | None = None,
    fill_na: float | None = None,
    zero_to_na: bool = False,
    sort_obs_by_annotation: bool = False,
    verbose: bool = False,
) -> ad.AnnData:
    """Convert protein-level pandas DataFrame tables into an AnnData
    container.

    Requires ``intensities_df`` to contain columns for ``sample_id``,
    ``intensity``, and ``protein_id``.

    ``sample_annotation_df``, when provided, must contain a
    ``sample_id`` column. Its remaining columns are merged into
    ``adata.obs``.

    ``protein_annotation_df``, when provided, must contain a
    ``protein_id`` column. Its remaining columns are merged into
    ``adata.var``. Duplicate proteins in either annotation are deduplicated
    by keeping the first occurrence.

    Column names may deviate from these defaults by supplying
    ``column_map``.
    """
    # -- Validate inputs and normalize column names
    column_aliases = _validate_column_map(
        column_map,
        {"protein_id", "sample_id", "intensity"},
    )

    df = _validate_intensities_df(
        intensities_df,
        column_aliases=column_aliases,
        required_keys=list(column_aliases),
        id_columns=["sample_id", "protein_id"],
        duplicate_subset=["sample_id", "protein_id"],
        fill_na=fill_na,
        zero_to_na=zero_to_na,
    )

    default_obs_order = (
        df["sample_id"].drop_duplicates().tolist()
    )
    annotation_order = None

    # -- Build .X
    intensity_matrix = df.pivot(
        index="sample_id",
        columns="protein_id",
        values="intensity",
    )
    intensity_matrix = intensity_matrix.astype(float)
    intensity_matrix = (
        intensity_matrix.sort_index().sort_index(axis=1)
    )
    if fill_na is not None:
        intensity_matrix = intensity_matrix.fillna(
            float(fill_na)
        )
    intensity_matrix.index.name = None
    intensity_matrix.columns.name = None

    # -- Build .obs
    obs = pd.DataFrame(index=intensity_matrix.index)
    obs["sample_id"] = obs.index

    if sample_annotation_df is not None:
        obs, annotation_order = _merge_sample_annotations(
            obs, sample_annotation_df,
            column_aliases, verbose,
        )

    # -- Build .var
    var = pd.DataFrame(index=intensity_matrix.columns)
    var.index.name = None
    var["protein_id"] = var.index

    if protein_annotation_df is not None:
        var = _merge_var_annotations(
            var, protein_annotation_df,
            id_column="protein_id",
            column_aliases=column_aliases,
            rename_keys=["protein_id"],
            entity_name="protein",
            verbose=verbose,
        )

    # -- Reorder observations
    if sort_obs_by_annotation:
        intensity_matrix, obs = _reorder_observations(
            intensity_matrix, obs,
            annotation_order, default_obs_order,
        )

    # -- Build AnnData
    return _finalize_adata(
        intensity_matrix, obs, var,
        zero_to_na, "protein",
    )


def long(
    intensities: str | Path | pd.DataFrame,
    level: Literal["peptide", "protein"] | None = None,
    *,
    sample_annotation: str | Path | pd.DataFrame | None = None,
    var_annotation: str | Path | pd.DataFrame | None = None,
    column_map: dict[str, str] | None = None,
    sep: str | None = None,
    fill_na: float | None = None,
    zero_to_na: bool = False,
    sort_obs_by_annotation: bool = False,
    verbose: bool = False,
) -> ad.AnnData:
    """Read long-format peptide or protein tabular data into an
    AnnData container.

    The ``intensities`` table must be in long format with one row per
    (sample, feature) measurement. Required columns differ by level:

    - **Peptide level**: ``sample_id``, ``intensity``, and
      ``peptide_id`` must be present. ``protein_id`` may come from
      the intensities table or from ``var_annotation``; see below.
    - **Protein level**: ``sample_id``, ``intensity``, and
      ``protein_id`` must all be present.

    At peptide level, ``protein_id`` is resolved in two steps. If
    the intensities table already contains ``protein_id``, it is
    used directly. Otherwise, ``var_annotation`` must be supplied
    and contain both ``peptide_id`` and ``protein_id``.

    ``sample_annotation``, when supplied, must contain a
    ``sample_id`` column and is merged into ``adata.obs``.

    ``var_annotation``, when supplied, must contain a ``peptide_id``
    column (peptide level) or a ``protein_id`` column (protein level)
    and is merged into ``adata.var``.

    Column names that differ from the defaults above can be mapped
    to the canonical names via ``column_map``.

    Parameters
    ----------
    intensities : str | Path | pd.DataFrame
        Long-form intensities data. Accepts a file path
        (str or Path) or a :class:`pandas.DataFrame`.
    level : {"peptide", "protein"}, default None
        Select whether to process peptide- or protein-level
        inputs. This argument is required.
    sample_annotation : str | Path | pd.DataFrame, optional
        Optional obs annotations. Accepts a file path or
        DataFrame.
    var_annotation : str | Path | pd.DataFrame, optional
        Optional var annotations. Accepts a file path or
        DataFrame. Interpreted as peptide annotations when
        ``level="peptide"`` and as protein annotations when
        ``level="protein"``.
    column_map : dict, optional
        Optional mapping that specifies custom column names
        for the expected keys: peptide_id, protein_id,
        sample_id, intensity.
    sep : str, optional
        Delimiter passed to `pandas.read_csv`. If None (the
        default), the separator is auto-detected from the
        file extension. Ignored when input is a DataFrame.
    fill_na : float, optional
        Optional replacement value for missing intensity
        entries.
    zero_to_na : bool, optional
        If True, zeros in the AnnData X matrix will be
        replaced with ``np.nan``.
    sort_obs_by_annotation : bool, default False
        When True, reorder observations to match the order
        of samples in the annotation (if supplied) or the
        original intensity table.
    verbose : bool, optional
        If True, print status messages.

    Returns
    -------
    AnnData
        Structured representation of the long-form
        intensities ready for downstream analysis.

    Examples
    --------
    **Example 1**: Minimal peptide-level read with
    ``protein_id`` in the intensities DataFrame.

    >>> import pandas as pd
    >>> import proteopy as pr
    >>> intensities = pd.DataFrame({
    ...     "sample_id": [
    ...         "S1", "S1", "S2", "S2",
    ...     ],
    ...     "peptide_id": [
    ...         "PEP1", "PEP2", "PEP1", "PEP2",
    ...     ],
    ...     "protein_id": [
    ...         "PROT1", "PROT1", "PROT1", "PROT1",
    ...     ],
    ...     "intensity": [
    ...         12450.0, 8730.0, 15320.0, 6890.0,
    ...     ],
    ... })
    >>> adata = pr.read.long(
    ...     intensities, level="peptide",
    ... )
    >>> adata
    AnnData object with n_obs × n_vars = 2 × 2
        obs: 'sample_id'
        var: 'peptide_id', 'protein_id'

    **Example 2**: Peptide-level read with ``protein_id``
    supplied via ``var_annotation`` instead of the intensities
    DataFrame.

    >>> intensities = pd.DataFrame({
    ...     "sample_id": [
    ...         "S1", "S1", "S2", "S2",
    ...     ],
    ...     "peptide_id": [
    ...         "PEP1", "PEP2", "PEP1", "PEP2",
    ...     ],
    ...     "intensity": [
    ...         12450.0, 8730.0, 15320.0, 6890.0,
    ...     ],
    ... })
    >>> var_ann = pd.DataFrame({
    ...     "peptide_id": ["PEP1", "PEP2"],
    ...     "protein_id": ["PROT1", "PROT1"],
    ... })
    >>> adata = pr.read.long(
    ...     intensities,
    ...     level="peptide",
    ...     var_annotation=var_ann,
    ... )
    >>> adata
    AnnData object with n_obs × n_vars = 2 × 2
        obs: 'sample_id'
        var: 'peptide_id', 'protein_id'

    **Example 3**: Peptide-level read with non-standard column
    names remapped via ``column_map``.

    >>> intensities = pd.DataFrame({
    ...     "run": ["S1", "S1", "S2", "S2"],
    ...     "seq": [
    ...         "PEP1", "PEP2", "PEP1", "PEP2",
    ...     ],
    ...     "prot": [
    ...         "PROT1", "PROT1", "PROT1", "PROT1",
    ...     ],
    ...     "quant": [
    ...         12450.0, 8730.0, 15320.0, 6890.0,
    ...     ],
    ... })
    >>> adata = pr.read.long(
    ...     intensities,
    ...     level="peptide",
    ...     column_map={
    ...         "sample_id": "run",
    ...         "peptide_id": "seq",
    ...         "protein_id": "prot",
    ...         "intensity": "quant",
    ...     },
    ... )
    >>> adata
    AnnData object with n_obs × n_vars = 2 × 2
        obs: 'sample_id'
        var: 'peptide_id', 'protein_id'
    """
    # -- Validate arguments
    if level is None:
        raise ValueError(
            "level is required; expected 'peptide' or "
            "'protein'."
        )

    level_normalised = level.lower()
    if level_normalised not in {"peptide", "protein"}:
        raise ValueError(
            "level must be one of {'peptide', 'protein'}; "
            f"got {level!r} instead."
        )

    if fill_na is not None and zero_to_na:
        raise ValueError(
            "fill_na and zero_to_na are mutually exclusive."
        )

    if column_map:
        if level_normalised == "peptide":
            valid_keys = {
                "sample_id", "intensity",
                "peptide_id", "protein_id",
            }
        else:
            valid_keys = {
                "sample_id", "intensity", "protein_id",
            }
        invalid = set(column_map).difference(valid_keys)
        if invalid:
            raise ValueError(
                "column_map contains keys not supported "
                f"at {level_normalised} level: "
                f"{', '.join(sorted(invalid))}"
            )

    # -- Load data
    df = load_dataframe(intensities, sep)

    sample_annotation_df = (
        load_dataframe(sample_annotation, sep)
        if sample_annotation is not None
        else None
    )

    var_annotation_df = (
        load_dataframe(var_annotation, sep)
        if var_annotation is not None
        else None
    )

    # -- Dispatch to level-specific helper
    if level_normalised == "peptide":
        adata = _peptides_long_from_df(
            df,
            sample_annotation_df=sample_annotation_df,
            peptide_annotation_df=var_annotation_df,
            fill_na=fill_na,
            zero_to_na=zero_to_na,
            column_map=column_map,
            sort_obs_by_annotation=sort_obs_by_annotation,
            verbose=verbose,
        )
        return adata
    else:
        adata = _proteins_long_from_df(
            df,
            sample_annotation_df=sample_annotation_df,
            protein_annotation_df=var_annotation_df,
            fill_na=fill_na,
            zero_to_na=zero_to_na,
            column_map=column_map,
            sort_obs_by_annotation=sort_obs_by_annotation,
            verbose=verbose,
        )
        return adata
