from __future__ import annotations

import warnings
from typing import Dict, Literal

import anndata as ad
import pandas as pd
from pathlib import Path

from copro.utils.anndata import check_proteodata
from copro.utils.pandas import load_dataframe


def _peptides_long_from_df(
    intensities_df: pd.DataFrame,
    *,
    sample_annotation_df: pd.DataFrame | None = None,
    peptide_annotation_df: pd.DataFrame | None = None,
    fill_na: float | None = None,
    column_map: Dict[str, str] | None = None,
    sort_obs_by_annotation: bool = False,
    ) -> ad.AnnData:
    """Convert pre-loaded peptide-level tables into an AnnData container."""
    # Normalize user-specified column names so downstream code can rely on a
    # fixed set of canonical fields (peptide_id, protein_id, sample_id,
    # intensity) regardless of the input header names.
    column_aliases = {
        "peptide_id": "peptide_id",
        "protein_id": "protein_id",
        "sample_id": "sample_id",
        "intensity": "intensity",
        }
    if column_map:
        unexpected = set(column_map).difference(column_aliases)
        if unexpected:
            raise ValueError(
                "column_map contains unsupported keys: "
                f"{', '.join(sorted(unexpected))}"
                )
        if len(set(column_map.values())) != len(column_map.values()):
            raise ValueError(
                "column_map must map each canonical key to a unique source column."
                )
        column_aliases.update(column_map)

    df = intensities_df.copy()

    required_actual_columns = {column_aliases[key] for key in column_aliases}
    missing_columns = required_actual_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(
            "DataFrame is missing required columns: "
            f"{', '.join(sorted(missing_columns))}"
            )

    # Rename columns now so all later checks (duplicates, mapping) operate on
    # canonical labels instead of alias-specific column names.
    rename_map_main = {actual: canonical for canonical, actual in column_aliases.items()}
    df = df.rename(columns=rename_map_main)

    try:
        df["intensity"] = pd.to_numeric(df["intensity"], errors="raise")
    except Exception as exc:
        raise ValueError("intensity column must contain numeric values.") from exc

    sample_column = "sample_id"
    duplicate_mask = df.duplicated(subset=[sample_column, "peptide_id"])
    if duplicate_mask.any():
        duplicated = df.loc[duplicate_mask, [sample_column, "peptide_id"]]
        n_duplicates = len(duplicated)
        sample_duplicates = duplicated.head(5).to_dict(orient='records')
        extra = f" (showing first 5 of {n_duplicates})" if n_duplicates > 5 else ""
        raise ValueError(
            f"Duplicate peptide entries per sample detected{extra}: "
            f"{sample_duplicates}"
            )

    protein_counts = df.groupby("peptide_id")["protein_id"].nunique()
    inconsistent = protein_counts[protein_counts > 1]
    if not inconsistent.empty:
        raise ValueError(
            "Each peptide_id must map to exactly one protein_id; conflicts for: "
            f"{', '.join(map(str, inconsistent.index.tolist()))}"
            )

    if fill_na is not None:
        fill_value = float(fill_na)

    default_obs_order = df[sample_column].drop_duplicates().tolist()
    annotation_order: list[str] | None = None

    # Pivot long-form rows into a deterministic samples x proteins matrix;
    # sorting both axes ensures stable ordering even if the input table was
    # shuffled.
    intensity_matrix = df.pivot(
        index=sample_column,
        columns="peptide_id",
        values="intensity",
    )
    intensity_matrix = intensity_matrix.astype(float)
    intensity_matrix = intensity_matrix.sort_index().sort_index(axis=1)
    if fill_na is not None:
        intensity_matrix = intensity_matrix.fillna(fill_value)
    intensity_matrix.index.name = None
    intensity_matrix.columns.name = None

    peptide_to_protein = (
        df.groupby("peptide_id", sort=False)["protein_id"]
        .first()
        .reindex(intensity_matrix.columns)
        )

    obs = pd.DataFrame(index=intensity_matrix.index)
    obs["sample_id"] = obs.index

    if sample_annotation_df is not None:
        annotation_df = sample_annotation_df.copy()
        actual_sample_id = column_aliases["sample_id"]
        rename_map_annotation = (
            {actual_sample_id: "sample_id"}
            if actual_sample_id in annotation_df.columns
            and actual_sample_id != "sample_id"
            else {}
            )
        annotation_df = annotation_df.rename(columns=rename_map_annotation)

        if "sample_id" not in annotation_df.columns:
            raise ValueError(
                "Annotation file is missing the required `sample_id` column."
                )

        duplicate_mask = annotation_df.duplicated(subset=["sample_id"], keep=False)
        if duplicate_mask.any():
            duplicate_count = annotation_df.loc[duplicate_mask, "sample_id"].nunique()
            warnings.warn(
                "Duplicate sample entries found in annotation file; keeping the "
                f"first occurrence for {duplicate_count} sample IDs.",
                UserWarning,
                )

        annotation_df_unique = annotation_df.drop_duplicates(
            subset=["sample_id"], keep="first"
            )

        obs_samples = set(obs["sample_id"])
        annotation_samples = set(annotation_df_unique["sample_id"])

        ignored_annotations_count = len(annotation_samples.difference(obs_samples))
        if ignored_annotations_count:
            print(
                f"{ignored_annotations_count} sample_id entries in the annotation "
                "file were not present in the intensity table and were ignored."
                )

        missing_annotation_count = len(obs_samples.difference(annotation_samples))
        if missing_annotation_count:
            print(
                f"{missing_annotation_count} sample_id entries in the intensity "
                "table did not have a matching annotation."
                )

        annotation_order = [
            name for name in annotation_df_unique["sample_id"] if name in obs_samples
            ]

        obs_reset = obs.reset_index(names="_obs_index")
        merged_obs = obs_reset.merge(
            annotation_df_unique,
            how="left",
            on="sample_id",
            suffixes=("", "_annotation"),
            )
        merged_obs.set_index("_obs_index", inplace=True)
        merged_obs.index.name = None
        obs = merged_obs

    # Initialise var with peptide/protein identifiers.
    var = pd.DataFrame(index=intensity_matrix.columns)
    var.index.name = None
    var["peptide_id"] = var.index
    var["protein_id"] = peptide_to_protein.loc[var.index].values

    if peptide_annotation_df is not None:
        peptide_annotation_df = peptide_annotation_df.copy()
        rename_map_peptide = {
            column_aliases[key]: key
            for key in ("peptide_id", "protein_id")
            if column_aliases[key] in peptide_annotation_df.columns
            and column_aliases[key] != key
            }
        peptide_annotation_df = peptide_annotation_df.rename(columns=rename_map_peptide)

        if "peptide_id" not in peptide_annotation_df.columns:
            raise ValueError(
                "Peptide annotation file is missing the required `peptide_id` column."
                )

        duplicate_mask = peptide_annotation_df.duplicated(
            subset=["peptide_id"], keep=False
            )
        if duplicate_mask.any():
            duplicate_count = peptide_annotation_df.loc[
                duplicate_mask, "peptide_id"
            ].nunique()
            warnings.warn(
                "Duplicate peptide entries found in peptide annotation file; "
                f"keeping the first occurrence for {duplicate_count} peptides.",
                UserWarning,
                )

        peptide_annotation_unique = peptide_annotation_df.drop_duplicates(
            subset=["peptide_id"], keep="first"
            )

        var_peptides = set(var["peptide_id"])
        annotation_peptides = set(peptide_annotation_unique["peptide_id"])

        ignored_peptide_annotations = len(
            annotation_peptides.difference(var_peptides)
            )
        if ignored_peptide_annotations:
            print(
                f"{ignored_peptide_annotations} peptide entries in the annotation "
                "file were not present in the intensity matrix and were ignored."
                )

        missing_peptide_annotations = len(
            var_peptides.difference(annotation_peptides)
        )
        if missing_peptide_annotations:
            print(
                f"{missing_peptide_annotations} peptide entries in the intensity "
                "matrix did not have a matching peptide annotation."
                )

        var_reset = var.reset_index(names="_var_index")
        merged_var = var_reset.merge(
            peptide_annotation_unique,
            how="left",
            on="peptide_id",
            suffixes=("", "_annotation"),
            )
        merged_var.set_index("_var_index", inplace=True)
        merged_var.index.name = None
        var = merged_var

    if sort_obs_by_annotation:
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
        intensity_matrix = intensity_matrix.reindex(final_order)
        obs = obs.loc[final_order]

    adata = ad.AnnData(
        X=intensity_matrix.to_numpy(copy=True),
        obs=obs,
        var=var,
        )
    adata.strings_to_categoricals()
    check_proteodata(adata)

    return adata


def _proteins_long_from_df(
    intensities_df: pd.DataFrame,
    *,
    sample_annotation_df: pd.DataFrame | None = None,
    protein_annotation_df: pd.DataFrame | None = None,
    fill_na: float | None = None,
    column_map: Dict[str, str] | None = None,
    sort_obs_by_annotation: bool = False,
    ) -> ad.AnnData:
    """Convert pre-loaded protein-level tables into an AnnData container."""
    # Normalize user-specified column names so downstream code can rely on a
    # fixed set of canonical fields (protein_id, sample_id, intensity)
    # regardless of the input header names.
    column_aliases = {
        "protein_id": "protein_id",
        "sample_id": "sample_id",
        "intensity": "intensity",
        }
    if column_map:
        unexpected = set(column_map).difference(column_aliases)
        if unexpected:
            raise ValueError(
                "column_map contains unsupported keys: "
                f"{', '.join(sorted(unexpected))}"
                )
        if len(set(column_map.values())) != len(column_map.values()):
            raise ValueError(
                "column_map must map each canonical key to a unique source column."
                )
        column_aliases.update(column_map)

    df = intensities_df.copy()

    required_columns = {column_aliases[key] for key in column_aliases}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(
            "DataFrame is missing required columns: "
            f"{', '.join(sorted(missing_columns))}"
            )

    # Rename columns now so all later checks (duplicates, mapping) operate on
    # canonical labels instead of alias-specific column names.
    rename_map_main = {actual: canonical for canonical, actual in column_aliases.items()}
    df = df.rename(columns=rename_map_main)

    try:
        df["intensity"] = pd.to_numeric(df["intensity"], errors="raise")
    except Exception as exc:
        raise ValueError("intensity column must contain numeric values.") from exc

    sample_column = "sample_id"
    duplicate_mask = df.duplicated(subset=[sample_column, "protein_id"])
    if duplicate_mask.any():
        duplicated = df.loc[duplicate_mask, [sample_column, "protein_id"]]
        n_duplicates = len(duplicated)
        sample_duplicates = duplicated.head(5).to_dict(orient='records')
        extra = f" (showing first 5 of {n_duplicates})" if n_duplicates > 5 else ""
        raise ValueError(
            f"Duplicate protein entries per sample detected{extra}: "
            f"{sample_duplicates}"
            )

    if fill_na is not None:
        fill_value = float(fill_na)

    default_obs_order = df[sample_column].drop_duplicates().tolist()
    annotation_order: list[str] | None = None

    # Pivot long-form rows into a deterministic samples x proteins matrix;
    # sorting both axes ensures stable ordering even if the input table was
    # shuffled.
    intensity_matrix = df.pivot(
        index=sample_column,
        columns="protein_id",
        values="intensity",
    )
    intensity_matrix = intensity_matrix.astype(float)
    intensity_matrix = intensity_matrix.sort_index().sort_index(axis=1)
    if fill_na is not None:
        intensity_matrix = intensity_matrix.fillna(fill_value)
    intensity_matrix.index.name = None
    intensity_matrix.columns.name = None

    # Keep sample_id both as index and explicit column to preserve the original
    # identifiers and enable merges with optional sample annotations.
    obs = pd.DataFrame(index=intensity_matrix.index)
    obs["sample_id"] = obs.index

    if sample_annotation_df is not None:
        annotation_df = sample_annotation_df.copy()
        actual_sample_id = column_aliases["sample_id"]
        rename_map_annotation = (
            {actual_sample_id: "sample_id"}
            if actual_sample_id in annotation_df.columns
            and actual_sample_id != "sample_id"
            else {}
            )
        annotation_df = annotation_df.rename(columns=rename_map_annotation)

        if "sample_id" not in annotation_df.columns:
            raise ValueError(
                "Annotation file is missing the required `sample_id` column."
                )

        duplicate_mask = annotation_df.duplicated(subset=["sample_id"], keep=False)
        if duplicate_mask.any():
            duplicate_count = annotation_df.loc[duplicate_mask, "sample_id"].nunique()
            warnings.warn(
                "Duplicate sample entries found in annotation file; keeping the "
                f"first occurrence for {duplicate_count} sample IDs.",
                UserWarning,
                )

        annotation_df_unique = annotation_df.drop_duplicates(
            subset=["sample_id"], keep="first"
            )

        obs_samples = set(obs["sample_id"])
        annotation_samples = set(annotation_df_unique["sample_id"])

        ignored_annotations = len(annotation_samples.difference(obs_samples))
        if ignored_annotations:
            print(
                f"{ignored_annotations} sample_id entries in the annotation file "
                "were not present in the intensity table and were ignored."
                )

        missing_annotations = len(obs_samples.difference(annotation_samples))
        if missing_annotations:
            print(
                f"{missing_annotations} sample_id entries in the intensity table "
                "did not have a matching annotation."
                )

        annotation_order = [
            name for name in annotation_df_unique["sample_id"] if name in obs_samples
            ]

        obs_reset = obs.reset_index(names="_obs_index")
        merged_obs = obs_reset.merge(
            annotation_df_unique,
            how="left",
            on="sample_id",
            suffixes=("", "_annotation"),
            )
        merged_obs.set_index("_obs_index", inplace=True)
        merged_obs.index.name = None
        obs = merged_obs

    # Initialise var with protein identifiers.
    var = pd.DataFrame(index=intensity_matrix.columns)
    var.index.name = None
    var["protein_id"] = var.index

    if protein_annotation_df is not None:
        protein_annotation_df = protein_annotation_df.copy()
        actual_protein_id = column_aliases["protein_id"]
        rename_map_protein = (
            {actual_protein_id: "protein_id"}
            if actual_protein_id in protein_annotation_df.columns
            and actual_protein_id != "protein_id"
            else {}
            )
        protein_annotation_df = protein_annotation_df.rename(
            columns=rename_map_protein
            )

        if "protein_id" not in protein_annotation_df.columns:
            raise ValueError(
                "Protein annotation file is missing the required `protein_id` column."
                )

        duplicate_mask = protein_annotation_df.duplicated(
            subset=["protein_id"], keep=False
            )
        if duplicate_mask.any():
            duplicate_count = protein_annotation_df.loc[
                duplicate_mask, "protein_id"
                ].nunique()
            warnings.warn(
                "Duplicate protein entries found in protein annotation file; "
                f"keeping the first occurrence for {duplicate_count} proteins.",
                UserWarning,
                )

        protein_annotation_unique = protein_annotation_df.drop_duplicates(
            subset=["protein_id"], keep="first"
            )

        var_proteins = set(var["protein_id"])
        annotation_proteins = set(protein_annotation_unique["protein_id"])

        ignored_protein_annotations = len(
            annotation_proteins.difference(var_proteins)
            )
        if ignored_protein_annotations:
            print(
                f"{ignored_protein_annotations} protein entries in the annotation "
                "file were not present in the intensity matrix and were ignored."
                )

        missing_protein_annotations = len(
            var_proteins.difference(annotation_proteins)
            )
        if missing_protein_annotations:
            print(
                f"{missing_protein_annotations} protein entries in the intensity "
                "matrix did not have a matching protein annotation."
                )

        var_reset = var.reset_index(names="_var_index")
        merged_var = var_reset.merge(
            protein_annotation_unique,
            how="left",
            on="protein_id",
            suffixes=("", "_annotation"),
            )
        merged_var.set_index("_var_index", inplace=True)
        merged_var.index.name = None
        var = merged_var

    if sort_obs_by_annotation:
        desired_order = annotation_order or default_obs_order
        seen = set()
        final_order: list[str] = []
        for name in desired_order:
            if name in intensity_matrix.index and name not in seen:
                final_order.append(name)
                seen.add(name)
        for name in intensity_matrix.index:
            if name not in seen:
                final_order.append(name)
                seen.add(name)
        intensity_matrix = intensity_matrix.reindex(final_order)
        obs = obs.loc[final_order]

    adata = ad.AnnData(
        X=intensity_matrix.to_numpy(copy=True),
        obs=obs,
        var=var,
        )
    adata.strings_to_categoricals()
    check_proteodata(adata)

    return adata


def long(
    intensities: str | Path | pd.DataFrame,
    *,
    level: Literal["peptide", "protein"] | None = None,
    sep: str | None = None,
    sample_annotation: str | Path | pd.DataFrame | None = None,
    var_annotation: str | Path | pd.DataFrame | None = None,
    fill_na: float | None = None,
    column_map: Dict[str, str] | None = None,
    sort_obs_by_annotation: bool = False,
    ) -> ad.AnnData:
    """Read long-format peptide or protein files into an AnnData container.

    Parameters
    ----------
    intensities : str | Path | pd.DataFrame
        Long-form intensities data. Accepts a file path (str or Path) or a
        pandas DataFrame. When a file path is provided, the separator is
        auto-detected from the extension (.csv -> ',', .tsv -> '\\t') unless
        `sep` is explicitly specified.
    level : {"peptide", "protein"}, default None
        Select whether to process peptide- or protein-level inputs. This
        argument is required.
    sep : str, optional
        Delimiter passed to `pandas.read_csv`. If None (the default),
        the separator is auto-detected from the file extension. Ignored
        when input is a DataFrame.
    sample_annotation : str | Path | pd.DataFrame, optional
        Optional obs annotations. Accepts a file path or DataFrame.
    var_annotation : str | Path | pd.DataFrame, optional
        Optional var annotations merged into ``adata.var``. Accepts a file
        path or DataFrame. Interpreted as peptide annotations when
        ``level="peptide"`` and as protein annotations when ``level="protein"``.
    fill_na : float, optional
        Optional replacement value for missing intensity entries.
    column_map : dict, optional
        Optional mapping that specifies custom column names for the expected
        keys: peptide_id, protein_id, sample_id, intensity.
    sort_obs_by_annotation : bool, default False
        When True, reorder observations to match the order of samples in the
        annotation (if supplied) or the original intensity table.

    Returns
    -------
    AnnData
        Structured representation of the long-form intensities ready for
        downstream analysis.
    """
    if level is None:
        raise ValueError("level is required; expected 'peptide' or 'protein'.")

    level_normalised = level.lower()
    if level_normalised not in {"peptide", "protein"}:
        raise ValueError(
            "level must be one of {'peptide', 'protein'}; "
            f"got {level!r} instead."
            )

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

    if level_normalised == "peptide":
        adata = _peptides_long_from_df(
            df,
            sample_annotation_df=sample_annotation_df,
            peptide_annotation_df=var_annotation_df,
            fill_na=fill_na,
            column_map=column_map,
            sort_obs_by_annotation=sort_obs_by_annotation,
            )
        return adata
    else:
        adata = _proteins_long_from_df(
            df,
            sample_annotation_df=sample_annotation_df,
            protein_annotation_df=var_annotation_df,
            fill_na=fill_na,
            column_map=column_map,
            sort_obs_by_annotation=sort_obs_by_annotation,
            )
        return adata
