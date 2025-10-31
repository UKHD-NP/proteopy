from __future__ import annotations

import warnings
from typing import Dict, Literal

import anndata as ad
import pandas as pd


def peptides_long_from_df(
    intensities_df: pd.DataFrame,
    *,
    filename_annotation_df: pd.DataFrame | None = None,
    peptide_annotation_df: pd.DataFrame | None = None,
    fill_na: float | None = None,
    column_map: Dict[str, str] | None = None,
    sort_obs_by_annotation: bool = False,
    ) -> ad.AnnData:
    """Convert pre-loaded peptide-level tables into an AnnData container."""
    # Normalise the user-supplied column names to internal canonical labels.
    column_aliases = {
        "peptide_id": "peptide_id",
        "protein_id": "protein_id",
        "filename": "filename",
        "intensity": "intensity",
        }
    if column_map:
        unexpected = set(column_map).difference(column_aliases)
        if unexpected:
            raise ValueError(
                "column_map contains unsupported keys: "
                f"{', '.join(sorted(unexpected))}"
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

    # Rename columns so downstream logic can rely on canonical labels.
    rename_map_main = {actual: canonical for canonical, actual in column_aliases.items()}
    df = df.rename(columns=rename_map_main)

    sample_column = "filename"
    duplicate_mask = df.duplicated(subset=[sample_column, "peptide_id"])
    if duplicate_mask.any():
        duplicated = df.loc[duplicate_mask, [sample_column, "peptide_id"]]
        raise ValueError(
            "Duplicate peptide entries per sample detected: "
            f"{duplicated.to_dict(orient='records')}"
            )

    protein_counts = df.groupby("peptide_id")["protein_id"].nunique()
    inconsistent = protein_counts[protein_counts > 1]
    if not inconsistent.empty:
        raise ValueError(
            "Each peptide_id must map to exactly one protein_id; conflicts for: "
            f"{', '.join(map(str, inconsistent.index.tolist()))}"
            )

    # Optionally fill _missing intensity values, keeping original data untouched.
    if fill_na is not None:
        fill_value = float(fill_na)
        df_work = df.copy()
        df_work["intensity"] = df_work["intensity"].fillna(fill_value)
    else:
        df_work = df

    default_obs_order = df_work[sample_column].drop_duplicates().tolist()
    annotation_order = None

    # Reshape to samples x peptides matrix.
    intensity_matrix = df_work.pivot(
        index=sample_column,
        columns="peptide_id",
        values="intensity",
    )
    intensity_matrix = intensity_matrix.sort_index().sort_index(axis=1)
    if fill_na is not None:
        intensity_matrix = intensity_matrix.fillna(fill_value)
    intensity_matrix.index.name = None
    intensity_matrix.columns.name = None

    peptide_to_protein = (
        df_work.groupby("peptide_id", sort=False)["protein_id"]
        .first()
        .reindex(intensity_matrix.columns)
        )

    # Build obs with the sample identifier retained as a column.
    obs = pd.DataFrame(index=intensity_matrix.index)
    obs["filename"] = obs.index

    if filename_annotation_df is not None:
        annotation_df = filename_annotation_df.copy()
        rename_map_annotation = {
            actual: canonical
            for canonical, actual in column_aliases.items()
            if actual in annotation_df.columns and canonical != actual
            }
        annotation_df = annotation_df.rename(columns=rename_map_annotation)

        if "filename" not in annotation_df.columns:
            raise ValueError(
                "Annotation file is missing the required `filename` column."
                )

        duplicate_mask = annotation_df.duplicated(subset=["filename"], keep=False)
        if duplicate_mask.any():
            duplicate_count = annotation_df.loc[duplicate_mask, "filename"].nunique()
            warnings.warn(
                "Duplicate filename entries found in annotation file; keeping "
                f"the first occurrence for {duplicate_count} filenames.",
                UserWarning,
                )

        annotation_df_unique = annotation_df.drop_duplicates(
            subset=["filename"], keep="first"
            )

        obs_filenames = set(obs["filename"])
        annotation_filenames = set(annotation_df_unique["filename"])

        ignored_annotations_count = len(annotation_filenames.difference(obs_filenames))
        if ignored_annotations_count:
            print(
                f"{ignored_annotations_count} filename entries in the annotation "
                "file were not present in the intensity table and were ignored."
                )

        missing_annotation_count = len(obs_filenames.difference(annotation_filenames))
        if missing_annotation_count:
            print(
                f"{missing_annotation_count} filename entries in the intensity "
                "table did not have a matching annotation."
                )

        annotation_order = [
            name for name in annotation_df_unique["filename"] if name in obs_filenames
            ]

        obs_reset = obs.reset_index().rename(columns={"index": "_obs_index"})
        merged_obs = obs_reset.merge(
            annotation_df_unique,
            how="left",
            on="filename",
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
            actual: canonical
            for canonical, actual in column_aliases.items()
            if actual in peptide_annotation_df.columns and canonical != actual
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

        var_reset = var.reset_index().rename(columns={"index": "_var_index"})
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
        seen = set()
        final_order = []
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

    return adata


def proteins_long_from_df(
    intensities_df: pd.DataFrame,
    *,
    filename_annotation_df: pd.DataFrame | None = None,
    protein_annotation_df: pd.DataFrame | None = None,
    fill_na: float | None = None,
    column_map: Dict[str, str] | None = None,
    sort_obs_by_annotation: bool = False,
    ) -> ad.AnnData:
    """Convert pre-loaded protein-level tables into an AnnData container."""

    column_aliases = {
        "protein_id": "protein_id",
        "filename": "filename",
        "intensity": "intensity",
        }
    if column_map:
        unexpected = set(column_map).difference(column_aliases)
        if unexpected:
            raise ValueError(
                "column_map contains unsupported keys: "
                f"{', '.join(sorted(unexpected))}"
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

    rename_map_main = {actual: canonical for canonical, actual in column_aliases.items()}
    df = df.rename(columns=rename_map_main)

    sample_column = "filename"
    duplicate_mask = df.duplicated(subset=[sample_column, "protein_id"])
    if duplicate_mask.any():
        duplicated = df.loc[duplicate_mask, [sample_column, "protein_id"]]
        raise ValueError(
            "Duplicate protein entries per sample detected: "
            f"{duplicated.to_dict(orient='records')}"
            )

    if fill_na is not None:
        fill_value = float(fill_na)
        df_work = df.copy()
        df_work["intensity"] = df_work["intensity"].fillna(fill_value)
    else:
        df_work = df

    default_obs_order = df_work[sample_column].drop_duplicates().tolist()
    annotation_order: list[str] | None = None

    intensity_matrix = df_work.pivot(
        index=sample_column,
        columns="protein_id",
        values="intensity",
    )
    intensity_matrix = intensity_matrix.sort_index().sort_index(axis=1)
    if fill_na is not None:
        intensity_matrix = intensity_matrix.fillna(fill_value)
    intensity_matrix.index.name = None
    intensity_matrix.columns.name = None

    obs = pd.DataFrame(index=intensity_matrix.index)
    obs["filename"] = obs.index

    if filename_annotation_df is not None:
        annotation_df = filename_annotation_df.copy()
        rename_map_annotation = {
            actual: canonical
            for canonical, actual in column_aliases.items()
            if actual in annotation_df.columns and canonical != actual
            }
        annotation_df = annotation_df.rename(columns=rename_map_annotation)

        if "filename" not in annotation_df.columns:
            raise ValueError(
                "Annotation file is missing the required `filename` column."
                )

        duplicate_mask = annotation_df.duplicated(subset=["filename"], keep=False)
        if duplicate_mask.any():
            duplicate_count = annotation_df.loc[duplicate_mask, "filename"].nunique()
            warnings.warn(
                "Duplicate filename entries found in annotation file; keeping "
                f"the first occurrence for {duplicate_count} filenames.",
                UserWarning,
                )

        annotation_df_unique = annotation_df.drop_duplicates(
            subset=["filename"], keep="first"
            )

        obs_filenames = set(obs["filename"])
        annotation_filenames = set(annotation_df_unique["filename"])

        ignored_annotations = len(annotation_filenames.difference(obs_filenames))
        if ignored_annotations:
            print(
                f"{ignored_annotations} filename entries in the annotation file "
                "were not present in the intensity table and were ignored."
                )

        missing_annotations = len(obs_filenames.difference(annotation_filenames))
        if missing_annotations:
            print(
                f"{missing_annotations} filename entries in the intensity table "
                "did not have a matching annotation."
                )

        annotation_order = [
            name for name in annotation_df_unique["filename"] if name in obs_filenames
            ]

        obs_reset = obs.reset_index().rename(columns={"index": "_obs_index"})
        merged_obs = obs_reset.merge(
            annotation_df_unique,
            how="left",
            on="filename",
            )
        merged_obs.set_index("_obs_index", inplace=True)
        merged_obs.index.name = None
        obs = merged_obs

    var = pd.DataFrame(index=intensity_matrix.columns)
    var.index.name = None
    var["protein_id"] = var.index

    if protein_annotation_df is not None:
        protein_annotation_df = protein_annotation_df.copy()
        rename_map_protein = {
            actual: canonical
            for canonical, actual in column_aliases.items()
            if actual in protein_annotation_df.columns and canonical != actual
            }
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

        var_reset = var.reset_index().rename(columns={"index": "_var_index"})
        merged_var = var_reset.merge(
            protein_annotation_unique,
            how="left",
            on="protein_id",
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

    return adata


def long(
    intensities_path: str,
    *,
    level: Literal["peptide", "protein"] | None = None,
    sep: str = "\t",
    filename_annotation_path: str | None = None,
    annotation_path: str | None = None,
    fill_na: float | None = None,
    column_map: Dict[str, str] | None = None,
    sort_obs_by_annotation: bool = False,
    ) -> ad.AnnData:
    """Read long-format peptide or protein files into an AnnData container.

    Parameters
    ----------
    intensities_path : str
        Path to a delimited text file containing long-form intensities.
    level : {"peptide", "protein"}, default None
        Select whether to process peptide- or protein-level inputs. This argument is required.
    sep : str, default "\\t"
        Delimiter passed to `pandas.read_csv`.
    filename_annotation_path : str, optional
        Optional path to per-filename annotations to be injected into `adata.obs`.
    annotation_path : str, optional
        Optional path to feature-level annotations merged into `adata.var`.
        The file is interpreted as peptide annotations when `level="peptide"` and as
        protein annotations when `level="protein"`.
    fill_na : float, optional
        Optional replacement value for missing intensity entries.
    column_map : dict, optional
        Optional mapping that specifies custom column names for the expected keys.
    sort_obs_by_annotation : bool, default False
        When True, reorder observations to match the order of filenames in the
        annotation (if supplied) or the original intensity table.

    Returns
    -------
    AnnData
        Structured representation of the long-form intensities ready for downstream analysis.
    """
    if level is None:
        raise ValueError("level is required; expected 'peptide' or 'protein'.")

    level_normalised = level.lower()
    if level_normalised not in {"peptide", "protein"}:
        raise ValueError(
            "level must be one of {'peptide', 'protein'}; "
            f"got {level!r} instead."
            )

    df = pd.read_csv(intensities_path, sep=sep)

    filename_annotation_df = (
        pd.read_csv(filename_annotation_path, sep=sep)
        if filename_annotation_path
        else None
        )

    annotation_df = (
        pd.read_csv(annotation_path, sep=sep)
        if annotation_path
        else None
        )

    if level_normalised == "peptide":
        return peptides_long_from_df(
            df,
            filename_annotation_df=filename_annotation_df,
            peptide_annotation_df=annotation_df,
            fill_na=fill_na,
            column_map=column_map,
            sort_obs_by_annotation=sort_obs_by_annotation,
            )

    return proteins_long_from_df(
        df,
        filename_annotation_df=filename_annotation_df,
        protein_annotation_df=annotation_df,
        fill_na=fill_na,
        column_map=column_map,
        sort_obs_by_annotation=sort_obs_by_annotation,
        )
