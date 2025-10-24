from __future__ import annotations

import warnings
from typing import Dict

import anndata as ad
import pandas as pd


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


def proteins_long(
    intensities_path: str,
    *,
    sep: str = "\t",
    filename_annotation_path: str | None = None,
    protein_annotation_path: str | None = None,
    fill_na: float | None = None,
    column_map: Dict[str, str] | None = None,
    sort_obs_by_annotation: bool = False,
    ) -> ad.AnnData:
    """Read protein-level files and delegate to ``proteins_long_from_df``."""
    df = pd.read_csv(intensities_path, sep=sep)

    filename_annotation_df = (
        pd.read_csv(filename_annotation_path, sep=sep) if filename_annotation_path else None
        )

    protein_annotation_df = (
        pd.read_csv(protein_annotation_path, sep=sep)
        if protein_annotation_path
        else None
        )

    return proteins_long_from_df(
        df,
        filename_annotation_df=filename_annotation_df,
        protein_annotation_df=protein_annotation_df,
        fill_na=fill_na,
        column_map=column_map,
        sort_obs_by_annotation=sort_obs_by_annotation,
        )
