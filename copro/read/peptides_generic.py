from __future__ import annotations
from typing import Dict

import warnings
from functools import partial
import pandas as pd
import anndata as ad


def peptides_long(
	filepath: str,
	*,
	sep: str = "\t",
	filename_annotation: str | None = None,
	peptides_annotation_path: str | None = None,
	fill_na: float | None = None,
	column_map: Dict[str, str] | None = None,
	sort_obs_by_annotation: bool = False,
	) -> ad.AnnData:
	"""Convert a peptide-level intensity table into an AnnData container.

	Parameters
	----------
	filepath :
	    Path to a delimited text file containing peptide intensities.
	filename_annotation :
	    Optional path to per-filename annotations to be injected into `adata.obs`.
	peptides_annotation_path :
	    Optional path to per-peptide annotations merged into `adata.var`.
	sep :
	    Delimiter passed to `pandas.read_csv`; defaults to tab for TSV files.
	fill_na :
	    Optional replacement value for missing intensity entries.
	column_map :
	    Optional mapping that specifies custom column names for the keys
	    ``{"peptide_id", "protein_id", "filename", "intensities"}``.
	sort_obs_by_annotation :
	    When True, reorder observations to match the order of filenames in the
	    annotation (if supplied) or the original intensity table.

    Returns
    -------
    AnnData
        Structured representation of the peptide intensities ready for downstream analysis.
    """

	# Normalise the user-supplied column names to internal canonical labels.
	column_aliases = {
	    "peptide_id": "peptide_id",
	    "protein_id": "protein_id",
	    "filename": "filename",
	    "intensities": "intensities",
	}
	if column_map:
	    unexpected = set(column_map).difference(column_aliases)
	    if unexpected:
	        raise ValueError(
	            "column_map contains unsupported keys: "
	            f"{', '.join(sorted(unexpected))}"
	        )
	    column_aliases.update(column_map)

	df = pd.read_csv(filepath, sep=sep)

	required_actual_columns = {
	    column_aliases[key] for key in column_aliases
	}
	missing_columns = required_actual_columns.difference(df.columns)
	if missing_columns:
	    raise ValueError(
	        "DataFrame is missing required columns: "
	        f"{', '.join(sorted(missing_columns))}"
	    )

	# Rename columns so downstream logic can rely on canonical labels.
	rename_map_main = {
	    actual: canonical for canonical, actual in column_aliases.items()
	}
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

	# Optionally fill missing intensity values, keeping original data untouched.
	if fill_na is not None:
	    fill_value = float(fill_na)
	    df_work = df.copy()
	    df_work["intensities"] = df_work["intensities"].fillna(fill_value)
	else:
	    df_work = df

	default_obs_order = df_work[sample_column].drop_duplicates().tolist()
	annotation_order = None

	# Reshape to samples x peptides matrix.
	intensity_matrix = df_work.pivot(
	    index=sample_column,
	    columns="peptide_id",
	    values="intensities",
	)
	intensity_matrix = intensity_matrix.sort_index().sort_index(axis=1)
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

	if filename_annotation is not None:
	    annotation_df = pd.read_csv(filename_annotation, sep=sep)
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
	        duplicate_count = annotation_df.loc[
	            duplicate_mask, "filename"
	        ].nunique()
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

	    ignored_annotations_count = len(
	        annotation_filenames.difference(obs_filenames)
	    )
	    if ignored_annotations_count:
	        print(
	            f"{ignored_annotations_count} filename entries in the annotation "
	            "file were not present in the intensity table and were ignored."
	        )

	    missing_annotation_count = len(
	        obs_filenames.difference(annotation_filenames)
	    )
	    if missing_annotation_count:
	        print(
	            f"{missing_annotation_count} filename entries in the intensity "
	            "table did not have a matching annotation."
	        )

	    annotation_order = [
	        name for name in annotation_df_unique["filename"]
	        if name in obs_filenames
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

	if peptides_annotation_path is not None:
	    peptide_annotation_df = pd.read_csv(peptides_annotation_path, sep=sep)
	    rename_map_peptide = {
	        actual: canonical
	        for canonical, actual in column_aliases.items()
	        if actual in peptide_annotation_df.columns and canonical != actual
	    }
	    peptide_annotation_df = peptide_annotation_df.rename(
	        columns=rename_map_peptide
	    )

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
   

def peptides_long_legacy(
    intensities,
    sample_annotation,
    sep=',',
    sort_obs_by_sample_annotation=True,
    fill_na=None,
    obs_id_col='filename',
    ):
    '''
    Read in current typical NP proteomics output format.

    Args:
        peptide_intensities (str): Path to tabular file with columns
            protein_id, peptide_id, obs_id_col (default: filename) and intensity.
        sample_annotation (str): Path to tabular file with the column
            defined by obs_id_col matching the corresponding values of the
            peptide_intensities and other columns with annotations for
            the samples.
        sep (str | list): separators of the tabular files.
        obs_id_col (str): Column in both inputs representing the observation
            identifiers. Defaults to 'filename'.

    Returns:
        anndata.AnnData
    '''
    if isinstance(sep, str):
        sep_intensities = sep
        sep_sample_ann = sep

    elif isinstance(sep, list):
        assert len(sep) == 2
        sep_intensities, sep_sample_ann = sep

    else:
        raise ValueError('Invalid sep argument format.')

    # Peptide intensities (X)
    if isinstance(intensities, pd.DataFrame):
        intensities = intensities.copy()
    else:
        intensities = pd.read_csv(intensities, sep=sep_intensities)

    if obs_id_col not in intensities.columns:
        raise ValueError(f"Missing required column '{obs_id_col}' in intensities.")
    intensities = intensities.rename(columns={obs_id_col: 'sample_id'})
    
    # Decide feature dimension: peptide if available, otherwise protein
    has_peptide = 'peptide_id' in intensities.columns
    feature_col = 'peptide_id' if has_peptide else 'protein_id'

    required_cols = {'protein_id', 'sample_id', 'intensity'}
    missing = required_cols.difference(intensities.columns)
    if missing:
        raise ValueError(f"Missing required column(s) in intensities: {missing}")

    # Check: unique sample_id -> feature_col combination
    if intensities.duplicated(subset=['sample_id',feature_id]).any():
        raise ValueError(f'Repeated {obs_id_col}-{feature_id} value entries')

    intensities = pd.pivot_table(intensities, index='sample_id', columns=feature_col, values='intensity', aggfunc='sum')
    intensities = intensities.sort_index(axis=0).sort_index(axis=1)
    intensities.index.name = None
    intensities.columns.name = None

    if fill_na is not None:
        intensities = intensities.fillna(fill_na)
    
    intensities.index = intensities.index.astype(str).str.strip()

    assert len(intensities.columns) == len(intensities.columns.unique())
    assert len(intensities.index) == len(intensities.index.unique())

    # Variable annotation (.var)
    if has_peptide:
        print(intensities.columns)
        peptides = intensities[['protein_id', 'peptide_id']].copy()
        peptides = peptides.drop_duplicates(subset='peptide_id')
        peptides = peptides.set_index('peptide_id', drop=False)
    else:
        peptides = intensities[['protein_id']].copy() if isinstance(intensities, pd.DataFrame) else pd.read_csv(intensities, sep=sep_intensities)[['protein_id']]
        peptides = peptides.drop_duplicates(subset='protein_id')
        peptides = peptides.set_index('protein_id', drop=False)
    peptides.index.name = None
    peptides = peptides.loc[intensities.columns,]

    # Observation annotation (.obs)
    if isinstance(sample_annotation, pd.DataFrame):
        sample_annotation = sample_annotation.copy()
    else:
        sample_annotation = pd.read_csv(sample_annotation, sep=sep_sample_ann)
    if obs_id_col not in sample_annotation.columns:
        raise ValueError(
            f"sample_annotation must contain the '{obs_id_col}' column matching intensities."
            )
    sample_annotation = sample_annotation.rename(columns={obs_id_col: 'sample_id'})
    sample_annotation['sample_id'] = sample_annotation['sample_id'].astype(str).str.strip()
    sample_annotation = sample_annotation.set_index('sample_id', drop=False)
    sample_annotation.index.name = None

    assert len(sample_annotation.index) == len(sample_annotation.index.unique())
    assert len(sample_annotation.columns) == len(sample_annotation.columns.unique())

    if len(sample_annotation.index.difference(intensities.index)):
        diff = sample_annotation.index.difference(intensities.index)
        overlap = sample_annotation.index.intersection(intensities.index)
        sample_annotation =  sample_annotation.loc[overlap,]
        warnings.warn((
            f'There are {len(diff)} rows/obs in sample_annotation '
            f'which are not found in intensities. They were ignored.'
            ))

    if len(intensities.index.difference(sample_annotation.index)):
        diff = intensities.index.difference(sample_annotation.index)
        sample_annotation = sample_annotation.reindex(sample_annotation.index.append(diff))
        warnings.warn((
            f'There are {len(diff)} obs in intensities which are not found in '
            f'sample_annotation, which were filled with nan.'
            ))

    if sort_obs_by_sample_annotation:
        intensities = intensities.loc[sample_annotation.index]
    else:
        sample_annotation = sample_annotation.loc[intensities.index,]



    adata = ad.AnnData(
        intensities,
        obs=sample_annotation,
        var=peptides,
        )

    adata.strings_to_categoricals()
    adata.obs_names_make_unique()

    return adata
