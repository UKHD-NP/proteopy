import warnings
import pandas as pd
import anndata as ad

def peptides_long(
        intensities_path,
        sample_annotation_path,
        sep = ',',
        sort_obs_by_sample_annotation = True,
        ):
    '''
    Read in current typical NP proteomics output format.

    Args:
        peptide_intensities_path (str): Path to tabular file with columns
            protein_id, peptide_id, filename and intensity.
        sample_annotation_path (str): Path to tabular file with the column
            filename matching filename values of the peptide)intensities_path
            and other columns with annotations for the samples.
        sep (str | list): separators of the tabular files.

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
    intensities = pd.read_csv(intensities_path, sep=sep_intensities)
    intensities = intensities.rename(columns={'filename': 'sample_id'})
    peptides = intensities[['protein_id', 'peptide_id']].copy()
    intensities = pd.pivot(intensities, index=['sample_id'], columns='peptide_id', values='intensity')
    intensities = intensities.sort_index(axis=0).sort_index(axis=1)
    intensities.index.name = None
    intensities.columns.name = None

    assert len(intensities.columns) == len(intensities.columns.unique())
    assert len(intensities.index) == len(intensities.index.unique())

    # Variable annotation (.var)
    peptides = peptides.drop_duplicates(subset='peptide_id')
    peptides = peptides.set_index('peptide_id', drop=False)
    peptides.index.name = None
    peptides = peptides.loc[intensities.columns,]

    # Observation annotation (.obs)
    sample_annotation = pd.read_csv(sample_annotation_path, sep=sep_sample_ann)
    sample_annotation = sample_annotation.rename(columns={'filename': 'sample_id'}).set_index('sample_id', drop=False)
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
