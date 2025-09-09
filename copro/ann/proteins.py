import warnings
import pandas as pd

def protein_ids(
    adata,
    annotation_file,
    var_protein_id_col = None,
    file_protein_id_col = 'protein_id',
    sep = '\t',
    ):
    '''Annotate AnnData protein_ids with external tabular file.

    Args:
        adata (AnnData):
        annotation_file (str): tabular format with first row being the column names.
        obs_protein_id_col (str): Default is 'protein_id', index for AnnData.var index can also be used.
        file_protein_id_col (str):

    Returns:
        AnnData with extended .var
    '''
    adata = adata.copy()
    var = adata.var.copy()
    var = var.reset_index()

    ann = pd.read_csv(annotation_file, sep=sep)

    ann[file_protein_id_col] = ann[file_protein_id_col].astype(str)

    if len(ann[file_protein_id_col].unique()) != len(ann):
        ann = ann.drop_duplicates(subset=file_protein_id_col, keep='first')
        warnings.warn(f'Non-unique protein_id row were collapsed to the first occurance.')

    new_var = pd.merge(
        var,
        ann,
        left_on=var_protein_id_col,
        right_on=file_protein_id_col,
        how='left',
        suffixes=('', '_annotation_file'),
        )

    new_var = new_var.drop(columns=[file_protein_id_col])
    new_var = new_var.set_index('index')
    new_var.index.name = None
    new_var[var_protein_id_col] = new_var[var_protein_id_col].astype('category')
    adata.var = new_var

    return adata
