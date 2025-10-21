import pandas as pd

def proteoforms_df(
    adata,
    proteins=None,
    only_proteins=False,
    score_threshold=None,
    pval_threshold=None,
    pval_adj_threshold=None,
    ):

    if proteins:
        if isinstance(proteins, str):
            proteins = [proteins]
        elif isinstance(proteins, list):
            pass
        else:
            raise ValueError('arg: proteins, must me str or list of strings')

    cols = [
        'protein_id',
        'peptide_id',
        'cluster_id',
        'proteoform_score',
        'proteoform_score_pval',
        'proteoform_score_pval_adj',
        'is_proteoform']

    proteoforms = adata.var[adata.var['protein_id'].isin(proteins)][cols].copy()

    if proteins:
        proteoforms = proteoforms[proteoforms['protein_id'].isin(proteins)]

    mask_notna = proteoforms['proteoform_score_pval'].notna()
    proteoforms = proteoforms.loc[mask_notna,]
    proteoforms = proteoforms.sort_values(['proteoform_score_pval_adj', 'proteoform_score', 'cluster_id'])
    
    # Filter on thresholds
    if score_threshold:
        proteoforms = proteoforms[proteoforms['proteoform_score'] > score_threshold]

    if pval_threshold:
        proteoforms = proteoforms[proteoforms['proteoform_score_pval'] < pval_threshold]

    if pval_adj_threshold:
        proteoforms = proteoforms[proteoforms['proteoform_score_pval_adj'] < pval_adj_threshold]

    if only_proteins:
        proteoform_proteins = proteoforms.drop(columns=['peptide_id', 'cluster_id']).reset_index(drop=True).drop_duplicates(ignore_index=True)
        return proteoform_proteins
    else:
        return proteoforms
