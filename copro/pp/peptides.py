def filter_genes_by_peptide_count(
    adata,
    min = None,
    max = None,
    gene_col = 'protein_id',
    ):
    if not min and not max:
        raise ValueError('Must pass either min or max arguments.')

    genes = adata.var[gene_col]
    counts = genes.value_counts()

    if min:
        gene_ids_filt = counts[counts >= min].index
    elif max:
        gene_ids_filt = counts[counts <= max].index
    elif min and max:
        gene_ids_filt = counts[(counts >= min) & (counts <= max)]
    
    var_filt = adata.var[genes.isin(gene_ids_filt)].index
    new_adata = adata[:,var_filt]

    n_genes_removed = len(set(genes.unique()) - set(gene_ids_filt.unique()))
    n_peptides_removed = sum(~genes.isin(gene_ids_filt))
    print(f'Removed {str(n_genes_removed)} genes and {str(n_peptides_removed)} peptides.')

    return new_adata
