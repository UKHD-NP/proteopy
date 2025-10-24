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
    else:
        raise ValueError('Pass at least one argument: min | max')
    
    var_filt = adata.var[genes.isin(gene_ids_filt)].index
    new_adata = adata[:,var_filt]

    n_genes_removed = len(set(genes.unique()) - set(gene_ids_filt.unique()))
    n_peptides_removed = sum(~genes.isin(gene_ids_filt))
    print(f'Removed {str(n_genes_removed)} genes and {str(n_peptides_removed)} peptides.')

    return new_adata

import pandas as pd
from collections import defaultdict
from typing import List, Dict

def _group_peptides(peptides: List[str]) -> Dict[str, List[str]]:
    """
    Group peptides so that any peptide fully contained in another belongs to the same group.
    Returns {representative_longest: [members...]} with members sorted by (-len, lexicographically).
    """
    peptides = [p for p in pd.Series(peptides).dropna().astype(str).tolist()]
    peptides = list(dict.fromkeys(peptides))  # remove duplicates, preserve order

    # --- union–find setup
    parent = {p: p for p in peptides}
    rank = {p: 0 for p in peptides}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    # --- check containment (only compare longer → shorter)
    peps_by_len = sorted(peptides, key=len, reverse=True)
    for i, longp in enumerate(peps_by_len):
        for shortp in peps_by_len[i+1:]:
            if shortp in longp:
                union(longp, shortp)

    # --- collect groups
    buckets = defaultdict(list)
    for p in peptides:
        buckets[find(p)].append(p)

    # --- representative = longest; sort members deterministically
    groups = {}
    for _, members in buckets.items():
        rep = max(members, key=len)
        groups[rep] = sorted(members, key=lambda s: (-len(s), s))
    return groups


def extract_peptide_groups(
    adata,
    peptide_col: str = "peptide_id",
    inplace: bool = True,
):
    """
    Create a new column `adata.var['peptide_group']` with all overlapping (substring) peptide_ids
    joined by ';' for each row in `adata.var`.

    Parameters
    ----------
    adata : AnnData
        Must have `adata.var[peptide_col]` with peptide sequences (already normalized).
    peptide_col : str
        Column in `adata.var` containing peptide sequences.
    inplace : bool
        If True, modifies `adata` in place. If False, returns a modified copy.

    Returns
    -------
    If inplace=True: None
    If inplace=False: AnnData (modified copy)
    """
    if peptide_col not in adata.var.columns:
        raise KeyError(f"'{peptide_col}' not found in adata.var")

    peptides = adata.var[peptide_col].astype(str).tolist()
    groups = _group_peptides(peptides)

    # map each peptide to its group string
    pep_to_group = {
        p: ";".join(groups.get(rep, [rep])) for rep, members in groups.items() for p in members
    }
    group_col = adata.var[peptide_col].map(pep_to_group).fillna(adata.var[peptide_col])

    if inplace:
        adata.var["peptide_group"] = group_col.values
        return None
    else:
        adata_copy = adata.copy()
        adata_copy.var["peptide_group"] = group_col.values
        return adata_copy