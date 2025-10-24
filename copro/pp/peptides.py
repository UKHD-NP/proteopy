import numpy as np
import pandas as pd
import anndata as ad

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

def summarize_overlapping_peptides(
    adata: ad.AnnData,
    peptide_col: str = "peptide_id",
    group_col: str = "peptide_group",
    inplace: bool = True,
):
    """
    Aggregate intensities in adata.X across peptides sharing the same `group_col`.

    - Sums intensities in adata.X within each group (NaN-aware).
    - Uses the longest peptide_id in each group as the new var_name.
    - Keeps both the representative peptide_id as a column and index.
    - Retains the grouping key as 'peptide_group'.
    - Concatenates differing .var annotations using ';'.

    Parameters
    ----------
    adata : AnnData
        Input AnnData object with quantitative data and annotations.
    peptide_col : str
        Column in adata.var containing peptide identifiers (or will be created from var_names).
    group_col : str
        Column in adata.var specifying grouping (e.g. 'peptide_group' or 'proteoform_id').
    inplace : bool
        If True, modifies adata in place. Otherwise returns a new AnnData.

    Returns
    -------
    AnnData | None
        Aggregated AnnData if inplace=False, else modifies in place.
    """
    # --- safety checks
    if group_col not in adata.var.columns:
        raise KeyError(f"'{group_col}' not found in adata.var")
    if peptide_col not in adata.var.columns:
        # fallback: use var_names as peptide identifiers
        adata.var[peptide_col] = adata.var_names.astype(str)

    # --- matrix as DataFrame (obs × vars)
    vals = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)

    # --- group columns by group_col and sum (NaN-aware; future-proof syntax)
    group_keys = adata.var[group_col].astype(str)
    vals.columns = group_keys.values
    agg_vals = vals.T.groupby(level=0).sum(min_count=1).T  # obs × unique groups

    # --- build new var table (aggregate annotations)
    groups = adata.var.assign(_var_index=adata.var_names).groupby(group_col, sort=True)
    records, group_to_peptide = [], {}

    for gkey, df_g in groups:
        # pick longest peptide_id (tie-break lexicographically)
        longest_pep = sorted(df_g[peptide_col].astype(str), key=lambda s: (-len(s), s))[0]
        group_to_peptide[str(gkey)] = longest_pep

        rec = {
            group_col: str(gkey),
            peptide_col: longest_pep,
            "n_grouped": len(df_g),
        }

        # aggregate all other var columns
        for col in adata.var.columns:
            if col in (group_col, peptide_col):
                continue
            unique_vals = sorted(set(df_g[col].dropna().astype(str)))
            if len(unique_vals) == 0:
                rec[col] = np.nan
            elif len(unique_vals) == 1:
                rec[col] = unique_vals[0]
            else:
                rec[col] = ";".join(unique_vals)
        records.append(rec)

    var_new = pd.DataFrame.from_records(records).set_index(peptide_col)

    # --- rename aggregated matrix columns from group key → longest peptide
    agg_vals.columns = [group_to_peptide[k] for k in agg_vals.columns]
    var_new = var_new.loc[agg_vals.columns]  # ensure same order

    # --- ensure 'peptide_id' column always matches index
    var_new[peptide_col] = var_new.index

    # --- rebuild AnnData (shape consistency guaranteed)
    if inplace:
        adata._init_as_actual(
            X=agg_vals.values,
            obs=adata.obs.copy(),
            var=var_new,
            uns=adata.uns,
            obsm=adata.obsm,
            varm={},     # drop since vars changed
            layers={},   # reset layers for safety
            obsp=adata.obsp if hasattr(adata, "obsp") else None,
        )
        adata.var_names = var_new.index
        return None
    else:
        out = ad.AnnData(
            X=agg_vals.values,
            obs=adata.obs.copy(),
            var=var_new.copy(),
            uns=adata.uns.copy(),
            obsm=adata.obsm.copy(),
            obsp=adata.obsp.copy() if hasattr(adata, "obsp") else None,
        )
        out.var_names = var_new.index
        return out
