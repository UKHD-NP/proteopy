import numpy as np
import pandas as pd
import anndata as ad
from typing import Optional

def quantify_by_var(
    adata: ad.AnnData,
    group_col: str = "proteoform_id",
    inplace: bool = True,
) -> Optional[ad.AnnData]:
    """
    Sum intensities in adata.X for all peptide_ids sharing the same `group_col`,
    aggregate annotations in adata.var by concatenating unique values with ';',
    and set `group_col` as the new index (var_names).

    Parameters
    ----------
    adata : AnnData
        Input AnnData with .X (obs x vars) and .var annotations.
    group_col : str
        Column in adata.var to group by (e.g. 'proteoform_id').
    inplace : bool
        If True, modify `adata` in place; else return a new AnnData.

    Returns
    -------
    AnnData | None
        Aggregated AnnData if inplace=False; otherwise None.
    """
    if group_col not in adata.var.columns:
        raise KeyError(f"'{group_col}' not found in adata.var")

    # --- Matrix as DataFrame (obs × vars)
    vals = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)

    # --- Group columns by group_col and sum (NaN-aware; future-proof syntax)
    group_keys = adata.var[group_col].astype(str)
    # transpose, group by the grouping Series, sum, then transpose back
    agg_vals = vals.T.groupby(group_keys).sum(min_count=1).T  # (obs × unique groups)

    # --- Build new var table (aggregate annotations per group)
    records = []
    for gkey, df_g in adata.var.groupby(group_col, sort=True):
        rec = {group_col: str(gkey)}
        for col in adata.var.columns:
            if col == group_col:
                continue
            non_na = df_g[col].dropna().astype(str)
            uniq = sorted(set(non_na))
            if len(uniq) == 0:
                rec[col] = np.nan
            elif len(uniq) == 1:
                rec[col] = uniq[0]
            else:
                rec[col] = ";".join(uniq)
        records.append(rec)

    var_new = pd.DataFrame.from_records(records).set_index(group_col)
    # align var rows to aggregated matrix columns
    var_new = var_new.loc[agg_vals.columns]
    var_new[group_col] = var_new.index
    var_new.index.name = None  

    # --- Rebuild AnnData so X and var change together
    if inplace:
        adata._init_as_actual(
            X=agg_vals.values,
            obs=adata.obs.copy(),
            var=var_new,
            uns=adata.uns,
            obsm=adata.obsm,
            varm={},     # vars changed
            layers={},   # reset layers unless you implement layer aggregation
            obsp=adata.obsp if hasattr(adata, "obsp") else None,
        )
        adata.var_names = var_new.index  # now group_col (e.g. proteoform_id)
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
