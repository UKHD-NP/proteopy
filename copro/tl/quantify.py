from functools import partial
import numpy as np
import pandas as pd
import anndata as ad
from typing import Callable, Optional, Union

def quantify_by_var(
    adata: ad.AnnData,
    group_by: str = None,
    layer=None,
    func: Union[str, Callable] = "sum",
    inplace: bool = True,
) -> Optional[ad.AnnData]:
    """
    Aggregate intensities in adata.X (or selected layer) by .var[group_col],
    aggregate annotations in adata.var by concatenating unique values with ';', and set `group_col` as the new index
    (var_names).

    Parameters
    ----------
    adata : AnnData
        Input AnnData with .X (obs x vars) and .var annotations.
    group_by : str
        Column in adata.var to group by (e.g. 'protein_id').
    layer : str | None
        Optional key in `adata.layers`; when set, quantification uses that layer
        instead of the default `adata.X`.
    func : {'sum', 'median', 'max'} | Callable
        Aggregation to apply across grouped variables.
    inplace : bool
        If True, modify `adata` in place; else return a new AnnData.

    Returns
    -------
    AnnData | None
        Aggregated AnnData if inplace=False; otherwise None.
    """
    if group_by is None or group_by not in adata.var.columns:
        raise KeyError(f"'{group_by}' not found in adata.var")

    # --- Matrix as DataFrame (obs × vars)
    vals = pd.DataFrame(
        adata.layers[layer] if layer is not None else adata.X,
        index=adata.obs_names,
        columns=adata.var_names,
    )

    # --- Group columns by group_col and apply requested aggregation
    group_keys = adata.var[group_by].astype(str)
    grouped = vals.groupby(group_keys, axis=1, observed=True, sort=True)
    if isinstance(func, str):
        if func == "sum":
            agg_vals = grouped.sum(min_count=1)
        elif func == "median":
            agg_vals = grouped.median()
        elif func == "max":
            agg_vals = grouped.max()
        else:
            raise ValueError(
                f"Unsupported func '{func}'. Choose from 'sum', 'median', or 'max'."
            )
    elif callable(func):
        agg_vals = grouped.aggregate(func)
        if isinstance(agg_vals, pd.Series):
            agg_vals = agg_vals.to_frame().T
        if not isinstance(agg_vals, pd.DataFrame):
            raise TypeError(
                "Callable `func` must return a pandas DataFrame when aggregated over groups."
            )
    else:
        raise TypeError("`func` must be either a string identifier or a callable.")

    # --- Build new var table (aggregate annotations per group)
    records = []
    for gkey, df_g in adata.var.groupby(group_by, sort=True, observed=True):
        rec = {group_by: str(gkey)}
        for col in adata.var.columns:
            if col == group_by:
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

    var_new = pd.DataFrame.from_records(records).set_index(group_by)
    # align var rows to aggregated matrix columns
    var_new = var_new.loc[agg_vals.columns]
    var_new[group_by] = var_new.index
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
            varp={},
            layers={},   # reset layers unless you implement layer aggregation
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

quantify_proteins = partial(
    quantify_by_var,
    group_by='protein_id',
    )

quantify_proteoforms = partial(
    quantify_by_var,
    group_by='proteoform_id',
    )
