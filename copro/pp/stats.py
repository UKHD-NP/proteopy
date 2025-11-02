import numpy as np
import pandas as pd

from copro.utils.array import is_log_transformed


def calculate_groupwise_cv(
    adata,
    group_by: str,
    layer: str | None = None,
    zero_to_na: bool = True,
    min_n: int = 1,
    inplace: bool = True,
):
    """
    Compute the coefficient of variation (CV = std / mean)
    for each variable (var) within each group defined by adata.obs[group_by].

    Adds new columns 'cv_<group>' to adata.var.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing quantitative data.
    group_by : str
        Column in adata.obs defining groups.
    layer : str or None, optional
        Layer to use for data. If None, uses adata.X.
    zero_to_na : bool, optional
        Treat zeros as missing values (NaN). Default = True.
    min_n : int, optional
        Minimum number of non-NaN samples required to compute a CV. Default = 1.
    inplace : bool, optional
        If True (default), modify adata.var in place.
        If False, return a copy of adata with added columns.

    Returns
    -------
    adata : AnnData
        Modified AnnData object (only if inplace=False).
        Nothing is returned if inplace=True.
    """
    # --- test if the data is raw
    is_log, stat = is_log_transformed(adata)
    if is_log:
        print(stat)
        raise ValueError("The data appears to be log-transformed. CVs should be computed on raw data.")

    # --- extract data matrix (obs × var)
    vals = adata.to_df(layer=layer).copy()
    if zero_to_na:
        vals = vals.replace(0, np.nan)

    groups = adata.obs[group_by].astype(str)
    group_names = groups.unique()

    # --- compute CVs per group
    cv_dict = {}
    for g in group_names:
        idx = groups == g
        sub = vals.loc[idx]
        mean_ = sub.mean(axis=0, skipna=True)
        std_ = sub.std(axis=0, ddof=1, skipna=True)
        n_ = sub.notna().sum(axis=0)
        cv = std_ / mean_
        cv[n_ < min_n] = np.nan
        cv_dict[g] = cv

    cv_df = pd.DataFrame(cv_dict)
    cv_df.index.name = "var_name"

    # --- handle inplace vs return copy
    target = adata if inplace else adata.copy()

    for g in group_names:
        col_name = f"cv_{g}"
        target.var[col_name] = cv_df[g].values

    if not inplace:
        return target
