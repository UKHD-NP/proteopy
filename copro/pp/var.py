import warnings
from functools import partial
import numpy as np
import pandas as pd
from scipy import sparse 
import warnings

def filter_axis(
    adata,
    axis,
    min_fraction = None,
    min_nr = None,
    group_by = None,
    zero_to_na = False,
    inplace = True,
    ):
    '''
    Filter obs or var by completeness.
    Args:
        axis (int, [0,1]): 0 = obs and 1 = var
        group_by (str|None): Column used to compute completeness/counts within
            groups; takes the maximum across groups.
    '''
    # Check input
    if not min_fraction and not min_nr:
        warnings.warn(
            'Neither min_fraction nor min_nr were provided so '
            'function does nothing.'
            )

    vals = adata.to_df().copy()
    if zero_to_na:
        vals = vals.replace(0, np.nan)

    axes_i = [1, 0]
    axis_i = axes_i[axis]
    axis_labels = [vals.index, vals.columns][axis]

    counts = None
    completeness = None

    if group_by is not None:
        metadata = adata.obs if axis == 1 else adata.var
        if group_by not in metadata.columns:
            raise KeyError(
                f'group_by "{group_by}" not present in '
                f'adata.{"obs" if axis == 1 else "var"}'
                )
        grouping = metadata[group_by]
        group_labels = grouping.dropna().unique().tolist()
        counts_by_group = []
        fractions_by_group = []

        for label in group_labels:
            if axis == 1:
                subset = vals.loc[grouping == label, :]
            else:
                subset = vals.loc[:, grouping == label]
            if subset.shape[axis_i] == 0:
                continue

            group_counts = subset.count(axis_i)
            counts_by_group.append(group_counts)

            if min_fraction:
                fractions_by_group.append( group_counts / subset.shape[axis_i])

        if counts_by_group:
            counts = pd.concat(counts_by_group, axis=1).max(axis=1)
        else:
            counts = pd.Series(0, index=axis_labels, dtype=float)

        if min_fraction:
            if fractions_by_group:
                completeness = pd.concat(fractions_by_group, axis=1).max(axis=1)
            else:
                completeness = pd.Series(0, index=axis_labels, dtype=float)
    else:
        counts = vals.count(axis_i)
        if min_fraction:
            completeness = counts / adata.shape[axis_i]

    if min_fraction:
        mask_fraction = completeness >= min_fraction
    else:
        mask_fraction = pd.Series(True, index=axis_labels)

    if min_nr:
        mask_nr = counts >= min_nr
    else:
        mask_nr = pd.Series(True, index=axis_labels)

    mask_filt = mask_fraction & mask_nr

    n_removed = (~mask_filt).sum()
    axes_names = ['obs', 'var']
    axis_name = axes_names[axis]
    print(f'{n_removed} {axis_name} removed')

    var_mask = [True] * adata.n_vars
    obs_mask = [True] * adata.n_obs

    if axis == 0:
        obs_mask = mask_filt.tolist()
    elif axis == 1:
        var_mask = mask_filt.tolist()

    if inplace:
        adata._inplace_subset_var(var_mask)
        adata._inplace_subset_obs(obs_mask)
    else:
        adata = adata[obs_mask:,var_mask].copy()
        return adata

filter_obs_completeness = partial(
    filter_axis,
    axis=0,
    min_nr=None,
    )

filter_var_completeness = partial(
    filter_axis,
    axis=1,
    min_nr=None,
    )

filter_obs_by_min_nr_var = partial(
    filter_axis,
    axis=0,
    min_fraction=None,
    )

filter_var_by_min_nr_obs = partial(
    filter_axis,
    axis=1,
    min_fraction=None,
    )

def is_log_transformed(
        adata, 
        layer=None, 
        neg_frac_thresh=5e-3, 
        p95_thresh=100.0
        ):
    """
    Heuristic detector for log-transformed matrices.

    Returns
    -------
    is_log : bool
        True if the matrix looks log-transformed.
    stats : dict
        {'frac_negative', 'p95', 'p5', 'dynamic_range_ratio', 'n_finite'}
    """
    Xsrc = adata.layers[layer] if layer is not None else adata.X
    X = Xsrc.toarray() if sparse.issparse(Xsrc) else np.asarray(Xsrc)
    X = X.astype(float, copy=False)

    finite = np.isfinite(X)
    vals = X[finite]
    if vals.size == 0:
        raise ValueError("No finite values found.")

    frac_negative = float(np.mean(vals < 0))
    p95 = float(np.nanpercentile(vals, 95))
    p5  = float(np.nanpercentile(vals, 5))
    # avoid divide-by-zero in very degenerate cases
    dr_ratio = float((p95 - p5) / max(abs(p5), 1e-12))

    # Simple decision
    is_log = (frac_negative >= neg_frac_thresh) or (p95 <= p95_thresh)

    stats = dict(
        frac_negative=frac_negative,
        p95=p95,
        p5=p5,
        dynamic_range_ratio=dr_ratio,
        n_finite=int(vals.size),
    )
    return bool(is_log), stats

def median_normalize(
    adata,
    layer=None,
    inplace: bool = True,
):
    """
    Row-wise median normalization via log2-wrap.

    - If input looks RAW: log2-transform (zeros -> NA), median-center per sample in log space, exp2 back.
    - If input looks LOG: median-center per sample in log space (no back-transform).
    - Zeros are ALWAYS treated as missing (NA). No pseudocount is used.

    Writes per-sample effect to:
      - RAW path:  adata.obs['median_scale']      (multiplicative factor on raw scale)
      - LOG path:  adata.obs['log_median_shift']  (additive shift in log2 space)
    """
    # pull matrix
    Xsrc = adata.layers[layer] if layer is not None else adata.X
    was_sparse = sparse.issparse(Xsrc)
    X = Xsrc.toarray() if was_sparse else np.asarray(Xsrc)
    X = X.astype(float, copy=True)

    # detect raw/log
    is_log, stats = is_log_transformed(adata, layer=layer)

    # masks
    nan_mask  = ~np.isfinite(X)
    zero_mask = (X == 0)

    # treat zeros as NA for normalization (always)
    M = X.copy()
    M[zero_mask] = np.nan
    M[nan_mask]  = np.nan

    # log2 space
    if is_log:
        Y = M  # already log; zeros now NaN
    else:
        # raw -> log2; negative values (if any) become NaN automatically
        Y = np.log2(M)

    # row-wise medians (ignore NaNs)
    row_meds = np.nanmedian(Y, axis=1)
    # rows with all-NaN -> no shift
    row_meds[~np.isfinite(row_meds)] = 0.0
    target = np.nanmedian(row_meds)
    if not np.isfinite(target):
        raise ValueError("Global median undefined (no finite values after masking).")

    shifts = target - row_meds
    Y_norm = Y + shifts[:, None]

    # back to output scale
    if is_log:
        Z = Y_norm
        obs_col, obs_vals = "log_median_shift", shifts
    else:
        Z = np.exp2(Y_norm)
        obs_col, obs_vals = "median_scale", np.exp2(shifts)

    # restore original NaNs; keep original zeros as NA (by design)
    Z[nan_mask]  = np.nan
    Z[zero_mask] = np.nan

    if not inplace:
        return Z, {"started_in": ("log" if is_log else "raw"), "detection": stats}

    # write back
    out = sparse.csr_matrix(Z) if was_sparse else Z
    if layer is None:
        adata.X = out
    else:
        adata.layers[layer] = out
    adata.obs[obs_col] = obs_vals

    # bookkeeping
    adata.uns.setdefault("normalization", {})
    adata.uns["normalization"].update({
        "method": "median",
        "started_in": ("log" if is_log else "raw"),
        "zeros_treated_as_na": True,
        "layer": layer,
        "detection": stats,
        "log_base": 2.0,
    })
    return adata

def impute_downshift(
    adata,
    layer=None,
    width: float = 0.3,
    downshift: float = 1.8,
    zero_to_nan: bool = True,
    inplace: bool = True,
    random_state: int | None = 42,
):
    """
    Left-censored imputation in log2 space with downshifted normal sampling.
    Adds `adata.layers["bool_imputed"]` marking imputed positions (True).
    """
    # --- pull matrix ---
    Xsrc = adata.layers[layer] if layer is not None else adata.X
    X = Xsrc.toarray() if sparse.issparse(Xsrc) else np.asarray(Xsrc)
    X = X.astype(float, copy=True)

    # --- decide log vs raw ---
    is_log, stats = is_log_transformed(adata, layer=layer)

    # Build log2 working matrix Y (NaN = missing) and capture missing mask
    if is_log:
        Y = X.copy()
        if zero_to_nan:
            Y[Y == 0] = np.nan
        Y[~np.isfinite(Y)] = np.nan
    else:
        W = X.copy()
        if zero_to_nan:
            W[W == 0] = np.nan
        W[~np.isfinite(W) | (W <= 0)] = np.nan
        Y = np.log2(W)

    miss_mask = ~np.isfinite(Y)              # True where we will impute (log-space definition)
    n_missing = int(miss_mask.sum())

    n_samples, n_feats = Y.shape
    rng = np.random.default_rng(random_state)

    # Global fallback stats
    y_finite = Y[np.isfinite(Y)]
    if y_finite.size < 3:
        raise ValueError("Not enough finite values to estimate imputation parameters.")
    g_mean = float(np.nanmean(y_finite))
    g_sd   = float(np.nanstd(y_finite))
    if not np.isfinite(g_sd) or g_sd <= 0:
        g_sd = 1.0

    # --- per-sample imputation ---
    Y_imp = Y.copy()
    for i in range(n_samples):
        row = Y[i, :]
        miss = miss_mask[i, :]
        if not miss.any():
            continue
        obs = row[np.isfinite(row)]
        if obs.size >= 3:
            r_mean = float(np.nanmean(obs))
            r_sd   = float(np.nanstd(obs))
            if not np.isfinite(r_sd) or r_sd <= 0:
                r_mean, r_sd = g_mean, g_sd
        else:
            r_mean, r_sd = g_mean, g_sd

        mu = r_mean - downshift * r_sd
        sd = max(width * r_sd, 1e-6)
        Y_imp[i, miss] = rng.normal(loc=mu, scale=sd, size=int(miss.sum()))

    # --- back to output scale ---
    Z = Y_imp if is_log else np.exp2(Y_imp)

    if not inplace:
        return Z, miss_mask

    # write back
    if layer is None:
        adata.X = Z
    else:
        adata.layers[layer] = Z

    # store boolean mask (dense bool array)
    adata.layers["bool_imputed"] = miss_mask.astype(bool)

    # bookkeeping
    adata.uns.setdefault("imputation", {})
    adata.uns["imputation"].update(dict(
        method="downshift_normal",
        layer=layer,
        worked_in_log2=True,
        started_in=("log" if is_log else "raw"),
        width=float(width),
        downshift=float(downshift),
        zero_to_nan=bool(zero_to_nan),
        random_state=(None if random_state is None else int(random_state)),
        detection=stats,
        n_imputed=int(n_missing),
        pct_imputed=float(n_missing / (Y.size) * 100.0),
    ))
    return adata

def calculate_groupwise_cv(
    adata,
    groupby: str,
    layer: str | None = None,
    zero_to_na: bool = True,
    min_n: int = 1,
    inplace: bool = True,
):
    """
    Compute the coefficient of variation (CV = std / mean)
    for each variable (var) within each group defined by adata.obs[groupby].

    Adds new columns 'cv_<group>' to adata.var.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing quantitative data.
    groupby : str
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

    groups = adata.obs[groupby].astype(str)
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
