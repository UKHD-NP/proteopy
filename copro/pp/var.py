import numpy as np
from scipy import sparse 
import warnings

def filter_var_completeness(
    adata,
    min_fraction,
    zero_to_na = False,
    inplace = True,
    ):
    vals = adata.to_df().copy()

    if zero_to_na:
        vals = vals.replace(0, np.nan)

    completeness = vals.count() / adata.n_obs
    var_subs = completeness[completeness > min_fraction].index

    n_removed = sum(~(completeness > min_fraction))
    print(f'{n_removed} var removed')

    if inplace:
        adata._inplace_subset_var(var_subs)
    else:
        adata = adata[:,var_subs].copy()
        return adata

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
