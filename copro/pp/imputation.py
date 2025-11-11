import numpy as np
from scipy import sparse

from copro.utils.array import is_log_transformed


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
    Adds `adata.layers["imputation_mask_<layer>"]` marking imputed positions
    (True) for the targeted layer/X matrix.
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

    # store boolean mask (dense bool array) with layer-specific name
    mask_layer = str(layer) if layer is not None else "X"
    mask_name = f"imputation_mask_{mask_layer}"
    adata.layers[mask_name] = miss_mask.astype(bool)

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
