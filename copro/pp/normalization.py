import numpy as np
import pandas as pd
from scipy import sparse 

def normalize_bulk(
    adata,
    method,
    log_space,
    fill_na = None,
    zeros_to_na = False,
    batch_id = None,
    inplace = True,
    force = False,
    ):
    '''
    Normalize intensity values stored in adata.X.

    Args
    ----
    method : {'median_max','median_median'}
    log_space : bool
    fill_na : float, optional
        If provided, temporarily replace NaS with this value *for the median computation only*.
        Original NAs are restored after normalization.
    zeros_to_na : bool, default False
        Temporarily treat zeros as NaN *for the median computation only*.
        Original zeros are restored after normalization.
    batch_id : str, optional
        Column name in adata.obs containing batch IDs.
        If provided, normalization is computed within each batch.
    inplace : bool, default True
    force : bool, default False
        Avoid warning when log_space=False but X contains <= 0 values

    Notes:
        Median normalization:
            - log_space=True: intensity + ref - sample_median
            - log_space=True: intensity * ref / sample_median
            - 'median_max': reference = max of sample medians (within batch if per_batch)
            - 'median_median': reference = median of sample medians (within batch if per_batch)

    Returns
    -------
    If inplace is False, returns a new AnnData with normalized X.
    Also returns a DataFrame of per-sample factors when inplace is False.
    When inplace is True, factors are stored in adata.uns['normalize_factors'] and the function returns None.
    '''
    per_batch = batch_id

    if fill_na and zeros_to_na:
        raise ValueError('Cannot set both zeros_to_na and fill_na to True.')

    def _is_log_like(x):
        return np.nanmin(x) <= 0

    X = adata.X

    n_samples, n_features = X.shape

    X_new = X.copy()

    na_mask = np.isnan(X_new)
    zero_mask = (X_new == 0)

    if zeros_to_na:
        X_new[zero_mask] = np.nan
    else:
        if fill_na is not None:
            X_new = np.where(np.isnan(X_new), fill_na, X_new)

        is_log_like = _is_log_like(X)
        if (not log_space) and is_log_like and not force:
            raise ValueError(
                "You passed log_space=False but X contains values <= 0, which often indicates log space. "
                "If you are sure your data are linear, set force=True."
            )


    def _normalize_sample_indices(
        sample_idx,
        method,
        log_space,
        ):
        '''
        Returns (norm_values, factors, pre_medians) for the given sample rows.

        - norm_values: normalized X for sample_idx
        - factors: per-sample shifts (log) or scales (linear) length = len(sample_idx)
        '''
        sub = X_new[sample_idx, :]

        with np.errstate(invalid='ignore'):
            sample_medians = np.nanmedian(sub, axis=1)

        if method == 'median_median':
            ref = float(np.nanmedian(sample_medians))
        elif method == 'median_max':
            ref = float(np.nanmax(sample_medians))
        else:
            raise ValueError("method must be one of {'median_median','median_max'}")

        if log_space:
            factors = (ref - sample_medians)[:, None]
            sub_norm = X[sample_idx, :] + factors
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                factors = (ref / sample_medians)[:, None]
            sub_norm = X[sample_idx, :] * factors

        return sub_norm, np.squeeze(factors)

    all_norm = np.empty_like(X)
    all_factors = np.empty((n_samples,), dtype=float)

    if per_batch is None:
        idx = np.arange(n_samples)
        sub_norm, sub_fac = _normalize_sample_indices(idx, method.lower(), log_space)
        all_norm[idx, :] = sub_norm
        all_factors[idx] = sub_fac if log_space else np.squeeze(sub_fac)
    else:
        if per_batch not in adata.obs.columns:
            raise KeyError(f"per_batch='{per_batch}' not found in adata.obs columns.")
        batches = adata.obs[per_batch].astype('category')
        for b in batches.cat.categories:
            idx = np.where(batches.values == b)[0]
            if idx.size == 0:
                continue
            sub_norm, sub_fac, sub_med = _normalize_sample_indices(idx, method.lower(), log_space)
            all_norm[idx, :] = sub_norm
            all_factors[idx] = sub_fac if log_space else np.squeeze(sub_fac)

    # --------------------------
    # 4) Restore original NaNs and zeros (only if we changed them)
    # --------------------------
    # We always respect user's original missingness and zeros in the *output* matrix.
    # If a value was originally NaN, keep it NaN after normalization.
    all_norm[na_mask] = np.nan
    # If a value was originally zero and user asked zeros_to_na for computation only,
    # restore zeros in the output.
    if zeros_to_na:
        all_norm[zero_mask] = 0.0

    # --------------------------
    # 5) Build factors frame
    # --------------------------
    if log_space:
        factor_name = "shift_log"
    else:
        factor_name = "scale_linear"

    factors_df = pd.DataFrame({
        "sample_index": np.arange(n_samples),
        factor_name: all_factors,
    })

    if per_batch is not None:
        factors_df[per_batch] = adata.obs[per_batch].values

    # Warn if any sample median was 0/NaN leading to NaN factor
    if np.isnan(all_factors).any():
        bad = np.where(np.isnan(all_factors))[0]
        print(f"Warning: {bad.size} sample(s) had undefined median; factors are NaN for indices {bad.tolist()}.")

    if inplace:
        adata.X = all_norm
        adata.uns["normalization_factors"] = factors_df
    else:
        adata = adata.copy()
        adata.X = all_norm
        return adata, factors_df


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
