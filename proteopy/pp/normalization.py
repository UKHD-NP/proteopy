import numpy as np
import pandas as pd
from scipy import sparse

from proteopy.utils.anndata import check_proteodata
from proteopy.utils.array import is_log_transformed


def normalize_median(
    adata,
    *,
    log_space: bool,
    target: str = "median",
    fill_na: float | None = None,
    zeros_to_na: bool = False,
    batch_id: str | None = None,
    inplace: bool = True,
    force: bool = False,
):
    """
    Median normalization of intensities.

    NAs are ignored when computing sample medians.

    Parameters
    ----------
    adata : AnnData
        Input AnnData.
    log_space : bool
        Whether the input intensities are log-transformed. Mismatches with
        automatic detection raise unless ``force=True``.
    target : {'max', 'median'}
        How to compute the normalization target from sample medians.
        ``'max'`` uses the maximum sample median, ``'median'`` uses the
        median of sample medians. Defaults to ``'median'``.
    fill_na : float, optional
        Temporarily replace non-finite entries with this value for the median
        computation only; original values are restored afterward.
    zeros_to_na : bool, default False
        Treat zeros as missing for the median computation only; original zeros
        are restored afterward.
    batch_id : str, optional
        Column in ``adata.obs`` to perform normalization within batches.
    inplace : bool, default True
        Modify ``adata`` in place. If False, return a copy.
    force : bool, default False
        Proceed even if ``log_space`` disagrees with automatic log detection.

    Returns
    -------
    AnnData or None
        Normalized AnnData when ``inplace`` is False; otherwise None.
    pandas.DataFrame, optional
        Per-sample factors when ``inplace`` is False.

    Notes
    -----
    Median normalization:
        ``log_space=True``
            ``X + target - sample_median``
        ``log_space=False``
            ``X * target / sample_median``
        ``'max'``
            target = max of sample medians (within batch if per_batch)
        ``'median'``
            target = median of sample medians (within batch if per_batch)

    Examples
    --------
    >>> import proteopy as pr
    >>> adata = pr.datasets.karayel_2020()

    Normalize using the median of sample medians (default):

    >>> pr.pp.normalize_median(adata, log_space=False)

    Normalize using the maximum of sample medians:

    >>> pr.pp.normalize_median(adata, target='max', log_space=False)
    """
    check_proteodata(adata)
    per_batch = batch_id

    target = target.lower()
    allowed_targets = {"max", "median"}
    if target not in allowed_targets:
        raise ValueError(f"target must be one of {allowed_targets!r}")

    if fill_na is not None and zeros_to_na:
        raise ValueError('Cannot set both zeros_to_na and fill_na to True.')

    Xsrc = adata.X
    was_sparse = sparse.issparse(Xsrc)
    X = Xsrc.toarray() if was_sparse else np.asarray(Xsrc)
    X = X.astype(float, copy=True)

    is_log, _ = is_log_transformed(adata)
    mismatch = (log_space != is_log)
    if mismatch and not force:
        if log_space:
            raise ValueError(
                "You passed log_space=True but the data do not look log-transformed. "
                "Set force=True to override the automatic detection."
            )
        else:
            raise ValueError(
                "You passed log_space=False but the data look log-transformed. "
                "Set force=True to override the automatic detection."
            )

    n_samples, _ = X.shape

    X_new = X.copy()  # X with replaces values as per user parameters

    # Track original missingness/zeros to restore later
    na_mask = ~np.isfinite(X)
    zero_mask = (X == 0)

    if zeros_to_na:
        X_new[zero_mask] = np.nan
    else:
        if fill_na is not None:
            X_new = np.where(~np.isfinite(X_new), fill_na, X_new)


    def _normalize_samples(
        X_work,
        target,
        log_space,
        ):
        """Normalize a subset of samples and return normalized values and factors."""
        with np.errstate(invalid='ignore'):
            sample_medians = np.nanmedian(X_work, axis=1)

        if target == 'median':
            target_val = float(np.nanmedian(sample_medians))
        elif target == 'max':
            target_val = float(np.nanmax(sample_medians))
        else:
            raise ValueError("target must be one of {'median', 'max'}")

        if log_space:
            factors = (target_val - sample_medians)[:, None]
            sub_norm = X_work + factors
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                factors = (target_val / sample_medians)[:, None]
            sub_norm = X_work * factors

        return sub_norm, np.squeeze(factors)

    all_norm = np.empty_like(X, dtype=float)
    all_factors = np.empty((n_samples,), dtype=float)

    if per_batch is None:
        idx = np.arange(n_samples)
        X_work = X_new[idx, :]
        sub_norm, sub_fac = _normalize_samples(X_work, target, log_space)
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
            X_work = X_new[idx, :]
            sub_norm, sub_fac = _normalize_samples(X_work, target, log_space)
            all_norm[idx, :] = sub_norm
            all_factors[idx] = sub_fac if log_space else np.squeeze(sub_fac)

    # Restore original NaNs and zeros in the output
    all_norm[na_mask] = np.nan
    if zeros_to_na:
        all_norm[zero_mask] = 0.0

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

    # Surface problematic medians via warnings
    if np.isnan(all_factors).any():
        bad = np.where(np.isnan(all_factors))[0]
        print(f"Warning: {bad.size} sample(s) had undefined median; factors are NaN for indices {bad.tolist()}.")
    if np.isinf(all_factors).any():
        bad = np.where(np.isinf(all_factors))[0]
        print(f"Warning: {bad.size} sample(s) had zero median; factors are inf for indices {bad.tolist()}.")

    out = sparse.csr_matrix(all_norm) if was_sparse else all_norm

    if inplace:
        adata.X = out
        adata.uns["normalization_factors"] = factors_df
        check_proteodata(adata)
    else:
        adata_out = adata.copy()
        adata_out.X = out
        adata_out.uns["normalization_factors"] = factors_df
        check_proteodata(adata_out)
        return adata_out, factors_df
