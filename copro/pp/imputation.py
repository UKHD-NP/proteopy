import numpy as np
from scipy import sparse

from copro.utils.anndata import check_proteodata
from copro.utils.array import is_log_transformed


def impute_downshift(
    adata,
    downshift: float = 1.8,
    width: float = 0.3,
    zero_to_nan: bool = True,
    inplace: bool = True,
    force: bool = False,
    random_state: int | None = 42,
):
    """
    Left-censored imputation in log space with downshifted normal sampling.

    Parameters
    ----------
    adata : AnnData
        Input proteomics AnnData.
    width : float, optional
        Standard deviation scaling factor for the normal sampler.
    downshift : float, optional
        Number of standard deviations to shift the mean downward.
    zero_to_nan : bool, optional
        Treat zeros as missing values before imputation.
    inplace : bool, optional
        Modify ``adata`` in place. If False, return a copied AnnData.
    force : bool, optional
        If False (default), raise a ValueError when data are detected as
        non-log. Set True to impute even when data appear non-log.
    random_state : int or None, optional
        Seed for the random number generator.

    Returns
    -------
    AnnData
        Modified AnnData (or a copy if ``inplace=False``).
    """
    check_proteodata(adata)

    if width <= 0:
        raise ValueError("`width` must be positive.")

    Xsrc = adata.X
    was_sparse = sparse.issparse(Xsrc)
    X = Xsrc.toarray() if was_sparse else np.asarray(Xsrc)
    X = X.astype(float, copy=True)

    is_log, stats = is_log_transformed(adata)
    if not is_log and not force:
        raise ValueError(
            "Imputation expects log-transformed intensities. "
            "Set force=True to proceed nevertheless."
        )

    # Build working matrix Y (NaN = missing) and capture missing mask
    Y = X.copy()
    if zero_to_nan:
        Y[Y == 0] = np.nan
    Y[~np.isfinite(Y)] = np.nan

    miss_mask = ~np.isfinite(Y)
    n_missing = int(miss_mask.sum())

    n_samples, _ = Y.shape
    rng = np.random.default_rng(random_state)

    # Global fallback stats
    y_finite = Y[np.isfinite(Y)]
    if y_finite.size < 3:
        raise ValueError("Not enough finite values to estimate imputation parameters.")
    g_mean = float(np.mean(y_finite))
    g_sd = float(np.std(y_finite))
    if not np.isfinite(g_sd) or g_sd <= 0:
        g_sd = 1.0

    # Per-sample imputation
    Y_imp = Y.copy()
    for i in range(n_samples):
        row = Y[i, :]
        miss = miss_mask[i, :]
        if not miss.any():
            continue
        obs = row[np.isfinite(row)]
        if obs.size >= 3:
            r_mean = float(np.mean(obs))
            r_sd = float(np.std(obs))
            if not np.isfinite(r_sd) or r_sd <= 0:
                r_mean, r_sd = g_mean, g_sd
        else:
            r_mean, r_sd = g_mean, g_sd

        mu = r_mean - downshift * r_sd
        sd = max(width * r_sd, 1e-6)
        Y_imp[i, miss] = rng.normal(loc=mu, scale=sd, size=int(miss.sum()))

    Z_out = sparse.csr_matrix(Y_imp) if was_sparse else Y_imp

    def _book_keeping(target):
        target.layers["imputation_mask_X"] = miss_mask.astype(bool)
        target.uns.setdefault("imputation", {})
        target.uns["imputation"].update(dict(
            method="downshift_normal",
            width=float(width),
            downshift=float(downshift),
            random_state=(None if random_state is None else int(random_state)),
            n_imputed=int(n_missing),
            pct_imputed=float(n_missing / (miss_mask.size) * 100.0),
        ))

    if not inplace:
        adata_out = adata.copy()
        adata_out.X = Z_out
        _book_keeping(adata_out)
        check_proteodata(adata_out)
        return adata_out
    else:
        adata.X = Z_out
        _book_keeping(adata)
        check_proteodata(adata)
        return None
