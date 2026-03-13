import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from anndata import AnnData

from proteopy.utils.anndata import check_proteodata
from proteopy.utils.array import is_log_transformed
from proteopy.utils.string import sanitize_string


def _compute_cv_stats(X, zero_to_na=True, fill_na=None):
    """
    Compute mean, std, and count across observations for CV calculation.

    Parameters
    ----------
    X : ndarray or sparse matrix
        Data matrix (obs x var).
    zero_to_na : bool
        If True, treat zeros as missing values.
    fill_na : float | int | None
        If not None, replace NaN values with this constant before
        computing statistics.

    Returns
    -------
    mean_ : ndarray
        Mean values for each variable.
    std_ : ndarray
        Standard deviation (ddof=1) for each variable.
    n_ : ndarray
        Count of non-missing values for each variable.
    """
    if sparse.issparse(X):
        X_arr = X.toarray().astype(float)
    else:
        X_arr = np.asarray(X, dtype=float)

    X_arr = X_arr.copy()
    if zero_to_na:
        X_arr[X_arr == 0] = np.nan
    if fill_na is not None:
        X_arr[np.isnan(X_arr)] = fill_na

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        mean_ = np.nanmean(X_arr, axis=0)
        std_ = np.nanstd(X_arr, axis=0, ddof=1)
    n_ = np.sum(~np.isnan(X_arr), axis=0)
    return mean_, std_, n_


def _safe_cv_divide(std_, mean_, verbose=False, label=None):
    """
    Compute CV = std / mean, replacing inf/nan from zero-mean
    division with np.nan.

    Parameters
    ----------
    std_ : ndarray
        Standard deviations per variable.
    mean_ : ndarray
        Means per variable.
    verbose : bool
        If True, warn about zero-mean occurrences.
    label : str or None
        Label for the warning message (e.g. group name).

    Returns
    -------
    cv : ndarray
    """
    zero_mean = mean_ == 0
    if verbose and np.any(zero_mean):
        n_zero_both = int(np.sum(zero_mean & (std_ == 0)))
        n_zero_mean_nonzero_std = int(
            np.sum(zero_mean & (std_ != 0))
        )
        context = (
            f" in group '{label}'" if label is not None else ""
        )
        warnings.warn(
            f"CV undefined for {int(np.sum(zero_mean))} "
            f"variable(s){context} where the mean is zero: "
            f"{n_zero_both} with std=0 (constant zero), "
            f"{n_zero_mean_nonzero_std} with std!=0."
        )

    with np.errstate(divide='ignore', invalid='ignore'):
        cv = std_ / mean_
    cv[zero_mean] = np.nan
    return cv


def calculate_cv(
    adata: AnnData,
    group_by: str | None = None,
    layer: str | None = None,
    zero_to_na: bool = True,
    fill_na: float | int | None = None,
    min_samples: int = 2,
    force: bool = False,
    key_added: str | None = None,
    inplace: bool = True,
    verbose: bool = False,
) -> AnnData | None:
    """
    Compute the coefficient of variation (CV = std / mean) for each variable.

    Performed within ``group_by`` groups optionally.
    CV is calculated ignoring NaNs. Variables where the group mean is
    zero are skipped (CV set to NaN), since the CV is undefined when the
    mean is zero.

    Parameters
    ----------
    adata : AnnData
        :class:`~anndata.AnnData` object containing data.
    group_by : str or None, optional
        Column in ``adata.obs`` defining groups. If None, computes CV
        across all samples without grouping.
    layer : str or None, optional
        Layer to use for data. If None, uses ``adata.X``.
    zero_to_na : bool, optional
        Treat zeros as missing values (NaN).
    fill_na : float | int | None, optional
        Replace NaN values in the data matrix with this constant before
        computing CV statistics. Cannot be used together with
        ``zero_to_na``.
    min_samples : int, optional
        Minimum number of non-NaN samples (obs) required to compute a CV.
    force : bool, optional
        If True, bypass the log-transform check. Use when you are certain
        the data is on the appropriate scale for CV calculation.
    key_added : str or None, optional
        Key under which to store results. When ``group_by`` is provided,
        defaults to ``'cv_by_<group_by>_<layer>'`` and stores in
        ``adata.varm``. When ``group_by`` is None, defaults to
        ``'cv_<layer>'`` and stores in ``adata.var``.
    inplace : bool, optional
        If True, modify ``adata`` in place. If False, return a copy of
        ``adata`` with the added key.
    verbose : bool, optional
        If True, emit a warning summarising how many variables had a
        zero mean (and therefore undefined CV), broken down by whether
        the standard deviation was also zero (constant zero) or
        non-zero (all-zero mean with non-zero spread).

    Returns
    -------
    AnnData or None
        Returns the modified AnnData object if ``inplace=False``,
        otherwise returns None.

    Raises
    ------
    ValueError
        If the data appears to be log-transformed. CVs should be computed
        on raw (linear-scale) data.
        If both ``zero_to_na`` and ``fill_na`` are set.
    KeyError
        If ``group_by`` is provided but does not exist in ``adata.obs``.
    """
    check_proteodata(adata)

    if fill_na is not None and zero_to_na:
        raise ValueError(
            "Cannot set both 'zero_to_na' and 'fill_na'. "
            "'zero_to_na' converts zeros to NaN, while 'fill_na' "
            "replaces NaN with a constant."
        )
    if min_samples <= 1:
        raise ValueError(
            f"min_samples must be > 1, got {min_samples}. "
            "At least one observation is required to compute a CV."
        )
    if group_by is not None and group_by not in adata.obs.columns:
        raise KeyError(
            f"Column '{group_by}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )

    # CV requires linear-scale data
    if not force:
        is_log, _ = is_log_transformed(adata)
        if is_log:
            raise ValueError(
                "The data appears to be log-transformed. "
                "CVs should be computed on raw data. "
                "Set force=True to bypass this check."
            )

    X = adata.layers[layer] if layer is not None else adata.X
    layer_suffix = sanitize_string(layer) if layer is not None else "X"
    target = adata if inplace else adata.copy()

    if group_by is None:
        mean_, std_, n_ = _compute_cv_stats(
            X, zero_to_na=zero_to_na, fill_na=fill_na,
        )
        cv = _safe_cv_divide(std_, mean_, verbose=verbose)
        cv[n_ < min_samples] = np.nan

        if key_added is None:
            key_added = f"cv_{layer_suffix}"
        if key_added in target.var.columns:
            warnings.warn(
                f"Key '{key_added}' already exists in adata.var and will be "
                "overwritten.",
            )
        target.var[key_added] = cv

    else:
        groups = adata.obs[group_by].astype(str)
        group_names = groups.unique()

        cv_dict = {}
        for g in group_names:
            mask = (groups == g).values
            X_sub = X[mask, :]
            mean_, std_, n_ = _compute_cv_stats(
                X_sub,
                zero_to_na=zero_to_na,
                fill_na=fill_na,
            )
            cv = _safe_cv_divide(
                std_, mean_, verbose=verbose, label=g,
            )
            cv[n_ < min_samples] = np.nan
            cv_dict[g] = cv

        cv_df = pd.DataFrame(cv_dict, index=adata.var_names)

        if key_added is None:
            key_added = f"cv_by_{sanitize_string(group_by)}_{layer_suffix}"
        if key_added in target.varm.keys():
            warnings.warn(
                f"Key '{key_added}' already exists in adata.varm and will be "
                "overwritten.",
            )
        target.varm[key_added] = cv_df

    check_proteodata(target)

    if not inplace:
        return target
    return None
