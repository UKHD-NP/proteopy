import warnings

import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse

from proteopy.utils.anndata import check_proteodata
from proteopy.utils.array import _is_log_transformed_array


def _validate_normalize_median_input(
    adata,
    log_space,
    target,
    fill_na,
    zero_to_na,
    group_by,
    key_added,
    inplace,
    force,
    verbose,
):
    """Validate and type-check arguments for ``normalize_median``.

    Returns
    -------
    str
        The lower-cased, validated ``target`` value.
    """
    if not isinstance(adata, ad.AnnData):
        raise TypeError(
            f"`adata` must be an AnnData object, "
            f"got {type(adata).__name__}."
        )
    # -- Sparse input is no longer supported; densify first
    if sparse.issparse(adata.X):
        raise TypeError(
            "Sparse `.X` is not supported by `normalize_median`. "
            "Densify the matrix first, e.g. "
            "`adata.X = adata.X.toarray()`."
        )
    check_proteodata(adata)
    if not isinstance(log_space, bool):
        raise TypeError(
            f"`log_space` must be a bool, got {type(log_space).__name__}."
        )
    if not isinstance(target, str):
        raise TypeError(
            f"`target` must be a string, got {type(target).__name__}."
        )
    target = target.lower()
    allowed_targets = {"max", "median"}
    if target not in allowed_targets:
        raise ValueError(f"`target` must be one of {allowed_targets!r}.")
    if fill_na is not None and zero_to_na:
        raise ValueError(
            "`fill_na` and `zero_to_na` are mutually exclusive; "
            "set at most one of them."
        )
    if fill_na is not None and (
        isinstance(fill_na, bool) or not isinstance(fill_na, (int, float))
    ):
        raise TypeError(
            f"`fill_na` must be a numeric value or None, "
            f"got {type(fill_na).__name__}."
        )
    if fill_na is not None and not np.isfinite(fill_na):
        raise ValueError("`fill_na` must be a finite value (not inf/nan).")
    if not isinstance(zero_to_na, bool):
        raise TypeError(
            f"`zero_to_na` must be a bool, "
            f"got {type(zero_to_na).__name__}."
        )
    if not isinstance(key_added, str):
        raise TypeError(
            f"`key_added` must be a string, "
            f"got {type(key_added).__name__}."
        )
    if not key_added:
        raise ValueError("`key_added` must be a non-empty string.")
    if not isinstance(inplace, bool):
        raise TypeError(
            f"`inplace` must be a bool, got {type(inplace).__name__}."
        )
    if not isinstance(force, bool):
        raise TypeError(f"`force` must be a bool, got {type(force).__name__}.")
    if not isinstance(verbose, bool):
        raise TypeError(
            f"`verbose` must be a bool, got {type(verbose).__name__}."
        )
    if group_by is not None:
        if not isinstance(group_by, str):
            raise TypeError(
                f"`group_by` must be a string or None, "
                f"got {type(group_by).__name__}."
            )
        if group_by not in adata.obs.columns:
            raise KeyError(f"`group_by`='{group_by}' not found in adata.obs")
        if adata.obs[group_by].isna().any():
            raise ValueError(
                f"`group_by`='{group_by}' column in "
                f"adata.obs contains NaN values."
            )
    return target


def _normalize_samples(X_work, sample_ids, target, log_space):
    """Normalize a subset of samples; return values and factors.

    Parameters
    ----------
    X_work : np.ndarray
        Sub-matrix (samples x vars) to normalize.
    sample_ids : array-like of str
        ``obs_names`` for the rows of ``X_work``; used for error
        messages naming offending samples.
    target : {'median', 'max'}
        How to compute the normalization target from sample medians.
    log_space : bool
        If ``True``, normalize additively; otherwise multiplicatively.

    Returns
    -------
    sub_norm : np.ndarray
        Normalized sub-matrix.
    factors : np.ndarray
        Per-sample factor (shift in log space, scale in linear space).
    """
    # NumPy raises a RuntimeWarning ("All-NaN slice encountered") when a
    # sample, or an entire group, is all-NaN. Such cases are expected and
    # intentionally yield NaN medians/factors (surfaced via verbose), so
    # the warning is suppressed. (np.errstate does NOT catch this one.)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sample_medians = np.nanmedian(X_work, axis=1)
        if target == "median":
            target_val = float(np.nanmedian(sample_medians))
        else:  # target == "max"
            target_val = float(np.nanmax(sample_medians))

    if log_space:
        factors = (target_val - sample_medians)[:, None]
        sub_norm = X_work + factors
    else:  # linear space
        # A median of exactly 0 in linear space yields inf factors.
        # Treat it as an error rather than silently producing inf.
        zero_med = sample_medians == 0
        if zero_med.any():
            bad = [str(sample_ids[i]) for i in np.where(zero_med)[0]]
            raise ValueError(
                "Cannot normalize in linear space: sample median is "
                "exactly 0 for sample(s): "
                f"{', '.join(bad)}. Consider filtering or imputing."
            )
        factors = (target_val / sample_medians)[:, None]
        sub_norm = X_work * factors

    return sub_norm, factors[:, 0]


def _report_normalize_median(
    adata,
    X,
    obs_names,
    all_factors,
    log_space,
    is_log,
    group_by,
    key_added,
):
    """Print a verbose summary of a ``normalize_median`` run."""
    space = "log" if log_space else "linear"
    detect = (
        "passed" if log_space == is_log else "passed (auto-detect differed)"
    )
    print(f"Normalizing in {space} space (log_space {detect}).")

    # -- Samples that are entirely NaN (these yield NaN factors). Derive
    #    from the matrix, not from the factors: a NaN factor can also
    #    arise from a NaN target, which would misattribute samples.
    nan_mask = np.isnan(X).all(axis=1)
    if nan_mask.any():
        if group_by is None:
            ids = ", ".join(str(s) for s in obs_names[nan_mask])
            print(
                f"{int(nan_mask.sum())} sample(s) had an all-NaN "
                f"median; their factors are NaN: {ids}"
            )
        else:
            groups = adata.obs[group_by].astype("category")
            for g in groups.cat.categories:
                idx = np.where(groups.values == g)[0]
                bad = idx[nan_mask[idx]]
                if bad.size == 0:
                    continue
                ids = ", ".join(str(s) for s in obs_names[bad])
                print(
                    f"Group '{g}': {bad.size} sample(s) had an "
                    f"all-NaN median; their factors are NaN: {ids}"
                )

    # -- Count groups actually normalized (1 when ungrouped)
    if group_by is None:
        n_groups = 1
    else:
        groups = adata.obs[group_by].astype("category")
        n_groups = int(
            sum((groups.values == g).any() for g in groups.cat.categories)
        )

    print(f"Stored per-sample factors in " f"adata.uns['{key_added}'].")
    print(
        f"Summary: normalized {len(obs_names)} sample(s) across "
        f"{n_groups} group(s)."
    )


def normalize_median(
    adata,
    *,
    log_space: bool = True,
    target: str = "median",
    fill_na: float | None = None,
    zero_to_na: bool = False,
    group_by: str | None = None,
    key_added: str = "normalization_factors",
    inplace: bool = True,
    force: bool = False,
    verbose: bool = False,
):
    r"""
    Median normalization of intensities.

    Each sample is rescaled so that its median intensity matches a
    common target. Let :math:`m_s` be the median over the finite
    (non-NaN) features of sample :math:`s`; NaNs are ignored when
    computing it.

    In log space (``log_space=True``) the rescaling is additive; in
    linear space (``log_space=False``) it is multiplicative:

    .. math::

        X'_{s,i} = X_{s,i} + (t - m_s)
        \qquad
        X'_{s,i} = X_{s,i} \cdot \frac{t}{m_s}

    The target :math:`t` is derived from the per-sample medians
    (within each group when ``group_by`` is set):

    .. math::

        t = \operatorname{median}_s(m_s)
        \qquad
        t = \max_s(m_s)

    for ``target='median'`` and ``target='max'`` respectively. The
    ``zero_to_na`` and ``fill_na`` transforms (mutually exclusive) are
    applied to ``.X`` before normalization and persist in the output.
    A sample of only NaNs yields :math:`m_s = \mathrm{NaN}` and thus a
    NaN factor; this is not an error and is surfaced through
    ``verbose``.

    Parameters
    ----------
    adata : AnnData
        Input AnnData in proteodata format.
    log_space : bool
        Whether the input intensities are log-transformed. Mismatches
        with automatic detection raise unless ``force=True``. Defaults
        to ``True``.
    target : {'max', 'median'}
        How to compute the scaling target from the per-sample
        medians. ``'max'`` uses the maximum sample median,
        ``'median'`` the median of sample medians. Defaults to
        ``'median'``.
    fill_na : float, optional
        Replace non-finite entries in ``.X`` with this value before
        normalization.
    zero_to_na : bool, default False
        Treat zeros in ``.X`` as missing (``NaN``) before
        normalization (replaces zeros with ``np.nan``).
    group_by : str, optional
        Column in ``adata.obs`` defining sample groups; when set,
        normalization is performed independently within each group
        (e.g. batch, condition, or any other sample grouping).
    key_added : str, default 'normalization_factors'
        Key of the ``adata.uns`` slot in which the per-sample factors
        DataFrame is stored.
    inplace : bool, default True
        Modify ``adata`` in place. If False, return a copy.
    force : bool, default False
        Proceed even if ``log_space`` disagrees with automatic log
        detection.
    verbose : bool, default False
        If True, print the resolved log space, samples whose median is
        NaN (per group when ``group_by`` is set), where the factors are
        stored (``adata.uns[key_added]``), and a run summary.

    Returns
    -------
    AnnData or None
        Normalized AnnData when ``inplace`` is False; otherwise None.
    pandas.DataFrame, optional
        Per-sample factors when ``inplace`` is False.

    Raises
    ------
    TypeError
        If any argument has an unexpected type, or if ``.X`` is sparse.
    ValueError
        If ``target`` is invalid, ``key_added`` is empty, ``fill_na``
        is non-finite, ``fill_na`` and ``zero_to_na`` are both set,
        ``group_by`` contains NaN, ``log_space`` disagrees with
        automatic detection and ``force=False``, a sample median is
        exactly 0 in linear space (``log_space=False``), or the
        normalization produces infinite values.
    KeyError
        If ``group_by`` is not a column in ``adata.obs``.

    Examples
    --------
    Build a minimal log-space, protein-level proteodata object:

    >>> import numpy as np
    >>> import pandas as pd
    >>> import anndata as ad
    >>> import proteopy as pr
    >>> adata = ad.AnnData(
    ...     X=np.array([[18.0, 20.0, 25.0],
    ...                 [19.0, 21.0, 22.0],
    ...                 [16.0, 19.0, 28.0]]),
    ...     obs=pd.DataFrame({"sample_id": ["S0", "S1", "S2"]},
    ...                      index=["S0", "S1", "S2"]),
    ...     var=pd.DataFrame({"protein_id": ["P0", "P1", "P2"]},
    ...                      index=["P0", "P1", "P2"]),
    ... )

    Normalize using the median of sample medians (defaults), returning
    a copy together with the per-sample factors:

    >>> adata_norm, factors = pr.pp.normalize_median(
    ...     adata, inplace=False)
    >>> adata_norm.X
    array([[18., 20., 25.],
           [18., 20., 21.],
           [17., 20., 29.]])

    Normalize in place using the maximum of sample medians:

    >>> pr.pp.normalize_median(adata, target="max")
    >>> adata.X
    array([[19., 21., 26.],
           [19., 21., 22.],
           [18., 21., 30.]])
    """
    target = _validate_normalize_median_input(
        adata,
        log_space,
        target,
        fill_na,
        zero_to_na,
        group_by,
        key_added,
        inplace,
        force,
        verbose,
    )

    X = np.asarray(adata.X, dtype=float).copy()

    if zero_to_na:
        X[X == 0] = np.nan
    elif fill_na is not None:
        X[~np.isfinite(X)] = fill_na

    # Detect on the working matrix so any zero_to_na/fill_na transform
    # already applied is reflected in the log-space heuristic.
    is_log, _ = _is_log_transformed_array(X)
    mismatch = log_space != is_log
    if mismatch and not force:
        if log_space:
            raise ValueError(
                "You passed log_space=True but the data do not look "
                "log-transformed. Set force=True to override the "
                "automatic detection."
            )
        raise ValueError(
            "You passed log_space=False but the data look "
            "log-transformed. Set force=True to override the "
            "automatic detection."
        )

    n_samples = X.shape[0]
    obs_names = np.asarray(adata.obs_names)

    all_norm = np.full_like(X, np.nan, dtype=float)
    all_factors = np.full(n_samples, np.nan, dtype=float)

    if group_by is None:
        idx = np.arange(n_samples)
        sub_norm, sub_fac = _normalize_samples(
            X[idx, :], obs_names[idx], target, log_space
        )
        all_norm[idx, :] = sub_norm
        all_factors[idx] = sub_fac
    else:
        groups = adata.obs[group_by].astype("category")
        for g in groups.cat.categories:
            idx = np.where(groups.values == g)[0]
            if idx.size == 0:
                continue
            sub_norm, sub_fac = _normalize_samples(
                X[idx, :], obs_names[idx], target, log_space
            )
            all_norm[idx, :] = sub_norm
            all_factors[idx] = sub_fac

    factor_name = "shift_log" if log_space else "scale_linear"

    factors_df = pd.DataFrame(
        {
            "sample_index": np.arange(n_samples),
            "sample_id": obs_names,
            factor_name: all_factors,
        }
    )
    # Avoid clobbering an existing factors_df column (e.g. group_by
    # == 'sample_id', which is always a valid obs column).
    if group_by is not None and group_by not in factors_df.columns:
        factors_df[group_by] = adata.obs[group_by].values

    if verbose:
        _report_normalize_median(
            adata,
            X,
            obs_names,
            all_factors,
            log_space,
            is_log,
            group_by,
            key_added,
        )

    # -- Compute -> validate -> assign: reject a non-finite result before
    #    touching `adata`, so a failure cannot leave it partially modified.
    if np.isinf(all_norm).any():
        raise ValueError(
            "Normalization produced infinite values; check the input "
            "intensities and `fill_na`."
        )

    if inplace:
        adata.X = all_norm
        adata.uns[key_added] = factors_df
        check_proteodata(adata)
        return None

    adata_out = adata.copy()
    adata_out.X = all_norm
    adata_out.uns[key_added] = factors_df
    check_proteodata(adata_out)
    return adata_out, factors_df
