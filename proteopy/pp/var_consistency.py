import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import sparse
from anndata import AnnData
from pandas.api.types import (
    is_bool_dtype,
    is_float_dtype,
    is_integer_dtype,
)

from proteopy.utils.anndata import check_proteodata
from proteopy.utils.string import sanitize_string

_INFERRED_KEY_SUFFIX = "_corr_min"


def _validate_var_consistency_input(
    adata,
    batch_key,
    group_by,
    min_groups,
    layer,
    key_added,
    inplace,
    verbose,
):
    """
    Validate inputs for ``var_consistency``.

    Returns
    -------
    list[str]
        ``group_by`` as a list of column names.
    """
    if not isinstance(batch_key, str):
        raise TypeError(
            f"`batch_key` must be a string, "
            f"got {type(batch_key).__name__}."
        )
    if isinstance(group_by, str):
        group_by = [group_by]
    if not isinstance(group_by, (list, tuple)) or not all(
        isinstance(k, str) for k in group_by
    ):
        raise TypeError(
            "`group_by` must be a string or a list of strings, "
            f"got {group_by!r}."
        )
    group_by = list(group_by)
    if not group_by:
        raise ValueError("`group_by` must name at least one column.")
    if len(set(group_by)) != len(group_by):
        raise ValueError(
            f"`group_by` must not contain duplicates, got {group_by}."
        )
    if batch_key in group_by:
        raise ValueError(
            f"`batch_key`='{batch_key}' must not be part of "
            f"`group_by`={group_by}."
        )

    keys = [batch_key, *group_by]
    missing_keys = [k for k in keys if k not in adata.obs.columns]
    if missing_keys:
        raise KeyError(
            f"Column(s) {missing_keys} not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )
    na_keys = [k for k in keys if adata.obs[k].isna().any()]
    if na_keys:
        raise ValueError(
            f"Column(s) {na_keys} in adata.obs contain missing values. "
            "Every sample needs a batch and a group."
        )

    if isinstance(min_groups, (bool, np.bool_)) or not isinstance(
        min_groups, (int, np.integer)
    ):
        raise TypeError(
            f"`min_groups` must be an int, "
            f"got {type(min_groups).__name__}."
        )
    if min_groups < 2:
        raise ValueError(f"`min_groups` must be at least 2, got {min_groups}.")
    if layer is not None and layer not in adata.layers:
        raise KeyError(
            f"Layer '{layer}' not found in adata.layers. "
            f"Available layers: {list(adata.layers.keys())}"
        )
    if key_added is not None and not isinstance(key_added, str):
        raise TypeError(
            f"`key_added` must be a string or None, "
            f"got {type(key_added).__name__}."
        )
    if not isinstance(inplace, bool):
        raise TypeError(
            f"`inplace` must be a bool, " f"got {type(inplace).__name__}."
        )
    if not isinstance(verbose, bool):
        raise TypeError(
            f"`verbose` must be a bool, " f"got {type(verbose).__name__}."
        )
    return group_by


def _pairwise_profile_corr(p1, p2, min_groups):
    """
    Column-wise Pearson correlation between two batch profiles.

    Parameters
    ----------
    p1, p2 : pd.DataFrame
        Profiles (groups by variables) of two batches. Only groups
        present in both are compared.
    min_groups : int
        Minimum number of shared non-missing groups per variable.

    Returns
    -------
    r : ndarray
        Pearson correlation per variable; NaN when fewer than
        ``min_groups`` groups are shared or either profile is constant.
    n : ndarray
        Number of shared non-missing groups per variable.
    """
    shared = p1.index.intersection(p2.index)
    a = p1.loc[shared].to_numpy(dtype=float)
    b = p2.loc[shared].to_numpy(dtype=float)

    mask = ~np.isnan(a) & ~np.isnan(b)
    n = mask.sum(axis=0)

    # A profile is constant when its smallest and largest shared
    # values coincide (exact test, unaffected by rounding)
    a_const = np.where(mask, a, np.inf).min(axis=0) == np.where(
        mask, a, -np.inf
    ).max(axis=0)
    b_const = np.where(mask, b, np.inf).min(axis=0) == np.where(
        mask, b, -np.inf
    ).max(axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        a_mean = np.where(mask, a, 0).sum(axis=0) / n
        b_mean = np.where(mask, b, 0).sum(axis=0) / n
        a_dev = np.where(mask, a - a_mean, 0)
        b_dev = np.where(mask, b - b_mean, 0)
        r = (a_dev * b_dev).sum(axis=0) / np.sqrt(
            (a_dev**2).sum(axis=0) * (b_dev**2).sum(axis=0)
        )

    valid = (n >= min_groups) & ~a_const & ~b_const
    r[~valid] = np.nan
    return r, n


def var_consistency(
    adata: AnnData,
    batch_key: str,
    group_by: str | list[str],
    min_groups: int = 4,
    layer: str | None = None,
    key_added: str | None = None,
    inplace: bool = True,
    verbose: bool = False,
) -> AnnData | None:
    """
    Compute the consistency of each variable's profile across batches.

    Batches (e.g. replicates) are expected to reproduce the same
    profile across groups (e.g. conditions and fractions). For every
    variable:

    1. construct its profile across the ``group_by`` groups for each
       batch
    2. compute the Pearson correlation between the profiles of every
       batch pair
    3. summarize the pairwise correlations by their median, minimum
       and mean

    The resulting scores can be used to remove variables with poorly
    reproducible profiles via
    :func:`~proteopy.pp.filter_by_var_consistency`.

    Parameters
    ----------
    adata : AnnData
        :class:`~anndata.AnnData` object in proteodata format.
    batch_key : str
        Column in ``adata.obs`` identifying the batches whose profiles
        are compared, e.g. replicates.
    group_by : str | list[str]
        Column(s) in ``adata.obs`` defining the groups that make up a
        profile. With several columns, each combination of their
        values is one group, e.g. ``["condition", "fraction"]``.
    min_groups : int
        Minimum number of groups with a value in both batches required
        to compute a correlation. Must be at least 2.
    layer : str | None
        Layer to use for the computation. If None, uses ``adata.X``.
    key_added : str | None
        Prefix of the keys under which results are stored. Defaults to
        ``'<batch_key>_corr'``. Stored are, in ``adata.var``,
        ``'<key_added>_median'``, ``'<key_added>_min'``,
        ``'<key_added>_mean'`` and ``'<key_added>_n_valid'`` (the
        number of batch pairs with a defined correlation), and, in
        ``adata.varm``, ``'<key_added>_pairs'``: a DataFrame with the
        correlation (``'corr_<batch1>_vs_<batch2>'``) and the number of
        shared groups (``'n_<batch1>_vs_<batch2>'``) of every batch
        pair.
    inplace : bool
        If True, modify ``adata`` in place. If False, return a copy of
        ``adata`` with the added keys.
    verbose : bool
        If True, print the matrix used and where results are stored.

    Returns
    -------
    AnnData | None
        The annotated copy of ``adata`` if ``inplace=False``, otherwise
        None.

    Raises
    ------
    KeyError
        If ``batch_key`` or a ``group_by`` column is not in
        ``adata.obs``, or ``layer`` is not in ``adata.layers``.
    ValueError
        If ``group_by`` is empty, has duplicates or contains
        ``batch_key``, the ``batch_key`` / ``group_by`` columns contain
        missing values, ``min_groups`` is smaller than 2, or fewer
        than two batches are present.
    TypeError
        If an argument has the wrong type.

    Warns
    -----
    UserWarning
        If the selected matrix is sparse and is densified, or if an
        output key already exists and is overwritten.

    Notes
    -----
    If a batch holds more than one sample for the same group, their
    values are averaged. A correlation is undefined (NaN) when the two
    batches share fewer than ``min_groups`` non-missing groups or
    either profile is constant. Zeros are used as values, not treated
    as missing; convert them to NaN beforehand if they denote missing
    measurements.

    Examples
    --------
    >>> import proteopy as pr
    >>> pr.pp.var_consistency(
    ...     adata,
    ...     batch_key="replicate",
    ...     group_by=["condition", "fraction"],
    ... )
    >>> adata.var["replicate_corr_min"].describe()
    """
    group_by = _validate_var_consistency_input(
        adata,
        batch_key,
        group_by,
        min_groups,
        layer,
        key_added,
        inplace,
        verbose,
    )
    check_proteodata(adata, layers=layer)

    Xsrc = adata.layers[layer] if layer is not None else adata.X
    matrix_name = (
        f"adata.layers['{layer}']" if layer is not None else "adata.X"
    )
    if sparse.issparse(Xsrc):
        warnings.warn(
            f"{matrix_name} is sparse and is being densified for "
            "computation.",
            UserWarning,
            stacklevel=2,
        )
        X = Xsrc.toarray()
    else:
        X = np.asarray(Xsrc)

    if key_added is None:
        key_added = f"{sanitize_string(batch_key)}_corr"

    # -- Batch x group profiles
    intensity = pd.DataFrame(
        X.astype(float),
        index=adata.obs_names,
        columns=adata.var_names,
    )
    # Average multiple samples of the same batch and group
    profiles = intensity.groupby(
        [adata.obs[k] for k in [batch_key, *group_by]],
        observed=True,
    ).mean()

    batches = profiles.index.get_level_values(batch_key).unique()
    if len(batches) < 2:
        raise ValueError(
            f"At least two batches in adata.obs['{batch_key}'] are "
            f"required, found {len(batches)}."
        )
    batch_pairs = list(combinations(batches, 2))

    if verbose:
        print(
            f"Using {matrix_name}: {len(batches)} batches "
            f"('{batch_key}'), {len(batch_pairs)} batch pairs, "
            f"groups from {group_by}."
        )

    # -- Pairwise batch correlations
    pairs = {}
    corrs = np.empty((adata.n_vars, len(batch_pairs)))
    for i, (batch1, batch2) in enumerate(batch_pairs):
        r, n = _pairwise_profile_corr(
            profiles.xs(batch1, level=batch_key),
            profiles.xs(batch2, level=batch_key),
            min_groups,
        )
        corrs[:, i] = r
        pairs[f"corr_{batch1}_vs_{batch2}"] = r
        pairs[f"n_{batch1}_vs_{batch2}"] = n

    # -- Summaries across batch pairs
    with warnings.catch_warnings():
        # All-NaN rows yield NaN, which is intended
        warnings.simplefilter("ignore", RuntimeWarning)
        summary = {
            "median": np.nanmedian(corrs, axis=1),
            "min": np.nanmin(corrs, axis=1),
            "mean": np.nanmean(corrs, axis=1),
        }
    summary["n_valid"] = (~np.isnan(corrs)).sum(axis=1)

    # -- Store results
    target = adata if inplace else adata.copy()
    for stat, values in summary.items():
        column = f"{key_added}_{stat}"
        if column in target.var.columns:
            warnings.warn(
                f"Key '{column}' already exists in adata.var and will "
                "be overwritten.",
                UserWarning,
                stacklevel=2,
            )
        target.var[column] = values

    varm_key = f"{key_added}_pairs"
    if varm_key in target.varm.keys():
        warnings.warn(
            f"Key '{varm_key}' already exists in adata.varm and will "
            "be overwritten.",
            UserWarning,
            stacklevel=2,
        )
    target.varm[varm_key] = pd.DataFrame(pairs, index=adata.var_names)

    if verbose:
        stored = ", ".join(f"'{key_added}_{s}'" for s in summary)
        print(f"Stored in adata.var: {stored}.")
        print(f"Stored in adata.varm: '{varm_key}'.")

    check_proteodata(target, layers=layer)

    if not inplace:
        return target
    return None


def _validate_filter_by_var_consistency_input(
    adata,
    threshold,
    key,
    remove_na,
    verbose,
    inplace,
):
    """
    Validate inputs for ``filter_by_var_consistency``.

    Returns
    -------
    str
        The score column, inferred when ``key`` is None.
    """
    if key is None:
        candidates = [
            c
            for c in adata.var.columns
            if str(c).endswith(_INFERRED_KEY_SUFFIX)
        ]
        if len(candidates) != 1:
            found = f"found {candidates}" if candidates else "found none"
            raise ValueError(
                "`key` was not given and could not be inferred: "
                "exactly one adata.var column ending in "
                f"'{_INFERRED_KEY_SUFFIX}' is required, {found}. "
                "Pass `key` explicitly."
            )
        key = candidates[0]
    if not isinstance(key, str):
        raise TypeError(
            f"`key` must be a string or None, " f"got {type(key).__name__}."
        )
    if key not in adata.var.columns:
        raise KeyError(
            f"Key '{key}' not found in adata.var. "
            f"Available columns: {list(adata.var.columns)}"
        )
    score = adata.var[key]
    if is_bool_dtype(score) or not (
        is_integer_dtype(score) or is_float_dtype(score)
    ):
        raise TypeError(
            f"adata.var['{key}'] has dtype '{score.dtype}'. "
            "Only integer and float columns can be used as a score."
        )
    if isinstance(threshold, (bool, np.bool_)) or not isinstance(
        threshold, (int, float, np.integer, np.floating)
    ):
        raise TypeError(
            f"`threshold` must be an int or float, "
            f"got {type(threshold).__name__}."
        )
    for name, value in [
        ("remove_na", remove_na),
        ("verbose", verbose),
        ("inplace", inplace),
    ]:
        if not isinstance(value, bool):
            raise TypeError(
                f"`{name}` must be a bool, " f"got {type(value).__name__}."
            )
    return key


def filter_by_var_consistency(
    adata: AnnData,
    threshold: int | float,
    key: str | None = None,
    remove_na: bool = True,
    verbose: bool = False,
    inplace: bool = True,
) -> AnnData | None:
    """
    Filter variables by a per-variable consistency score.

    Keeps the variables whose ``adata.var[key]`` is at least
    ``threshold``, e.g. the minimum batch-pair correlation computed by
    :func:`~proteopy.pp.var_consistency`.

    Parameters
    ----------
    adata : AnnData
        :class:`~anndata.AnnData` object in proteodata format.
    threshold : int | float
        Minimum score a variable needs to be kept
        (``score >= threshold``).
    key : str | None
        Numeric column in ``adata.var`` holding the score. If None, it
        is inferred as the single ``adata.var`` column ending in
        ``'_corr_min'``, i.e. the minimum pairwise correlation stored
        by :func:`~proteopy.pp.var_consistency`.
    remove_na : bool
        Remove variables whose score is NaN (undefined). If False,
        they are kept.
    verbose : bool
        Print how many variables are kept, removed and undefined.
    inplace : bool
        If True, modify ``adata`` in place. If False, return a filtered
        copy of ``adata``.

    Returns
    -------
    AnnData | None
        The filtered copy of ``adata`` if ``inplace=False``, otherwise
        None.

    Raises
    ------
    KeyError
        If ``key`` is not a column of ``adata.var``.
    ValueError
        If ``key`` is None and not exactly one ``adata.var`` column
        ends in ``'_corr_min'``.
    TypeError
        If ``key`` is not an integer or float column, ``threshold`` is
        not an int or float, or a flag is not a bool.

    Examples
    --------
    >>> import proteopy as pr
    >>> pr.pp.var_consistency(
    ...     adata,
    ...     batch_key="replicate",
    ...     group_by=["condition", "fraction"],
    ... )
    >>> pr.pp.filter_by_var_consistency(adata, threshold=0.25)

    Pass ``key`` when several consistency scores are stored, or to
    filter on another summary:

    >>> pr.pp.filter_by_var_consistency(
    ...     adata,
    ...     threshold=0.5,
    ...     key="replicate_corr_median",
    ... )
    """
    check_proteodata(adata)
    key = _validate_filter_by_var_consistency_input(
        adata,
        threshold,
        key,
        remove_na,
        verbose,
        inplace,
    )

    score = adata.var[key]
    keep = score.ge(threshold)
    if not remove_na:
        keep = keep | score.isna()

    if verbose:
        n_total = len(score)
        n_na = int(score.isna().sum())
        n_pass = int((score >= threshold).sum())
        n_fail = int((score < threshold).sum())
        n_valid = n_total - n_na
        print(f"Threshold:  {key} >= {threshold}")
        print(f"Total:      {n_total}")
        print(f"Keep:       {n_pass}")
        print(f"Remove:     {n_fail}")
        print(
            f"Undefined:  {n_na}" + (" (removed)" if remove_na else " (kept)")
        )
        if n_valid:
            print(
                f"Kept:       {100 * n_pass / n_valid:.1f}% "
                "of variables with a defined score"
            )

    if inplace:
        adata._inplace_subset_var(keep.to_numpy())
        check_proteodata(adata)
        return None

    filtered = adata[:, keep.to_numpy()].copy()
    check_proteodata(filtered)
    return filtered
