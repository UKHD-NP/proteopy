import numpy as np
import anndata as ad
from scipy import sparse

from proteopy.utils.anndata import check_proteodata
from proteopy.utils.array import _is_log_transformed_array


def _validate_impute_downshift_input(  # noqa: C901
    adata,
    downshift,
    width,
    zero_to_na,
    inplace,
    force,
    random_state,
    group_by,
    verbose,
    Y=None,
):
    """Validate and type-check arguments for ``impute_downshift``."""
    if not isinstance(adata, ad.AnnData):
        raise TypeError(
            f"`adata` must be an AnnData object, "
            f"got {type(adata).__name__}."
        )
    if isinstance(downshift, bool) or not isinstance(
        downshift, (int, float)
    ):
        raise TypeError(
            f"`downshift` must be a numeric value, "
            f"got {type(downshift).__name__}."
        )
    if isinstance(width, bool) or not isinstance(
        width, (int, float)
    ):
        raise TypeError(
            f"`width` must be a numeric value, "
            f"got {type(width).__name__}."
        )
    if width <= 0:
        raise ValueError("`width` must be positive.")
    if not isinstance(zero_to_na, bool):
        raise TypeError(
            f"`zero_to_na` must be a bool, "
            f"got {type(zero_to_na).__name__}."
        )
    if not isinstance(inplace, bool):
        raise TypeError(
            f"`inplace` must be a bool, "
            f"got {type(inplace).__name__}."
        )
    if not isinstance(force, bool):
        raise TypeError(
            f"`force` must be a bool, "
            f"got {type(force).__name__}."
        )
    if random_state is not None and not isinstance(
        random_state, int
    ):
        raise TypeError(
            f"`random_state` must be an int or None, "
            f"got {type(random_state).__name__}."
        )
    if group_by is not None and not isinstance(group_by, str):
        raise TypeError(
            f"`group_by` must be a string or None, "
            f"got {type(group_by).__name__}."
        )
    if not isinstance(verbose, bool):
        raise TypeError(
            f"`verbose` must be a bool, "
            f"got {type(verbose).__name__}."
        )
    if group_by is not None:
        if group_by not in adata.obs.columns:
            raise KeyError(
                f"`group_by`='{group_by}' not found "
                f"in adata.obs"
            )
    # -- Log-transform check on cleaned matrix so that
    #    zeros (now NaN) don't bias the heuristic
    if not force and Y is not None:
        is_log, _ = _is_log_transformed_array(Y)
        if not is_log:
            raise ValueError(
                "Imputation expects log-transformed "
                "intensities. Set force=True to "
                "proceed nevertheless."
            )


def _impute_rows(
    Y_imp, miss_mask, row_indices,
    median, sd, downshift, width, rng,
):
    """Impute NaNs in the given rows from a downshifted normal.

    Draws random samples from
    ``N(median - downshift*sd, (width*sd)^2)`` and writes them into
    ``Y_imp`` wherever ``miss_mask`` is ``True``.

    Parameters
    ----------
    Y_imp : np.ndarray
        Output matrix (obs x vars); imputed values are written
        here in-place.
    miss_mask : np.ndarray
        Boolean mask of shape (obs x vars); ``True`` marks positions
        to be imputed.
    row_indices : array-like of int
        Row indices to process.
    median : float
        Median of the reference distribution (center before shifting).
    sd : float
        Standard deviation of the reference distribution.
    downshift : float
        Number of standard deviations to shift the center leftward.
    width : float
        Scaling factor applied to ``sd`` to set the sampler width.
    rng : np.random.Generator
        NumPy random generator used for reproducible sampling.
    """
    mu = median - downshift * sd
    scale = max(width * sd, 1e-6)
    for i in row_indices:
        miss = miss_mask[i, :]
        if not miss.any():
            continue
        Y_imp[i, miss] = rng.normal(
            loc=mu, scale=scale, size=int(miss.sum()),
        )


def _impute_by_group(
    Y, Y_imp, miss_mask, groups,
    g_median, g_sd, downshift, width, rng,
):
    """Impute per group, falling back to global stats."""
    for label in groups.unique():
        row_idx = np.where(groups == label)[0]
        grp_vals = Y[row_idx, :][
            np.isfinite(Y[row_idx, :])
        ]
        if grp_vals.size >= 3:
            grp_median = float(np.median(grp_vals))
            grp_sd = float(np.std(grp_vals))
            if not np.isfinite(grp_sd) or grp_sd <= 0:
                grp_median, grp_sd = g_median, g_sd
        else:
            grp_median, grp_sd = g_median, g_sd
        _impute_rows(
            Y_imp, miss_mask, row_idx,
            grp_median, grp_sd,
            downshift, width, rng,
        )


def _store_downshift_imputation_metadata(
    target, miss_mask, n_missing,
    width, downshift, group_by, random_state,
):
    """Write imputation mask and run metadata to ``target``."""
    target.layers["imputation_mask_X"] = (
        miss_mask.astype(bool)
    )
    target.uns.setdefault("imputation", {})
    target.uns["imputation"].update(dict(
        method="downshift_normal",
        width=float(width),
        downshift=float(downshift),
        group_by=group_by,
        random_state=(
            None if random_state is None
            else int(random_state)
        ),
        n_imputed=int(n_missing),
        pct_imputed=float(
            n_missing / miss_mask.size * 100.0
        ),
    ))


def impute_downshift(
    adata,
    zero_to_na: bool = True,
    downshift: float = 1.8,
    width: float = 0.3,
    group_by: str | None = None,
    inplace: bool = True,
    force: bool = False,
    random_state: int | None = 42,
    verbose: bool = False,
):
    """Impute missing values via a downshifted Gaussian.

    Replaces ``NaN`` (and optionally zero) entries by sampling from a
    Gaussian centered at ``median - downshift * sd`` with standard
    deviation ``width * sd``, simulating expression signals below the
    detection limit as popularised by the Perseus platform [1]_.
    The median and standard deviation are estimated from the observed
    values of the global distribution or distributions defined
    by the ``group_by`` parameter:

    - ``group_by=None`` — global distribution (all finite values
      in ``.X``). Recommended when sample-level distributions are similar.
    - ``group_by=<obs column>`` — per-group distribution pooled across
      all samples sharing the same label in that column.

    When a sample or group contains fewer than three finite values, the
    global distribution (all finite values in ``.X``) is used as a
    fallback.

    The function records an imputation mask in
    ``.layers["imputation_mask_X"]`` (``True`` where values were
    imputed) and stores run metadata in ``.uns["imputation"]``.

    It is recommended to work on the log-transformed intensities space.

    Parameters
    ----------
    adata : ad.AnnData
        Proteodata-formatted :class:`~anndata.AnnData`.
    zero_to_na : bool, optional
        If ``True``, replace zeros in ``.X`` with ``NaN`` before
        imputation so they are treated as missing values.
    downshift : float, optional
        Number of standard deviations to shift the distribution
        center leftward from the observed median.
    width : float, optional
        Scaling factor applied to the observed standard deviation to
        set the width of the sampling distribution.
    group_by : str | None, optional
        Column in ``adata.obs`` defining groups over which the
        reference distribution is pooled. When ``None``, the global
        distribution across all samples is used.
    inplace : bool, optional
        If ``True``, modify ``adata`` in place and return ``None``.
        If ``False``, return an imputed copy without altering ``adata``.
    force : bool, optional
        If ``False``, raise a ``ValueError`` when the data are
        detected as non-log-transformed. Set to ``True`` to bypass
        this check and impute regardless.
    random_state : int | None, optional
        Seed for the NumPy random generator. Pass ``None`` for a
        non-deterministic run.
    verbose : bool, optional
        Print stats.

    Returns
    -------
    ad.AnnData | None
        Imputed ``AnnData`` when ``inplace=False``; ``None`` otherwise.
        The returned or modified object contains:

        - ``.X`` — imputed intensity matrix (sparse if input was
          sparse).
        - ``.layers["imputation_mask_X"]`` — boolean mask; ``True``
          marks positions that were imputed.
        - ``.uns["imputation"]`` — dict with keys ``method``,
          ``downshift``, ``width``, ``group_by``, ``random_state``,
          ``n_imputed``, and ``pct_imputed``.

    Raises
    ------
    TypeError
        If any argument has an unexpected type.
    ValueError
        If ``width`` is not positive, fewer than three finite values
        exist globally, or the data appear non-log-transformed and
        ``force=False``.
    KeyError
        If ``group_by`` is not a column in ``adata.obs``.

    References
    ----------
    .. [1] Tyanova S, Temu T, Sinitcyn P, Carlson A, Hein MY,
       Geiger T, Mann M, and Cox J. "The Perseus computational
       platform for comprehensive analysis of (prote)omics
       data." *Nature Methods*, 2016, 13(9):731-740.

    Examples
    --------
    >>> import numpy as np
    >>> import proteopy as pr
    >>> adata = pr.datasets.karayel_2020()
    >>> adata.layers["raw"] = adata.X
    >>> adata.X[adata.X == 0] = np.nan
    >>> adata.X = np.log2(adata.X)

    Simple imputation as popularized by Tyanova et. al 2016
    (downshift=1.8, width=0.3)

    >>> pr.pp.impute_downshift(adata)

    Impute by drawing from sample-level Gaussian distributions
    instead of global:

    >>> pr.pp.impute_downshift(adata, group_by="sample_id")
    """
    check_proteodata(adata)

    Xsrc = adata.X
    was_sparse = sparse.issparse(Xsrc)
    X = Xsrc.toarray() if was_sparse else np.asarray(Xsrc)
    X = X.astype(float, copy=True)

    # -- Build working matrix (NaN = missing)
    Y = X.copy()
    if zero_to_na:
        Y[Y == 0] = np.nan
    Y[~np.isfinite(Y)] = np.nan

    _validate_impute_downshift_input(
        adata, downshift, width, zero_to_na,
        inplace, force, random_state, group_by,
        verbose, Y=Y,
    )

    miss_mask = ~np.isfinite(Y)
    n_missing = int(miss_mask.sum())

    rng = np.random.default_rng(random_state)

    # -- Global fallback stats
    y_finite = Y[np.isfinite(Y)]
    if y_finite.size < 3:
        raise ValueError(
            "Not enough finite values to estimate "
            "imputation parameters."
        )
    g_median = float(np.median(y_finite))
    g_sd = float(np.std(y_finite))
    if not np.isfinite(g_sd) or g_sd <= 0:
        raise ValueError(
            "Global standard deviation is zero or "
            "non-finite; cannot estimate imputation "
            "parameters. The data may lack variation."
        )

    # -- Imputation
    Y_imp = Y.copy()

    if group_by is None:
        _impute_rows(
            Y_imp, miss_mask, range(Y.shape[0]),
            g_median, g_sd, downshift, width, rng,
        )
    else:
        _impute_by_group(
            Y, Y_imp, miss_mask, adata.obs[group_by],
            g_median, g_sd, downshift, width, rng,
        )

    Z_out = sparse.csr_matrix(Y_imp) if was_sparse else Y_imp

    if verbose:
        total = miss_mask.size
        measured_n = total - n_missing
        print(
            f"Measured: {measured_n:,} values "
            f"({100 * measured_n / total:.1f}%)"
        )
        print(
            f"Imputed: {n_missing:,} values "
            f"({100 * n_missing / total:.1f}%)"
        )

    if not inplace:
        adata_out = adata.copy()
        adata_out.X = Z_out
        _store_downshift_imputation_metadata(
            adata_out, miss_mask, n_missing,
            width, downshift, group_by, random_state,
        )
        check_proteodata(adata_out)
        return adata_out
    else:
        adata.X = Z_out
        _store_downshift_imputation_metadata(
            adata, miss_mask, n_missing,
            width, downshift, group_by, random_state,
        )
        check_proteodata(adata)
        return None
