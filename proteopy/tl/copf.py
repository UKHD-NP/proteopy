import itertools
import copy as copym
import numpy as np
import pandas as pd
import anndata as ad
from scipy import stats
from sklearn.cluster import AgglomerativeClustering
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

from proteopy.utils.anndata import check_proteodata, is_proteodata
from proteopy.utils.copf import reconstruct_corrs_df_symmetric_from_long_df
from proteopy.utils.data_structures import BinaryClusterTree
from proteopy.utils.hash import md5_hash_list
from proteopy.utils.pandas import long_pairs_to_symmetric_matrix
from proteopy.utils.slot_parsers import (
    parse_pairwise_peptide_correlations_result_legacy,
    parse_pairwise_var_correlations_result,
    resolve_pairwise_var_correlations_key,
)

NOISE = 1e6


def _resolve_var_subset(
    adata: ad.AnnData,
    var_subset,
) -> list | None:
    """
    Resolve ``var_subset`` to an ordered list of var names.

    Accepted forms:
      - ``None``: no subset; the function uses all vars.
      - Sequence of strings: names to include. Duplicates and names
        absent from ``adata.var_names`` raise ``ValueError``.
      - Sequence of booleans (length = ``adata.n_vars``): mask aligned
        with ``adata.var_names``.

    Returns
    -------
    list[str] | None
        Var names in ``adata.var_names`` order (a subset thereof), or
        ``None`` when no subset is requested. The returned order is
        independent of input order, so the column-order invariant of
        :func:`pairwise_var_correlations` holds across runs.
    """
    if var_subset is None:
        return None

    if isinstance(var_subset, str) or not hasattr(var_subset, "__iter__"):
        raise ValueError(
            "var_subset must be a sequence of var names or a "
            "boolean mask; got "
            f"{type(var_subset).__name__}."
        )

    items = list(var_subset)
    if not items:
        raise ValueError("var_subset is empty.")

    is_bool = [isinstance(x, (bool, np.bool_)) for x in items]
    is_str = [isinstance(x, str) for x in items]

    if all(is_bool):
        if len(items) != adata.n_vars:
            raise ValueError(
                f"Boolean var_subset has length {len(items)} but "
                f"adata.n_vars is {adata.n_vars}."
            )
        mask = np.asarray(items, dtype=bool)
        return adata.var_names[mask].tolist()

    if all(is_str):
        if len(set(items)) != len(items):
            raise ValueError("var_subset contains duplicate names.")
        known = set(adata.var_names)
        unknown = [x for x in items if x not in known]
        if unknown:
            preview = unknown[:10]
            suffix = " ..." if len(unknown) > 10 else ""
            raise ValueError(
                f"var_subset contains {len(unknown)} name(s) not "
                f"in adata.var_names: {preview}{suffix}"
            )
        # -- enforce adata.var_names order to preserve invariant
        items_set = set(items)
        return [v for v in adata.var_names if v in items_set]

    raise ValueError(
        "var_subset must contain only var names (str) or only "
        "booleans, not a mix."
    )


# pylint: disable=too-many-branches
def _validate_pairwise_var_correlations_params(  # noqa: C901
    adata: ad.AnnData,
    *,
    group_by: str | None,
    batch_key: str | None,
    layer: str | None,
    method: str,
    var_subset: list | None,
    fill_na: float | int | None,
    min_contrib_batches: int,
    min_wsum: float,
    key_added: str | None,
    inplace: bool,
) -> str:
    """
    Validate parameters for :func:`pairwise_var_correlations`.

    Returns
    -------
    str
        ``resolved_key_added``.
    """
    # -- scalar arg validation (no adata access)
    if method not in {"pearson", "spearman"}:
        raise ValueError(
            f"method must be 'pearson' or 'spearman'; got {method!r}"
        )
    if not isinstance(inplace, bool):
        raise ValueError(
            f"inplace must be a bool; got {type(inplace).__name__}"
        )

    # -- validate layer existence BEFORE check_proteodata
    if layer is not None:
        if not isinstance(layer, str) or not layer:
            raise ValueError("layer must be a non-empty string or None.")
        if layer not in adata.layers:
            raise ValueError(f"layer '{layer}' not found in .layers.")

    # -- proteodata structure & matrix validation
    check_proteodata(
        adata,
        layers=[layer] if layer is not None else None,
    )

    # -- validate remaining optional string keys against containers
    if group_by is not None:
        if not isinstance(group_by, str) or not group_by:
            raise ValueError("group_by must be a non-empty string or None.")
        if group_by not in adata.var.columns:
            raise ValueError(
                f"group_by '{group_by}' not found in .var.columns."
            )
        # -- COPF requires >= 2 vars per group: a single-var group
        # yields no pairs and is meaningless for proteoform work.
        # Apply var_subset BEFORE counting so the check reflects the
        # vars actually used downstream.
        if var_subset is not None:
            group_col = adata.var.loc[var_subset, group_by]
        else:
            group_col = adata.var[group_by]
        counts = group_col.value_counts(dropna=True)
        too_small = counts[counts < 2].index.tolist()
        if too_small:
            preview = too_small[:10]
            suffix = " ..." if len(too_small) > 10 else ""
            raise ValueError(
                f"group_by '{group_by}' requires >= 2 vars per "
                f"group for COPF; {len(too_small)} group(s) "
                f"violate this: {preview}{suffix}"
            )
    else:
        # -- ungrouped path: at least 2 vars required to form a pair
        n_vars_eff = (
            len(var_subset) if var_subset is not None else adata.n_vars
        )
        if n_vars_eff < 2:
            raise ValueError(
                "pairwise_var_correlations requires >= 2 vars; "
                f"got n_vars={n_vars_eff}."
            )
    if batch_key is not None:
        if not isinstance(batch_key, str) or not batch_key:
            raise ValueError("batch_key must be a non-empty string or None.")
        if batch_key not in adata.obs.columns:
            raise ValueError(
                f"batch_key '{batch_key}' not found in .obs.columns."
            )

    # -- validate fill_na
    if fill_na is not None:
        if isinstance(fill_na, bool) or not isinstance(fill_na, (int, float)):
            raise ValueError(
                "fill_na must be a number (int or float) or None; "
                f"got {type(fill_na).__name__}."
            )
        if not np.isfinite(fill_na):
            raise ValueError(f"fill_na must be finite; got {fill_na!r}.")

    # -- top-level NaN guard when fill_na is None. fill_na substitution
    # is applied later in the public function body, so a fill_na value
    # makes NaNs legitimate input; only flag them when no fill is set.
    if fill_na is None:
        mat = adata.X if layer is None else adata.layers[layer]
        mat_arr = np.asarray(mat)
        if var_subset is not None:
            col_idx = adata.var_names.get_indexer(var_subset)
            mat_arr = mat_arr[:, col_idx]
        if np.isnan(mat_arr).any():
            src = "adata.X" if layer is None else f"adata.layers[{layer!r}]"
            raise ValueError(
                f"{src} contains NaN values but fill_na is None. "
                "Either pass fill_na=<value> to replace NaNs with "
                "a constant prior to correlation, or preprocess "
                "adata (e.g. via proteopy.pp.impute_*) to remove "
                "NaNs before calling pairwise_var_correlations."
            )

    # -- validate Fisher-pooling thresholds. Booleans subclass int in
    # Python, so isinstance(True, int) is True; reject them explicitly
    # to match fill_na's behaviour and avoid silent True->1 coercion.
    if (
        isinstance(min_contrib_batches, bool)
        or not isinstance(min_contrib_batches, int)
        or min_contrib_batches < 1
    ):
        raise ValueError(
            "min_contrib_batches must be an int >= 1; got "
            f"{min_contrib_batches!r}"
        )
    if (
        isinstance(min_wsum, bool)
        or not isinstance(min_wsum, (int, float))
        or min_wsum < 0
    ):
        raise ValueError(
            f"min_wsum must be a non-negative number; got " f"{min_wsum!r}"
        )

    # -- resolve key_added (build default or validate provided)
    if key_added is None:
        if var_subset is not None:
            var_hash = md5_hash_list(var_subset, n_chars=7)
        else:
            var_hash = ""
        resolved_key_added = (
            "pairwise_correlations"
            f";{group_by or ''}"
            f";{batch_key or ''}"
            f";{layer or ''}"
            f";{var_hash}"
        )
    else:
        if not isinstance(key_added, str) or not key_added:
            raise ValueError("key_added must be a non-empty string or None.")
        resolved_key_added = key_added

    return resolved_key_added


# pylint: enable=too-many-branches


def _pairwise_var_correlations(
    df: pd.DataFrame,
    *,
    method: str = "pearson",
) -> pd.DataFrame:
    """
    Compute pairwise correlations between columns of an obs x vars frame.

    Emits the upper triangle of the correlation matrix (excluding the
    diagonal). For ``n_vars < 2`` no pairs exist and the function
    returns an empty frame with the documented column schema; the
    validator in :func:`pairwise_var_correlations` rejects this case
    upstream, so in normal use the helper sees ``n_vars >= 2``.

    Parameters
    ----------
    df : pandas.DataFrame
        Obs x vars frame. Index is observations, columns are variables
        (e.g. peptides for a single protein, or proteins).
    method : str
        Correlation method, one of ``{"pearson", "spearman"}``.

    Returns
    -------
    pandas.DataFrame
        Flat frame with columns ``["varA", "varB", "corr", "pval"]``.
        Empty when ``n_vars < 2``.

    Raises
    ------
    ValueError
        If ``method`` is not recognised, if ``df`` contains NaNs, if
        ``n_obs < 3``, or if any column has variance below
        ``1e-12`` (effectively constant).
    """
    # -- input validation
    if method not in {"pearson", "spearman"}:
        raise ValueError(
            f"method must be 'pearson' or 'spearman'; got {method!r}"
        )
    if df.isna().any().any():
        raise ValueError(
            "Input frame contains NaNs; impute or drop them first."
        )

    n_obs, _ = df.shape

    if n_obs < 3:
        raise ValueError(
            f"Need at least 3 observations to compute correlations; "
            f"got n_obs={n_obs}."
        )
    # -- tolerance-based zero-variance check. Strict equality lets
    # near-constant columns (var ~ 1e-20) through, and pearsonr then
    # returns NaN which silently poisons Fisher pooling downstream.
    zero_var_eps = 1e-12
    variances = df.var(axis=0, ddof=0)
    low_var = variances[variances < zero_var_eps]
    if not low_var.empty:
        preview_items = list(low_var.head(10).items())
        preview = ", ".join(
            f"{col!r}: var={val:.3g}" for col, val in preview_items
        )
        suffix = " ..." if len(low_var) > 10 else ""
        raise ValueError(
            f"{len(low_var)} column(s) have variance below "
            f"{zero_var_eps:g} and cannot be correlated:\n"
            f"  {preview}{suffix}\n"
            "Preprocess the data to drop or impute these columns "
            "before calling pairwise_var_correlations."
        )

    # -- upper-triangle enumeration (excluding diagonal)
    corr_fn = stats.pearsonr if method == "pearson" else stats.spearmanr
    cols = df.columns.tolist()
    rows = []
    for a, b in itertools.combinations(cols, 2):
        res = corr_fn(df[a].to_numpy(), df[b].to_numpy())
        # scipy returns (statistic, pvalue) for both pearsonr and
        # spearmanr; recent scipy versions return a result object
        # with .statistic / .pvalue attributes -- tuple-unpacking
        # still works.
        r, p = float(res[0]), float(res[1])
        rows.append((a, b, r, p))

    return pd.DataFrame(rows, columns=["varA", "varB", "corr", "pval"])


def _iter_groups(
    X_df: pd.DataFrame,
    var: pd.DataFrame,
    *,
    group_col: str | None,
):
    """
    Yield ``(group_id, sub_df)`` pairs.

    If ``group_col`` is None, yield a single ``(None, X_df)``.
    Otherwise, group ``var`` by ``group_col`` and yield, for
    each group, the column-subset of ``X_df`` restricted to
    that group's vars.
    """
    if group_col is None:
        yield None, X_df
        return

    for gid, sub_var in var.groupby(group_col, observed=True, sort=False):
        cols = sub_var.index.tolist()
        yield gid, X_df.loc[:, cols]


def _compute_pooled_nonbatched(
    X_df: pd.DataFrame,
    var: pd.DataFrame,
    *,
    group_col: str | None,
    method: str,
) -> pd.DataFrame:
    """
    Compute pairwise correlations without batch pooling.

    Returns
    -------
    pandas.DataFrame
        If ``group_col`` is None: columns
        ``["varA", "varB", "corr", "pval"]``.
        Otherwise: columns
        ``["group_id", "varA", "varB", "corr", "pval"]``.
    """
    corr_dfs = []
    for gid, sub_df in _iter_groups(X_df, var, group_col=group_col):
        out = _pairwise_var_correlations(sub_df, method=method)
        if group_col is not None:
            out = out.copy()
            out.insert(0, "group_id", gid)
        corr_dfs.append(out)

    if not corr_dfs:
        cols = (
            ["group_id", "varA", "varB", "corr", "pval"]
            if group_col is not None
            else ["varA", "varB", "corr", "pval"]
        )
        return pd.DataFrame(columns=cols)

    return pd.concat(corr_dfs, axis=0, ignore_index=True)


def _compute_pooled_batched(
    X_df: pd.DataFrame,
    var: pd.DataFrame,
    obs_batch: pd.Series,
    *,
    group_col: str | None,
    method: str,
    min_contrib_batches: int,
    min_wsum: float,
    verbose: bool,
) -> pd.DataFrame:
    """
    Compute Fisher-pooled pairwise correlations across batches.

    Returns
    -------
    pandas.DataFrame
        If ``group_col`` is None: columns
        ``["varA", "varB", "corr", "pval", "var_z_between"]``.
        Otherwise: columns
        ``["group_id", "varA", "varB", "corr", "pval",
        "var_z_between"]``.
        ``pval`` is always NaN in the batched output.
    """
    out_cols = (
        ["group_id", "varA", "varB", "corr", "pval", "var_z_between"]
        if group_col is not None
        else ["varA", "varB", "corr", "pval", "var_z_between"]
    )

    corr_dfs, batch_sizes = _compute_per_batch_correlations(
        X_df,
        var,
        obs_batch,
        group_col=group_col,
        method=method,
        verbose=verbose,
    )

    # -- global preconditions: every (varA, varB) pair is pooled
    # across the same set of eligible batches (n_obs >= 4 + no
    # zero-variance columns), so min_contrib_batches and min_wsum
    # are enforced once here as global thresholds, not per pair.
    eligible_batches = {b: n for b, n in batch_sizes.items() if n >= 4}
    wsum_eligible = sum(max(n - 3.0, 0.0) for n in eligible_batches.values())
    if len(eligible_batches) < min_contrib_batches:
        raise ValueError(
            f"Only {len(eligible_batches)} batch(es) have "
            f"n_obs >= 4, but min_contrib_batches="
            f"{min_contrib_batches} requires at least that many. "
            f"Batch sizes: {batch_sizes}"
        )
    if wsum_eligible < min_wsum:
        raise ValueError(
            f"Sum of (n_b - 3) over eligible batches is "
            f"{wsum_eligible}, but min_wsum={min_wsum} requires "
            f"at least that much. Eligible batch sizes: "
            f"{eligible_batches}"
        )
    if verbose:
        print(
            f"pairwise_var_correlations: pooling over "
            f"{len(eligible_batches)} eligible batches "
            f"({eligible_batches})"
        )

    if not corr_dfs:
        return pd.DataFrame(columns=out_cols)

    cross_batch_df = pd.concat(corr_dfs, axis=0, ignore_index=True)
    # Per-batch inverse-variance weights for Fisher z meta-analysis.
    # Fisher z = arctanh(r) has asymptotic Var(z_b) ~ 1 / (n_b - 3),
    # so the inverse-variance weight is w_b = n_b - 3.
    batch_weights = {b: max(n - 3.0, 0.0) for b, n in batch_sizes.items()}
    group_keys = (
        ["group_id", "varA", "varB"]
        if group_col is not None
        else ["varA", "varB"]
    )

    rows = []
    for keys, gdf in cross_batch_df.groupby(
        group_keys, observed=True, sort=False
    ):
        rhat, var_z_between = _pool_one_pair_fisher(gdf, batch_weights)
        if group_col is not None:
            gid, va, vb = keys
            rows.append((gid, va, vb, rhat, np.nan, var_z_between))
        else:
            va, vb = keys
            rows.append((va, vb, rhat, np.nan, var_z_between))

    if not rows:
        return pd.DataFrame(columns=out_cols)
    return pd.DataFrame(rows, columns=out_cols)


def _compute_per_batch_correlations(
    X_df: pd.DataFrame,
    var: pd.DataFrame,
    obs_batch: pd.Series,
    *,
    group_col: str | None,
    method: str,
    verbose: bool,
) -> tuple[list[pd.DataFrame], dict]:
    """
    Compute pairwise correlations independently within each batch.

    Iterates over batches defined by ``obs_batch``; for each batch
    with ``n_obs >= 4`` it restricts ``X_df`` to that batch's
    observations and computes per-group pairwise correlations via
    ``_iter_groups`` + ``_pairwise_var_correlations``. Batches with
    ``n_obs < 4`` are skipped (a notice is printed if ``verbose``
    is True), but their sample count is still recorded in
    ``batch_sizes`` so callers can report what was excluded. The
    ``n_obs < 4`` floor matches the Fisher-pooling weight formula
    ``w_b = max(n_b - 3, 0)``: smaller batches would contribute
    zero weight downstream anyway.

    No pooling is performed here -- the per-batch frames are handed
    to ``_compute_pooled_batched`` for Fisher-space aggregation.

    Returns
    -------
    corr_dfs : list[pandas.DataFrame]
        One frame per ``(batch, group)`` combination with non-empty
        results. Columns: ``["varA", "varB", "corr", "pval",
        "batch_id"]``, with a leading ``"group_id"`` column when
        ``group_col`` is not None.
    batch_sizes : dict
        Maps each batch id to its observation count, including
        batches that were skipped.
    """
    corr_dfs: list[pd.DataFrame] = []
    batch_sizes: dict = {}

    for batch_id, obs_positions in obs_batch.groupby(
        obs_batch, observed=True, sort=False
    ):
        n_b = len(obs_positions)
        batch_sizes[batch_id] = n_b
        # Fisher pooling weight is max(n_b - 3, 0), so batches with
        # n_b < 4 contribute zero weight regardless. Skip them up
        # front to avoid wasted correlation computation.
        if n_b < 4:
            if verbose:
                print(f"Skipping batch={batch_id}: n_obs={n_b} < 4")
            continue

        df_b = X_df.loc[obs_positions.index, :]

        for gid, sub_df in _iter_groups(df_b, var, group_col=group_col):
            try:
                out = _pairwise_var_correlations(sub_df, method=method)
            except ValueError as e:
                context = f"In batch {batch_id!r}"
                if group_col is not None:
                    context += f", group {gid!r}"
                raise ValueError(f"{context}: {e}") from e
            if out.empty:
                continue
            out = out.copy()
            out["batch_id"] = batch_id
            if group_col is not None:
                out.insert(0, "group_id", gid)
            corr_dfs.append(out)

    return corr_dfs, batch_sizes


def _pool_one_pair_fisher(
    gdf: pd.DataFrame,
    batch_weights: dict,
) -> tuple[float, float]:
    """
    Fisher-pool a single (group, varA, varB) row group.

    Under the contract enforced by
    :func:`_compute_pooled_batched` (every eligible batch contributes
    a row for every pair; ``min_contrib_batches`` / ``min_wsum`` are
    global preconditions), all batches in ``gdf`` have positive
    weight, so this function unconditionally returns a pooled
    estimate.

    Returns
    -------
    tuple
        ``(rhat, var_z_between)``.
    """
    r = gdf["corr"].to_numpy(dtype=float)
    bids = gdf["batch_id"].to_numpy()
    # Clip strict +/-1 correlations so arctanh stays finite.
    r = np.clip(r, -0.999999, 0.999999)
    z = np.arctanh(r)
    # Per-batch weights w_b are inverse variances of Fisher z which
    # simplifies to w_b = n_b - 3
    w = np.array(
        [batch_weights.get(b, 0.0) for b in bids],
        dtype=float,
    )
    wsum = float(w.sum())
    zbar_fe = float((w * z).sum() / wsum)
    # Cochran's Q: weighted dispersion of per-batch z's
    q_stat = float((w * (z - zbar_fe) ** 2).sum())
    # heterogeneity penalty: estimator of between-batch variance of z
    var_z_between = q_stat / wsum
    rhat = float(np.tanh(zbar_fe - var_z_between))
    return rhat, var_z_between


def pairwise_var_correlations(
    adata: ad.AnnData,
    *,
    group_by: str | None = None,
    batch_key: str | None = None,
    layer: str | None = None,
    method: str = "pearson",
    var_subset=None,
    fill_na: float | int | None = None,
    min_contrib_batches: int = 1,
    min_wsum: float = 1.0,
    key_added: str | None = None,
    inplace: bool = True,
    verbose: bool = False,
) -> ad.AnnData | None:
    """
    Compute pairwise correlations between vars, optionally pooled
    across batches via Fisher z-transformation.

    Upper-triangle pairwise correlations are computed between
    columns of ``adata.X`` (or ``adata.layers[layer]``). When
    ``group_by`` is set, pairs are restricted to within-group
    vars (e.g. peptides within the same protein); each group must
    contain at least 2 vars. When ``batch_key`` is set,
    per-batch correlations are pooled using inverse-variance
    weighted Fisher z-transform; batches with fewer than 4
    observations are excluded as ineligible.

    Missing values in the input matrix raise an error unless
    ``fill_na`` is set.

    Results are stored in ``adata.uns[key_added]``.

    Parameters
    ----------
    adata : AnnData
        Proteodata-conforming :class:`~anndata.AnnData` (see
        :func:`proteopy.utils.anndata.is_proteodata`).
    group_by : str | None, optional
        Column in ``.var`` to restrict pairing to within-group
        vars (e.g. ``"protein_id"`` for COPF). Each group must
        have >= 2 vars.
    batch_key : str | None, optional
        Column in ``.obs`` identifying batches. When set,
        per-batch correlations are pooled using Fisher
        z-transform meta-analysis with inverse-variance weights
        ``w_b = max(n_b - 3, 0)``.
    layer : str | None, optional
        Key in ``adata.layers``; when set, uses that layer
        instead of ``.X``.
    method : str, optional
        Correlation method: ``"pearson"`` or ``"spearman"``.
    var_subset : list[str] | str | None, optional
        Subset of ``adata.var_names`` to restrict the
        correlation to.
    fill_na : float | int | None, optional
        If set, NAs in the AnnData X matrix are replaced with
        this value prior to correlation.
    min_contrib_batches : int, optional
        Minimum number of eligible batches (``n_obs >= 4``)
        required globally. Raises ``ValueError`` if not met.
        Applied once to the whole call, not per pair.
    min_wsum : float, optional
        Minimum total inverse-variance weight
        ``sum_b max(n_b - 3, 0)``.
    key_added : str | None, optional
        Key in ``adata.uns`` to store results. When ``None``,
        defaults to
        ``"pairwise_correlations;<group_by>;<batch_key>;``
        ``<layer>;<var_subset_md5>"``.
    inplace : bool, optional
        If True, modify ``adata`` in place; else return a copy.
    verbose : bool, optional
        If True, print status messages about batches used and
        where results are stored.

    Returns
    -------
    AnnData | None
        ``None`` when ``inplace=True``. A new
        :class:`~anndata.AnnData` when ``inplace=False``.

        The result DataFrame is stored in ``adata.uns[key_added]``.

        Non-batched (``batch_key`` is ``None``): columns
        ``["varA", "varB", "corr", "pval"]``. When ``group_by``
        is set, a leading ``"group_id"`` column is prepended.

        Batched (``batch_key`` is not ``None``): columns
        ``["varA", "varB", "corr", "pval", "var_z_between"]``
        where ``pval`` is always ``NaN`` and ``var_z_between``
        is the Cochran Q-derived between-batch heterogeneity.
        When ``group_by`` is set, a leading ``"group_id"``
        column is prepended.

    Raises
    ------
    ValueError
        If ``adata`` does not conform to ProteoPy assumptions,
        ``group_by`` or ``batch_key`` columns are missing or
        malformed, any group has fewer than 2 vars, NaNs are
        present and ``fill_na`` is ``None``, ``method`` is
        invalid, a column has zero variance, or ``batch_key``
        is set and the global ``min_contrib_batches`` /
        ``min_wsum`` preconditions are not satisfied.

    Notes
    -----
    Batch pooling uses Fisher z-transform meta-analysis. The
    per-batch correlation ``r_b`` is mapped to
    ``z_b = arctanh(r_b)``, whose asymptotic variance is
    ``Var(z_b) ~ 1 / (n_b - 3)``. The inverse-variance weight
    is therefore ``w_b = max(n_b - 3, 0)``, and only batches
    with ``n_b >= 4`` contribute (``w_b >= 1``).

    The total precision of the pooled estimate is

        wsum = sum_{b : n_b >= 4} (n_b - 3)

    and ``min_wsum`` is the floor below which pooling is
    refused. Example: three eligible batches of sizes
    ``n_b = 4, 5, 6`` yield weights ``w_b = 1, 2, 3`` and
    ``wsum = 6``. ``min_wsum=6`` passes; ``min_wsum=7`` raises.

    Examples
    --------
    Build a small peptide-level dataset: one protein (``A``) with
    five peptides forming two within-protein correlation clusters
    -- ``pep1``/``pep2`` and ``pep3``/``pep4``/``pep5`` -- across
    two batches.

    >>> import numpy as np
    >>> import pandas as pd
    >>> import anndata as ad
    >>> import proteopy as pr
    >>> X = np.array([
    ...     [ 1,  1,  1,  2,  1],
    ...     [ 2,  3,  4,  4,  5],
    ...     [ 3,  2,  5,  5,  5],
    ...     [ 4,  4,  5,  5,  5],
    ...     [ 5,  6,  4,  4,  5],
    ...     [ 6,  5,  1,  2,  1],
    ...     [ 7,  7,  1,  2,  1],
    ...     [ 8,  9,  4,  4,  5],
    ...     [ 9,  8,  5,  6,  6],
    ...     [10, 11,  5,  6,  6],
    ...     [11, 10,  4,  4,  5],
    ...     [12, 12,  1,  2,  1],
    ... ], dtype=float)
    >>> obs = pd.DataFrame(
    ...     {"sample_id": [f"S{i}" for i in range(12)],
    ...      "batch": ["b1"] * 6 + ["b2"] * 6},
    ...     index=[f"S{i}" for i in range(12)],
    ... )
    >>> var = pd.DataFrame(
    ...     {"peptide_id": [f"pep{i}" for i in range(1, 6)],
    ...      "protein_id": ["A"] * 5},
    ...     index=[f"pep{i}" for i in range(1, 6)],
    ... )
    >>> adata = ad.AnnData(X=X, obs=obs, var=var)

    Apply canonical preprocessing (no effect on this dataset --
    the protein already has 5 peptides and no var has zero
    variance):

    >>> pr.pp.filter_proteins_by_peptide_count(adata, min_count=4)
    Removed 0 proteins and 0 peptides.
    >>> pr.pp.remove_zero_variance_vars(adata)

    Within-protein pairwise peptide correlations:

    >>> adata_grouped = pr.tl.pairwise_var_correlations(
    ...     adata,
    ...     group_by="protein_id",
    ...     inplace=False,
    ... )
    >>> key_g = "pairwise_correlations;protein_id;;;"
    >>> print(adata_grouped.uns[key_g].round(3))
      group_id  varA  varB   corr   pval
    0        A  pep1  pep2  0.972  0.000
    1        A  pep1  pep3  0.000  1.000
    2        A  pep1  pep4  0.099  0.759
    3        A  pep1  pep5  0.071  0.826
    4        A  pep2  pep3  0.028  0.930
    5        A  pep2  pep4  0.116  0.721
    6        A  pep2  pep5  0.119  0.713
    7        A  pep3  pep4  0.961  0.000
    8        A  pep3  pep5  0.980  0.000
    9        A  pep4  pep5  0.943  0.000

    Fisher-pooled correlations across batches (>= 2 contributing
    batches required):

    >>> adata_batched = pr.tl.pairwise_var_correlations(
    ...     adata,
    ...     group_by="protein_id",
    ...     batch_key="batch",
    ...     min_contrib_batches=2,
    ...     inplace=False,
    ... )
    >>> key_b = "pairwise_correlations;protein_id;batch;;"
    >>> print(adata_batched.uns[key_b].round(3))
      group_id  varA  varB   corr  pval  var_z_between
    0        A  pep1  pep2  0.886   NaN          0.000
    1        A  pep1  pep3  0.000   NaN          0.000
    2        A  pep1  pep4  0.000   NaN          0.000
    3        A  pep1  pep5  0.000   NaN          0.000
    4        A  pep2  pep3  0.054   NaN          0.003
    5        A  pep2  pep4  0.038   NaN          0.002
    6        A  pep2  pep5  0.094   NaN          0.011
    7        A  pep3  pep4  0.976   NaN          0.322
    8        A  pep3  pep5  0.979   NaN          0.590
    9        A  pep4  pep5  0.945   NaN          0.000

    Ungrouped: correlate all vars pairwise (same pair set as
    grouped, since there is only one protein):

    >>> adata_ungrouped = pr.tl.pairwise_var_correlations(
    ...     adata,
    ...     inplace=False,
    ... )
    >>> key_u = "pairwise_correlations;;;;"
    >>> print(adata_ungrouped.uns[key_u].round(3))
       varA  varB   corr   pval
    0  pep1  pep2  0.972  0.000
    1  pep1  pep3  0.000  1.000
    2  pep1  pep4  0.099  0.759
    3  pep1  pep5  0.071  0.826
    4  pep2  pep3  0.028  0.930
    5  pep2  pep4  0.116  0.721
    6  pep2  pep5  0.119  0.713
    7  pep3  pep4  0.961  0.000
    8  pep3  pep5  0.980  0.000
    9  pep4  pep5  0.943  0.000
    """
    # -- resolve var_subset to an ordered list of names (or None)
    resolved_var_subset = _resolve_var_subset(adata, var_subset)

    # -- validate
    resolved_key = _validate_pairwise_var_correlations_params(
        adata,
        group_by=group_by,
        batch_key=batch_key,
        layer=layer,
        method=method,
        var_subset=resolved_var_subset,
        fill_na=fill_na,
        min_contrib_batches=min_contrib_batches,
        min_wsum=min_wsum,
        key_added=key_added,
        inplace=inplace,
    )

    # -- build obs x vars frame and var lookup
    # INVARIANT: column order is fixed here once. Every batch reuses
    # this `var`, so itertools.combinations downstream emits identical
    # (varA, varB) tuples across batches -- required for Fisher
    # pooling to align rows by pair. Don't re-derive cols per batch.
    # When var_subset is applied below, the resolver has already put
    # it in adata.var_names order so the invariant still holds.
    mat = adata.X if layer is None else adata.layers[layer]
    X_df = pd.DataFrame(
        np.asarray(mat),
        index=adata.obs_names.copy(),
        columns=adata.var_names.copy(),
    )
    var = adata.var.copy()

    if resolved_var_subset is not None:
        X_df = X_df.loc[:, resolved_var_subset]
        var = var.loc[resolved_var_subset, :]

    # -- replace NAs with fill_na if requested. Validator guarantees
    # the matrix is NaN-free when fill_na is None, so this branch only
    # fires for the explicit-fill path.
    if fill_na is not None:
        X_df = X_df.fillna(fill_na)

    # -- dispatch on batch_key
    if batch_key is None:
        result = _compute_pooled_nonbatched(
            X_df,
            var,
            group_col=group_by,
            method=method,
        )
    else:
        obs_batch = adata.obs[batch_key]
        result = _compute_pooled_batched(
            X_df,
            var,
            obs_batch,
            group_col=group_by,
            method=method,
            min_contrib_batches=min_contrib_batches,
            min_wsum=min_wsum,
            verbose=verbose,
        )

    if verbose:
        print(
            f"pairwise_var_correlations: stored result under "
            f".uns['{resolved_key}'] "
            f"(group_by={group_by}, "
            f"batch_key={batch_key}, layer={layer}, "
            f"n_rows={len(result)})"
        )

    # -- in-place vs return
    if inplace:
        adata.uns[resolved_key] = result
        check_proteodata(
            adata,
            layers=[layer] if layer is not None else None,
        )
        return None

    adata_out = adata.copy()
    adata_out.uns[resolved_key] = result
    check_proteodata(
        adata_out,
        layers=[layer] if layer is not None else None,
    )
    return adata_out


def pairwise_peptide_correlations(
    adata: ad.AnnData,
    *,
    protein_id: str | None = "protein_id",
    batch_key: str | None = None,
    layer: str | None = None,
    method: str = "pearson",
    var_subset=None,
    fill_na: float | int | None = None,
    min_contrib_batches: int = 1,
    min_wsum: float = 1.0,
    key_added: str | None = None,
    inplace: bool = True,
    verbose: bool = False,
) -> ad.AnnData | None:
    """
    Compute pairwise peptide correlations within proteins,
    optionally pooled across batches via Fisher z-transformation.

    Upper-triangle pairwise peptide correlations are computed
    between columns of ``adata.X`` (or ``adata.layers[layer]``),
    restricted to peptide pairs mapping to the same protein. Each
    protein must contain at least 2 peptides. When ``batch_key``
    is set, per-batch correlations are pooled using
    inverse-variance weighted Fisher z-transform; batches with
    fewer than 4 observations are excluded as ineligible.

    Missing values in the input matrix raise an error unless
    ``fill_na`` is set.

    Results are stored in ``adata.uns[key_added]``.

    Parameters
    ----------
    adata : AnnData
        Peptide-level proteodata-conforming
        :class:`~anndata.AnnData` (see
        :func:`proteopy.utils.anndata.is_proteodata`). Must have
        ``.var['peptide_id']`` and ``.var['protein_id']``.
    protein_id : str | None, optional
        Column in ``.var`` mapping peptides to proteins. Pairs are
        restricted to within-protein peptides. Each protein must
        have >= 2 peptides. When ``None``, falls back to the
        ungrouped behaviour of
        :func:`pairwise_var_correlations`.
    batch_key : str | None, optional
        Column in ``.obs`` identifying batches. When set,
        per-batch correlations are pooled using Fisher
        z-transform meta-analysis with inverse-variance weights
        ``w_b = max(n_b - 3, 0)``.
    layer : str | None, optional
        Key in ``adata.layers``; when set, uses that layer
        instead of ``.X``.
    method : str, optional
        Correlation method: ``"pearson"`` or ``"spearman"``.
    var_subset : list[str] | str | None, optional
        Subset of ``adata.var_names`` (peptide IDs) to restrict
        the correlation to.
    fill_na : float | int | None, optional
        If set, NAs in the AnnData X matrix are replaced with
        this value prior to correlation.
    min_contrib_batches : int, optional
        Minimum number of eligible batches (``n_obs >= 4``)
        required globally. Raises ``ValueError`` if not met.
        Applied once to the whole call, not per pair.
    min_wsum : float, optional
        Minimum total inverse-variance weight
        ``sum_b max(n_b - 3, 0)``.
    key_added : str | None, optional
        Key in ``adata.uns`` to store results. When ``None``,
        defaults to
        ``"pairwise_correlations;<protein_id>;<batch_key>;``
        ``<layer>;<var_subset_md5>"``.
    inplace : bool, optional
        If True, modify ``adata`` in place; else return a copy.
    verbose : bool, optional
        If True, print status messages about batches used and
        where results are stored.

    Returns
    -------
    AnnData | None
        ``None`` when ``inplace=True``. A new
        :class:`~anndata.AnnData` when ``inplace=False``.

        The result DataFrame is stored in ``adata.uns[key_added]``.

        Non-batched (``batch_key`` is ``None``): columns
        ``["group_id", "varA", "varB", "corr", "pval"]`` where
        ``group_id`` is the protein.

        Batched (``batch_key`` is not ``None``): columns
        ``["group_id", "varA", "varB", "corr", "pval", ``
        ``"var_z_between"]`` where ``pval`` is always ``NaN`` and
        ``var_z_between`` is the Cochran Q-derived between-batch
        heterogeneity.

    Raises
    ------
    ValueError
        If ``adata`` is not peptide-level proteodata,
        ``protein_id`` or ``batch_key`` columns are missing or
        malformed, any protein has fewer than 2 peptides, NaNs are
        present and ``fill_na`` is ``None``, ``method`` is
        invalid, a column has zero variance, or ``batch_key``
        is set and the global ``min_contrib_batches`` /
        ``min_wsum`` preconditions are not satisfied.

    Notes
    -----
    Batch pooling uses Fisher z-transform meta-analysis. The
    per-batch correlation ``r_b`` is mapped to
    ``z_b = arctanh(r_b)``, whose asymptotic variance is
    ``Var(z_b) ~ 1 / (n_b - 3)``. The inverse-variance weight
    is therefore ``w_b = max(n_b - 3, 0)``, and only batches
    with ``n_b >= 4`` contribute (``w_b >= 1``).

    The total precision of the pooled estimate is

        wsum = sum_{b : n_b >= 4} (n_b - 3)

    and ``min_wsum`` is the floor below which pooling is
    refused. Example: three eligible batches of sizes
    ``n_b = 4, 5, 6`` yield weights ``w_b = 1, 2, 3`` and
    ``wsum = 6``. ``min_wsum=6`` passes; ``min_wsum=7`` raises.

    Examples
    --------
    Build a small peptide-level dataset: one protein (``A``) with
    five peptides forming two within-protein correlation clusters
    -- ``pep1``/``pep2`` and ``pep3``/``pep4``/``pep5`` -- across
    two batches.

    >>> import numpy as np
    >>> import pandas as pd
    >>> import anndata as ad
    >>> import proteopy as pr
    >>> X = np.array([
    ...     [ 1,  1,  1,  2,  1],
    ...     [ 2,  3,  4,  4,  5],
    ...     [ 3,  2,  5,  5,  5],
    ...     [ 4,  4,  5,  5,  5],
    ...     [ 5,  6,  4,  4,  5],
    ...     [ 6,  5,  1,  2,  1],
    ...     [ 7,  7,  1,  2,  1],
    ...     [ 8,  9,  4,  4,  5],
    ...     [ 9,  8,  5,  6,  6],
    ...     [10, 11,  5,  6,  6],
    ...     [11, 10,  4,  4,  5],
    ...     [12, 12,  1,  2,  1],
    ... ], dtype=float)
    >>> obs = pd.DataFrame(
    ...     {"sample_id": [f"S{i}" for i in range(12)],
    ...      "batch": ["b1"] * 6 + ["b2"] * 6},
    ...     index=[f"S{i}" for i in range(12)],
    ... )
    >>> var = pd.DataFrame(
    ...     {"peptide_id": [f"pep{i}" for i in range(1, 6)],
    ...      "protein_id": ["A"] * 5},
    ...     index=[f"pep{i}" for i in range(1, 6)],
    ... )
    >>> adata = ad.AnnData(X=X, obs=obs, var=var)

    Apply canonical preprocessing (no effect on this dataset --
    the protein already has 5 peptides and no var has zero
    variance):

    >>> pr.pp.filter_proteins_by_peptide_count(adata, min_count=4)
    Removed 0 proteins and 0 peptides.
    >>> pr.pp.remove_zero_variance_vars(adata)

    Within-protein pairwise peptide correlations:

    >>> adata_grouped = pr.tl.pairwise_peptide_correlations(
    ...     adata,
    ...     inplace=False,
    ... )
    >>> key_g = "pairwise_correlations;protein_id;;;"
    >>> print(adata_grouped.uns[key_g].round(3))
      group_id  varA  varB   corr   pval
    0        A  pep1  pep2  0.972  0.000
    1        A  pep1  pep3  0.000  1.000
    2        A  pep1  pep4  0.099  0.759
    3        A  pep1  pep5  0.071  0.826
    4        A  pep2  pep3  0.028  0.930
    5        A  pep2  pep4  0.116  0.721
    6        A  pep2  pep5  0.119  0.713
    7        A  pep3  pep4  0.961  0.000
    8        A  pep3  pep5  0.980  0.000
    9        A  pep4  pep5  0.943  0.000

    Fisher-pooled correlations across batches (>= 2 contributing
    batches required):

    >>> adata_batched = pr.tl.pairwise_peptide_correlations(
    ...     adata,
    ...     batch_key="batch",
    ...     min_contrib_batches=2,
    ...     inplace=False,
    ... )
    >>> key_b = "pairwise_correlations;protein_id;batch;;"
    >>> print(adata_batched.uns[key_b].round(3))
      group_id  varA  varB   corr  pval  var_z_between
    0        A  pep1  pep2  0.886   NaN          0.000
    1        A  pep1  pep3  0.000   NaN          0.000
    2        A  pep1  pep4  0.000   NaN          0.000
    3        A  pep1  pep5  0.000   NaN          0.000
    4        A  pep2  pep3  0.054   NaN          0.003
    5        A  pep2  pep4  0.038   NaN          0.002
    6        A  pep2  pep5  0.094   NaN          0.011
    7        A  pep3  pep4  0.976   NaN          0.322
    8        A  pep3  pep5  0.979   NaN          0.590
    9        A  pep4  pep5  0.945   NaN          0.000
    """
    # -- reject non-peptide-level input up front. Without this, a
    # protein-level adata would produce singleton groups (one var
    # per protein_id) and surface as an opaque "no group has >= 2
    # vars" error from the validator.
    is_proteo, level = is_proteodata(adata)
    if not is_proteo:
        raise ValueError(
            "pairwise_peptide_correlations requires peptide-level "
            "proteodata; the input does not conform to proteodata "
            "conventions. See proteopy.utils.anndata.is_proteodata "
            "for the required structure."
        )
    if level != "peptide":
        raise ValueError(
            "pairwise_peptide_correlations requires peptide-level "
            f"proteodata; got level={level!r}. Use "
            "pairwise_var_correlations for protein-level inputs."
        )
    return pairwise_var_correlations(
        adata,
        group_by=protein_id,
        batch_key=batch_key,
        layer=layer,
        method=method,
        var_subset=var_subset,
        fill_na=fill_na,
        min_contrib_batches=min_contrib_batches,
        min_wsum=min_wsum,
        key_added=key_added,
        inplace=inplace,
        verbose=verbose,
    )


# -- Legacy
def pairwise_peptide_correlations_(
    df,
    sample_column="filename",
    peptide_column="peptide_id",
    value_column="intensity",
):
    """
    Calculate pairwise peptide correlations.
    Only outputs unique (non-symmetrical) correlations.

    Parameters:
    - df (pandas.DataFrame): The input DataFrame containing the data.
    - sample_column (str): The name of the column in `df` representing the samples.
    - peptide_column (str): The name of the column in `df` representing the peptides.
    - value_column (str): The name of the column in `df` representing the values.

    Returns:
    - result (pandas.DataFrame): A DataFrame containing the pairwise peptide
        correlations. Columns: 'pepA', 'pepB', 'PCC' (Pearson correlation coefficient).
        Only outputs unique (non-symmetrical) correlations (AB, not AB, B-A, AA, BB).
    """

    # TODO: modify df input to be obs x vars. Here we have redundant steps with
    # AnnDataTrces pairwise_peptide_correlations()
    df = df[[sample_column, peptide_column, value_column]]

    pivot_df = df.pivot_table(
        index=sample_column, columns=peptide_column, values=value_column
    )
    columns = pivot_df.columns.tolist()

    corr_dict = {}

    for col_a, col_b in itertools.combinations(columns, 2):

        pivot_col_a = pivot_df.loc[:, col_a]
        pivot_col_b = pivot_df.loc[:, col_b]
        corr_dict[col_a + "_" + col_b] = stats.pearsonr(
            pivot_col_a, pivot_col_b
        )

    corr_df = pd.DataFrame.from_dict(corr_dict, orient="index")
    corr_df.columns = ["PCC", "p-value"]
    corr_df["peptide_pair"] = corr_df.index
    corr_df[["pepA", "pepB"]] = corr_df["peptide_pair"].str.split(
        "_", expand=True
    )
    corr_df = corr_df[["pepA", "pepB", "PCC"]]
    corr_df = corr_df.reset_index(drop=True)

    return corr_df


def pairwise_peptide_correlations_legacy(
    adata,
    protein_id="protein_id",
    inplace=True,
    copy=False,
    batch_key: str | None = None,  # per-batch if provided → always pooled
    min_contrib_batches: int = 1,  # pooling threshold
    min_wsum: float = 0.0,  # pooling threshold on sum(n_b-3)
):

    if inplace and copy:
        raise ValueError("Arguments raise and copy are mutually exclusive")

    if protein_id not in adata.var.columns:
        raise ValueError(f"protein_id: {protein_id} not in .var.columns")

    STORE_KEY = "pairwise_peptide_correlations"
    PER_BATCH_STORE_KEY = "pairwise_peptide_correlations_by_batch"

    def _finalize(out, per_batch=None):
        if copy:
            adata_new = adata.copy()
            adata_new.uns[STORE_KEY] = out
            if per_batch is not None:
                adata_new.uns[PER_BATCH_STORE_KEY] = per_batch
            return adata_new
        if inplace:
            adata.uns[STORE_KEY] = out
            if per_batch is not None:
                adata.uns[PER_BATCH_STORE_KEY] = per_batch
            return
        return out

    def compute_corrs(df):
        corrs = pairwise_peptide_correlations_(
            df,
            sample_column="obs_id",
            peptide_column="var_id",
            value_column="intensity",
        )

        return corrs

    anns = adata.var[["protein_id"]].reset_index()
    traces_df = adata.to_df().T.reset_index()
    traces_df = traces_df.merge(anns, on="index")
    traces_df = traces_df.rename(columns={"index": "var_id"})

    # TODO: remove unnecessary step of melting which gets unmelted
    #   in protein-level function

    traces_df = pd.melt(
        traces_df,
        id_vars=["protein_id", "var_id"],
        var_name="obs_id",
        value_name="intensity",
    )

    if batch_key is None:
        corrs = traces_df.groupby("protein_id", observed=True).apply(
            compute_corrs, include_groups=False
        )
        corrs = corrs.droplevel(1, axis=0)
        corrs = corrs.sort_values(["pepA", "pepB"]).sort_index()
        return _finalize(corrs)

    if batch_key not in adata.obs.columns:
        raise ValueError(f"batch_key '{batch_key}' not found in .obs.columns")

    batches = (
        adata.obs[[batch_key]]
        .reset_index()
        .rename(columns={"index": "obs_id", batch_key: "batch_id"})
    )
    long = traces_df.merge(batches, on="obs_id", how="left")

    batch_sizes = adata.obs[batch_key].value_counts().to_dict()
    batch_weights = {b: max(n - 3.0, 0.0) for b, n in batch_sizes.items()}

    per_batch = long.groupby([protein_id, "batch_id"], observed=True).apply(
        compute_corrs, include_groups=False
    )

    if per_batch.empty:
        per_batch_df = pd.DataFrame(columns=["pepA", "pepB", "PCC"])
        per_batch_df.index = pd.MultiIndex.from_tuples(
            [], names=[protein_id, "batch_id"]
        )
    else:
        per_batch_df = (
            per_batch.reset_index(level=2, drop=True)
            .sort_values(["pepA", "pepB"])
            .sort_index()
        )

    # Fisher pooling across batches
    rows = []
    for prot, gprot in per_batch_df.reset_index().groupby(
        protein_id, observed=True, sort=False
    ):
        for (pa, pb), gp in gprot.groupby(
            ["pepA", "pepB"], observed=True, sort=False
        ):
            r = gp["PCC"].to_numpy(dtype=float)
            bids = gp["batch_id"].to_numpy()
            r = np.clip(r, -0.999999, 0.999999)
            z = np.arctanh(r)
            w = np.array(
                [batch_weights.get(b, 0.0) for b in bids], dtype=float
            )
            mask = w > 0
            if not np.any(mask):
                continue
            w = w[mask]
            z = z[mask]
            wsum = float(w.sum())
            if (mask.sum() >= min_contrib_batches) and (wsum >= min_wsum):
                # Fixed-effects mean (zbar_fe) and weighted between-batch variance (var_z_between)
                zbar_fe = float((w * z).sum() / wsum)
                Q = float((w * (z - zbar_fe) ** 2).sum())
                var_z_between = Q / wsum

                # Conservative PCC from fixed-effects mean (no DL): shift by var_z_between
                rhat = float(np.tanh(zbar_fe - var_z_between))

                rows.append((prot, pa, pb, rhat, var_z_between))

    if rows:
        pooled_df = (
            pd.DataFrame(
                rows,
                columns=[protein_id, "pepA", "pepB", "PCC", "var_z_between"],
            )
            .set_index(protein_id)
            .sort_values(["pepA", "pepB"])
            .sort_index()
        )
    else:
        pooled_df = pd.DataFrame(
            columns=["pepA", "pepB", "PCC", "var_z_between"]
        )
        pooled_df.index.name = protein_id

    return _finalize(pooled_df, per_batch=per_batch_df)


def peptide_dendograms_by_correlation_(
    df,
    method: str = "agglomerative-hierarchical-clustering",
):
    """
    Perform peptide clustering grouped by protein annotation.


    Parameters:
    ----------
    df : pandas.DataFrame
        Data frame with pairwise correlations annotated with the protein they belong to.]

    method : str
        Which clustering method to apply.

    Returns:
    -------
    dict
        Dictionary with clustering method output.
        - 'agglomerative-hierarchical-clustering'
            => {protein_id: {'labels': list, 'height': list, 'merge': list(list)}}
            - labels: list of peptides
            - merge: steps in which different peptides are merged.
                     n_steps == n_samples - 1
                     The two ids included for every step represent the index of the peptide in 'labels'.
            - heights: The height of each merging step in 'merge'.
                       The idx of the height corresponds to the index of the step in 'merge'.
    """

    assert all(df.index == df.columns)

    model = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=0,
        compute_distances=True,
    )

    model.fit(df)

    # pylint: disable=no-member
    dendogram = {
        "type": "sklearn_agglomerative_clustering",
        "labels": model.feature_names_in_.tolist(),
        "heights": model.distances_.tolist(),
        "merge": model.children_.tolist(),
    }
    # pylint: enable=no-member

    return dendogram


def peptide_dendograms_by_correlation(
    adata,
    method="agglomerative-hierarchical-clustering",
    inplace=True,
    copy=False,
    legacy: bool = False,
    corrs_key: str | None = None,
):
    """
    Build per-protein peptide dendograms from stored pairwise
    peptide correlations.

    Reads a long-form pairwise peptide correlation frame from
    ``adata.uns[corrs_key]``, converts each protein's correlations
    into a symmetric distance matrix (``1 - corr``), and fits an
    agglomerative hierarchical clustering model per protein. The
    resulting dendograms are stored in ``adata.uns['dendograms']``
    as a ``{protein_id: dendogram_dict}`` mapping.

    Parameters
    ----------
    adata : AnnData
        Proteodata-conforming :class:`~anndata.AnnData` carrying the
        stored correlations under ``adata.uns[corrs_key]``.
    method : str
        Clustering method to apply. Currently only
        ``'agglomerative-hierarchical-clustering'`` is supported.
    inplace : bool
        If True, store the result in ``adata.uns['dendograms']``
        and return ``None``.
    copy : bool
        If True, return a new :class:`~anndata.AnnData` with the
        result attached under ``.uns['dendograms']``. Mutually
        exclusive with ``inplace``.
    legacy : bool
        Selects which correlation-frame format to consume.
        If ``False`` (default), parse output produced by
        :func:`pairwise_peptide_correlations` (columns
        ``group_id, varA, varB, corr, ...``) via
        :func:`parse_pairwise_var_correlations_result`.
        If ``True``, parse output produced by
        :func:`pairwise_peptide_correlations_legacy` (index is
        ``protein_id``; columns ``pepA, pepB, PCC``) via
        :func:`parse_pairwise_peptide_correlations_result_legacy`.
    corrs_key : str | None
        Key in ``adata.uns`` holding the long-form correlation
        frame. In non-legacy mode, ``None`` triggers auto-inference
        via :func:`resolve_pairwise_var_correlations_key`: the call
        succeeds when exactly one matching slot is present in
        ``adata.uns`` and raises otherwise. When ``legacy=True`` and
        ``corrs_key is None``, defaults to
        ``'pairwise_peptide_correlations'``.

    Returns
    -------
    AnnData | dict | None
        ``None`` when ``inplace=True``. A new
        :class:`~anndata.AnnData` when ``copy=True``. Otherwise a
        ``{protein_id: dendogram_dict}`` dict.

    Raises
    ------
    ValueError
        If ``inplace`` and ``copy`` are both True, or if the stored
        correlation frame cannot be parsed.
    KeyError
        If ``corrs_key`` is not present in ``adata.uns``.
    """
    check_proteodata(adata)

    if inplace and copy:
        raise ValueError("Arguments inplace and copy are mutually exclusive")

    if legacy:
        resolved_key = (
            corrs_key
            if corrs_key is not None
            else "pairwise_peptide_correlations"
        )
        parser = parse_pairwise_peptide_correlations_result_legacy
    else:
        resolved_key = resolve_pairwise_var_correlations_key(
            adata, corrs_key
        )
        parser = parse_pairwise_var_correlations_result

    if resolved_key not in adata.uns:
        raise KeyError(f"corrs_key '{resolved_key}' not found in adata.uns.")

    # -- enumerate group_ids from the stored frame
    df = adata.uns[resolved_key]
    if legacy:
        group_ids = pd.Index(df.index).unique().tolist()
    else:
        if "group_id" not in df.columns:
            raise ValueError(
                f"adata.uns['{resolved_key}'] has no 'group_id' "
                "column; cannot iterate per-protein dendograms."
            )
        group_ids = df["group_id"].unique().tolist()

    dends = {}
    for group_id in group_ids:
        _, corr_sym = parser(
            adata,
            corrs_key=resolved_key,
            group_id=group_id,
        )
        corr_dists = 1 - corr_sym
        dends[group_id] = peptide_dendograms_by_correlation_(
            corr_dists,
            method=method,
        )

    if inplace:
        adata.uns["dendograms"] = dends
        check_proteodata(adata)
        return None

    if copy:
        adata_new = adata.copy()
        adata_new.uns["dendograms"] = dends
        check_proteodata(adata_new)
        return adata_new

    return dends


def peptide_clusters_from_dendograms_(
    dendogram,
    n_clusters=2,
    min_peptides_per_cluster=2,
    noise=1e6,
):
    """
    Cut clusters from cluster_peptides into N clusters with more than 1 peptide.
    """
    n_peptides = len(dendogram["labels"])
    n_real_clusters = 0
    k = n_clusters
    cluster_tree = BinaryClusterTree(constructor=dendogram)

    while n_real_clusters < n_clusters:
        clusters = cluster_tree.cut(k, use_labels=True)
        n_per_cluster = clusters.value_counts()
        is_multipep = n_per_cluster >= min_peptides_per_cluster
        n_real_clusters = is_multipep.sum()
        k += 1

        single_pep_clusters = n_per_cluster[~is_multipep].index
        clusters[clusters.isin(single_pep_clusters)] = noise

        if k >= n_peptides:
            clusters[:] = noise
            break

    # Rename cluster_ids to systematic format
    max_cluster = clusters.max()
    cats = clusters.astype("category").cat.categories
    n_clusters = len(cats)

    if max_cluster != n_clusters:
        for i in range(n_clusters):
            clusters[clusters == cats[i]] = i

    if noise in cats:
        clusters[clusters == max(clusters)] = noise

    return clusters


def peptide_clusters_from_dendograms(
    adata,
    n_clusters=2,
    min_peptides_per_cluster=2,
    noise=NOISE,
    inplace=True,
    copy=False,
):

    if inplace and copy:
        raise ValueError("Arguments raise and copy are mutually exclusive")

    if "dendograms" not in adata.uns:
        raise ValueError(f"dendograms not in .uns")

    var = adata.var.copy()
    var["cluster_id"] = np.nan

    clusters_ann = {}

    dends = adata.uns["dendograms"]
    for prot, dend in dends.items():
        dend_upd = copym.deepcopy(dend)
        dend_upd["type"] = "sklearn_agglomerative_clustering"

        clusters = peptide_clusters_from_dendograms_(
            dend_upd, n_clusters=2, min_peptides_per_cluster=2, noise=noise
        )

        mask = (var["protein_id"] == prot) & (var.index.isin(clusters.index))
        var.loc[mask, "cluster_id"] = clusters.reindex(var.index[mask])

        clusters_ann[prot] = clusters

    assert not any((var["cluster_id"] == -1).tolist())

    var["proteoform_id"] = (
        var["protein_id"].astype(str)
        + "_"
        + var["cluster_id"].astype(int).astype(str)
    )

    if inplace:
        adata.uns["clusters"] = clusters_ann
        adata.var = var

    elif copy:
        adata_new = adata.copy()
        adata_new.uns["clusters"] = clusters_ann
        return adata_new

    else:
        return clusters_ann


def proteoform_scores_(
    corrs, clusters, n_fractions, summary_func=np.mean, noise=NOISE
):
    """
    Calculates a score for proteoforms based on the difference of within
    cluster distances and between cluster distances.

    IMPORTANT: currently only implemented properly for n_clusters = 2

    Args:
        corrs (pd.DataFrame): correlation between peptides.
            In symmetrical matrix form (index == columns)
        clusters (pd.Series | pd.DataFrame): vector of cluster_ids with indexes
            corresponding to the peptides for a specific protein.
        n_fractions (int): Number of samples.
        summary_func (Callable): Summary function to apply to intra- and inter-
            cluster correlation coefficients.
    """

    def replace_upper_triangle(df, replacement, k=0):
        arr = df.to_numpy().astype(float)
        rows, cols = np.triu_indices_from(arr, k=k)
        arr[rows, cols] = replacement

        new_df = pd.DataFrame(arr, columns=df.columns, index=df.index)

        return new_df

    if isinstance(clusters, pd.DataFrame):
        clusters = clusters["cluster"]

    if np.issubdtype(clusters.dtype, np.floating):
        clusters = clusters.astype(int)

    assert any(corrs.index == corrs.columns)
    assert all([i in clusters.index for i in corrs.index]), (
        f"clusters.index = {clusters.index}" f"\ncorrs_index = {corrs.index}"
    )

    if (clusters == noise).all().all():
        return np.array([0, np.nan, np.nan, np.nan])

    cluster_ids = clusters.unique()
    cluster_ids = cluster_ids[cluster_ids != noise].tolist()

    if len(cluster_ids) > 2:

        raise ValueError(
            "Functionality with n_clusters > 2 not implemented yet."
        )

        mat = corrs.copy(deep=True)
        stat_v = []

        for c in cluster_ids:
            cluster_ids_inv = cluster_ids[cluster_ids != c]
            clust1_ids = clusters[clusters == cluster_ids_inv[0]]
            clust2_ids = clusters[clusters == cluster_ids_inv[1]]
            clust_ids_ord = clust1_ids + clust2_ids
            mat_inv = corrs.loc[clust_ids_ord, clust_ids_ord]

            cross = mat_inv.loc[
                clust1_ids, clust2_ids
            ]  # QUESTION: why no diagonal removal as below?
            values = cross.to_numpy().flatten()
            values = values[~np.isnan(values)]
            stat_across = np.apply_along_axis(summary_func, 0, cross)

            rows, cols = np.triu_indices_from(
                mat_inv, k=0
            )  # k=1 excludes diagonal
            mat_inv.to_numpy()[rows, cols] = np.nan

            within_c1 = mat_inv.loc[clust1_ids, clust1_ids]
            values = within_c1.to_numpy().flatten()
            values = values[~np.isnan(values)]
            stat_within_c1 = np.apply_along_axis(summary_func, 0, values)

            within_c2 = mat_inv.loc[clust2_ids, clust2_ids]
            values = within_c2.to_numpy().flatten()
            values = values[~np.isnan(values)]
            stat_within_c2 = np.apply_along_axis(summary_func, 0, values)

            stat_within = min([stat_within_c1, stat_within_c2])

            diff_stat = stat_within - stat_across

            z_stat_within = np.atanh(stat_within)
            z_stat_across = np.atanh(stat_across)
            z_diff_stat = z_stat_within - z_stat_across

            dz = z_diff_stat / (
                np.sqrt((1 / (n_fractions - 3)) + (1 / (n_fractions - 3)))
            )
            pval = 2 * (1 - norm.cdf(np.abs(dz)))

            stat_v.append([diff_stat, z_diff_stat, dz, pval])

        diff_stats = [i[0] for i in stat_v]
        sel_min_diff = np.which(diff_stats == diff_stats.min(skip_na=True))[0]

        return stat_v[sel_min_diff]

    else:
        clust1_ids = clusters[clusters == cluster_ids[0]].index.to_list()
        clust2_ids = clusters[clusters == cluster_ids[1]].index.to_list()
        clust_ids_ord = clust1_ids + clust2_ids
        mat = corrs.loc[clust_ids_ord, clust_ids_ord]

        # Cross-cluster statistic
        cross = corrs.loc[clust1_ids, clust2_ids]
        values = cross.to_numpy().flatten()
        stat_across = np.apply_along_axis(summary_func, 0, values)

        mat = replace_upper_triangle(mat, np.nan, k=0)

        # Within cluster statistic
        within_c1 = mat.loc[clust1_ids, clust1_ids]
        wc1_values = within_c1.to_numpy().flatten()
        wc1_values = wc1_values[~np.isnan(wc1_values)]
        stat_within_c1 = np.apply_along_axis(summary_func, 0, wc1_values)

        within_c2 = mat.loc[clust2_ids, clust2_ids]
        wc2_values = within_c2.to_numpy().flatten()
        wc2_values = wc2_values[~np.isnan(wc2_values)]
        stat_within_c2 = np.apply_along_axis(summary_func, 0, wc2_values)

        stat_within = min([stat_within_c1, stat_within_c2])

        diff_stat = stat_within - stat_across

        # Fisher's z-transformation to norm distr. and rationally scaled values
        z_stat_within = np.atanh(stat_within)
        z_stat_across = np.atanh(stat_across)
        z_diff_stat = z_stat_within - z_stat_across

        # T-test: intra-cluster peptide correlations are significantly different
        #   from cross-cluster peptide correlations
        dz = z_diff_stat / np.sqrt(
            (1 / (n_fractions - 3)) + (1 / (n_fractions - 3))
        )
        pval = 2 * (1 - norm.cdf(np.abs(dz)))

        return np.array([diff_stat, z_diff_stat, dz, pval])


def proteoform_scores(
    adata,
    *,
    min_pval_adj=None,
    min_score=None,
    summary_func=np.mean,
    noise=NOISE,
    inplace=True,
    copy=False,
    legacy: bool = False,
    corrs_key: str | None = None,
):

    check_proteodata(adata)

    if inplace and copy:
        raise ValueError("Arguments inplace and copy are mutually exclusive")

    if legacy:
        resolved_key = (
            corrs_key
            if corrs_key is not None
            else "pairwise_peptide_correlations"
        )
        parser = parse_pairwise_peptide_correlations_result_legacy
    else:
        resolved_key = resolve_pairwise_var_correlations_key(
            adata, corrs_key
        )
        parser = parse_pairwise_var_correlations_result

    if resolved_key not in adata.uns:
        raise KeyError(
            f"corrs_key '{resolved_key}' not found in adata.uns."
        )

    if "dendograms" not in adata.uns:
        raise ValueError("dendograms not in .uns")

    columns = [
        "protein_id",
        "proteoform_score",
        "proteoform_score_z",
        "proteoform_score_dz",
        "proteoform_score_pval",
    ]

    df = adata.uns[resolved_key]
    if legacy:
        group_ids = pd.Index(df.index).unique().tolist()
    else:
        if "group_id" not in df.columns:
            raise ValueError(
                f"adata.uns['{resolved_key}'] has no 'group_id' "
                "column; cannot iterate per-protein scores."
            )
        group_ids = df["group_id"].unique().tolist()

    # pylint: disable=access-member-before-definition
    var = adata.var
    # pylint: enable=access-member-before-definition
    n_fractions = adata.n_obs

    proteoform_scores_list = []

    for group_id in group_ids:

        _, corrs_mat = parser(
            adata,
            corrs_key=resolved_key,
            group_id=group_id,
        )

        clusters = var.loc[var["protein_id"] == group_id, "cluster_id"]

        scores = proteoform_scores_(
            corrs_mat, clusters, n_fractions, summary_func=summary_func
        )

        scores_entry = {
            column: value for column, value in zip(columns[1:5], scores)
        }
        scores_entry["protein_id"] = group_id
        scores_entry = pd.DataFrame([scores_entry])
        proteoform_scores_list.append(scores_entry)

    proteoform_scores = pd.concat(proteoform_scores_list, ignore_index=True)
    proteoform_scores = proteoform_scores[columns]

    # Perform multiple-testing correction

    mask_nonan = proteoform_scores["proteoform_score_pval"].notna()
    pvals = proteoform_scores.loc[mask_nonan, "proteoform_score_pval"]

    bh_alpha = min_pval_adj if min_pval_adj is not None else 0.05
    _, corrected_pvals, _, _ = multipletests(
        pvals,
        alpha=bh_alpha,
        method="fdr_bh",
    )

    proteoform_scores["proteoform_score_pval_adj"] = np.nan
    proteoform_scores["is_proteoform"] = np.nan

    proteoform_scores.loc[pvals.index, "proteoform_score_pval_adj"] = (
        corrected_pvals
    )

    if min_pval_adj is not None or min_score is not None:
        is_pf = pd.Series(True, index=pvals.index)
        if min_pval_adj is not None:
            is_pf &= corrected_pvals <= min_pval_adj
        if min_score is not None:
            scores = proteoform_scores.loc[pvals.index, "proteoform_score"]
            is_pf &= scores >= min_score
        proteoform_scores.loc[pvals.index, "is_proteoform"] = is_pf.astype(
            int
        ).values

    # --- drop existing score columns before merge (safe for re-runs) ---
    score_cols = [
        "proteoform_score",
        "proteoform_score_z",
        "proteoform_score_dz",
        "proteoform_score_pval",
        "proteoform_score_pval_adj",
        "is_proteoform",
    ]
    var = var.drop(columns=[c for c in score_cols if c in var.columns])
    # Add all new scores to .var
    var_upd = pd.merge(
        var,
        proteoform_scores,
        on="protein_id",
        how="left",
        validate="many_to_one",
    )

    var_upd = var_upd.set_index("peptide_id", drop=False)
    var_upd.index.name = None

    assert (var.index == var_upd.index).all()

    if inplace:
        adata.var = var_upd
        check_proteodata(adata)
        return None

    if copy:
        adata_new = adata.copy()
        adata_new.var = var_upd
        check_proteodata(adata_new)
        return adata_new

    return proteoform_scores
