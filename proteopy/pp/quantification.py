import re
from collections import defaultdict
from functools import partial
from typing import Callable, List, Dict, Union, Optional

import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse

from proteopy.utils.anndata import check_proteodata, is_proteodata


def _aggregate_var_value(series):
    """Collapse a var-annotation series into a single value.

    Unique non-NaN values are joined with ``';'``.
    """
    uniq = sorted(set(series.dropna().astype(str)))
    if len(uniq) == 0:
        return np.nan
    if len(uniq) == 1:
        return uniq[0]
    return ";".join(uniq)


def _rebuild_adata(adata, X_new, var_new, inplace):
    """Create or reinitialise an AnnData from aggregated data."""
    if inplace:
        adata._init_as_actual(
            X=X_new,
            obs=adata.obs.copy(),
            var=var_new,
            uns=adata.uns,
            obsm=adata.obsm,
            varm={},
            varp={},
            layers={},
            obsp=(
                adata.obsp
                if hasattr(adata, "obsp") else None
            ),
        )
        adata.var_names = var_new.index
        return None
    out = ad.AnnData(
        X=X_new,
        obs=adata.obs.copy(),
        var=var_new.copy(),
        uns=adata.uns.copy(),
        obsm=adata.obsm.copy(),
        obsp=(
            adata.obsp.copy()
            if hasattr(adata, "obsp") else None
        ),
    )
    out.var_names = var_new.index
    return out


def _apply_grouped_func(grouped, func):
    """Dispatch aggregation for *func* (str or callable)."""
    if isinstance(func, str):
        if func == "sum":
            return grouped.sum(min_count=1)
        if func == "mean":
            return grouped.mean()
        if func == "median":
            return grouped.median()
        if func == "max":
            return grouped.max()
        raise ValueError(
            "Unsupported func. Choose from "
            "'sum', 'mean', 'median', or 'max'."
        )
    if callable(func):
        result = grouped.aggregate(func)
        if isinstance(result, pd.Series):
            result = result.to_frame().T
        if not isinstance(result, pd.DataFrame):
            raise TypeError(
                "Callable `func` must return a "
                "DataFrame or Series per group."
            )
        return result
    raise TypeError(
        "`func` must be either a string "
        "identifier or a callable."
    )


def _find_root(parent, x):
    """Find root of *x* with path compression (iterative)."""
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _union_find_groups(peptides):
    """Return ``{representative: [members]}`` via union-find
    on substring containment."""
    parent = {p: p for p in peptides}
    rank = {p: 0 for p in peptides}

    peps_by_len = sorted(peptides, key=len, reverse=True)
    for i, longp in enumerate(peps_by_len):
        for shortp in peps_by_len[i + 1:]:
            if shortp in longp:
                ra = _find_root(parent, longp)
                rb = _find_root(parent, shortp)
                if ra != rb:
                    if rank[ra] < rank[rb]:
                        ra, rb = rb, ra
                    parent[rb] = ra
                    if rank[ra] == rank[rb]:
                        rank[ra] += 1

    buckets = defaultdict(list)
    for p in peptides:
        buckets[_find_root(parent, p)].append(p)

    groups = {}
    for _, members in buckets.items():
        rep = max(members, key=len)
        groups[rep] = sorted(
            members, key=lambda s: (-len(s), s),
        )
    return groups


def _group_peptides(peptides: List[str]) -> Dict[str, List[str]]:
    """
    Group peptides so that any peptide fully contained
    in another belongs to the same group.

    Returns {representative_longest: [members...]} with
    members sorted by (-len, lexicographically).
    """
    peptides = [
        p for p in
        pd.Series(peptides).dropna().astype(str).tolist()
    ]
    peptides = list(dict.fromkeys(peptides))
    return _union_find_groups(peptides)


def extract_peptide_groups(
    adata,
    peptide_col: str = "peptide_id",
    inplace: bool = True,
):
    """
    Create a new column ``adata.var['peptide_group']``
    with all overlapping (substring) peptide_ids joined
    by ``';'`` for each row in ``adata.var``.

    Parameters
    ----------
    adata : AnnData
        Must have ``adata.var[peptide_col]`` with peptide
        sequences (already normalized).
    peptide_col : str
        Column in `adata.var` containing peptide sequences.
    inplace : bool
        If True, modifies `adata` in place. If False, returns a modified copy.

    Returns
    -------
    If inplace=True: None
    If inplace=False: AnnData (modified copy)
    """
    if peptide_col not in adata.var.columns:
        raise KeyError(f"'{peptide_col}' not found in adata.var")

    peptides = adata.var[peptide_col].astype(str).tolist()
    groups = _group_peptides(peptides)

    # map each peptide to its group string
    pep_to_group = {
        p: ";".join(groups.get(rep, [rep]))
        for rep, members in groups.items()
        for p in members
    }
    group_col = (
        adata.var[peptide_col]
        .map(pep_to_group)
        .fillna(adata.var[peptide_col])
    )

    if inplace:
        adata.var["peptide_group"] = group_col.values
        return None
    else:
        adata_copy = adata.copy()
        adata_copy.var["peptide_group"] = group_col.values
        return adata_copy


def summarize_overlapping_peptides(
    adata: ad.AnnData,
    peptide_col: str = "peptide_id",
    group_by: str = "peptide_group",
    layer: str | None = None,
    func: Union[str, Callable] = "sum",
    inplace: bool = True,
):
    """
    Aggregate intensities across peptides sharing the
    same ``group_col``.

    - Sums intensities in ``adata.X`` within each group
      (NaN-aware).
    - Uses the longest peptide_id in each group as the
      new var_name.
    - Keeps both the representative peptide_id as a column
      and index.
    - Retains the grouping key as ``'peptide_group'``.
    - Concatenates differing ``.var`` annotations using
      ``';'``.

    Parameters
    ----------
    adata : AnnData
        Input AnnData with quantitative data and
        annotations.
    peptide_col : str
        Column in ``adata.var`` containing peptide
        identifiers (or will be created from var_names).
    group_by : str
        Column in ``adata.var`` specifying grouping
        (e.g. ``'peptide_group'`` or ``'proteoform_id'``).
    layer : str, optional
        Key in ``adata.layers`` specifying which matrix
        to aggregate; defaults to ``.X``.
    func : {'sum', 'mean', 'median', 'max'} or Callable
        Aggregation applied across peptides in each group.
    inplace : bool
        If True, modifies adata in place. Otherwise returns a new AnnData.

    Returns
    -------
    AnnData | None
        Aggregated AnnData if inplace=False, else modifies in place.
    """
    # --- safety checks
    if group_by not in adata.var.columns:
        raise KeyError(f"'{group_by}' not found in adata.var")
    if peptide_col not in adata.var.columns:
        # fallback: use var_names as peptide identifiers
        adata.var[peptide_col] = adata.var_names.astype(str)

    # --- matrix as DataFrame (obs × vars)
    vals = pd.DataFrame(
        adata.layers[layer] if layer is not None else adata.X,
        index=adata.obs_names,
        columns=adata.var_names,
    )

    # --- group columns and aggregate
    group_keys = adata.var[group_by].astype(str)
    grouped = vals.T.groupby(
        group_keys, sort=True, observed=True,
    )
    agg_vals = _apply_grouped_func(grouped, func).T

    # --- build new var table (aggregate annotations)
    groups = adata.var.groupby(
        group_by, sort=True, observed=True,
    )
    records, group_to_peptide = [], {}

    for gkey, df_g in groups:
        # pick longest peptide_id (tie-break lexicographic)
        longest_pep = sorted(
            df_g[peptide_col].astype(str),
            key=lambda s: (-len(s), s),
        )[0]
        group_to_peptide[str(gkey)] = longest_pep

        rec = {
            group_by: str(gkey),
            peptide_col: longest_pep,
            "n_grouped": len(df_g),
        }

        # aggregate all other var columns
        for col in adata.var.columns:
            if col in (group_by, peptide_col):
                continue
            rec[col] = _aggregate_var_value(df_g[col])
        records.append(rec)

    var_new = pd.DataFrame.from_records(records).set_index(peptide_col)

    # --- rename aggregated matrix columns from group key → longest peptide
    agg_vals.columns = [group_to_peptide[k] for k in agg_vals.columns]
    var_new = var_new.loc[agg_vals.columns]  # ensure same order

    # --- ensure 'peptide_id' column always matches index
    var_new[peptide_col] = var_new.index

    # --- rebuild AnnData
    return _rebuild_adata(
        adata, agg_vals.values, var_new, inplace,
    )


def quantify_by_category(
    adata: ad.AnnData,
    group_by: str = None,
    layer=None,
    func: Union[str, Callable] = "sum",
    inplace: bool = True,
) -> Optional[ad.AnnData]:
    """
    Aggregate intensities in ``adata.X`` (or selected
    layer) by ``.var[group_col]``, aggregate annotations
    in ``adata.var`` by concatenating unique values with
    ``';'``, and set ``group_col`` as the new index
    (``var_names``).

    Parameters
    ----------
    adata : AnnData
        Input AnnData with .X (obs x vars) and .var annotations.
    group_by : str
        Column in adata.var to group by (e.g. 'protein_id').
    layer : str, optional
        Key in ``adata.layers``; when set, quantification
        uses that layer instead of ``adata.X``.
    func : {'sum', 'median', 'max'} | Callable
        Aggregation to apply across grouped variables.
    inplace : bool
        If True, modify `adata` in place; else return a new AnnData.

    Returns
    -------
    AnnData | None
        Aggregated AnnData if inplace=False; otherwise None.
    """
    if group_by is None or group_by not in adata.var.columns:
        raise KeyError(f"'{group_by}' not found in adata.var")

    # --- Matrix as DataFrame (obs × vars)
    vals = pd.DataFrame(
        adata.layers[layer] if layer is not None else adata.X,
        index=adata.obs_names,
        columns=adata.var_names,
    )

    # --- Group columns and aggregate
    group_keys = adata.var[group_by].astype(str)
    grouped = vals.T.groupby(
        group_keys, sort=True, observed=True,
    )
    agg_vals = _apply_grouped_func(grouped, func).T

    # --- Build new var table (aggregate annotations per group)
    records = []
    for gkey, df_g in adata.var.groupby(
        group_by, sort=True, observed=True,
    ):
        rec = {group_by: str(gkey)}
        for col in adata.var.columns:
            if col == group_by:
                continue
            rec[col] = _aggregate_var_value(df_g[col])
        records.append(rec)

    var_new = pd.DataFrame.from_records(records).set_index(group_by)
    # align var rows to aggregated matrix columns
    var_new = var_new.loc[agg_vals.columns]
    var_new[group_by] = var_new.index
    var_new.index.name = None

    # --- Rebuild AnnData
    return _rebuild_adata(
        adata, agg_vals.values, var_new, inplace,
    )


quantify_proteins = partial(
    quantify_by_category,
    group_by='protein_id',
)

quantify_proteoforms = partial(
    quantify_by_category,
    group_by='proteoform_id',
)


def _validate_summarize_mods_input(
    adata, method, zero_to_na, fill_na, keep_var_cols,
):
    """Validate arguments for ``summarize_modifications``."""
    _, level = is_proteodata(adata)
    if level != "peptide":
        raise ValueError(
            "summarize_modifications requires "
            "peptide-level proteodata (adata.var "
            "must contain 'peptide_id')."
        )

    allowed = {"sum", "mean", "median", "max"}
    if method not in allowed:
        raise ValueError(
            f"method must be one of {allowed!r}, "
            f"got '{method}'."
        )
    if zero_to_na and fill_na is not None:
        raise ValueError(
            "Cannot set both zero_to_na and fill_na."
        )
    if keep_var_cols is not None:
        missing = [
            c for c in keep_var_cols
            if c not in adata.var.columns
        ]
        if missing:
            raise KeyError(
                f"keep_var_cols entries not found in "
                f"adata.var: {missing}"
            )
        _reserved = {
            "peptide_id", "protein_id",
            "n_peptidoforms", "n_modifications",
        }
        overlap = [
            c for c in keep_var_cols
            if c in _reserved
        ]
        if overlap:
            raise ValueError(
                f"keep_var_cols must not include "
                f"reserved columns: {overlap}"
            )


def _count_modifications(peptide_ids, pattern):
    """Count unique (position, text) modification sites."""
    all_mods = set()
    for pid in peptide_ids:
        removed = 0
        for m in pattern.finditer(pid):
            pos = m.start() - removed
            all_mods.add((pos, m.group()))
            removed += len(m.group())
    return len(all_mods)


def _build_var_table_mods(
    var_src, stripped, sort, pattern, keep_var_cols,
):
    """Build the new ``.var`` table for
    ``summarize_modifications``."""
    records = []
    for gkey, df_g in var_src.groupby(
        stripped, sort=sort, observed=True,
    ):
        rec = {"peptide_id": gkey}
        pids = df_g["protein_id"].unique()
        if len(pids) > 1:
            raise ValueError(
                f"Stripped sequence '{gkey}' maps to "
                f"multiple protein_ids: "
                f"{pids.tolist()}. Cannot summarize "
                f"modifications across different "
                f"proteins."
            )
        rec["protein_id"] = pids[0]
        rec["n_peptidoforms"] = len(df_g)
        rec["n_modifications"] = _count_modifications(
            df_g.index, pattern,
        )
        for col in (keep_var_cols or []):
            rec[col] = _aggregate_var_value(df_g[col])
        records.append(rec)

    var_new = pd.DataFrame.from_records(records)
    var_new = var_new.set_index("peptide_id")
    var_new.index.name = None
    var_new["peptide_id"] = var_new.index
    return var_new


def _apply_str_method(grouped, method):
    """Dispatch named aggregation *method* (string only)."""
    if method == "sum":
        return grouped.sum(min_count=1)
    if method == "mean":
        return grouped.mean()
    if method == "median":
        return grouped.median()
    return grouped.max()


def summarize_modifications(
    adata: ad.AnnData,
    mod_regex: str = r"\s*\(.*?\)",
    method: str = "sum",
    layer: str | None = None,
    zero_to_na: bool = False,
    fill_na: float | int | None = None,
    skip_na: bool = True,
    sort: bool = True,
    keep_var_cols: list[str] | None = None,
    inplace: bool = True,
    verbose: bool = False,
) -> ad.AnnData | None:
    """
    Aggregate modified peptides by their stripped sequence.

    Removes modification annotations from peptide identifiers
    using a regular expression, groups peptides sharing the
    same stripped sequence, and summarizes intensities with the
    chosen aggregation method.

    Parameters
    ----------
    adata : AnnData
        :class:`~anndata.AnnData` with peptide-level data.
    mod_regex : str, optional
        Regular expression matching modification annotations
        to strip from peptide identifiers.
    method : {'sum', 'mean', 'median', 'max'}, optional
        Aggregation applied to each group of modified
        peptides.
    layer : str, optional
        Key in ``adata.layers`` to use instead of
        ``adata.X``.
    zero_to_na : bool, optional
        If True, replace zeros with ``np.nan`` before
        aggregation.
    fill_na : float or int, optional
        Replace ``np.nan`` values with this constant before
        aggregation.
    skip_na : bool, optional
        If True, ignore ``np.nan`` during aggregation. If
        False, any ``np.nan`` in a group produces ``np.nan``
        in the result.
    sort : bool, optional
        If True, sort the output variables alphabetically
        by stripped sequence. If False, preserve the order
        of first appearance.
    keep_var_cols : list of str, optional
        Additional ``.var`` columns to carry over to the
        output. Values are aggregated per group (unique
        values joined by ``';'``). ``'peptide_id'`` and
        ``'protein_id'`` are always included.
    inplace : bool, optional
        If True, modify ``adata`` in place. Otherwise return
        a new AnnData.
    verbose : bool, optional
        Print status messages.

    Returns
    -------
    AnnData or None
        Aggregated AnnData when ``inplace=False``; otherwise
        None. Two columns are added to ``.var``:
        ``n_peptidoforms`` (total variants per stripped
        sequence) and ``n_modifications`` (unique
        modification sites, identified by their position
        in the stripped sequence and the matched
        annotation text, collected across all
        peptidoforms in the group).

    Examples
    --------
    **UniMod notation** (default ``mod_regex``).

    >>> import proteopy as pr
    >>> import numpy as np
    >>> import pandas as pd
    >>> from anndata import AnnData
    >>> pids = [
    ...     "AAADLM(UniMod:35)AYC(UniMod:4)EAHAKEDPLLTPVPASENPFR",
    ...     "AAADLMAYC(UniMod:4)EAHAKEDPLLTPVPASENPFR",
    ...     "AAADLMAYCEAHAKEDPLLTPVPASENPFR",
    ...     "VVDLMR",
    ... ]
    >>> var = pd.DataFrame(
    ...     {
    ...         "peptide_id": pids,
    ...         "protein_id": ["P1", "P1", "P1", "P2"],
    ...     },
    ...     index=pids,
    ... )
    >>> obs = pd.DataFrame(
    ...     {"sample_id": ["s1", "s2"]},
    ...     index=["s1", "s2"],
    ... )
    >>> X = np.array([
    ...     [100.0, 200.0, np.nan, 50.0],
    ...     [150.0, 100.0, 300.0,  75.0],
    ... ])
    >>> adata = AnnData(X=X, obs=obs, var=var)
    >>> result = pr.pp.summarize_modifications(
    ...     adata,
    ...     method="sum",
    ...     inplace=False,
    ... )
    >>> result.var_names.tolist()
    ['AAADLMAYCEAHAKEDPLLTPVPASENPFR', 'VVDLMR']
    >>> result.X
    array([[300.,  50.],
           [550.,  75.]])
    >>> result.var["n_peptidoforms"].tolist()
    [3, 1]
    >>> result.var["n_modifications"].tolist()
    [2, 0]

    With ``skip_na=False``, any NaN in a group propagates:

    >>> result = pr.pp.summarize_modifications(
    ...     adata,
    ...     method="sum",
    ...     skip_na=False,
    ...     inplace=False,
    ... )
    >>> result.X
    array([[ nan,  50.],
           [550.,  75.]])

    With ``method="max"``, the maximum intensity per
    group is retained:

    >>> result = pr.pp.summarize_modifications(
    ...     adata,
    ...     method="max",
    ...     inplace=False,
    ... )
    >>> result.X
    array([[200.,  50.],
           [300.,  75.]])

    **Mass-shift notation** (e.g. ``M[+16]``) requires a
    custom ``mod_regex``:

    >>> pids_ms = [
    ...     "AAADLM[+16]AYC[+57]R",
    ...     "AAADLMAYC[+57]R",
    ...     "AAADLMAYCR",
    ... ]
    >>> var_ms = pd.DataFrame(
    ...     {
    ...         "peptide_id": pids_ms,
    ...         "protein_id": ["P1", "P1", "P1"],
    ...     },
    ...     index=pids_ms,
    ... )
    >>> obs_ms = pd.DataFrame(
    ...     {"sample_id": ["s1"]},
    ...     index=["s1"],
    ... )
    >>> X_ms = np.array([[10.0, 20.0, 30.0]])
    >>> adata_ms = AnnData(
    ...     X=X_ms,
    ...     obs=obs_ms,
    ...     var=var_ms,
    ... )
    >>> result = pr.pp.summarize_modifications(
    ...     adata_ms,
    ...     mod_regex="\\[.*?\\]",
    ...     method="sum",
    ...     inplace=False,
    ... )
    >>> result.var_names.tolist()
    ['AAADLMAYCR']
    >>> result.X
    array([[60.]])
    >>> result.var["n_peptidoforms"].tolist()
    [3]
    >>> result.var["n_modifications"].tolist()
    [2]
    """
    # --- validate input
    check_proteodata(adata, layers=layer)
    _validate_summarize_mods_input(
        adata, method, zero_to_na, fill_na, keep_var_cols,
    )

    # --- extract matrix
    Xsrc = (
        adata.layers[layer] if layer is not None
        else adata.X
    )
    was_sparse = sparse.issparse(Xsrc)
    X = Xsrc.toarray() if was_sparse else np.asarray(Xsrc)
    X = X.astype(float, copy=True)

    if zero_to_na:
        X[X == 0] = np.nan
    if fill_na is not None:
        X[np.isnan(X)] = fill_na

    # --- strip modifications from peptide identifiers
    peptide_ids = adata.var["peptide_id"].astype(str).values
    try:
        pattern = re.compile(mod_regex)
    except re.error as exc:
        raise ValueError(
            f"Invalid mod_regex '{mod_regex}': {exc}"
        ) from exc
    stripped = np.array([
        pattern.sub("", pid) for pid in peptide_ids
    ])

    if verbose:
        n_unique = len(np.unique(stripped))
        print(
            f"Stripping modifications: "
            f"{len(peptide_ids)} peptides -> "
            f"{n_unique} unique stripped sequences "
            f"(method='{method}')."
        )

    # --- aggregate matrix by stripped sequence
    df = pd.DataFrame(
        X,
        index=adata.obs_names,
        columns=peptide_ids,
    )
    grouped = df.T.groupby(stripped, sort=sort)
    agg_vals = _apply_str_method(grouped, method).T

    if not skip_na:
        has_nan = df.isna().T.groupby(
            stripped, sort=sort,
        ).any().T
        agg_vals[has_nan] = np.nan

    # --- build new .var table
    var_new = _build_var_table_mods(
        adata.var.copy(), stripped, sort,
        pattern, keep_var_cols,
    )

    # --- result matrix
    X_new = agg_vals.values
    if was_sparse:
        X_new = sparse.csr_matrix(X_new)

    # --- rebuild AnnData
    result = _rebuild_adata(
        adata, X_new, var_new, inplace,
    )
    if inplace:
        check_proteodata(adata)
    else:
        check_proteodata(result)
    return result
