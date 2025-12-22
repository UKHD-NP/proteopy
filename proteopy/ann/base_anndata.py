import warnings

from anndata import AnnData
import pandas as pd

from proteopy.utils import check_proteodata


def obs(
    adata: AnnData,
    df: pd.DataFrame,
    obs_on: str,
    df_on: str,
    *,
    suffix: str = "_annotated",
    sort_obs_by_ann: bool = False,
    inplace: bool = True,
) -> AnnData | None:
    """Annotate ``adata.obs`` with rows from ``df`` matched on a key.

    Parameters
    ----------
    adata : AnnData
        Input AnnData object.
    df : pandas.DataFrame
        Annotation table that supplies additional columns.
    obs_on : str
        Name of the key column in ``adata.obs`` (or the obs index name /
        literal ``"index"``) used for matching.
    df_on : str
        Name of the key column in ``df``.
    suffix : str, optional
        Suffix applied to colliding column names from ``df``.
    sort_obs_by_ann : bool, optional
        Reorder observations by the order of matching keys in ``df``.
    inplace : bool, optional
        If ``True``, modify ``adata`` and return ``None``. Otherwise return a
        new AnnData copy.

    Returns
    -------
    AnnData or None
        Updated AnnData when ``inplace`` is ``False``; otherwise ``None``.
    """
    check_proteodata(adata)

    if df_on not in df.columns:
        raise ValueError(f"Column '{df_on}' not found in annotation dataframe.")

    adata_target = adata if inplace else adata.copy()
    obs = adata_target.obs.copy()
    obs_reset = obs.reset_index()

    index_col = obs_reset.columns[0]
    index_name = obs.index.name

    if obs_on == "index":
        merge_col = index_col
    elif obs_on in obs_reset.columns:
        merge_col = obs_on
    else:
        raise ValueError(
            f"Column '{obs_on}' not present in adata.obs or as obs index."
        )

    df_local = df.copy()
    df_local["_obs_merge_key"] = df_local[df_on].astype(str)
    obs_reset["_obs_merge_key"] = obs_reset[merge_col].astype(str)

    duplicated_mask = df_local["_obs_merge_key"].duplicated(keep=False)
    if duplicated_mask.any():
        duplicated_values = (
            df_local.loc[duplicated_mask, "_obs_merge_key"]
            .drop_duplicates()
            .tolist()
        )
        warnings.warn(
            f"{len(duplicated_values)} duplicate key(s) in '{df_on}' detected; "
            "keeping the first occurrence for each.",
            RuntimeWarning,
            stacklevel=2,
        )
        df_local = df_local.drop_duplicates(subset="_obs_merge_key", keep="first")

    obs_keys = set(obs_reset["_obs_merge_key"].tolist())
    df_keys = set(df_local["_obs_merge_key"].tolist())

    diff_df = df_keys.difference(obs_keys)
    if diff_df:
        warnings.warn(
            f"{len(diff_df)} unique value(s) in '{df_on}' were absent in '{obs_on}' "
            "and were ignored.",
            RuntimeWarning,
            stacklevel=2,
        )

    diff_obs = obs_keys.difference(df_keys)
    if diff_obs:
        warnings.warn(
            f"{len(diff_obs)} value(s) in '{obs_on}' had no match in '{df_on}' "
            "and were filled with NaN.",
            RuntimeWarning,
            stacklevel=2,
        )

    merged = pd.merge(
        obs_reset,
        df_local,
        on="_obs_merge_key",
        how="left",
        suffixes=("", suffix),
        validate="many_to_one",
    )
    merged = merged.drop(columns=["_obs_merge_key"])
    if merge_col != df_on and df_on in merged.columns:
        merged = merged.drop(columns=[df_on])

    merged = merged.set_index(index_col)
    merged.index.name = None

    if sort_obs_by_ann:
        df_order = pd.unique(df_local["_obs_merge_key"])
        key_lookup = pd.Series(
            obs_reset["_obs_merge_key"].values, index=obs_reset[index_col]
        )

        ordered_obs = []
        seen_obs = set()

        for key in df_order:
            matches = key_lookup[key_lookup == key].index
            for obs_name in matches:
                if obs_name not in seen_obs:
                    ordered_obs.append(obs_name)
                    seen_obs.add(obs_name)

        for obs_name in key_lookup.index:
            if obs_name not in seen_obs:
                ordered_obs.append(obs_name)

        order_idx = adata_target.obs.index.get_indexer(ordered_obs)
        if (order_idx < 0).any():
            raise RuntimeError(
                "Failed to align annotation order with observations."
            )
        adata_target._inplace_subset_obs(order_idx)
        merged = merged.reindex(adata_target.obs.index)

    adata_target.obs = merged
    check_proteodata(adata_target)

    if inplace:
        return None

    return adata_target


def var(
    adata: AnnData,
    df: pd.DataFrame,
    var_on: str,
    df_on: str,
    *,
    suffix: str = "_annotated",
    sort_var_by_ann: bool = False,
    inplace: bool = True,
) -> AnnData | None:
    """Annotate ``adata.var`` with rows from ``df`` matched on a key.

    Parameters
    ----------
    adata : AnnData
        Input AnnData object.
    df : pandas.DataFrame
        Annotation table that supplies additional columns.
    var_on : str
        Name of the key column in ``adata.var`` (or the var index name /
        literal ``"index"``) used for matching.
    df_on : str
        Name of the key column in ``df``.
    suffix : str, optional
        Suffix applied to colliding column names from ``df``.
    sort_var_by_ann : bool, optional
        Reorder variables by the order of matching keys in ``df``.
    inplace : bool, optional
        If ``True``, modify ``adata`` and return ``None``. Otherwise return a
        new AnnData copy.

    Returns
    -------
    AnnData or None
        Updated AnnData when ``inplace`` is ``False``; otherwise ``None``.
    """
    check_proteodata(adata)

    if df_on not in df.columns:
        raise ValueError(f"Column '{df_on}' not found in annotation dataframe.")

    adata_target = adata if inplace else adata.copy()
    var_df = adata_target.var.copy()
    var_reset = var_df.reset_index()

    index_col = var_reset.columns[0]

    if var_on == "index":
        merge_col = index_col
    elif var_on in var_reset.columns:
        merge_col = var_on
    else:
        raise ValueError(
            f"Column '{var_on}' not present in adata.var or as var index."
        )

    df_local = df.copy()
    df_local["_var_merge_key"] = df_local[df_on].astype(str)
    var_reset["_var_merge_key"] = var_reset[merge_col].astype(str)

    duplicated_mask = df_local["_var_merge_key"].duplicated(keep=False)
    if duplicated_mask.any():
        duplicated_values = (
            df_local.loc[duplicated_mask, "_var_merge_key"]
            .drop_duplicates()
            .tolist()
        )
        warnings.warn(
            f"{len(duplicated_values)} duplicate key(s) in '{df_on}' detected; "
            "keeping the first occurrence for each.",
            RuntimeWarning,
            stacklevel=2,
        )
        df_local = df_local.drop_duplicates(subset="_var_merge_key", keep="first")

    var_keys = set(var_reset["_var_merge_key"].tolist())
    df_keys = set(df_local["_var_merge_key"].tolist())

    diff_df = df_keys.difference(var_keys)
    if diff_df:
        warnings.warn(
            f"{len(diff_df)} unique value(s) in '{df_on}' were absent in '{var_on}' "
            "and were ignored.",
            RuntimeWarning,
            stacklevel=2,
        )

    diff_var = var_keys.difference(df_keys)
    if diff_var:
        warnings.warn(
            f"{len(diff_var)} value(s) in '{var_on}' had no match in '{df_on}' "
            "and were filled with NaN.",
            RuntimeWarning,
            stacklevel=2,
        )

    merged = pd.merge(
        var_reset,
        df_local,
        on="_var_merge_key",
        how="left",
        suffixes=("", suffix),
        validate="many_to_one",
    )
    merged = merged.drop(columns=["_var_merge_key"])
    if merge_col != df_on and df_on in merged.columns:
        merged = merged.drop(columns=[df_on])

    merged = merged.set_index(index_col)
    merged.index.name = None

    if sort_var_by_ann:
        df_order = pd.unique(df_local["_var_merge_key"])
        key_lookup = pd.Series(
            var_reset["_var_merge_key"].values, index=var_reset[index_col]
        )

        ordered_vars = []
        seen_vars = set()

        for key in df_order:
            matches = key_lookup[key_lookup == key].index
            for var_name in matches:
                if var_name not in seen_vars:
                    ordered_vars.append(var_name)
                    seen_vars.add(var_name)

        for var_name in key_lookup.index:
            if var_name not in seen_vars:
                ordered_vars.append(var_name)

        order_idx = adata_target.var.index.get_indexer(ordered_vars)
        if (order_idx < 0).any():
            raise RuntimeError(
                "Failed to align annotation order with variables."
            )
        adata_target._inplace_subset_var(order_idx)
        merged = merged.reindex(adata_target.var.index)

    adata_target.var = merged
    check_proteodata(adata_target)

    if inplace:
        return None

    return adata_target


def samples(
    adata: AnnData,
    df: pd.DataFrame,
    obs_on: str,
    df_on: str,
    *,
    suffix: str = "_annotated",
    sort_obs_by_ann: bool = False,
    inplace: bool = True,
) -> AnnData | None:
    """Annotate ``adata.obs`` with rows from ``df`` matched on a key.

    This function is an alias for :func:`~proteopy.ann.obs` and accepts the same
    parameters. In proteomics, observations (rows in ``adata.obs``) often
    represent samples, so this alias provides a more intuitive name for the
    same functionality.

    Parameters
    ----------
    adata : AnnData
        Input AnnData object.
    df : pandas.DataFrame
        Annotation table that supplies additional columns.
    obs_on : str
        Name of the key column in ``adata.obs`` (or the obs index name /
        literal ``"index"``) used for matching.
    df_on : str
        Name of the key column in ``df``.
    suffix : str, optional
        Suffix applied to colliding column names from ``df``.
    sort_obs_by_ann : bool, optional
        Reorder observations by the order of matching keys in ``df``.
    inplace : bool, optional
        If ``True``, modify ``adata`` and return ``None``. Otherwise return a
        new AnnData copy.

    Returns
    -------
    AnnData or None
        Updated AnnData when ``inplace`` is ``False``; otherwise ``None``.

    See Also
    --------
    obs : Original function with identical functionality
    """
    return obs(
        adata=adata,
        df=df,
        obs_on=obs_on,
        df_on=df_on,
        suffix=suffix,
        sort_obs_by_ann=sort_obs_by_ann,
        inplace=inplace,
    )
