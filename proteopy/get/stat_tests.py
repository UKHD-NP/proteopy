from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from anndata import AnnData

from proteopy.utils.anndata import check_proteodata
from proteopy.utils.parsers import parse_stat_test_varm_slot


def differential_abundance_df(
    adata: AnnData,
    keys: Sequence[str] | str | None = None,
    key_group: str | None = None,
    min_logfc: float | None = None,
    max_logfc: float | None = None,
    max_pval: float | None = None,
    sort_by: str | None = None,
) -> pd.DataFrame:
    """
    Retrieve differential abundance results from ``.varm`` as a long-format DataFrame.

    Merges one or more test result DataFrames stored in ``adata.varm`` into a
    single tidy DataFrame with an added column identifying the source test.

    Parameters
    ----------
    adata : :class:`~anndata.AnnData`
        Annotated data object containing differential abundance results in
        ``.varm``.
    keys : str | Sequence[str] | None
        One or more keys in ``adata.varm`` corresponding to differential
        abundance test results (e.g., ``"ttest_two_sample_treated-control"``
        or ``"welch_A-vs-rest"``). Mutually exclusive with ``key_group``.
    key_group : str | None
        Alternative to ``keys``. A key group identifier (e.g.,
        ``"welch_one_vs_rest"``) that selects all ``.varm`` keys belonging
        to that group. Use :func:`tests` to see available key groups.
        Mutually exclusive with ``keys``.
    min_logfc : float | None
        If provided, filter to rows where ``logfc >= min_logfc``.
    max_logfc : float | None
        If provided, filter to rows where ``logfc <= max_logfc``.
    max_pval : float | None
        If provided, filter to rows where adjusted p-value <= ``max_pval``.
        Uses ``pval_adj`` column if present, otherwise falls back to ``pval``.
    sort_by : str | None
        Column name to sort by in descending order (e.g., ``"logfc"``).

    Returns
    -------
    pandas.DataFrame
        Long-format DataFrame with columns:

        - ``var_id``: Variable identifier (from ``adata.var_names``).
        - ``test_type``: The ``.varm`` key identifying the source test.
        - ``test_type_label``: Human-readable label for the test type.
        - ``design``: Underscore-separated design identifier (e.g., ``"A_vs_rest"``).
        - ``design_label``: Human-readable description of what the test compares.
        - ``mean1``: Mean expression in group 1.
        - ``mean2``: Mean expression in group 2.
        - ``logfc``: Log fold change.
        - ``tstat``: t-statistic.
        - ``pval``: Raw p-value.
        - ``pval_adj``: Adjusted p-value.
        - ``is_diff_abundant``: Boolean indicating significance.

    Raises
    ------
    ValueError
        If both ``keys`` and ``key_group`` are provided, or if neither is
        provided.
    TypeError
        If ``keys`` is neither a string nor a sequence of strings.
    KeyError
        If any specified key is not found in ``adata.varm``, or if
        ``key_group`` does not match any test group.

    Examples
    --------
    >>> import proteopy as pp
    >>> # Using explicit keys
    >>> df = pp.get.differential_abundance_df(
    ...     adata,
    ...     keys=["welch_treated-control", "welch_A-vs-rest"],
    ... )
    >>> sig_proteins = df[df["is_diff_abundant"]]
    >>>
    >>> # Using key_group to select all tests in a group
    >>> df = pp.get.differential_abundance_df(
    ...     adata,
    ...     key_group="welch_one_vs_rest",
    ... )
    """
    check_proteodata(adata)

    # Validate mutually exclusive parameters
    if keys is not None and key_group is not None:
        raise ValueError(
            "Cannot specify both `keys` and `key_group`. "
            "Please provide only one."
        )
    if keys is None and key_group is None:
        raise ValueError(
            "Must specify either `keys` or `key_group`."
        )

    # Resolve keys from key_group if provided
    if key_group is not None:
        tests_df = tests(adata)
        matching = tests_df[tests_df["key_group"] == key_group]
        if matching.empty:
            available_groups = tests_df["key_group"].unique().tolist()
            raise KeyError(
                f"key_group '{key_group}' not found. "
                f"Available key groups: {available_groups}"
            )
        keys_list = matching["key"].tolist()
    elif isinstance(keys, str):
        keys_list = [keys]
    elif isinstance(keys, Sequence):
        if not all(isinstance(k, str) for k in keys):
            raise TypeError(
                "`keys` must contain only strings; received "
                f"{keys!r}."
            )
        keys_list = list(keys)
    else:
        raise TypeError(
            "`keys` must be a string or a sequence of strings, "
            f"received {type(keys)!r}."
        )

    # Validate all keys exist in varm
    missing_keys = [k for k in keys_list if k not in adata.varm]
    if missing_keys:
        available = list(adata.varm.keys())
        raise KeyError(
            f"Keys not found in adata.varm: {missing_keys}. "
            f"Available keys: {available}"
        )

    # Merge DataFrames
    frames = []
    for key in keys_list:
        df = adata.varm[key].copy()
        df["var_id"] = df.index
        df["test_type"] = key
        parsed = parse_stat_test_varm_slot(key, adata=adata)
        df["test_type_label"] = parsed["test_type_label"]
        df["design"] = parsed["design"]
        df["design_label"] = parsed["design_label"]
        frames.append(df)

    result = pd.concat(frames, ignore_index=True)

    # Reorder columns: var_id, test_type, test_type_label, design, design_label,
    # then the rest
    col_order = ["var_id", "test_type", "test_type_label", "design",
                 "design_label", "mean1", "mean2", "logfc", "tstat", "pval",
                 "pval_adj", "is_diff_abundant"]
    # Include any extra columns that might be present
    extra_cols = [c for c in result.columns if c not in col_order]
    result = result[col_order + extra_cols]

    # Apply filters
    if min_logfc is not None:
        result = result[result["logfc"] >= min_logfc]
    if max_logfc is not None:
        result = result[result["logfc"] <= max_logfc]
    if max_pval is not None:
        pval_col = "pval_adj" if "pval_adj" in result.columns else "pval"
        result = result[result[pval_col] <= max_pval]

    # Apply sorting
    if sort_by is not None:
        if sort_by not in result.columns:
            raise KeyError(
                f"sort_by column '{sort_by}' not found in result. "
                f"Available columns: {result.columns.tolist()}"
            )
        result = result.sort_values(by=sort_by, ascending=True)
        result = result.reset_index(drop=True)

    return result


def tests(adata: AnnData) -> pd.DataFrame:
    """
    Retrieve a summary of all differential abundance tests stored in ``.varm``.

    Scans the ``.varm`` slots of the AnnData object for statistical test results
    and returns a DataFrame summarizing the tests performed.

    Parameters
    ----------
    adata : :class:`~anndata.AnnData`
        Annotated data object containing differential abundance results in
        ``.varm``.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns:

        - ``key``: The ``.varm`` slot name.
        - ``key_group``: String identifier for the test group in format
          ``"<test_type>_<design_mode>"`` or ``"<test_type>_<design_mode>_<layer>"``
          if a layer was used.
        - ``test_type``: The statistical test type (e.g., ``"ttest_two_sample"``).
        - ``design``: Underscore-separated design identifier (e.g., ``"A_vs_rest"``).
        - ``design_label``: Human-readable description of what the test compares.
        - ``design_mode``: Either ``"one_vs_rest"`` or ``"one_vs_one"``.
        - ``layer``: The layer used for the test, or ``None`` if ``.X`` was used.

    Examples
    --------
    >>> import proteopy as pp
    >>> # After running differential abundance tests
    >>> tests_df = pp.get.tests(adata)
    >>> tests_df
              key          key_group  ...  design_mode
    0  welch_A_vs_rest  welch_one_vs_rest  ...  one_vs_rest
    1  welch_B_vs_rest  welch_one_vs_rest  ...  one_vs_rest
    """
    from proteopy.utils.parsers import parse_stat_test_varm_slot

    check_proteodata(adata)

    records = []
    for key in adata.varm.keys():
        try:
            parsed = parse_stat_test_varm_slot(key, adata=adata)
            design = parsed["design"]
            design_mode = "one_vs_rest" if design.endswith("_vs_rest") else "one_vs_one"
            records.append({
                "key": key,
                "test_type": parsed["test_type"],
                "design": design,
                "design_label": parsed["design_label"],
                "design_mode": design_mode,
                "layer": parsed["layer"],
            })
        except ValueError:
            # Not a stat-test slot, skip
            continue

    if not records:
        return pd.DataFrame(
            columns=["key", "key_group", "test_type", "design", "design_label",
                     "design_mode", "layer"]
        )

    df = pd.DataFrame(records)

    # Build key_group string: "<test_type>_<design_mode>" or
    # "<test_type>_<design_mode>_<layer>" if layer is not None
    def build_key_group(row):
        parts = [row["test_type"], row["design_mode"]]
        if row["layer"] is not None:
            parts.append(row["layer"])
        return "_".join(parts)

    df["key_group"] = df.apply(build_key_group, axis=1)
    df = df[["key", "key_group", "test_type", "design", "design_label",
             "design_mode", "layer"]]

    return df
