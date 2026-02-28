"""
Statistical tests for differential abundance analysis.
"""

import warnings

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse
from scipy import stats
from statsmodels.stats.multitest import multipletests

from proteopy.utils.anndata import check_proteodata
from proteopy.utils.array import is_log_transformed
from proteopy.utils.string import sanitize_string


SUPPORTED_METHODS = {
    "ttest_two_sample": {
        "space": "log",
        "type": "two_group",
        "executor": "_execute_two_group_ttest",
        "equal_var": True,
    },
    "welch": {
        "space": "log",
        "type": "two_group",
        "executor": "_execute_two_group_ttest",
        "equal_var": False,
    },
    "anova": {
        "space": "log",
        "type": "1vrest",
        "executor": "_execute_anova",
    },
}

SUPPORTED_CORRECTIONS = [
    "bonferroni",
    "fdr_bh", "fdr", "bh", "benjamini_hochberg",
]

MIN_SAMPLES_PER_GROUP = 3


def _validate_setup_two_group(
    setup: dict,
    group_by: str,
    obs_column,
    method: str,
) -> None:
    """
    Validate setup dictionary for two-group comparison methods.

    Parameters
    ----------
    setup : dict
        Setup dictionary to validate.
    group_by : str
        Name of the grouping column (for error messages).
    obs_column : pd.Series
        The actual column from adata.obs[group_by].
    method : str
        Name of the method (for error messages).

    Raises
    ------
    ValueError
        If setup is missing required keys, contains extra keys, or values
        don't exist in the column.
    """
    required_keys = {"group1", "group2"}
    missing_keys = required_keys - set(setup.keys())
    if missing_keys:
        raise ValueError(
            f"For method '{method}', setup must contain keys "
            f"{required_keys}. Missing: {missing_keys}"
        )

    # Check for extra keys
    extra_keys = set(setup.keys()) - required_keys
    if extra_keys:
        raise ValueError(
            f"For method '{method}', setup must only contain keys "
            f"{required_keys}. Unexpected keys: {extra_keys}"
        )

    unique_values = set(obs_column.unique())
    for key in ["group1", "group2"]:
        value = setup[key]
        if value not in unique_values:
            raise ValueError(
                f"Value '{value}' for setup['{key}'] not found in "
                f"adata.obs['{group_by}']. "
                f"Available values: {sorted(unique_values)}"
            )


def _validate_setup_1vrest(
    setup: dict,
    group_by: str,
    obs_column: pd.Series,
) -> None:
    """
    Validate setup dictionary for one-vs-rest comparison methods.

    Parameters
    ----------
    setup : dict
        Setup dictionary to validate.
    group_by : str
        Name of the grouping column (for error messages).
    obs_column : pd.Series
        The actual column from adata.obs[group_by].

    Raises
    ------
    ValueError
        If setup contains unexpected keys, if groups are not unique, or if
        specified groups are not found in the column.
    """
    # Validate setup contains only allowed keys
    allowed_keys = {"groups"}
    extra_keys = set(setup.keys()) - allowed_keys
    if extra_keys:
        raise ValueError(
            f"For one-vs-rest mode, setup must only contain 'groups' key "
            f"(or be empty). Unexpected keys: {extra_keys}"
        )

    # If groups key is present, validate it
    if "groups" in setup:
        groups_spec = setup["groups"]

        # Validate that groups_spec is "all" or a list
        if groups_spec != "all" and not isinstance(groups_spec, list):
            raise ValueError(
                f"setup['groups'] must be 'all' or a list, "
                f"got {type(groups_spec).__name__}."
            )

        # If it's a list, validate uniqueness and existence
        if isinstance(groups_spec, list):
            # Check for duplicates
            if len(groups_spec) != len(set(groups_spec)):
                duplicates = [
                    g for g in set(groups_spec)
                    if groups_spec.count(g) > 1
                ]
                raise ValueError(
                    f"setup['groups'] contains duplicate values: {duplicates}"
                )

            # Check all groups exist in obs_column
            unique_values = set(obs_column.unique())
            missing_groups = set(groups_spec) - unique_values
            if missing_groups:
                raise ValueError(
                    f"Groups not found in adata.obs['{group_by}']: "
                    f"{sorted(missing_groups)}. "
                    f"Available groups: {sorted(unique_values)}"
                )


def _perform_ttest(
    X1: np.ndarray,
    X2: np.ndarray,
    group1_name: str,
    group2_name: str,
    effective_space: str,
    equal_var: bool,
) -> dict:
    """
    Perform t-test between two groups and compute summary statistics.

    Parameters
    ----------
    X1 : np.ndarray
        Expression matrix for group 1 (observations x variables).
    X2 : np.ndarray
        Expression matrix for group 2 (observations x variables).
    group1_name : str
        Name of group 1 (for error messages).
    group2_name : str
        Name of group 2 (for error messages).
    effective_space : str
        The space of the data ('log' or 'linear').
    equal_var : bool
        If True, use Student's t-test (equal variances).
        If False, use Welch's t-test (unequal variances).

    Returns
    -------
    dict
        Results dictionary containing 'mean1', 'mean2', 'logfc', 'tstat', 'pval'.

    Raises
    ------
    ValueError
        If either group has fewer than MIN_SAMPLES_PER_GROUP samples or
        contains NA values.
    """
    # Validate minimum sample counts
    n_group1 = X1.shape[0]
    n_group2 = X2.shape[0]

    if n_group1 < MIN_SAMPLES_PER_GROUP:
        raise ValueError(
            f"Group '{group1_name}' has {n_group1} samples, but at least "
            f"{MIN_SAMPLES_PER_GROUP} are required for t-test."
        )
    if n_group2 < MIN_SAMPLES_PER_GROUP:
        raise ValueError(
            f"Group '{group2_name}' has {n_group2} samples, but at least "
            f"{MIN_SAMPLES_PER_GROUP} are required for t-test."
        )

    # Strict NA validation
    if np.isnan(X1).any():
        raise ValueError(
            f"Expression matrix for group '{group1_name}' contains "
            "NA values. Please impute or filter missing values before "
            "running differential abundance analysis."
        )
    if np.isnan(X2).any():
        raise ValueError(
            f"Expression matrix for group '{group2_name}' contains "
            "NA values. Please impute or filter missing values before "
            "running differential abundance analysis."
        )

    # Compute means
    mean1 = X1.mean(axis=0)
    mean2 = X2.mean(axis=0)

    # Flatten if needed (in case of matrix return)
    if hasattr(mean1, 'A1'):
        mean1 = mean1.A1
    if hasattr(mean2, 'A1'):
        mean2 = mean2.A1

    # Compute logFC
    if effective_space == 'log':
        logfc = mean1 - mean2
    else:  # linear -> compute log2 fold change
        logfc = np.log2(mean1 / mean2)

    # Execute t-test
    tstats, pvals = stats.ttest_ind(
        X1, X2,
        axis=0,
        equal_var=equal_var,
    )

    return {
        'mean1': mean1,
        'mean2': mean2,
        'logfc': logfc,
        'tstat': tstats,
        'pval': pvals,
    }


def _execute_two_group_ttest(
    X: np.ndarray,
    obs_column: pd.Series,
    setup: dict,
    group_by: str,
    method: str,
    effective_space: str,
    equal_var: bool,
    **kwargs,
) -> tuple[dict, str]:
    """
    Execute two-sample t-test for differential abundance.

    Parameters
    ----------
    X : np.ndarray
        Processed expression matrix (observations x variables).
    obs_column : pd.Series
        Column from adata.obs containing group labels.
    setup : dict
        Setup dictionary with 'group1' and 'group2' keys.
    group_by : str
        Name of the grouping column (for error messages).
    method : str
        Name of the method (for error messages).
    effective_space : str
        The space of the data ('log' or 'linear').
    equal_var : bool
        If True, use Student's t-test (equal variances).
        If False, use Welch's t-test (unequal variances).
    **kwargs
        Additional method config (ignored).

    Returns
    -------
    list[tuple[dict, str]]
        A tuple of (results_dict, group_label) inside a list where
        results_dict contains 'mean1', 'mean2', 'logfc', 'tstat',
        'pval' and group_label is a sanitized string for naming the
        output.

    Raises
    ------
    ValueError
        If setup is invalid, groups have fewer than MIN_SAMPLES_PER_GROUP
        samples, or data contains NA values.
    """
    # Validate setup parameter
    _validate_setup_two_group(setup, group_by, obs_column, method)

    # Get data for each group
    group1_name = setup["group1"]
    group2_name = setup["group2"]

    idxs1 = obs_column == group1_name
    idxs2 = obs_column == group2_name

    X1 = X[idxs1.values, :]
    X2 = X[idxs2.values, :]

    # Perform t-test (includes validation for min samples and NA)
    test_results = _perform_ttest(
        X1=X1,
        X2=X2,
        group1_name=group1_name,
        group2_name=group2_name,
        effective_space=effective_space,
        equal_var=equal_var,
    )

    group_label = sanitize_string(f"{group1_name}_vs_{group2_name}")

    return [(test_results, group_label)]


def _execute_one_vs_rest_ttest(
    X: np.ndarray,
    obs_column: pd.Series,
    setup: dict,
    group_by: str,
    method: str,
    effective_space: str,
    equal_var: bool,
    **kwargs,
) -> list[tuple[dict, str]]:
    """
    Execute one-vs-rest t-tests for differential abundance.

    For each specified group, performs a t-test comparing that group against
    all other groups combined.

    Parameters
    ----------
    X : np.ndarray
        Processed expression matrix (observations x variables).
    obs_column : pd.Series
        Column from adata.obs containing group labels.
    setup : dict
        Setup dictionary containing optional 'groups' key:

        - ``"groups"``: Either ``"all"`` to test all unique groups, or a
          list of group labels to test. Defaults to ``"all"``.
    group_by : str
        Name of the grouping column (for error messages).
    method : str
        Name of the method (for error messages).
    effective_space : str
        The space of the data ('log' or 'linear').
    equal_var : bool
        If True, use Student's t-test (equal variances).
        If False, use Welch's t-test (unequal variances).
    **kwargs
        Additional method config (ignored).

    Returns
    -------
    list[tuple[dict, str]]
        A list of (results_dict, group_label) tuples, one per tested group.
        Each results_dict contains 'mean1', 'mean2', 'logfc', 'tstat', 'pval'.
        group_label is formatted as '{group}-vs-rest'.

    Raises
    ------
    ValueError
        If setup contains unexpected keys, if any group or its corresponding
        'rest' has fewer than MIN_SAMPLES_PER_GROUP samples, if data contains
        NA values, or if specified groups are not found in the data.
    """
    # Validate setup parameter
    _validate_setup_1vrest(setup, group_by, obs_column)

    unique_groups = set(obs_column.unique())

    # Determine which groups to test
    groups_spec = setup.get("groups", "all")
    if groups_spec == "all":
        groups_to_test = list(unique_groups)
    else:
        groups_to_test = groups_spec

    # Validate we have enough groups for one-vs-rest
    if len(unique_groups) < 2:
        raise ValueError(
            f"One-vs-rest mode requires at least 2 groups in "
            f"adata.obs['{group_by}'], but found {len(unique_groups)}."
        )

    all_results = []

    for group in groups_to_test:
        # Get data for this group vs rest
        idxs_group = obs_column == group
        idxs_rest = ~idxs_group

        X_group = X[idxs_group.values, :]
        X_rest = X[idxs_rest.values, :]

        # Perform t-test (includes validation for min samples and NA)
        test_results = _perform_ttest(
            X1=X_group,
            X2=X_rest,
            group1_name=str(group),
            group2_name="rest",
            effective_space=effective_space,
            equal_var=equal_var,
        )

        group_label = sanitize_string(f"{group}_vs_rest")
        all_results.append((test_results, group_label))

    return all_results


def _perform_anova(
    X: np.ndarray,
    obs_column: pd.Series,
    groups_to_test: list,
) -> dict:
    """
    Perform one-way ANOVA across groups for each variable.

    Parameters
    ----------
    X : np.ndarray
        Expression matrix (observations x variables).
    obs_column : pd.Series
        Column from adata.obs containing group labels.
    groups_to_test : list
        List of group labels to include in the ANOVA.

    Returns
    -------
    dict
        Results dictionary containing 'fstat', 'pval', and
        'mean_{group}' for each group.

    Raises
    ------
    ValueError
        If any group has fewer than MIN_SAMPLES_PER_GROUP samples
        or contains NA values.
    """
    group_arrays = {}
    for group in groups_to_test:
        idxs = (obs_column == group).values
        X_group = X[idxs, :]

        n_samples = X_group.shape[0]
        if n_samples < MIN_SAMPLES_PER_GROUP:
            raise ValueError(
                f"Group '{group}' has {n_samples} samples, "
                f"but at least {MIN_SAMPLES_PER_GROUP} are "
                "required for ANOVA."
            )

        if np.isnan(X_group).any():
            raise ValueError(
                f"Expression matrix for group '{group}' "
                "contains NA values. Please impute or filter "
                "missing values before running differential "
                "abundance analysis."
            )

        group_arrays[group] = X_group

    # Run one-way ANOVA per variable
    fstats, pvals = stats.f_oneway(
        *[group_arrays[g] for g in groups_to_test]
    )

    results = {
        'fstat': fstats,
        'pval': pvals,
    }

    # Add per-group means
    for group in groups_to_test:
        mean_vals = group_arrays[group].mean(axis=0)
        if hasattr(mean_vals, 'A1'):
            mean_vals = mean_vals.A1
        results[f'mean_{sanitize_string(str(group))}'] = mean_vals

    return results


def _execute_anova(
    X: np.ndarray,
    obs_column: pd.Series,
    setup: dict,
    group_by: str,
    method: str,
    effective_space: str,
    **kwargs,
) -> list[tuple[dict, str]]:
    """
    Execute one-way ANOVA for differential abundance.

    Parameters
    ----------
    X : np.ndarray
        Processed expression matrix (observations x variables).
    obs_column : pd.Series
        Column from adata.obs containing group labels.
    setup : dict
        Setup dictionary containing optional 'groups' key:

        - ``"groups"``: Either ``"all"`` to test all unique groups,
          or a list of group labels to test. Defaults to ``"all"``.
    group_by : str
        Name of the grouping column (for error messages).
    method : str
        Name of the method (for error messages).
    effective_space : str
        The space of the data ('log' or 'linear').
    **kwargs
        Additional method config (ignored).

    Returns
    -------
    list[tuple[dict, str]]
        A list containing a single (results_dict, group_label)
        tuple. results_dict contains 'fstat', 'pval', and
        'mean_{group}' for each group.

    Raises
    ------
    ValueError
        If setup contains unexpected keys, fewer than 3 groups
        are specified, or data validation fails.
    """
    _validate_setup_1vrest(setup, group_by, obs_column)

    unique_groups = list(obs_column.unique())

    # Determine which groups to test
    groups_spec = setup.get("groups", "all")
    if groups_spec == "all":
        groups_to_test = unique_groups
    else:
        groups_to_test = groups_spec

    if len(groups_to_test) < 3:
        raise ValueError(
            f"ANOVA requires at least 3 groups, but only "
            f"{len(groups_to_test)} were specified. Use "
            f"'ttest_two_sample' or 'welch' for two-group "
            f"comparisons."
        )

    results = _perform_anova(
        X=X,
        obs_column=obs_column,
        groups_to_test=groups_to_test,
    )

    # Build group label
    sorted_groups = sorted(str(g) for g in groups_to_test)
    if set(groups_to_test) == set(unique_groups):
        group_label = "all"
    else:
        group_label = sanitize_string(
            "_".join(sorted_groups)
        )

    return [(results, group_label)]


# Dispatcher mapping executor names to functions
METHOD_EXECUTORS = {
    "_execute_two_group_ttest": _execute_two_group_ttest,
    "_execute_one_vs_rest_ttest": _execute_one_vs_rest_ttest,
    "_execute_anova": _execute_anova,
}


def differential_abundance(
    adata: AnnData,
    method: str = "ttest_two_sample",
    group_by: str = None,
    setup: dict = None,
    layer: str | None = None,
    multitest_correction: str = "fdr_bh",
    alpha: float | int = 0.05,
    space: str = "auto",
    force: bool = False,
    fill_na: float | int | None = None,
    inplace: bool = True,
) -> AnnData | None:
    """
    Perform differential abundance analysis between sample groups.

    Compares expression values between groups using statistical tests.
    Computes log fold changes, p-values, and applies multiple testing
    correction. Results are stored in ``adata.varm`` as DataFrames.

    Parameters
    ----------
    adata : ad.AnnData
        :class:`~anndata.AnnData` object with expression data in ``.X``
        or a specified layer.
    method : str, optional
        Statistical test for differential abundance. Supported methods:

        - ``"ttest_two_sample"``: Independent two-sample Student's
          t-test assuming equal variances.
        - ``"welch"``: Welch's t-test without equal variance
          assumption. More robust when group variances differ.
        - ``"anova"``: One-way ANOVA testing whether any group
          differs from the others. Requires at least 3 groups.
          Only supports ``setup`` with ``"groups"`` key (or
          ``None``/``{}`` for all groups). Results contain
          ``fstat`` instead of ``tstat``/``logfc``, plus
          per-group mean columns (``mean_{group}``).
    group_by : str
        Column in ``adata.obs`` containing group labels for comparison.
    setup : dict | None, optional
        Dictionary specifying comparison mode. Two modes available:

        **Two-group mode** (keys ``"group1"`` and ``"group2"``
        present):
            Compare two specific groups. Required keys:

            - ``"group1"``: First group label (numerator in log fold
              change).
            - ``"group2"``: Second group label (denominator in log fold
              change).

            Example: ``{"group1": "treated", "group2": "control"}``

        **One-vs-rest mode** (default when ``setup`` is ``None`` or
        ``{}``):
            Compare each group against all other groups combined.
            Optional key:

            - ``"groups"``: ``"all"`` (default) to test all groups, or
              list of specific group labels to test.

            Examples: ``None``, ``{}``, or
            ``{"groups": ["A", "B"]}``.

        Each group and the combined "rest" must have at least 3
        samples.
    layer : str | None, optional
        Key in ``adata.layers`` to use. If ``None``, uses ``adata.X``.
    multitest_correction : str, optional
        Multiple testing correction method. Supported values:

        - ``"bonferroni"``: Bonferroni correction (family-wise error
          rate control).
        - ``"fdr_bh"``: Benjamini-Hochberg FDR correction (false
          discovery rate control).
        - ``"fdr"``, ``"bh"``, ``"benjamini_hochberg"``: Aliases for
          ``"fdr_bh"``.
    alpha : float | int, optional
        Significance threshold for labeling differential abundance.
        Must satisfy 0 < alpha <= 1.
    space : {'auto', 'log', 'linear'}, optional
        Intensity space of input data. When ``"auto"``, inferred via
        :func:`~proteopy.utils.array.is_log_transformed`. Two-sample
        methods require log space; linear data are converted to log2.
        When ``"log"`` or ``"linear"``, mismatch with inferred space
        raises error unless ``force=True``.
    force : bool, optional
        Skip space-mismatch validation and use declared ``space``. Use
        with caution when automatic detection is incorrect.
    fill_na : float | int | None, optional
        Replace ``np.nan`` values in expression matrix with this value
        before analysis. If ``None``, no replacement occurs.
    inplace : bool, optional
        If ``True``, modify ``adata`` in place and return ``None``. If
        ``False``, return modified copy of ``adata``.

    Returns
    -------
    ad.AnnData | None
        When ``inplace=False``, returns copy of
        :class:`~anndata.AnnData` with results in ``.varm``. When
        ``inplace=True``, returns ``None`` and modifies ``adata`` in
        place.

        **Storage format in** ``adata.varm``:

        Results stored as :class:`~pandas.DataFrame` with keys using
        the format ``{method};{group_by};{design}`` or
        ``{method};{group_by};{design};{layer}`` when a layer is used:

        - Two-group mode: ``"{method};{group_by};{group1}_vs_{group2}"``
          (e.g., ``"welch;condition;treated_vs_control"``).
        - One-vs-rest mode: ``"{method};{group_by};{group}_vs_rest"``
          for each tested group
          (e.g., ``"ttest_two_sample;cell_type;A_vs_rest"``).
        - When a layer is used, it is appended as the fourth component
          (e.g., ``"welch;condition;treated_vs_control;raw_intensities"``).

        Additionally, a sanitized version of the ``group_by`` column
        is added to ``adata.obs`` if not already present. This column
        contains sanitized versions of the group labels.

        **DataFrame columns**:

        Each results DataFrame contains the following columns indexed
        by variable names (matching ``adata.var_names``):

        - ``mean1``: Mean expression in group1 (focal group in
          one-vs-rest).
        - ``mean2``: Mean expression in group2 (rest in one-vs-rest).
        - ``logfc``: Log fold change (``mean1 - mean2`` in log space).
          Computed in log2 space for linear input data, otherwise in
          the data's existing log base.
        - ``tstat``: t-statistic from the statistical test.
        - ``pval``: Raw p-value from the test.
        - ``pval_adj``: Adjusted p-value using the specified
          ``multitest_correction`` method.
        - ``is_diff_abundant``: Boolean indicating
          ``pval_adj <= alpha``.

    Raises
    ------
    ValueError
        If ``group_by`` is ``None`` or not in ``adata.obs``.
    ValueError
        If ``layer`` is not in ``adata.layers``.
    ValueError
        If ``method`` is not supported.
    ValueError
        If ``multitest_correction`` is not supported.
    ValueError
        If ``alpha`` is not in range (0, 1].
    ValueError
        If ``space`` mismatches inferred space and ``force=False``.
    ValueError
        If groups have fewer than 3 samples.

    Examples
    --------
    Two-group comparison between treated and control samples:

    >>> import proteopy as pp
    >>> adata = pp.datasets.karayel_2020()
    >>> pp.tl.differential_abundance(
    ...     adata,
    ...     method="welch",
    ...     group_by="condition",
    ...     setup={"group1": "treated", "group2": "control"},
    ... )
    >>> results = adata.varm["welch;condition;treated_vs_control"]
    >>> sig_proteins = results[results["is_diff_abundant"]]

    One-vs-rest comparison for all cell types:

    >>> pp.tl.differential_abundance(
    ...     adata,
    ...     method="ttest_two_sample",
    ...     group_by="cell_type",
    ...     setup=None,
    ... )
    >>> # Results stored as "ttest_two_sample;cell_type;{celltype}_vs_rest"
    >>> for key in adata.varm.keys():
    ...     print(key, adata.varm[key]["is_diff_abundant"].sum())
    """
    check_proteodata(adata)
    target = adata if inplace else adata.copy()

    # Validate group_by
    if group_by is None:
        raise ValueError("Parameter 'group_by' is required.")
    if group_by not in target.obs.columns:
        raise ValueError(
            f"Column '{group_by}' not found in adata.obs. "
            f"Available columns: {list(target.obs.columns)}"
        )
    
    # Validate layer
    if layer is not None and layer not in target.layers:
        available_layers = list(target.layers.keys())
        raise ValueError(
            f"Layer '{layer}' not found in adata.layers. "
            f"Available layers: {available_layers}"
        )

    # Get data
    obs_column = target.obs[group_by]
    X_orig = target.layers[layer] if layer is not None else target.X
    X = X_orig.toarray() if sparse.issparse(X_orig) else np.asarray(X_orig, dtype=float)

    # Validate and apply fill_na
    if fill_na is not None:
        if not isinstance(fill_na, (int, float)):
            raise ValueError(
                f"Parameter 'fill_na' must be a number, got {type(fill_na).__name__}."
            )
        X = np.nan_to_num(X, fill_na)

    # Validate method
    if method not in SUPPORTED_METHODS:
        raise ValueError(
            f"Method '{method}' is not supported. "
            f"Supported methods: {list(SUPPORTED_METHODS.keys())}"
        )

    # Validate multitest_correction
    if multitest_correction not in SUPPORTED_CORRECTIONS:
        raise ValueError(
            f"Correction method '{multitest_correction}' is not supported. "
            f"Supported methods: {SUPPORTED_CORRECTIONS}"
        )

    # Validate alpha
    if not isinstance(alpha, (int, float)):
        raise ValueError("Parameter 'alpha' must be a number between 0 and 1.")
    alpha = float(alpha)
    if alpha <= 0 or alpha > 1:
        raise ValueError("Parameter 'alpha' must be: 0 < alpha <= 1.")

    # Validate and normalize setup
    if setup is None:
        setup = {}
    if not isinstance(setup, dict):
        raise ValueError(
            f"Parameter 'setup' must be a dictionary or None, "
            f"got {type(setup).__name__}."
        )

    # Validate space
    method_spaces = {cfg["space"] for cfg in SUPPORTED_METHODS.values()}
    allowed_spaces = list(method_spaces) + ["auto", "linear"]
    if space not in allowed_spaces:
        raise ValueError(
            f"Parameter 'space' must be one of {sorted(allowed_spaces)}, "
            f"got '{space}'."
        )

    final_space = SUPPORTED_METHODS[method]["space"]
    data_is_log, _ = is_log_transformed(target, layer=layer)

    if data_is_log:
        effective_space = 'log'
    else:
        effective_space = 'linear'

    if space != 'auto' and space != effective_space:
        if force:
            effective_space = space
        else:
            raise ValueError(
                f"Data appears to be in '{effective_space}' space, but "
                f"'space' was set to '{space}'. Set force=True to override "
                "or adjust the 'space' parameter."
            )

    X_proc = X.copy()
    if effective_space == 'linear' and final_space == 'log':
        pseudocount = 1
        if np.any((X_proc + pseudocount) <= 0):
            raise ValueError(
                "Non-positive values encountered after adding pseudocount; "
                "cannot compute log."
            )
        X_proc = np.log2(X_proc + pseudocount)
        warnings.warn(
            "Data treated as linear; applying log2 transform with "
            "pseudocount=1 for differential_abundance.",
        )
        effective_space = 'log'

    if final_space != effective_space:
        raise ValueError(
            f"Data are in '{effective_space}' space, but method '{method}' "
            f"requires '{final_space}' space."
        )

    # Determine executor
    method_config = SUPPORTED_METHODS[method]
    if method == 'ttest_two_sample' or method == 'welch':
        # Determine comparison mode based on setup contents
        has_group_keys = "group1" in setup and "group2" in setup
        is_one_vs_rest = not has_group_keys

        # Select executor based on mode
        if is_one_vs_rest:
            executor = METHOD_EXECUTORS["_execute_one_vs_rest_ttest"]
        else:
            executor_name = method_config["executor"]
            executor = METHOD_EXECUTORS[executor_name]
    elif method == 'anova':
        executor = METHOD_EXECUTORS["_execute_anova"]
    else:
        raise ValueError('Not implemented yet')

    results_list = executor(
        X=X_proc,
        obs_column=obs_column,
        setup=setup,
        group_by=group_by,
        method=method,
        effective_space=effective_space,
        **method_config,
    )

    # Determine correction method
    match multitest_correction:
        case "bonferroni":
            correction_method = "bonferroni"
        case "fdr_bh" | "fdr" | "bh" | "benjamini_hochberg":
            correction_method = "fdr_bh"

    # Add sanitized group_by column to .obs if not already present
    sanitized_group_by = sanitize_string(group_by)
    if sanitized_group_by not in target.obs.columns:
        target.obs[sanitized_group_by] = target.obs[group_by].apply(
            sanitize_string
        )

    # Process each comparison result
    method_label = sanitize_string(method)
    group_by_label = sanitize_string(group_by)
    layer_label = sanitize_string(layer) if layer is not None else None

    for results, group_label in results_list:
        # Multiple testing correction
        reject, pval_adj, _, _ = multipletests(
            results['pval'],
            alpha=alpha,
            method=correction_method,
        )

        results["pval_adj"] = pval_adj
        results["is_diff_abundant"] = reject

        # Create DataFrame and store in varm
        results_df = pd.DataFrame(results, index=target.var_names)

        # Format: <test_type>;<group_by>;<design> or
        # <test_type>;<group_by>;<design>;<layer> if layer is used
        if layer_label is not None:
            slot_name = f"{method_label};{group_by_label};{group_label};{layer_label}"
        else:
            slot_name = f"{method_label};{group_by_label};{group_label}"

        target.varm[slot_name] = results_df
        print(f"Saved test results in .varm['{slot_name}']")

    check_proteodata(target)

    if inplace:
        return None
    else:
        return target
