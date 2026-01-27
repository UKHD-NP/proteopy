"""Hierarchical clustering tools for proteomics data."""

import hashlib
import warnings

import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from proteopy.utils.anndata import check_proteodata
from proteopy.utils.parsers import (
    _is_standard_hclustv_key,
    _parse_hclustv_key_components,
    _resolve_hclustv_cluster_key,
)


def _validate_linkage_and_values(
    Z: np.ndarray,
    values_df: pd.DataFrame,
) -> None:
    """
    Validate linkage matrix and values DataFrame for clustering operations.

    Parameters
    ----------
    Z : np.ndarray
        Linkage matrix from hierarchical clustering.
    values_df : pd.DataFrame
        DataFrame with variables as columns used to obtain the linkage matrix.

    Raises
    ------
    TypeError
        If linkage matrix is not a numpy array or values is not a DataFrame.
    ValueError
        If linkage matrix has invalid shape or dimension mismatch with
        values DataFrame.
    """
    if not isinstance(Z, np.ndarray):
        raise TypeError(
            f"Expected linkage matrix to be numpy array, "
            f"got {type(Z).__name__}."
        )
    if Z.ndim != 2 or Z.shape[1] != 4:
        raise ValueError(
            f"Invalid linkage matrix shape {Z.shape}. "
            f"Expected (n-1, 4) for n observations."
        )

    if not isinstance(values_df, pd.DataFrame):
        raise TypeError(
            f"Expected values to be DataFrame, "
            f"got {type(values_df).__name__}."
        )

    n_vars = values_df.shape[1]
    expected_vars = Z.shape[0] + 1
    if n_vars != expected_vars:
        raise ValueError(
            f"Dimension mismatch: linkage matrix has {expected_vars} leaves "
            f"but values DataFrame has {n_vars} columns."
        )


def hclustv_tree(
    adata: ad.AnnData,
    selected_vars: list[str] | None = None,
    group_by: str | None = None,
    summary_method: str = "median",
    linkage_method: str = "average",
    distance_metric: str = "euclidean",
    layer: str | None = None,
    zero_to_na: bool = False,
    fill_na: float | int | None = None,
    z_transform: bool = True,
    inplace: bool = True,
    key_added: str | None = None,
    verbose: bool = True,
) -> ad.AnnData | None:
    """
    Perform hierarchical clustering on variables (peptides or proteins).

    Computes a linkage matrix from variable profiles across samples or groups,
    storing the result in ``adata.uns`` for downstream visualization or analysis.

    Parameters
    ----------
    adata : AnnData
        :class:`~anndata.AnnData` with proteomics annotations.
    selected_vars : list[str] | None
        Explicit list of variables to include. When ``None``, all variables
        are used.
    group_by : str | None
        Column in ``adata.obs`` used to group observations. When provided,
        computes a summary statistic for each group rather than using
        individual samples. Grouping can resolve NaN values through
        aggregation (e.g., median of [1, NaN, 3] = 2).
    summary_method : str
        Method for computing group summaries when ``group_by`` is specified.
        One of ``"median"`` or ``"mean"`` (alias ``"average"``).
    linkage_method : str
        Linkage criterion passed to :func:`scipy.cluster.hierarchy.linkage`.
        Common options include ``"average"``, ``"complete"``, ``"single"``,
        and ``"ward"``.
    distance_metric : str
        Distance metric for clustering. One of ``"euclidean"``, ``"manhattan"``,
        or ``"cosine"``.
    layer : str | None
        Optional ``adata.layers`` key to draw quantification values from.
        When ``None`` the primary matrix ``adata.X`` is used.
    zero_to_na : bool
        Replace zeros with ``NaN`` before computing profiles.
    fill_na : float | int | None
        Replace ``NaN`` values with the specified constant before summary
        computation.
    z_transform : bool
        Standardize values to mean 0 and variance 1 per variable before
        clustering. Variables with zero variance will be set to 0 (the mean)
        after transformation.
    inplace : bool
        If ``True``, store results in ``adata.uns`` and return ``None``.
        If ``False``, return a modified copy of ``adata``.
    key_added : str | None
        Custom key prefix for storing results in ``adata.uns``. When ``None``,
        uses the default format ``'hclustv_linkage;<group_by>;<var_hash>;<layer>'``.
    verbose : bool
        Print storage location keys after computation.

    Returns
    -------
    AnnData | None
        If ``inplace=True``, returns ``None``.
        If ``inplace=False``, returns a copy of ``adata`` with clustering
        results stored in ``.uns``.

    Notes
    -----
    The linkage matrix is stored at
    ``adata.uns['hclustv_linkage;<group_by>;<var_hash>;<layer>']``.
    The profile values DataFrame (after all transformations) is stored at
    ``adata.uns['hclustv_values;<group_by>;<var_hash>;<layer>']``.

    The ``var_hash`` is the first 8 characters of the MD5 hash of the sorted,
    semicolon-joined variable names used for clustering.
    When ``group_by`` is ``None``, the field is left empty in the key.
    When ``layer`` is ``None``, ``'X'`` is used in the key.

    Examples
    --------
    >>> import proteopy as pp
    >>> adata = pp.datasets.example_peptide_data()
    >>> pp.tl.hclustv_tree(adata, group_by="condition")
    >>> # Linkage matrix stored in adata.uns['hclustv_linkage;condition;a1b2c3d4;X']
    """
    check_proteodata(adata)

    # Validate summary_method
    summary_method = summary_method.lower()
    if summary_method == "average":
        summary_method = "mean"
    if summary_method not in ("median", "mean"):
        raise ValueError(
            f"summary_method must be 'median' or 'mean', got '{summary_method}'."
        )

    # Validate distance_metric
    distance_metric = distance_metric.lower()
    if distance_metric not in ("euclidean", "manhattan", "cosine"):
        raise ValueError(
            f"distance_metric must be 'euclidean', 'manhattan', or 'cosine', "
            f"got '{distance_metric}'."
        )

    # Map metric names to scipy pdist names
    metric_map = {
        "euclidean": "euclidean",
        "manhattan": "cityblock",
        "cosine": "cosine",
    }
    scipy_metric = metric_map[distance_metric]

    # Validate ward linkage compatibility
    if linkage_method == "ward" and distance_metric != "euclidean":
        raise ValueError(
            "linkage_method='ward' requires distance_metric='euclidean'. "
            f"Got distance_metric='{distance_metric}'."
        )

    # Extract matrix
    matrix = adata.layers[layer] if layer else adata.X

    if matrix is None:
        raise ValueError("Selected matrix is empty.")

    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    else:
        matrix = np.asarray(matrix)

    df = pd.DataFrame(
        matrix,
        index=adata.obs_names,
        columns=adata.var_names,
    )

    # Filter variables if specified
    if selected_vars is not None:
        seen = set()
        duplicates = [v for v in selected_vars if v in seen or seen.add(v)]
        if duplicates:
            raise ValueError(
                f"Duplicate variables in selected_vars: {list(set(duplicates))}"
            )
        missing_vars = [v for v in selected_vars if v not in df.columns]
        if missing_vars:
            raise KeyError(
                f"Variables not found in adata.var_names: {missing_vars}"
            )
        df = df[selected_vars]

    if zero_to_na:
        df = df.replace(0, np.nan)

    if fill_na is not None:
        df = df.fillna(fill_na)

    # Group by if specified
    if group_by is not None:
        if group_by not in adata.obs.columns:
            raise KeyError(f"Column '{group_by}' not found in adata.obs.")
        groups = adata.obs[group_by]
        df["__group__"] = groups.values

        # Compute group summaries
        if summary_method == "median":
            profile_df = df.groupby("__group__", observed=True).apply(
                lambda x: x.median(skipna=True),
                include_groups=False,
            )
        else:
            profile_df = df.groupby("__group__", observed=True).apply(
                lambda x: x.mean(skipna=True),
                include_groups=False,
            )
    else:
        profile_df = df

    # Validate sufficient rows (observations/groups) for clustering
    if profile_df.shape[0] < 2:
        raise ValueError(
            "At least 2 observations or groups are required for hierarchical "
            f"clustering. Got {profile_df.shape[0]} after grouping."
        )

    # Drop variables with all NaN (operates on columns since vars are columns)
    profile_df = profile_df.dropna(axis=1, how="all")

    if profile_df.empty or profile_df.shape[1] < 2:
        raise ValueError(
            "At least 2 variables are required for hierarchical clustering."
        )

    # Check for remaining NaN values - clustering does not support NaN
    nan_count = profile_df.isna().sum().sum()
    if nan_count > 0:
        nan_vars = profile_df.columns[profile_df.isna().any()].tolist()
        suffix = "..." if len(nan_vars) > 5 else ""
        raise ValueError(
            f"Clustering does not support NaN values. Found {nan_count} NaN "
            f"values in {len(nan_vars)} variables: {nan_vars[:5]}{suffix}. "
            "Use fill_na parameter or preprocess data to handle missing values."
        )

    # Optionally compute z-scores per variable (column)
    if z_transform:
        col_means = profile_df.mean(axis=0, skipna=True)
        col_stds = profile_df.std(axis=0, skipna=True, ddof=1)

        zero_var_cols = col_stds[col_stds == 0].index.tolist()
        if zero_var_cols:
            suffix = "..." if len(zero_var_cols) > 5 else ""
            warnings.warn(
                f"{len(zero_var_cols)} variables have zero variance and will "
                f"be set to 0 after z-transform: {zero_var_cols[:5]}{suffix}"
            )

        col_stds = col_stds.replace(0, np.nan)  # avoid division by zero
        profile_df = (profile_df - col_means) / col_stds
        # Fill NaN from zero-variance columns with 0 (the mean)
        profile_df = profile_df.fillna(0)

    profile_values_df = profile_df.copy()

    # Transpose for clustering: pdist computes distances between rows
    # We want distances between variables, so vars must be rows
    clustering_matrix = profile_df.T

    # Compute pairwise distances and linkage
    dist_matrix = pdist(clustering_matrix.values, metric=scipy_metric)
    Z = linkage(dist_matrix, method=linkage_method)

    # Compute var_hash from the actual clustered variables
    clustered_vars = list(profile_values_df.columns)
    var_hash = hashlib.md5(
        ";".join(sorted(clustered_vars)).encode()
    ).hexdigest()[:8]

    # Build storage keys
    group_by_str = group_by if group_by is not None else ""
    layer_str = layer if layer is not None else "X"

    if key_added is not None:
        linkage_key = key_added
        values_key = f"{key_added}_values"
    else:
        linkage_key = f"hclustv_linkage;{group_by_str};{var_hash};{layer_str}"
        values_key = f"hclustv_values;{group_by_str};{var_hash};{layer_str}"

    # Store results
    if inplace:
        adata.uns[linkage_key] = Z
        adata.uns[values_key] = profile_values_df
        check_proteodata(adata)
    else:
        adata_out = adata.copy()
        adata_out.uns[linkage_key] = Z
        adata_out.uns[values_key] = profile_values_df
        check_proteodata(adata_out)

    if verbose:
        print(
            f"Linkage matrix stored in adata.uns['{linkage_key}']\n"
            f"Profile values stored in adata.uns['{values_key}']"
        )

    if inplace:
        return None
    else:
        return adata_out


def hclustv_cluster_ann(
    adata: ad.AnnData,
    k: int,
    linkage_key: str = 'auto',
    values_key: str = 'auto',
    inplace: bool = True,
    key_added: str | None = None,
    verbose: bool = True,
) -> ad.AnnData | None:
    """
    Annotate variables with cluster assignments from hierarchical clustering.

    Uses :func:`scipy.cluster.hierarchy.fcluster` to cut the dendrogram
    at ``k`` clusters and stores cluster assignments in ``.var``.

    Parameters
    ----------
    adata : AnnData
        :class:`~anndata.AnnData` with hierarchical clustering results stored
        in ``.uns`` (from :func:`~proteopy.tl.hclustv_tree`).
    k : int
        Number of clusters to generate (required).
    linkage_key : str
        Key in ``adata.uns`` containing the linkage matrix. When ``'auto'``,
        auto-detects the linkage key if exactly one
        ``'hclustv_linkage;...'`` key exists. When multiple keys are present,
        must be specified explicitly.
    values_key : str
        Key in ``adata.uns`` containing the values DataFrame. When ``'auto'``,
        auto-detects the values key if exactly one
        ``'hclustv_values;...'`` key exists. When multiple keys are present,
        must be specified explicitly.
    inplace : bool
        If ``True``, store results in ``adata.var`` and return ``None``.
        If ``False``, return a modified copy of ``adata``.
    key_added : str | None
        Custom key for storing results in ``adata.var``. When ``None``,
        uses the default format
        ``'hclustv_cluster;<group_by>;<var_hash>;<layer>'`` derived from
        the linkage key components.
    verbose : bool
        Print storage location key after computation.

    Returns
    -------
    AnnData | None
        If ``inplace=True``, returns ``None``.
        If ``inplace=False``, returns a copy of ``adata`` with cluster
        annotations stored in ``.var``.

    Raises
    ------
    ValueError
        If no hierarchical clustering results are found in ``adata.uns``.
        If multiple clustering results exist and ``linkage_key`` is not
        specified.
        If linkage matrix has invalid shape.
        If ``k < 2`` (single cluster is semantically meaningless).
        If auto-generated storage key cannot be derived from a custom
        linkage key.
    TypeError
        If linkage matrix is not a numpy array.
    KeyError
        If specified ``linkage_key`` is not found in ``adata.uns``.

    Notes
    -----
    Cluster assignments are stored at
    ``adata.var['hclustv_cluster;<group_by>;<var_hash>;<layer>']``
    Variables not included in the clustering (e.g., filtered out due to
    NaN values) will have ``NaN`` in this column.

    Examples
    --------
    >>> import proteopy as pr
    >>> adata = pr.datasets.karayel_2020()
    >>> pr.tl.hclustv_tree(
    ...     adata, group_by="condition", selected_vars=adata.vars[0:1000]
    ... )
    >>> pr.tl.hclustv_cluster_ann(adata, 5)

    Access cluster assignments:

    >>> adata.var['hclustv_cluster;condition;a1b2c3d4;X']
    """
    check_proteodata(adata)

    # Resolve linkage and values keys
    linkage_key, values_key = _resolve_hclustv_keys(
        adata, linkage_key, values_key, verbose
    )

    Z = adata.uns[linkage_key]
    values_df = adata.uns[values_key]

    _validate_linkage_and_values(Z, values_df)

    n_vars = values_df.shape[1]
    if k < 2:
        raise ValueError(
            "k must be at least 2. A single cluster is semantically "
            "meaningless for cluster assignments."
        )
    if k > n_vars:
        if verbose:
            print(
                f"k={k} exceeds number of variables ({n_vars}). "
                f"Limiting to k={n_vars}."
            )
        k = n_vars

    labels = fcluster(Z, t=k, criterion="maxclust")

    var_names = values_df.columns.tolist()
    cluster_map = dict(zip(var_names, labels))

    # Zero-pad cluster numbers for correct alphanumeric sorting
    n_digits = len(str(k))

    # Build cluster annotation series for adata.var
    cluster_annotations = pd.Series(
        index=adata.var_names,
        dtype="object",
    )
    for var_name, cluster_id in cluster_map.items():
        cluster_annotations[var_name] = f"{cluster_id:0{n_digits}d}"

    # Build storage key
    if key_added is not None:
        cluster_key = key_added
    elif _is_standard_hclustv_key(linkage_key, "linkage"):
        components = _parse_hclustv_key_components(linkage_key)
        group_by_str, var_hash, layer_str = components
        cluster_key = f"hclustv_cluster;{group_by_str};{var_hash};{layer_str}"
    else:
        raise ValueError(
            f"Cannot auto-generate storage key from custom linkage_key "
            f"'{linkage_key}'. Please provide key_added explicitly."
        )

    if inplace:
        adata.var[cluster_key] = cluster_annotations
        check_proteodata(adata)
        if verbose:
            print(f"Cluster annotations stored in adata.var['{cluster_key}']")
        return None
    else:
        adata_out = adata.copy()
        adata_out.var[cluster_key] = cluster_annotations
        check_proteodata(adata_out)
        if verbose:
            print(f"Cluster annotations stored in adata.var['{cluster_key}']")
        return adata_out

