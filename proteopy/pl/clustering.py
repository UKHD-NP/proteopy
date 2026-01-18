"""Clustering visualization tools for proteomics data."""

import warnings
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import silhouette_score

from proteopy.utils.anndata import check_proteodata


def _resolve_hclust_keys(
    adata: ad.AnnData,
    linkage_key: str | None,
    values_key: str | None,
    verbose: bool,
) -> tuple[str, str]:
    """
    Resolve linkage and values keys from adata.uns.

    Auto-detects keys if not provided, validates existence, and returns
    the resolved key names.
    """
    linkage_candidates = [
        key for key in adata.uns.keys()
        if key.startswith("hclustv_linkage;")
    ]
    values_candidates = [
        key for key in adata.uns.keys()
        if key.startswith("hclustv_values;")
    ]

    if linkage_key is None:
        if len(linkage_candidates) == 0:
            raise ValueError(
                "No hierarchical clustering results found in adata.uns. "
                "Run proteopy.tl.hclustv_tree() first."
            )
        if len(linkage_candidates) > 1:
            raise ValueError(
                "Multiple linkage matrices found in adata.uns. "
                "Please specify linkage_key explicitly. "
                f"Available keys: {linkage_candidates}"
            )
        linkage_key = linkage_candidates[0]
        if verbose:
            print(f"Using linkage matrix: adata.uns['{linkage_key}']")
    else:
        if linkage_key not in adata.uns:
            raise KeyError(
                f"Linkage key '{linkage_key}' not found in adata.uns."
            )

    if values_key is None:
        if len(values_candidates) == 0:
            raise ValueError(
                "No profile values found in adata.uns. "
                "Run proteopy.tl.hclustv_tree() first."
            )
        if len(values_candidates) > 1:
            raise ValueError(
                "Multiple profile values found in adata.uns. "
                "Please specify values_key explicitly. "
                f"Available keys: {values_candidates}"
            )
        values_key = values_candidates[0]
        if verbose:
            print(f"Using profile values: adata.uns['{values_key}']")
    else:
        if values_key not in adata.uns:
            raise KeyError(
                f"Values key '{values_key}' not found in adata.uns."
            )

    return linkage_key, values_key


def _compute_wcss(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute within-cluster sum of squares.

    Parameters
    ----------
    X : np.ndarray
        Data matrix with samples as rows.
    labels : np.ndarray
        Cluster labels for each sample.

    Returns
    -------
    float
        Total within-cluster sum of squares.
    """
    wcss = 0.0
    unique_labels = np.unique(labels)
    for label in unique_labels:
        cluster_points = X[labels == label]
        centroid = cluster_points.mean(axis=0)
        wcss += np.sum((cluster_points - centroid) ** 2)
    return wcss


def hclustv_silhouette(
    adata: ad.AnnData,
    linkage_key: str = 'auto',
    values_key: str = 'auto',
    k: int = 15,
    figsize: tuple[float, float] = (6.0, 4.0),
    show: bool = True,
    ax: bool = False,
    save: str | Path | None = None,
    verbose: bool = True,
) -> Axes | None:
    """
    Plot silhouette scores for hierarchical clustering.

    Evaluates clustering quality by computing the average silhouette
    score for cluster counts ranging from 2 to ``k``. Higher silhouette
    scores indicate better-defined clusters.

    Parameters
    ----------
    adata : AnnData
        :class:`~anndata.AnnData` with clustering results from
        :func:`proteopy.tl.hclustv_tree` stored in ``.uns``.
    linkage_key : str
        Key in ``adata.uns`` for the linkage matrix. When ``'auto'``,
        auto-detects keys matching ``hclustv_linkage;*``.
    values_key : str
        Key in ``adata.uns`` for the profile values DataFrame. When
        ``'auto'``, auto-detects keys matching ``hclustv_values;*``.
    k : int
        Maximum number of clusters to evaluate. Silhouette scores are
        computed for cluster counts from 2 to ``k`` (inclusive).
    figsize : tuple[float, float]
        Matplotlib figure size in inches.
    show : bool
        Display the figure.
    ax : bool
        Return the Matplotlib Axes object instead of displaying.
    save : str | Path | None
        File path for saving the figure.
    verbose : bool
        Print status messages including auto-detected keys.

    Returns
    -------
    Axes | None
        Axes object when ``ax`` is ``True``; otherwise ``None``.

    Raises
    ------
    ValueError
        If no clustering results are found in ``adata.uns``, if
        multiple candidates exist and keys are not specified, or
        if ``k < 2``.
    KeyError
        If the specified ``linkage_key`` or ``values_key`` is not
        found.

    Examples
    --------
    >>> import proteopy as pp
    >>> adata = pp.datasets.example_peptide_data()
    >>> pr.tl.hclustv_tree(adata, group_by="condition")
    >>> pr.pl.hclustv_silhouette(adata, k=5)
    """
    check_proteodata(adata)

    if k < 2:
        raise ValueError("k must be at least 2 to compute silhouette scores.")

    # Auto-detect keys if not provided
    linkage_candidates = [
        key for key in adata.uns.keys()
        if key.startswith("hclust_var_linkage;")
    ]
    values_candidates = [
        key for key in adata.uns.keys()
        if key.startswith("hclust_var_values;")
    ]

    if linkage_key is None:
        if len(linkage_candidates) == 0:
            raise ValueError(
                "No hierarchical clustering results found in adata.uns. "
                "Run proteopy.tl.hclust_vars() first."
            )
        if len(linkage_candidates) > 1:
            raise ValueError(
                "Multiple linkage matrices found in adata.uns. "
                "Please specify linkage_key explicitly. "
                f"Available keys: {linkage_candidates}"
            )
        linkage_key = linkage_candidates[0]
        if verbose:
            print(f"Using linkage matrix: adata.uns['{linkage_key}']")
    else:
        if linkage_key not in adata.uns:
            raise KeyError(
                f"Linkage key '{linkage_key}' not found in adata.uns."
            )

    if values_key is None:
        if len(values_candidates) == 0:
            raise ValueError(
                "No profile values found in adata.uns. "
                "Run proteopy.tl.hclust_vars() first."
            )
        if len(values_candidates) > 1:
            raise ValueError(
                "Multiple profile values found in adata.uns. "
                "Please specify values_key explicitly. "
                f"Available keys: {values_candidates}"
            )
        values_key = values_candidates[0]
        if verbose:
            print(f"Using profile values: adata.uns['{values_key}']")
    else:
        if values_key not in adata.uns:
            raise KeyError(
                f"Values key '{values_key}' not found in adata.uns."
            )

    Z = adata.uns[linkage_key]
    profile_df = adata.uns[values_key]

    # profile_df has observations/groups as rows, variables as columns
    # For silhouette_score, we need samples (variables) as rows
    X = profile_df.T.values
    n_vars = X.shape[0]

    # Limit k to valid range
    max_k = n_vars - 1
    if k > max_k:
        if verbose:
            print(
                f"k={k} exceeds maximum valid clusters ({max_k}). "
                f"Limiting to k={max_k}."
            )
        k = max_k

    # Compute silhouette scores for k from 2 to k
    k_values = list(range(2, k + 1))
    silhouette_scores_list = []

    for n_clusters in k_values:
        labels = fcluster(Z, t=n_clusters, criterion="maxclust")
        score = silhouette_score(X, labels)
        silhouette_scores_list.append(score)

    # Create plot
    fig, _ax = plt.subplots(figsize=figsize)
    _ax.plot(k_values, silhouette_scores_list, marker="o", linewidth=1.5)
    _ax.set_xlabel("Number of clusters (k)")
    _ax.set_ylabel("Average silhouette score")
    _ax.set_title("Silhouette analysis for hierarchical clustering")

    # Set x-axis to show integer ticks only
    _ax.set_xticks(k_values)

    plt.tight_layout()

    if save is not None:
        fig.savefig(save, dpi=300, bbox_inches="tight")
        if verbose:
            print(f"Figure saved to: {save}")

    if show:
        plt.show()

    if ax:
        return _ax

    if not show and save is None and not ax:
        warnings.warn(
            "Function does not do anything. Enable `show`, provide a `save` "
            "path, or set `ax=True`."
        )
        plt.close(fig)

    return None
