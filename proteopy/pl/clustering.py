"""Clustering visualization tools for proteomics data."""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import silhouette_score

from proteopy.utils.anndata import check_proteodata
from proteopy.utils.parsers import (
    _resolve_hclustv_keys,
    _resolve_hclustv_profile_key,
)


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

    linkage_key, values_key = _resolve_hclustv_keys(
        adata,
        linkage_key,
        values_key,
        verbose
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


def hclustv_elbow(
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
    Plot within-cluster sum of squares (elbow plot) for hierarchical clustering.

    Evaluates clustering by computing WCSS for cluster counts ranging from
    1 to ``k``. The "elbow" point where WCSS reduction diminishes suggests
    an optimal cluster count.

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
        Maximum number of clusters to evaluate. WCSS is computed for
        cluster counts from 1 to ``k`` (inclusive).
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
        if ``k < 1``.
    KeyError
        If the specified ``linkage_key`` or ``values_key`` is not
        found.

    Examples
    --------
    >>> import proteopy as pr
    >>> adata = pr.datasets.example_peptide_data()
    >>> pr.tl.hclustv_tree(adata, group_by="condition")
    >>> pr.pl.hclustv_elbow(adata, k=10)
    """
    check_proteodata(adata)

    if k < 1:
        raise ValueError("k must be at least 1 to compute WCSS.")

    linkage_key, values_key = _resolve_hclustv_keys(
        adata,
        linkage_key,
        values_key,
        verbose
    )

    Z = adata.uns[linkage_key]
    profile_df = adata.uns[values_key]

    # profile_df has observations/groups as rows, variables as columns
    # For WCSS, we need samples (variables) as rows
    X = profile_df.T.values
    n_vars = X.shape[0]

    # Limit k to valid range
    max_k = n_vars
    if k > max_k:
        if verbose:
            print(
                f"k={k} exceeds maximum valid clusters ({max_k}). "
                f"Limiting to k={max_k}."
            )
        k = max_k

    # Compute WCSS for k from 1 to k
    k_values = list(range(1, k + 1))
    wcss_list = []

    for n_clusters in k_values:
        labels = fcluster(Z, t=n_clusters, criterion="maxclust")
        wcss = _compute_wcss(X, labels)
        wcss_list.append(wcss)

    # Create plot
    fig, _ax = plt.subplots(figsize=figsize)
    _ax.plot(k_values, wcss_list, marker="o", linewidth=1.5)
    _ax.set_xlabel("Number of clusters (k)")
    _ax.set_ylabel("Within-cluster sum of squares (WCSS)")
    _ax.set_title("Elbow plot for hierarchical clustering")

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


def hclustv_profile_intensities(
    adata: ad.AnnData,
    profiles: str | list[str] | None = None,
    profile_key: str = 'auto',
    group_by: str | pd.Series | dict | None = None,
    sort_by: str | pd.Series | dict | None = None,
    order: list[str] | None = None,
    n_cols: int = 2,
    n_rows: int = 3,
    title: str | None = None,
    titles: list[str] | dict[str, str] | None = None,
    xlabel_rotation: float = 45,
    sort_by_label_rotation: float = 0,
    ylabel: str = "Intensity",
    marker: str = 'o',
    markersize: float = 6,
    linewidth: float = 1.5,
    errorbar: str | tuple = 'se',
    color: str | None = None,
    figsize: tuple[float, float] | None = None,
    show: bool = True,
    ax: bool = False,
    save: str | Path | None = None,
    verbose: bool = True,
) -> list[Axes] | None:
    """
    Plot cluster profile intensities across observations.

    Displays line plots for each cluster profile showing how intensity
    varies across observations. When ``group_by`` is specified, observations
    are grouped and error bars are displayed.

    Parameters
    ----------
    adata : AnnData
        :class:`~anndata.AnnData` with cluster profiles stored in ``.uns``
        from :func:`proteopy.tl.hclustv_profiles`.
    profiles : str | list[str] | None
        Profile column(s) to plot from the profiles DataFrame. When ``None``,
        plots the first 6 profiles or fewer if not available. Can be a single
        profile name (e.g., ``"01"``) or a list of names.
    profile_key : str
        Key in ``adata.uns`` for the profiles DataFrame. When ``'auto'``,
        auto-detects keys matching ``hclustv_profiles;*``.
    group_by : str | pd.Series | dict | None
        Grouping for x-axis observations. When ``str``, uses the column
        from ``adata.obs`` to group observations and display error bars.
        When ``pd.Series``, uses Series index as observation keys and values
        as group labels. When ``dict``, maps observation indices to group
        labels directly. When ``None``, plots individual observations without
        grouping. If the column or Series is categorical, the category order
        is respected for x-axis ordering. Mutually exclusive with ``sort_by``.
    sort_by : str | pd.Series | dict | None
        Sort individual observations by group membership without aggregating.
        When ``str``, uses the column from ``adata.obs``. When ``pd.Series``,
        uses Series index as observation keys and values as sort groups.
        When ``dict``, maps observation indices to sort groups directly.
        Observations are ordered by their group, with group order determined
        by ``order`` (if provided) or categorical order (if categorical).
        Mutually exclusive with ``group_by``.
    order : list[str] | None
        Order of groups on the x-axis. When ``group_by`` is specified,
        controls the order of grouped categories. When ``sort_by`` is
        specified, controls the order in which sort groups appear.
        When ``None``, uses categorical order if available, otherwise
        sorted alphabetically.
    n_cols : int
        Number of columns in the subplot grid.
    n_rows : int
        Number of rows in the subplot grid.
    title : str | None
        Overall figure title. When ``None``, no suptitle is added.
    titles : list[str] | dict[str, str] | None
        Custom titles for each subplot. When ``list``, must have the same
        length as the number of plotted profiles. When ``dict``, maps
        profile/cluster names to custom titles. When ``None``, uses
        default titles (``"Cluster {profile_name}"``).
    xlabel_rotation : float
        Rotation angle (degrees) for x-axis tick labels.
    sort_by_label_rotation : float
        Rotation angle (degrees) for sort group labels when ``sort_by``
        is used.
    ylabel : str
        Label for the y-axis of each subplot.
    marker : str
        Marker style for data points.
    markersize : float
        Size of data point markers.
    linewidth : float
        Width of connecting lines.
    errorbar : str | tuple
        Error bar style for grouped data. Passed to ``sns.lineplot``.
        Common options: ``'se'`` (standard error), ``'sd'`` (standard
        deviation), ``'ci'`` (confidence interval), ``('ci', 95)``.
    color : str | None
        Color for the line and markers. When ``None``, uses default palette.
    figsize : tuple[float, float] | None
        Figure size. When ``None``, auto-computed based on grid dimensions.
    show : bool
        Display the figure.
    ax : bool
        Return the Matplotlib Axes objects instead of displaying.
    save : str | Path | None
        File path for saving the figure.
    verbose : bool
        Print status messages including auto-detected keys.

    Returns
    -------
    list[Axes] | None
        List of Axes objects when ``ax`` is ``True``; otherwise ``None``.

    Raises
    ------
    ValueError
        If no cluster profiles are found in ``adata.uns``, if multiple
        candidates exist and ``profile_key`` is not specified, or if
        specified profiles are not found in the DataFrame.
    KeyError
        If the specified ``profile_key`` is not found, or if ``group_by``
        column is not found in ``adata.obs``.
    TypeError
        If the profiles data is not a pandas DataFrame.

    Examples
    --------
    >>> import proteopy as pr
    >>> adata = pr.datasets.karayel_2020()
    >>> pr.tl.hclustv_tree(adata, group_by="condition")
    >>> pr.tl.hclustv_cluster_ann(adata, k=5)
    >>> pr.tl.hclustv_profiles(adata)
    >>> pr.pl.hclustv_profile_intensities(adata)

    Plot with grouping and error bars:

    >>> pr.pl.hclustv_profile_intensities(adata, group_by="condition")

    Plot specific profiles:

    >>> pr.pl.hclustv_profile_intensities(adata, profiles=["01", "03"])
    """
    import seaborn as sns

    check_proteodata(adata)

    # Resolve profiles key
    resolved_key = _resolve_hclustv_profile_key(
        adata, profile_key, verbose
    )

    profiles_df = adata.uns[resolved_key]

    # Validate profiles DataFrame
    if not isinstance(profiles_df, pd.DataFrame):
        raise TypeError(
            f"Expected profiles data to be DataFrame, "
            f"got {type(profiles_df).__name__}."
        )

    if profiles_df.empty:
        raise ValueError("Profiles DataFrame is empty.")

    available_profiles = profiles_df.columns.tolist()

    # Determine which profiles to plot
    if profiles is None:
        max_profiles = n_cols * n_rows
        selected_profiles = available_profiles[:min(6, max_profiles)]
    elif isinstance(profiles, str):
        selected_profiles = [profiles]
    else:
        selected_profiles = list(profiles)

    # Validate selected profiles exist
    missing_profiles = [
        p for p in selected_profiles if p not in available_profiles
    ]
    if missing_profiles:
        raise ValueError(
            f"Profiles not found in DataFrame: {missing_profiles}. "
            f"Available profiles: {available_profiles}"
        )

    if not selected_profiles:
        raise ValueError("No profiles to plot.")

    # Limit to grid capacity
    max_plots = n_cols * n_rows
    if len(selected_profiles) > max_plots:
        if verbose:
            print(
                f"Only plotting first {max_plots} profiles "
                f"(grid capacity: {n_rows}x{n_cols})."
            )
        selected_profiles = selected_profiles[:max_plots]

    # Validate titles parameter
    if titles is not None:
        if isinstance(titles, list):
            if len(titles) != len(selected_profiles):
                raise ValueError(
                    f"titles list length ({len(titles)}) must match "
                    f"number of profiles ({len(selected_profiles)})."
                )
        elif not isinstance(titles, dict):
            raise TypeError(
                f"titles must be list, dict, or None, "
                f"got {type(titles).__name__}."
            )

    # Validate mutually exclusive parameters
    if group_by is not None and sort_by is not None:
        raise ValueError(
            "group_by and sort_by are mutually exclusive. "
            "Use group_by to aggregate observations, or sort_by to "
            "order individual observations by group membership."
        )

    # Helper to extract mapping and category order
    def _extract_mapping(param, param_name):
        mapping = None
        cat_order = None
        if param is None:
            pass
        elif isinstance(param, str):
            if param not in adata.obs.columns:
                raise KeyError(
                    f"{param_name} column '{param}' not found in adata.obs."
                )
            obs_col_data = adata.obs[param]
            if hasattr(obs_col_data, 'cat'):
                cat_order = obs_col_data.cat.categories.tolist()
            obs_in_profiles = profiles_df.index.intersection(adata.obs_names)
            mapping = adata.obs.loc[obs_in_profiles, param].to_dict()
        elif isinstance(param, pd.Series):
            if hasattr(param, 'cat'):
                cat_order = param.cat.categories.tolist()
            mapping = param.to_dict()
        elif isinstance(param, dict):
            mapping = param
        else:
            raise TypeError(
                f"{param_name} must be str, pd.Series, dict, or None, "
                f"got {type(param).__name__}."
            )
        return mapping, cat_order

    group_mapping, group_category_order = _extract_mapping(group_by, 'group_by')
    sort_mapping, sort_category_order = _extract_mapping(sort_by, 'sort_by')

    # Build long-form DataFrame for seaborn
    plot_data = profiles_df[selected_profiles].copy()
    plot_data = plot_data.reset_index()
    plot_data = plot_data.melt(
        id_vars=[plot_data.columns[0]],
        var_name='profile',
        value_name='intensity',
    )
    obs_col = plot_data.columns[0]

    # Determine x variable and apply grouping/sorting
    if group_mapping is not None:
        plot_data['group'] = plot_data[obs_col].map(group_mapping)
        plot_data = plot_data.dropna(subset=['group'])
        x_var = 'group'
        category_order = group_category_order
    elif sort_mapping is not None:
        plot_data['_sort_group'] = plot_data[obs_col].map(sort_mapping)
        plot_data = plot_data.dropna(subset=['_sort_group'])
        x_var = obs_col
        category_order = sort_category_order
    else:
        x_var = obs_col
        category_order = None

    # Determine group/sort order
    if order is not None:
        group_order = order
    elif category_order is not None:
        if group_mapping is not None:
            present_values = set(plot_data['group'].unique())
        elif sort_mapping is not None:
            present_values = set(plot_data['_sort_group'].unique())
        else:
            present_values = set()
        group_order = [c for c in category_order if c in present_values]
    elif group_mapping is not None:
        group_order = sorted(plot_data['group'].unique())
    elif sort_mapping is not None:
        group_order = sorted(plot_data['_sort_group'].unique())
    else:
        group_order = None

    # Filter to only include specified groups
    if group_order is not None:
        if group_mapping is not None:
            plot_data = plot_data[plot_data['group'].isin(group_order)]
        elif sort_mapping is not None:
            plot_data = plot_data[plot_data['_sort_group'].isin(group_order)]

    # Determine x-axis order
    if group_mapping is not None:
        x_order = group_order
    elif sort_mapping is not None:
        # Sort observations by their group membership
        plot_data['_sort_group'] = pd.Categorical(
            plot_data['_sort_group'], categories=group_order, ordered=True
        )
        sorted_obs = (
            plot_data[[obs_col, '_sort_group']]
            .drop_duplicates()
            .sort_values('_sort_group')[obs_col]
            .tolist()
        )
        x_order = sorted_obs
    else:
        x_order = profiles_df.index.tolist()

    # Convert x variable to categorical with specified order
    plot_data[x_var] = pd.Categorical(
        plot_data[x_var], categories=x_order, ordered=True
    )
    plot_data = plot_data.sort_values(x_var)

    # Determine figure size
    if figsize is None:
        fig_width = 4 * n_cols
        fig_height = 3 * n_rows
        figsize = (fig_width, fig_height)

    # Create figure and axes
    n_profiles = len(selected_profiles)
    actual_rows = min(n_rows, (n_profiles + n_cols - 1) // n_cols)

    fig, axes_array = plt.subplots(
        actual_rows,
        n_cols,
        figsize=figsize,
        squeeze=False,
    )
    axes_flat = axes_array.flatten()

    returned_axes = []

    for idx, profile_name in enumerate(selected_profiles):
        _ax = axes_flat[idx]
        profile_data = plot_data[plot_data['profile'] == profile_name]

        if group_mapping is not None:
            sns.lineplot(
                data=profile_data,
                x=x_var,
                y='intensity',
                err_style='bars',
                errorbar=errorbar,
                err_kws={'capsize': 4},
                marker=marker,
                markersize=markersize,
                linewidth=linewidth,
                color=color,
                ax=_ax,
                sort=False,
            )
        else:
            sns.lineplot(
                data=profile_data,
                x=x_var,
                y='intensity',
                errorbar=None,
                marker=marker,
                markersize=markersize,
                linewidth=linewidth,
                color=color if color else '#4C78A8',
                ax=_ax,
                sort=False,
            )

            # Add sort group labels below plot area
            if sort_mapping is not None and group_order is not None:
                # Extend y-axis to make room for labels
                ymin, ymax = _ax.get_ylim()
                y_range = ymax - ymin
                new_ymin = ymin - 0.15 * y_range
                _ax.set_ylim(new_ymin, ymax)

                # Position for labels (just below original ymin)
                label_y = ymin - 0.05 * y_range

                # Build mapping of obs position in x_order
                obs_to_pos = {obs: i for i, obs in enumerate(x_order)}

                for group_label in group_order:
                    # Find observations belonging to this group
                    group_obs = [
                        obs for obs in x_order
                        if sort_mapping.get(obs) == group_label
                    ]
                    if not group_obs:
                        continue

                    # Get x positions for this group's observations
                    positions = [obs_to_pos[obs] for obs in group_obs]
                    center_x = (min(positions) + max(positions)) / 2

                    # Add label below plot area
                    _ax.text(
                        center_x,
                        label_y,
                        str(group_label),
                        ha='center',
                        va='top',
                        fontsize=9,
                        rotation=sort_by_label_rotation,
                    )

        # Set x-axis tick labels with rotation
        _ax.tick_params(axis='x', rotation=xlabel_rotation)
        for label in _ax.get_xticklabels():
            label.set_ha('right')

        _ax.set_xlabel('')
        _ax.set_ylabel(ylabel)

        # Set subplot title
        if titles is None:
            subplot_title = f"Profile {profile_name}"
        elif isinstance(titles, list):
            subplot_title = titles[idx]
        else:
            subplot_title = titles.get(profile_name, f"Profile {profile_name}")
        _ax.set_title(subplot_title)

        returned_axes.append(_ax)

    # Hide unused axes
    for idx in range(n_profiles, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Add overall title
    if title is not None:
        fig.suptitle(title, fontsize=12, y=1.02)

    plt.tight_layout()

    if save is not None:
        fig.savefig(save, dpi=300, bbox_inches="tight")
        if verbose:
            print(f"Figure saved to: {save}")

    if show:
        plt.show()

    if ax:
        return returned_axes

    if not show and save is None and not ax:
        warnings.warn(
            "Function does not do anything. Enable `show`, provide a `save` "
            "path, or set `ax=True`."
        )
        plt.close(fig)

    return None
