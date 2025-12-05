import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp

from copro.utils.functools import partial_with_docsig
from copro.utils.anndata import check_proteodata, is_proteodata


def filter_axis(
    adata,
    axis,
    min_fraction=None,
    min_count=None,
    group_by=None,
    zero_to_na=False,
    inplace=True,
):
    """
    Filter observations or variables based on non-missing value content.

    This function filters the AnnData object along a specified axis (observations
    or variables) based on the fraction or number of non-missing (np.nan) values.
    Filtering can be performed globally or within groups defined by the `group_by`
    parameter.

    Parameters
    ----------
    adata : anndata.AnnData
        The annotated data matrix to filter.
    axis : int
        The axis to filter on. `0` for observations, `1` for variables.
    min_fraction : float, optional
        The minimum fraction of non-missing values required to keep an observation
        or variable. If `group_by` is provided, this threshold is applied to the
        maximum completeness across all groups.
    min_count : int, optional
        The minimum number of non-missing values required to keep an observation
        or variable. If `group_by` is provided, this threshold is applied to the
        maximum count across all groups.
    group_by : str, optional
        A column key in `adata.obs` (if `axis=1`) or `adata.var` (if `axis=0`)
        used for grouping before applying the filter. The maximum completeness or
        count across the groups is used for filtering.
    zero_to_na : bool, optional
        If True, zeros in the data matrix are treated as missing values (NaN).
    inplace : bool, optional
        If True, modifies the `adata` object in place. Otherwise, returns a
        filtered copy.

    Returns
    -------
    anndata.AnnData or None
        If `inplace=False`, returns a new filtered AnnData object. Otherwise,
        returns `None`.

    Raises
    ------
    KeyError
        If the `group_by` key is not found in the corresponding annotation
        DataFrame.
    """
    check_proteodata(adata)

    if min_fraction is None and min_count is None:
        warnings.warn(
            "Neither `min_fraction` nor `min_count` were provided, so "
            "the function does nothing."
        )
        return None if inplace else adata.copy()

    X = adata.X.copy()
    if zero_to_na:
        if sp.issparse(X):
            X.data[X.data == 0] = np.nan
        else:
            X[X == 0] = np.nan

    if sp.issparse(X):
        X.eliminate_zeros()

    axis_i = 1 - axis
    axis_labels = adata.obs_names if axis == 0 else adata.var_names

    if group_by is not None:
        metadata = adata.obs if axis == 1 else adata.var
        if group_by not in metadata.columns:
            raise KeyError(
                f'`group_by`="{group_by}" not present in '
                f'adata.{"obs" if axis == 1 else "var"}'
            )
        grouping = metadata[group_by]
        unique_groups = grouping.dropna().unique()

        counts_by_group = []
        completeness_by_group = []
        for label in unique_groups:
            mask = (grouping == label).values
            subset = X[mask, :] if axis == 1 else X[:, mask]

            if subset.shape[axis_i] == 0:
                continue

            group_size = subset.shape[axis_i]

            if sp.issparse(subset):
                group_counts = subset.getnnz(axis=axis_i)
            else:
                group_counts = np.count_nonzero(~np.isnan(subset), axis=axis_i)

            df_counts = pd.DataFrame(group_counts, index=axis_labels)
            counts_by_group.append(df_counts)
            if min_fraction is not None:
                df_completeness = df_counts / group_size
                completeness_by_group.append(df_completeness)

        if not counts_by_group:
            counts = pd.Series(0, index=axis_labels, dtype=float)
        else:
            counts = pd.concat(counts_by_group, axis=1).max(axis=1)
        if min_fraction is not None:
            if not completeness_by_group:
                completeness = pd.Series(0, index=axis_labels, dtype=float)
            else:
                completeness = pd.concat(completeness_by_group, axis=1).max(axis=1)
    else:
        if sp.issparse(X):
            counts = pd.Series(X.getnnz(axis=axis_i), index=axis_labels)
        else:
            counts = pd.Series(
                np.count_nonzero(~np.isnan(X), axis=axis_i), index=axis_labels
            )
        if min_fraction is not None:
            num_total = adata.shape[axis_i]
            completeness = counts / num_total

    mask_filt = pd.Series(True, index=axis_labels)
    if min_fraction is not None:
        mask_filt &= completeness >= min_fraction

    if min_count is not None:
        mask_filt &= counts >= min_count

    n_removed = (~mask_filt).sum()
    axis_name = ["obs", "var"][axis]
    print(f"{n_removed} {axis_name} removed")

    if inplace:
        if axis == 0:
            adata._inplace_subset_obs(mask_filt.values)
        else:
            adata._inplace_subset_var(mask_filt.values)
        check_proteodata(adata)
        return None
    else:
        adata_filtered = adata[mask_filt, :] if axis == 0 else adata[:, mask_filt]
        check_proteodata(adata_filtered)
        return adata_filtered


docstr_header = """
Filter observations based on non-missing value content.

This function filters the AnnData object along the `obs` axis based on the
fraction or number of non-missing values (np.nan). Filtering can be performed
globally or within groups defined by the `group_by` parameter.
"""
filter_obs = partial_with_docsig(
    filter_axis,
    axis=0,
    docstr_header=docstr_header,
    )

docstr_header = """
Filter observations based on data completeness.

This function filters the AnnData object along a the obs axis based on the
fraction of non-missing values (np.nan). Filtering can be performed globally
or within groups defined by the `group_by` parameter.
"""
filter_obs_completeness = partial_with_docsig(
    filter_axis,
    axis=0,
    min_count=None,
    )

docstr_header = """
Filter variables based on non-missing value content.

This function filters the AnnData object along the `var` axis based on the
fraction or number of non-missing values (np.nan). Filtering can be performed
globally or within groups defined by the `group_by` parameter.
"""
filter_var = partial_with_docsig(
    filter_axis,
    axis=1,
    )

docstr_header = """
Filter variables based on data completeness.

This function filters the AnnData object along a the var axis based on the
fraction of non-missing values (np.nan). Filtering can be performed globally
or within groups defined by the `group_by` parameter.
"""
filter_var_completeness = partial_with_docsig(
    filter_axis,
    axis=1,
    min_count=None,
    )


def filter_proteins_by_peptide_count(
    adata,
    min_count=None,
    max_count=None,
    protein_col="protein_id",
    inplace=True,
    ):
    """
    Filter proteins by their peptide count.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix with a protein identifier column in ``adata.var``.
    min_count : int or None, optional
        Keep peptides whose proteins have at least this many peptides.
    max_count : int or None, optional
        Keep peptides whose proteins have at most this many peptides.
    protein_col : str, optional (default: "protein_id")
        Column in ``adata.var`` containing protein identifiers.
    inplace : bool, optional (default: True)
        If True, modify ``adata`` in place. Otherwise, return a filtered view.

    Returns
    -------
    None or anndata.AnnData
        ``None`` if ``inplace=True``; otherwise the filtered AnnData view.
    """
    check_proteodata(adata)
    if is_proteodata(adata)[1] != "peptide":
        raise ValueError((
            "`AnnData` object must be in ProteoData peptide format."
            ))

    if min_count is None and max_count is None:
        warnings.warn("Pass at least one argument: min_count | max_count")
        adata_copy = None if inplace else adata.copy()
        if adata_copy is not None:
            check_proteodata(adata_copy)
        return adata_copy

    if min_count is not None:
        if min_count < 0:
            raise ValueError("`min_count` must be non-negative.")
    if max_count is not None:
        if max_count < 0:
            raise ValueError("`max_count` must be non-negative.")
    if (min_count is not None and max_count is not None) and (min_count > max_count):
        raise ValueError("`min_count` cannot be greater than `max_count`.")

    if protein_col not in adata.var.columns:
        raise KeyError(f"`protein_col`='{protein_col}' not found in adata.var")

    proteins = adata.var[protein_col]
    counts = proteins.value_counts()

    keep_mask = pd.Series(True, index=counts.index)
    if min_count is not None:
        keep_mask &= counts >= min_count
    if max_count is not None:
        keep_mask &= counts <= max_count
    protein_ids_keep = counts.index[keep_mask]

    var_keep_mask = proteins.isin(protein_ids_keep)

    if inplace:
        adata._inplace_subset_var(var_keep_mask.values)
        check_proteodata(adata)
        n_proteins_removed = len(counts.index) - len(protein_ids_keep)
        n_peptides_removed = int((~var_keep_mask).sum())
        print(
            f"Removed {n_proteins_removed} proteins and "
            f"{n_peptides_removed} peptides."
        )
        return None

    else:
        new_adata = adata[:, var_keep_mask]
        check_proteodata(new_adata)
        n_proteins_removed = len(counts.index) - len(protein_ids_keep)
        n_peptides_removed = int((~var_keep_mask).sum())
        print(
            f"Removed {n_proteins_removed} proteins and "
            f"{n_peptides_removed} peptides."
        )
        return new_adata


def filter_obs_by_category_count(
    adata,
    category_col,
    min=None,
    max=None,
    ):
    obs = adata.obs[category_col].copy()
    counts = obs.value_counts()

    if min is None and max is None:
        raise ValueError('At least one argument must be passed: min | max')

    counts_filt = counts.copy()

    if min:
        counts_filt = counts_filt[counts_filt >= min]
    if max:
        counts_filt = counts_filt[counts_filt <= max]


    obs_filt = obs[obs.isin(counts_filt.index)].index
    new_adata = adata[obs_filt,:].copy()
    
    return new_adata 


def remove_zero_variance_variables(
    adata,
    group_by=None,
    atol=1e-8,
    inplace=False,
):
    """
    Remove variables (columns) with near-zero variance, skipping NaN values.

    This function removes variables (e.g., genes, peptides, or features) whose
    variance across observations is less than or equal to a given tolerance.
    If a grouping variable is provided via `group_by`, a variable is removed
    if it has near-zero variance (≤ `atol`) in **any** group.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix. The data matrix is taken from ``adata.X``, and
        grouping information (if any) from ``adata.obs``.
    group_by : str or None, optional (default: None)
        Column name in ``adata.obs`` to compute variance per group. If provided,
        variables are removed if their variance is ≤ `atol` within *any* group.
        If None, variance is computed across all observations.
    atol : float, optional (default: 1e-8)
        Absolute tolerance threshold. Variables with variance ≤ `atol` are
        considered to have zero variance and are removed.
    inplace : bool, optional (default: False)
        If True, modifies ``adata`` in place. Otherwise, returns a copy with
        low-variance variables removed.

    Returns
    -------
    None or anndata.AnnData
        If ``inplace=True``, returns None and modifies ``adata`` in place.
        Otherwise, returns a new AnnData object containing only variables
        with variance > `atol`.

    Notes
    -----
    - NaN values are ignored using ``np.nanvar`` (population variance, ddof=0).
    - For sparse matrices, ``adata.X`` is densified during computation. This
      guarantees NaN skipping but may use significant memory for large data.
    - If `group_by` is provided, any variable that has variance ≤ `atol` in
      *any* group is removed globally.
    """
    X = adata.X
    n_vars = adata.n_vars

    def _nanvar_axis0(M):
        if sp.issparse(M):
            M = M.toarray()
        else:
            M = np.asarray(M)

        variances = np.nanvar(
            M,
            axis=0,
            ddof=0,   # population variance (ddof=0)
            )
        return variances

    keep_mask = np.ones(n_vars, dtype=bool)

    if group_by is None:
        var_all = _nanvar_axis0(X)
        keep_mask &= (var_all > atol)
    else:
        if group_by not in adata.obs.columns:
            raise KeyError(f"`group_by`='{group_by}' not found in adata.obs")

        groups = adata.obs[group_by].astype("category")
        zero_any = np.zeros(n_vars, dtype=bool)

        for g in groups.cat.categories:
            idx = np.where(groups.values == g)[0]
            if idx.size == 0:
                continue
            Xg = X[idx, :]
            vg = _nanvar_axis0(Xg)
            zero_any |= (vg <= atol)

        keep_mask &= ~zero_any

    removed = int((~keep_mask).sum())
    print(f"Removed {removed} variables.")

    if inplace:
        adata._inplace_subset_var(keep_mask)
        return None
    else:
        return adata[:, keep_mask].copy()
