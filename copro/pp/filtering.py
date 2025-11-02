import warnings
from functools import partial
import numpy as np
import pandas as pd
import scipy.sparse as sp


def filter_axis(
    adata,
    axis,
    min_fraction = None,
    min_nr = None,
    group_by = None,
    zero_to_na = False,
    inplace = True,
    ):
    '''
    Filter obs or var by completeness.
    Args:
        axis (int, [0,1]): 0 = obs and 1 = var
        group_by (str|None): Column used to compute completeness/counts within
            groups; takes the maximum across groups.
    '''
    # Check input
    if not min_fraction and not min_nr:
        warnings.warn(
            'Neither min_fraction nor min_nr were provided so '
            'function does nothing.'
            )

    vals = adata.to_df().copy()
    if zero_to_na:
        vals = vals.replace(0, np.nan)

    axes_i = [1, 0]
    axis_i = axes_i[axis]
    axis_labels = [vals.index, vals.columns][axis]

    counts = None
    completeness = None

    if group_by is not None:
        metadata = adata.obs if axis == 1 else adata.var
        if group_by not in metadata.columns:
            raise KeyError(
                f'group_by "{group_by}" not present in '
                f'adata.{"obs" if axis == 1 else "var"}'
                )
        grouping = metadata[group_by]
        group_labels = grouping.dropna().unique().tolist()
        counts_by_group = []
        fractions_by_group = []

        for label in group_labels:
            if axis == 1:
                subset = vals.loc[grouping == label, :]
            else:
                subset = vals.loc[:, grouping == label]
            if subset.shape[axis_i] == 0:
                continue

            group_counts = subset.count(axis_i)
            counts_by_group.append(group_counts)

            if min_fraction:
                fractions_by_group.append( group_counts / subset.shape[axis_i])

        if counts_by_group:
            counts = pd.concat(counts_by_group, axis=1).max(axis=1)
        else:
            counts = pd.Series(0, index=axis_labels, dtype=float)

        if min_fraction:
            if fractions_by_group:
                completeness = pd.concat(fractions_by_group, axis=1).max(axis=1)
            else:
                completeness = pd.Series(0, index=axis_labels, dtype=float)
    else:
        counts = vals.count(axis_i)
        if min_fraction:
            completeness = counts / adata.shape[axis_i]

    if min_fraction:
        mask_fraction = completeness >= min_fraction
    else:
        mask_fraction = pd.Series(True, index=axis_labels)

    if min_nr:
        mask_nr = counts >= min_nr
    else:
        mask_nr = pd.Series(True, index=axis_labels)

    mask_filt = mask_fraction & mask_nr

    n_removed = (~mask_filt).sum()
    axes_names = ['obs', 'var']
    axis_name = axes_names[axis]
    print(f'{n_removed} {axis_name} removed')

    var_mask = [True] * adata.n_vars
    obs_mask = [True] * adata.n_obs

    if axis == 0:
        obs_mask = mask_filt.tolist()
    elif axis == 1:
        var_mask = mask_filt.tolist()

    if inplace:
        adata._inplace_subset_var(var_mask)
        adata._inplace_subset_obs(obs_mask)
    else:
        adata = adata[obs_mask:,var_mask].copy()
        return adata

filter_obs = partial(
    filter_axis,
    axis=0,
    )

filter_var = partial(
    filter_axis,
    axis=1,
    )

filter_obs_completeness = partial(
    filter_axis,
    axis=0,
    min_nr=None,
    )

filter_var_completeness = partial(
    filter_axis,
    axis=1,
    min_nr=None,
    )

filter_obs_by_min_nr_var = partial(
    filter_axis,
    axis=0,
    min_fraction=None,
    )

filter_var_by_min_nr_obs = partial(
    filter_axis,
    axis=1,
    min_fraction=None,
    )


def filter_genes_by_peptide_count(
    adata,
    min = None,
    max = None,
    gene_col = 'protein_id',
    ):
    if not min and not max:
        raise ValueError('Must pass either min or max arguments.')

    genes = adata.var[gene_col]
    counts = genes.value_counts()

    if min:
        gene_ids_filt = counts[counts >= min].index
    elif max:
        gene_ids_filt = counts[counts <= max].index
    elif min and max:
        gene_ids_filt = counts[(counts >= min) & (counts <= max)]
    else:
        raise ValueError('Pass at least one argument: min | max')
    
    var_filt = adata.var[genes.isin(gene_ids_filt)].index
    new_adata = adata[:,var_filt]

    n_genes_removed = len(set(genes.unique()) - set(gene_ids_filt.unique()))
    n_peptides_removed = sum(~genes.isin(gene_ids_filt))
    print(f'Removed {str(n_genes_removed)} genes and {str(n_peptides_removed)} peptides.')

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
