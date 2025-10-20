import numpy as np
import scipy.sparse as sp

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
