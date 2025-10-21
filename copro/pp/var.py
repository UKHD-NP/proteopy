import warnings
from functools import partial
import numpy as np
import pandas as pd

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
