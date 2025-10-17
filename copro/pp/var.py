import warnings
from functools import partial
import numpy as np
import pandas as pd

def filter_completeness(
    adata,
    axis,
    min_fraction = None,
    min_nr = None,
    zero_to_na = False,
    inplace = True,
    ):
    '''Filter obs or var by completeness.
    Args:
        axis (int, [0,1]): 0 = obs and 1 = var 
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

    if min_fraction:
        completeness = vals.count(axis_i) / adata.shape[axis_i]
        mask_fraction = completeness >= min_fraction
    else:
        mask_fraction = pd.Series([True] * adata.shape[axis])

    if min_nr:
        counts = vals.count(axis_i)
        mask_nr = counts >= min_nr
    else:
        mask_nr = pd.Series([True] * adata.shape[axis])

    mask_filt = pd.Series( mask_fraction.to_numpy() & mask_nr.to_numpy())

    n_removed = sum(~(mask_filt))
    axes_names = ['obs', 'var']
    axis_name = axes_names[axis]
    print(f'{n_removed} {axis_name} removed')

    var_mask = [True] * adata.n_vars
    obs_mask = [True] * adata.n_obs

    if axis == 0:
        obs_mask = mask_filt
    elif axis == 1:
        var_mask = mask_filt

    if inplace:
        adata._inplace_subset_var(var_mask)
        adata._inplace_subset_obs(obs_mask)
    else:
        adata = adata[obs_mask:,var_mask].copy()
        return adata

filter_obs_completeness = partial(
    filter_completeness,
    axis=0,
    )

filter_var_completeness = partial(
    filter_completeness,
    axis=1,
    )
