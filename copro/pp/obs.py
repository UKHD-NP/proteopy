def filter_category_count(
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
