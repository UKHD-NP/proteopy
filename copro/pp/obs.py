def filter_category_count(
    adata,
    category_col,
    min=None,
    max=None,
    ):
    obs = adata.obs[category_col].copy()
    counts = obs.value_counts()

    if min:
        counts_filt = counts[counts >= min]
    if max:
        counts_filt = counts[counts <= max]

    obs_filt = obs[obs.isin(counts_filt.index)].index
    new_adata = adata[obs_filt,:].copy()
    
    return new_adata 
