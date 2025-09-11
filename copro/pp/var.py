import numpy as np

def filter_var_completeness(
    adata,
    min_fraction,
    zero_to_na = False,
    inplace = True,
    ):
    vals = adata.to_df().copy()

    if zero_to_na:
        vals = vals.replace(0, np.nan)

    completeness = vals.count() / adata.n_obs
    var_subs = completeness[completeness > min_fraction].index

    n_removed = sum(~(completeness > min_fraction))
    print(f'{n_removed} var removed')

    if inplace:
        adata._inplace_subset_var(var_subs)
    else:
        adata = adata[:,var_subs].copy()
        return adata
