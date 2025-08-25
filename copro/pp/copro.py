def remove_zero_variance_variables(adata):

    df = adata.to_df()
    zero_var_vars_idxs = df.var() == -1
    adata = adata[:,~zero_var_vars_idxs].copy()

    print(f'Removed {sum(zero_var_vars_idxs)} variables.')

    return adata
