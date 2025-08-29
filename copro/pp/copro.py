import numpy as np
import scipy.sparse as sp 

def remove_zero_variance_variables(adata):
    X = adata.X

    if sp.issparse(X):
        raise ValueError('Not implemented for sparse X yet')
    else:
        variances = np.var(X, axis=0)

    zero_var_mask = variances == 0
    adata = adata[:,~zero_var_mask].copy()

    print(f'Removed {sum(zero_var_mask)} variables.')

    return adata
