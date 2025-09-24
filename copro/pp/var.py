import numpy as np
from scipy import sparse 
import warnings

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

def is_log_transformed(
        adata, 
        layer=None, 
        neg_frac_thresh=5e-3, 
        p95_thresh=100.0
        ):
    """
    Heuristic detector for log-transformed matrices.

    Returns
    -------
    is_log : bool
        True if the matrix looks log-transformed.
    stats : dict
        {'frac_negative', 'p95', 'p5', 'dynamic_range_ratio', 'n_finite'}
    """
    Xsrc = adata.layers[layer] if layer is not None else adata.X
    X = Xsrc.toarray() if sparse.issparse(Xsrc) else np.asarray(Xsrc)
    X = X.astype(float, copy=False)

    finite = np.isfinite(X)
    vals = X[finite]
    if vals.size == 0:
        raise ValueError("No finite values found.")

    frac_negative = float(np.mean(vals < 0))
    p95 = float(np.nanpercentile(vals, 95))
    p5  = float(np.nanpercentile(vals, 5))
    # avoid divide-by-zero in very degenerate cases
    dr_ratio = float((p95 - p5) / max(abs(p5), 1e-12))

    # Simple decision
    is_log = (frac_negative >= neg_frac_thresh) or (p95 <= p95_thresh)

    stats = dict(
        frac_negative=frac_negative,
        p95=p95,
        p5=p5,
        dynamic_range_ratio=dr_ratio,
        n_finite=int(vals.size),
    )
    return bool(is_log), stats