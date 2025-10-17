import pandas as pd
import numpy as np
import json

def sanitize_obs_cols(
    adata,
    jsonize_complex=True,
    #datetime_format='%Y-%m-%d %H:%M:%S'
):
    '''Sanitize anndata columns (in-place)

    Makes all columns of adata.obs HDF5-writable by converting unsupported types.

    - Keeps numeric/boolean columns as pandas nullable dtypes.
    #- Converts datetime columns to formatted string *object* dtype.
    - Object columns:
        • If all entries are strings/missing → cast to object dtype.
        • If mixed/complex → JSON-serialize (if jsonize_complex=True), then object dtype.

    Args:
        jsonize_complex (bool): JSON-serialize lists/dicts/sets in object columns.
        datetime_format (str): Format for datetime conversion.
    '''
    def _is_missing(x):
        try:
            return pd.isna(x)
        except Exception:
            return False

    def _looks_like_sequence(x):
        return isinstance(x, (list, tuple, set, np.ndarray))

    def _to_jsonish(x):

        if _is_missing(x):
            return np.nan

        if jsonize_complex and (isinstance(x, dict) or _looks_like_sequence(x)):

            if isinstance(x, set):
                x = sorted(list(x))
            try:
                return json.dumps(x, default=str, ensure_ascii=False)
            except Exception:
                return str(x)
        return str(x)

    def _coerce_series(s):
        if pd.api.types.is_bool_dtype(s):
            return s.astype('boolean')
        if pd.api.types.is_integer_dtype(s):
            return s.astype('int64')
        if pd.api.types.is_float_dtype(s):
            return s.astype('float64')
        if isinstance(s.dtype, pd.CategoricalDtype):
            return s

        # Datetime → formatted string (object dtype)
        #if pd.api.types.is_datetime64_any_dtype(s):
            #return s.dt.strftime(datetime_format).astype('object')

        # Object or mixed columns
        if pd.api.types.is_object_dtype(s):
            only_strings = s.map(lambda x: isinstance(x, (str, np.str_)) or _is_missing(x)).all()
            if only_strings:
                return s.astype('object')

            # Mixed/complex objects → JSON
            out = s.map(_to_jsonish).astype('object')
            return out

        # Fallback untouched
        return s

    if adata.obs is not None and len(adata.obs.columns):
        obs = adata.obs.copy()
        for c in obs.columns:
            obs[c] = _coerce_series(obs[c])
        adata.obs = obs

    return adata
