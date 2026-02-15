import warnings
import json
import re


import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse


def _has_infinite_values(X) -> bool:
    """
    Check if the matrix X contains any infinite values (np.inf or -np.inf).

    Handles both dense numpy arrays and scipy sparse matrices.
    """
    if X is None:
        return False
    if sparse.issparse(X):
        return np.any(np.isinf(X.data))
    return np.any(np.isinf(X))


def _axis_len(a, axis: int = 0) -> int:
    """
    returns the length along `axis` using .shape if available,
    otherwise falls back to len(a).
    """
    # Prefer shape if present (numpy, pandas, scipy.sparse, torch, etc.)
    shape = getattr(a, "shape", None)
    if shape is not None:
        try:
            return int(shape[axis])
        except Exception:
            pass
    # Fallback
    try:
        return int(len(a))
    except Exception as e:
        raise TypeError(
            (
                "Object of type "
                f"{type(a)!r} does not expose a usable "
                f"length along axis {axis}."
            )
        ) from e


def _check_2d_shape(adata: AnnData) -> None:
    """
    Ensure .X is 2-dimensional if present.
    """
    if adata.X is not None:
        shp = getattr(adata.X, "shape", ())
        if len(shp) != 2:
            raise ValueError(
                f"X needs to be 2-dimensional, not {len(shp)}-dimensional."
            )


def _check_axis_synchronization(adata: AnnData) -> None:
    """
    Ensure obs/var are synchronized with obs_names/var_names and that
    obsm/varm first dimensions match n_obs/n_vars respectively.
    """
    # obs axis
    if len(adata.obs) != len(adata.obs_names):
        raise ValueError(
            (
                "Length of obs "
                f"({len(adata.obs)}) does not match length of obs_names "
                f"({len(adata.obs_names)})."
            )
        )
    if not adata.obs.index.equals(adata.obs_names):
        raise ValueError("obs.index must exactly match obs_names.")

    # var axis
    if len(adata.var) != len(adata.var_names):
        raise ValueError(
            (
                "Length of var "
                f"({len(adata.var)}) does not match length of var_names "
                f"({len(adata.var_names)})."
            )
        )
    if not adata.var.index.equals(adata.var_names):
        raise ValueError("var.index must exactly match var_names.")

    # obsm dimensions
    for key, arr in adata.obsm.items():
        n0 = _axis_len(arr, 0)
        if n0 != adata.n_obs:
            raise ValueError(
                (
                    f"obsm['{key}'] must have first dimension equal to "
                    f"n_obs ({adata.n_obs}), but has {n0}."
                )
            )

    # varm dimensions
    for key, arr in adata.varm.items():
        n0 = _axis_len(arr, 0)
        if n0 != adata.n_vars:
            raise ValueError(
                (
                    f"varm['{key}'] must have first dimension equal to "
                    f"n_vars ({adata.n_vars}), but has {n0}."
                )
            )


def _check_dimensions(adata: AnnData) -> None:
    """
    Composite dimension/index checks for an AnnData object.
    """
    _check_2d_shape(adata)
    _check_axis_synchronization(adata)


def _check_uniqueness(adata: AnnData, warn_only: bool = False) -> None:
    """
    Check uniqueness of obs/var indices.
    Raises a ValueError by default if duplicates are found.

    Parameters
    ----------
    adata : AnnData
        The AnnData object to validate.
    warn_only : bool, optional (default: False)
        If True, duplicates will only trigger warnings instead of errors.
    """
    def _handle_duplicates(axis_name: str, index):
        dup_mask = index.duplicated()
        if dup_mask.any():
            dups = index[dup_mask].unique()
            shown = ", ".join(map(repr, dups[:10]))
            extra = "" if len(dups) <= 10 else f" … and {len(dups) - 10} more"
            msg = (
                f"Duplicate names detected in {axis_name} "
                f"axis: {shown}{extra}. "
                "Consider calling `.obs_names_make_unique()` or "
                "`.var_names_make_unique()` (depending on axis) to fix."
            )
            if warn_only:
                warnings.warn(msg, UserWarning, stacklevel=2)
            else:
                raise ValueError(msg)

    _handle_duplicates("obs", adata.obs.index)
    _handle_duplicates("var", adata.var.index)


def _check_structure(adata: AnnData) -> None:
    """
    High-level structure checks for an AnnData object.
    """
    _check_uniqueness(adata)
    _check_dimensions(adata)


def _var_column_matches_axis(adata: AnnData, column: str) -> bool:
    """Return True when the chosen .var column exactly
    matches both axis definitions."""
    if column not in adata.var.columns:
        return False

    series = adata.var[column]
    if series.isna().any():
        return False

    values = series.to_numpy()
    var_names = adata.var_names.to_numpy()
    var_index = adata.var.index.to_numpy()

    matches_names = np.array_equal(values, var_names)
    matches_index = np.array_equal(values, var_index)
    return matches_names and matches_index


def _has_multiple_values_per_cell(
    series: pd.Series, delimiters: str = " ,;"
) -> bool:
    """Return True when any entry contains more than one
    value separated by delimiters."""
    if series.isna().any():
        return True

    tokens = series.astype(str).str.strip()
    if (tokens == "").any():
        return True

    pattern = f"[{re.escape(delimiters)}]"
    return tokens.str.contains(pattern).any()


# ------------------------------------------------------------------
# is_proteodata helpers
# ------------------------------------------------------------------

_FAIL = (False, None)


def _validation_fail(msg, raise_error):
    """Raise ValueError or return ``_FAIL`` depending on *raise_error*."""
    if raise_error:
        raise ValueError(msg)
    return _FAIL


def _check_matrix_values(adata, raise_error, layers):
    """Validate .X and requested layers for infinite values."""
    if adata.X is not None and _has_infinite_values(adata.X):
        return _validation_fail(
            "AnnData.X contains infinite values (np.inf or "
            "-np.inf). Please remove or replace infinite values "
            "before proceeding.",
            raise_error,
        )

    if layers is not None:
        if isinstance(layers, str):
            layers = [layers]
        for layer_key in layers:
            if layer_key not in adata.layers:
                return _validation_fail(
                    f"Layer '{layer_key}' not found in "
                    f"adata.layers. Available layers: "
                    f"{list(adata.layers.keys())}.",
                    raise_error,
                )
            if _has_infinite_values(adata.layers[layer_key]):
                return _validation_fail(
                    f"adata.layers['{layer_key}'] contains "
                    f"infinite values (np.inf or -np.inf). "
                    f"Please remove or replace infinite values "
                    f"before proceeding.",
                    raise_error,
                )
    return None


def _check_obs_requirements(adata, raise_error):
    """Validate obs has sample_id and no misplaced var columns."""
    obs = adata.obs
    if "sample_id" not in obs.columns:
        return _validation_fail(
            "Missing required 'sample_id' column in adata.obs. "
            "Each observation (row) must have a sample "
            "identifier equal to the AnnData index.",
            raise_error,
        )

    sample_ids = obs["sample_id"].to_numpy()
    obs_names = adata.obs_names.to_numpy()
    obs_index = obs.index.to_numpy()
    if (
        not np.array_equal(sample_ids, obs_names)
        or not np.array_equal(sample_ids, obs_index)
    ):
        return _validation_fail(
            "adata.obs['sample_id'] does not match "
            "adata.obs_names. The 'sample_id' column "
            "must be identical to the obs index.",
            raise_error,
        )

    misplaced_in_obs = [
        col for col in ("protein_id", "peptide_id")
        if col in obs.columns
    ]
    if misplaced_in_obs:
        return _validation_fail(
            f"Found column(s) {misplaced_in_obs} in adata.obs. "
            "These columns belong in adata.var, not adata.obs. "
            "Observations (rows) represent samples, while "
            "variables (columns) represent peptides or proteins.",
            raise_error,
        )
    return None


def _check_var_requirements(adata, raise_error):
    """Validate var does not contain misplaced obs columns."""
    if "sample_id" in adata.var.columns:
        return _validation_fail(
            "Found 'sample_id' column in adata.var. "
            "This column belongs in adata.obs, not adata.var. "
            "Observations (rows) represent samples, while "
            "variables (columns) represent peptides or proteins.",
            raise_error,
        )
    return None


def _check_peptide_level(adata, raise_error):
    """Validate peptide-level specific requirements."""
    var = adata.var
    if "protein_id" not in var.columns:
        return _validation_fail(
            "Found a 'peptide_id' column but no 'protein_id' "
            "column. If working at peptide-level, a "
            "peptide_id -> protein_id mapping must be included.",
            raise_error,
        )

    if var["peptide_id"].isna().any():
        return _validation_fail(
            "Column 'peptide_id' in adata.var contains missing "
            "values (NaN/None). All peptide identifiers must "
            "be non-null.",
            raise_error,
        )

    if var["protein_id"].isna().any():
        return _validation_fail(
            "Column 'protein_id' in adata.var contains missing "
            "values (NaN/None). All protein identifiers must "
            "be non-null.",
            raise_error,
        )

    if not _var_column_matches_axis(adata, "peptide_id"):
        return _validation_fail(
            "Found a 'peptide_id' column but it does "
            "not match AnnData.var_names. "
            "If your data are protein-level, please rename "
            "or remove the 'peptide_id' column.",
            raise_error,
        )

    if _has_multiple_values_per_cell(var["protein_id"]):
        return _validation_fail(
            "Detected peptides mapping to multiple proteins. "
            "Ensure each peptide maps to exactly one protein_id.",
            raise_error,
        )
    return None


def _check_protein_level(adata, raise_error):
    """Validate protein-level specific requirements."""
    var = adata.var
    if var["protein_id"].isna().any():
        return _validation_fail(
            "Column 'protein_id' in adata.var contains missing "
            "values (NaN/None). All protein identifiers must "
            "be non-null.",
            raise_error,
        )

    if not _var_column_matches_axis(adata, "protein_id"):
        return _validation_fail(
            "Found a 'protein_id' column but it does "
            "not match AnnData.var_names.",
            raise_error,
        )
    return None


def is_proteodata(
    adata: AnnData,
    *,
    raise_error: bool = False,
    layers: str | list[str] | None = None,
) -> tuple[bool, str | None]:
    """
    Check whether the AnnData object stores peptide- or protein-level
    proteomics data.

    Parameters
    ----------
    adata
        AnnData object whose `.var` annotations will be inspected.
    raise_error
        If True, raise a ValueError for proteomics-specific validation
        failures instead of returning False.
    layers
        Optional layer key or list of layer keys in ``adata.layers`` to
        validate. Each specified layer matrix is checked for infinite
        values, the same way ``.X`` is checked.

    Returns
    -------
    tuple[bool, str | None]
        ``(True, "peptide")`` if the data satisfy the peptide-level
        assumptions, ``(True, "protein")`` if they satisfy the
        protein-level assumptions, otherwise ``(False, None)`` when
        ``raise_error`` is False.

    Notes
    -----
    Peptide-level data must provide both ``.var["peptide_id"]`` and
    ``.var["protein_id"]``. Every ``peptide_id`` value must be unique,
    and the column must match ``adata.var_names`` (and the ``.var``
    index) exactly. Each peptide must map to exactly one
    ``protein_id``. Neither column may contain missing values.

    Protein-level data must provide ``.var["protein_id"]``, must *not*
    contain a ``peptide_id`` column, and the ``protein_id`` values must
    match ``adata.var_names`` (and the ``.var`` index) exactly while
    also being unique. The column must not contain missing values.

    Set ``raise_error=True`` to raise a ValueError instead of returning
    False when the proteomics-specific validation fails.
    """
    if not isinstance(adata, AnnData):
        raise TypeError("is_proteodata expects an AnnData object.")

    if adata.var is None or adata.var.empty:
        return _FAIL

    _check_structure(adata)

    result = _check_matrix_values(adata, raise_error, layers)
    if result is not None:
        return result

    result = _check_obs_requirements(adata, raise_error)
    if result is not None:
        return result

    result = _check_var_requirements(adata, raise_error)
    if result is not None:
        return result

    var = adata.var
    has_peptide_id = "peptide_id" in var.columns

    if has_peptide_id:
        result = _check_peptide_level(adata, raise_error)
        if result is not None:
            return result
        return True, "peptide"

    if "protein_id" not in var.columns:
        return _FAIL

    result = _check_protein_level(adata, raise_error)
    if result is not None:
        return result
    return True, "protein"


def check_proteodata(
    adata: AnnData,
    *,
    layers: str | list[str] | None = None,
) -> tuple[bool, str | None]:
    """
    Validate that *adata* satisfies ProteoPy assumptions, raising on
    failure.

    Thin wrapper around :func:`is_proteodata` with
    ``raise_error=True``. See that function for full documentation.

    Parameters
    ----------
    adata
        AnnData object to validate.
    layers
        Optional layer key or list of layer keys to validate for
        infinite values.

    Returns
    -------
    tuple[bool, str | None]
        ``(True, "peptide")`` or ``(True, "protein")`` on success.

    Raises
    ------
    ValueError
        If any validation check fails.
    """
    return is_proteodata(
        adata,
        raise_error=True,
        layers=layers,
    )


# ------------------------------------------------------------------
# sanitize_obs_cols helpers
# ------------------------------------------------------------------

def _is_missing(x):
    """Return True if *x* is a pandas-recognised missing value."""
    try:
        return pd.isna(x)
    except Exception:
        return False


def _looks_like_sequence(x):
    """Return True for list, tuple, set, or ndarray."""
    return isinstance(x, (list, tuple, set, np.ndarray))


def _to_jsonish(x, jsonize_complex):
    """Convert *x* to a JSON-serialisable string representation."""
    if _is_missing(x):
        return np.nan

    is_dict = isinstance(x, dict)
    is_seq = _looks_like_sequence(x)
    if jsonize_complex and (is_dict or is_seq):
        if isinstance(x, set):
            x = sorted(list(x))
        try:
            return json.dumps(
                x, default=str, ensure_ascii=False,
            )
        except Exception:
            return str(x)
    return str(x)


def _coerce_series(s, jsonize_complex):
    """Coerce a single Series to an HDF5-writable dtype."""
    if pd.api.types.is_bool_dtype(s):
        return s.astype('boolean')
    if pd.api.types.is_integer_dtype(s):
        return s.astype('int64')
    if pd.api.types.is_float_dtype(s):
        return s.astype('float64')
    if isinstance(s.dtype, pd.CategoricalDtype):
        return s

    if pd.api.types.is_object_dtype(s):
        only_strings = s.map(
            lambda x: isinstance(
                x, (str, np.str_)
            ) or _is_missing(x)
        ).all()
        if only_strings:
            return s.astype('object')

        out = s.map(
            lambda x: _to_jsonish(x, jsonize_complex),
        ).astype('object')
        return out

    return s


def sanitize_obs_cols(
    adata,
    jsonize_complex=True,
):
    '''Sanitize anndata columns (in-place).

    Makes all columns of adata.obs HDF5-writable by
    converting unsupported types.

    - Keeps numeric/boolean columns as pandas nullable
      dtypes.
    - Object columns:
        - If all entries are strings/missing: cast to
          object dtype.
        - If mixed/complex: JSON-serialize (if
          jsonize_complex=True), then object dtype.

    Args:
        jsonize_complex (bool): JSON-serialize
            lists/dicts/sets in object columns.
    '''
    if adata.obs is not None and len(adata.obs.columns):
        obs = adata.obs.copy()
        for c in obs.columns:
            obs[c] = _coerce_series(obs[c], jsonize_complex)
        adata.obs = obs

    return adata
