import json
import re
import warnings

import numpy as np
import pandas as pd
from anndata import AnnData


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
                f"{type(a)!r} does not expose a usable length along axis {axis}."
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
                f"Duplicate names detected in {axis_name} axis: {shown}{extra}. "
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
    """Return True when the chosen .var column exactly matches both axis definitions."""
    if column not in adata.var.columns:
        return False

    series = adata.var[column]
    if series.isna().any():
        return False

    values = series.to_numpy()
    var_names = adata.var_names.to_numpy()
    var_index = adata.var.index.to_numpy()

    return np.array_equal(values, var_names) and np.array_equal(values, var_index)


def _has_multiple_values_per_cell(
    series: pd.Series, delimiters: str = " ,;"
) -> bool:
    """Return True when any entry contains more than one value separated by delimiters."""
    if series.isna().any():
        return True

    tokens = series.astype(str).str.strip()
    if (tokens == "").any():
        return True

    pattern = f"[{re.escape(delimiters)}]"
    return tokens.str.contains(pattern).any()


def is_proteodata(
    adata: AnnData
    ) -> tuple[bool, str | None]:
    """
    Check whether the AnnData object stores peptide- or protein-level proteomics data.

    Parameters
    ----------
    adata
        AnnData object whose `.var` annotations will be inspected.

    Returns
    -------
    tuple[bool, str | None]
        `(True, "peptide")` if the data satisfy the peptide-level assumptions,
        `(True, "protein")` if they satisfy the protein-level assumptions,
        otherwise `(False, None)`.

    Notes
    -----
    Peptide-level data must provide both `.var["peptide_id"]` and `.var["protein_id"]`.
    Every `peptide_id` value must be unique, and the column must match
    `adata.var_names` (and the `.var` index) exactly. Each peptide must map to
    exactly one `protein_id`.

    Protein-level data must provide `.var["protein_id"]`, must *not* contain a
    `peptide_id` column, and the `protein_id` values must match `adata.var_names`
    (and the `.var` index) exactly while also being unique.
    """
    if not isinstance(adata, AnnData):
        raise TypeError("is_proteodata expects an AnnData object.")

    if adata.var is None or adata.var.empty:
        return False, None

    _check_structure(adata)  # also checks if obs/vars are unique

    var = adata.var
    has_protein_id = "protein_id" in var.columns
    has_peptide_id = "peptide_id" in var.columns

    if has_peptide_id:
        if not has_protein_id:
            raise ValueError(
                    "Found a 'peptide_id' column but no 'protein_id' column."
                    "If working at peptide-level, a peptide_id -> protein_id mapping"
                    "must be included"
                )
            return False, None

        if not _var_column_matches_axis(adata, "peptide_id"):
            raise ValueError(
                "Found a 'peptide_id' column but it does not match AnnData.var_names. "
                "If your data are protein-level, please rename or remove the "
                "'peptide_id' column."
            )

        if _has_multiple_values_per_cell(var["protein_id"]):
            warnings.warn(
                "Detected peptides mapping to multiple proteins. "
                "Ensure each peptide maps to exactly one protein_id.",
                stacklevel=2,
            )
            return False, None

        return True, "peptide"

    if not has_protein_id:
        return False, None

    if not _var_column_matches_axis(adata, "protein_id"):
        raise ValueError(
            "Found a 'protein_id' column but it does not match AnnData.var_names."
        )
        return False, None

    return True, "protein"


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
