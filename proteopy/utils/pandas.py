from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from proteopy.utils.string import detect_separator_from_extension


def long_pairs_to_symmetric_matrix(
    df: pd.DataFrame,
    *,
    var_a_col: str | int = 0,
    var_b_col: str | int = 1,
    value_col: str | int = 2,
    diagonal_value: float = 1.0,
) -> pd.DataFrame:
    """Pivot a long-form pairwise frame into a symmetric matrix.

    Reconstructs a full symmetric matrix from a long DataFrame
    containing one row per unordered pair ``(varA, varB)`` with an
    associated value (e.g. a correlation coefficient). The diagonal is
    set to ``diagonal_value`` and missing pairs are filled by their
    transpose when available.

    Parameters
    ----------
    df : pandas.DataFrame
        Frame with one row per pair. Must contain the columns
        identified by ``var_a_col``, ``var_b_col`` and ``value_col``.
    var_a_col : str | int
        Name or positional index of the column holding the first
        variable identifier.
    var_b_col : str | int
        Name or positional index of the column holding the second
        variable identifier.
    value_col : str | int
        Name or positional index of the column holding the pairwise
        value (e.g. correlation coefficient).
    diagonal_value : float
        Value placed on the diagonal of the output matrix.

    Returns
    -------
    pandas.DataFrame
        Symmetric matrix with sorted unique variables as both row
        index and columns.

    Raises
    ------
    ValueError
        If a pair (and its transpose) is missing entirely from ``df``.
    """
    if isinstance(var_a_col, int):
        var_a_col = df.columns[var_a_col]
    if isinstance(var_b_col, int):
        var_b_col = df.columns[var_b_col]
    if isinstance(value_col, int):
        value_col = df.columns[value_col]

    all_vars = sorted(set(df[var_a_col]).union(set(df[var_b_col])))
    n = len(all_vars)
    var_to_idx = {v: i for i, v in enumerate(all_vars)}

    mat = np.full((n, n), np.nan)
    np.fill_diagonal(mat, diagonal_value)

    for _, row in df.iterrows():
        i = var_to_idx[row[var_a_col]]
        j = var_to_idx[row[var_b_col]]
        mat[i, j] = row[value_col]

    # -- mirror missing entries via the transpose when available
    for i in range(n):
        for j in range(i + 1, n):
            ij_nan = np.isnan(mat[i, j])
            ji_nan = np.isnan(mat[j, i])
            if ij_nan and not ji_nan:
                mat[i, j] = mat[j, i]
            elif ji_nan and not ij_nan:
                mat[j, i] = mat[i, j]
            elif ij_nan and ji_nan:
                rev = {idx: v for v, idx in var_to_idx.items()}
                raise ValueError(
                    f"Missing value for pair ({rev[i]}, {rev[j]}) "
                    "in long pairs frame."
                )

    return pd.DataFrame(mat, index=all_vars, columns=all_vars)


def load_dataframe(
    data: str | Path | pd.DataFrame,
    sep: str | None = None,
) -> pd.DataFrame:
    """Load data from file path or return DataFrame directly.

    Parameters
    ----------
    data : str | Path | pd.DataFrame
        Either a file path (str or Path) or a pandas DataFrame.
    sep : str | None
        Separator for reading files. If None, auto-detect from extension.

    Returns
    -------
    pd.DataFrame
        The loaded or passed-through DataFrame.
    """
    if isinstance(data, pd.DataFrame):
        return data
    else:
        # Input is a file path
        file_path = Path(data)
        if sep is None:
            sep = detect_separator_from_extension(file_path)
        df = pd.read_csv(file_path, sep=sep)
        return df
