from __future__ import annotations

from pathlib import Path

import pandas as pd

from proteopy.utils.string import detect_separator_from_extension


def _resolve_group_order(series: pd.Series) -> list:
    """Return observed, non-missing group labels in plotting order.

    Categorical labels keep their category order. Other labels are
    sorted lexicographically by their string representations. The
    original label values and the input Series are preserved.
    """
    observed = series.dropna()
    if isinstance(series.dtype, pd.CategoricalDtype):
        return list(observed.cat.remove_unused_categories().cat.categories)
    return sorted(pd.unique(observed), key=str)


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
