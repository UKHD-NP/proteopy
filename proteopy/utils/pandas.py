from __future__ import annotations

from pathlib import Path

import pandas as pd

from proteopy.utils.string import detect_separator


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
            sep = detect_separator(file_path)
        df = pd.read_csv(file_path, sep=sep)
        return df
