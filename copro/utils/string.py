from __future__ import annotations

import re
from pathlib import Path


def sanitize_string(s: str) -> str:
    """
    Sanitize a string for use as a column name or identifier.

    Replaces any character that is not alphanumeric or underscore with
    an underscore.

    Parameters
    ----------
    s : str
        The input string to sanitize.

    Returns
    -------
    str
        The sanitized string with non-alphanumeric characters (except
        underscores) replaced by underscores.

    Examples
    --------
    >>> sanitize_string("Group A")
    'Group_A'
    >>> sanitize_string("condition-1")
    'condition_1'
    >>> sanitize_string("sample/ctrl")
    'sample_ctrl'
    """
    return re.sub(r"[^a-zA-Z0-9_]", "_", str(s))


def detect_separator(file_path: str | Path) -> str:
    """Detect CSV/TSV separator from file extension.

    Parameters
    ----------
    file_path : str | Path
        Path to the file.

    Returns
    -------
    str
        Detected separator: ',' for .csv, '\\t' for .tsv.

    Raises
    ------
    ValueError
        If the file extension is not .csv or .tsv.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return ","
    elif suffix == ".tsv":
        return "\t"
    else:
        raise ValueError(
            f"Cannot auto-detect separator for extension '{suffix}'. "
            "Supported extensions: .csv, .tsv. "
            "Please provide the `sep` parameter explicitly."
            )
