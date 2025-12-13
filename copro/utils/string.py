import re


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
