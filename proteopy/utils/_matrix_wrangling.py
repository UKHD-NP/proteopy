import numpy as np
import pandas as pd


def reconstruct_symmetric_matrix_from_long(
    df,
    var_a_col=0,
    var_b_col=1,
    value_col=2,
    *,
    diagonal=1.0,
    allow_missing=False,
):
    """Reconstruct a square symmetric matrix from a long DataFrame.

    Build a full symmetric matrix from a long-format DataFrame that
    lists pairs of labels and an associated value. Each ``(a, b)`` pair
    fills both ``M[a, b]`` and ``M[b, a]``. Labels are the sorted union
    of the two label columns and become both the index and the columns
    of the result.

    Parameters
    ----------
    df : pandas.DataFrame
        Long-format table with a column for the first label, a column
        for the second label, and a column with the pair value.
    var_a_col : str | int
        Name or positional index of the first label column.
    var_b_col : str | int
        Name or positional index of the second label column.
    value_col : str | int
        Name or positional index of the value column.
    diagonal : float
        Value written on the matrix diagonal (``1.0`` by default, the
        correlation convention).
    allow_missing : bool
        If ``True``, a label pair without a value in either triangle
        (absent or ``NaN``) is left as ``NaN`` instead of raising.

    Returns
    -------
    pandas.DataFrame
        Square symmetric matrix with the sorted labels as both index
        and columns.

    Raises
    ------
    ValueError
        If a label pair is absent from both triangles (and
        ``allow_missing`` is ``False``), or if a pair is
        present in both triangles with conflicting values.
    """
    if isinstance(var_a_col, int):
        var_a_col = df.columns[var_a_col]

    if isinstance(var_b_col, int):
        var_b_col = df.columns[var_b_col]

    if isinstance(value_col, int):
        value_col = df.columns[value_col]

    labels = set(df[var_a_col]).union(set(df[var_b_col]))
    labels = sorted(labels)
    n = len(labels)

    label_to_idx = {label: i for i, label in enumerate(labels)}

    # -- Initialise with NaN and a fixed diagonal
    matrix = np.full((n, n), np.nan)
    np.fill_diagonal(matrix, diagonal)

    # -- Fill in the known values
    for _, row in df.iterrows():
        i = label_to_idx[row[var_a_col]]
        j = label_to_idx[row[var_b_col]]

        matrix[i, j] = row[value_col]

    # -- Mirror across the diagonal, validating any overlap
    idx_to_label = {i: label for label, i in label_to_idx.items()}
    for i in range(n):
        for j in range(i + 1, n):

            upper_nan = np.isnan(matrix[i, j])
            lower_nan = np.isnan(matrix[j, i])

            if upper_nan and not lower_nan:
                matrix[i, j] = matrix[j, i]
            elif lower_nan and not upper_nan:
                matrix[j, i] = matrix[i, j]
            elif upper_nan and lower_nan:
                if allow_missing:
                    continue
                raise ValueError(
                    "No value found for the combination of labels: "
                    f"{idx_to_label[i]} and {idx_to_label[j]}."
                )
            elif not np.isclose(matrix[i, j], matrix[j, i]):
                raise ValueError(
                    "Conflicting values for the combination of labels "
                    f"{idx_to_label[i]} and {idx_to_label[j]}: "
                    f"{matrix[i, j]} != {matrix[j, i]}."
                )

    return pd.DataFrame(matrix, index=labels, columns=labels)
