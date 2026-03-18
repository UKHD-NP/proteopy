import warnings
from pathlib import Path
from typing import Callable
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
from Bio import SeqIO

from proteopy.utils.functools import partial_with_docsig
from proteopy.utils.anndata import check_proteodata, is_proteodata


def filter_axis(
    adata,
    axis,
    min_fraction=None,
    min_count=None,
    group_by=None,
    zero_to_na=False,
    inplace=True,
):
    """
    Filter observations or variables based on non-missing value content.

    This function filters the AnnData object along a specified axis (observations
    or variables) based on the fraction or number of non-missing (np.nan) values.
    Filtering can be performed globally or within groups defined by the `group_by`
    parameter.

    Parameters
    ----------
    adata : anndata.AnnData
        The annotated data matrix to filter.
    axis : int
        The axis to filter on. `0` for observations, `1` for variables.
    min_fraction : float, optional
        The minimum fraction of non-missing values required to keep an observation
        or variable. If `group_by` is provided, this threshold is applied to the
        maximum completeness across all groups.
    min_count : int, optional
        The minimum number of non-missing values required to keep an observation
        or variable. If `group_by` is provided, this threshold is applied to the
        maximum count across all groups.
    group_by : str, optional
        A column key in `adata.obs` (if `axis=1`) or `adata.var` (if `axis=0`)
        used for grouping before applying the filter. The maximum completeness or
        count across the groups is used for filtering.
    zero_to_na : bool, optional
        If True, zeros in the data matrix are treated as missing values (NaN).
    inplace : bool, optional
        If True, modifies the `adata` object in place. Otherwise, returns a
        filtered copy.

    Returns
    -------
    anndata.AnnData or None
        If `inplace=False`, returns a new filtered AnnData object. Otherwise,
        returns `None`.

    Raises
    ------
    KeyError
        If the `group_by` key is not found in the corresponding annotation
        DataFrame.
    """
    check_proteodata(adata)

    if min_fraction is None and min_count is None:
        warnings.warn(
            "Neither `min_fraction` nor `min_count` were provided, so "
            "the function does nothing."
        )
        return None if inplace else adata.copy()

    X = adata.X.copy()
    if zero_to_na:
        if sp.issparse(X):
            X.data[X.data == 0] = np.nan
        else:
            X[X == 0] = np.nan

    if sp.issparse(X):
        X.eliminate_zeros()

    axis_i = 1 - axis
    axis_labels = adata.obs_names if axis == 0 else adata.var_names
    completeness = None # assigned below when min_fraction is set

    if group_by is not None:
        metadata = adata.obs if axis == 1 else adata.var
        if group_by not in metadata.columns:
            raise KeyError(
                f'`group_by`="{group_by}" not present in '
                f'adata.{"obs" if axis == 1 else "var"}'
            )
        grouping = metadata[group_by]
        unique_groups = grouping.dropna().unique()

        counts_by_group = []
        completeness_by_group = []
        for label in unique_groups:
            mask = (grouping == label).values
            subset = X[mask, :] if axis == 1 else X[:, mask]

            if subset.shape[axis_i] == 0:
                continue

            group_size = subset.shape[axis_i]

            if sp.issparse(subset):
                group_counts = subset.getnnz(axis=axis_i)
            else:
                group_counts = np.count_nonzero(~np.isnan(subset), axis=axis_i)

            df_counts = pd.DataFrame(group_counts, index=axis_labels)
            counts_by_group.append(df_counts)
            if min_fraction is not None:
                df_completeness = df_counts / group_size
                completeness_by_group.append(df_completeness)

        if not counts_by_group:
            counts = pd.Series(0, index=axis_labels, dtype=float)
        else:
            counts = pd.concat(counts_by_group, axis=1).max(axis=1)
        if min_fraction is not None:
            if not completeness_by_group:
                completeness = pd.Series(0, index=axis_labels, dtype=float)
            else:
                completeness = pd.concat(completeness_by_group, axis=1).max(axis=1)
    else:
        if sp.issparse(X):
            counts = pd.Series(X.getnnz(axis=axis_i), index=axis_labels)
        else:
            counts = pd.Series(
                np.count_nonzero(~np.isnan(X), axis=axis_i), index=axis_labels
            )
        if min_fraction is not None:
            num_total = adata.shape[axis_i]
            completeness = counts / num_total

    mask_filt = pd.Series(True, index=axis_labels)
    if min_fraction is not None:
        mask_filt &= completeness >= min_fraction

    if min_count is not None:
        mask_filt &= counts >= min_count

    n_removed = (~mask_filt).sum()
    axis_name = ["obs", "var"][axis]
    print(f"{n_removed} {axis_name} removed")

    if inplace:
        if axis == 0:
            adata._inplace_subset_obs(mask_filt.values)
        else:
            adata._inplace_subset_var(mask_filt.values)
        check_proteodata(adata)
        return None
    else:
        adata_filtered = adata[mask_filt, :] if axis == 0 else adata[:, mask_filt]
        check_proteodata(adata_filtered)
        return adata_filtered


docstr_header = """
Filter observations based on non-missing value content.

This function filters the AnnData object along the `obs` axis based on the
fraction or number of non-missing values (np.nan). Filtering can be performed
globally or within groups defined by the `group_by` parameter.
"""
filter_samples = partial_with_docsig(
    filter_axis,
    axis=0,
    docstr_header=docstr_header,
    )

docstr_header = """
Filter observations based on data completeness.

This function filters the AnnData object along a the obs axis based on the
fraction of non-missing values (np.nan). Filtering can be performed globally
or within groups defined by the `group_by` parameter.
"""
filter_samples_completeness = partial_with_docsig(
    filter_axis,
    axis=0,
    min_count=None,
    docstr_header=docstr_header,
    )

docstr_header = """
Filter variables based on non-missing value content.

This function filters the AnnData object along the `var` axis based on the
fraction or number of non-missing values (np.nan). Filtering can be performed
globally or within groups defined by the `group_by` parameter.
"""
filter_var = partial_with_docsig(
    filter_axis,
    axis=1,
    docstr_header=docstr_header,
    )

docstr_header = """
Filter variables based on data completeness.

This function filters the AnnData object along a the var axis based on the
fraction of non-missing values (np.nan). Filtering can be performed globally
or within groups defined by the `group_by` parameter.
"""
filter_var_completeness = partial_with_docsig(
    filter_axis,
    axis=1,
    min_count=None,
    docstr_header=docstr_header,
    )


def filter_proteins_by_peptide_count(
    adata,
    min_count=None,
    max_count=None,
    protein_col="protein_id",
    inplace=True,
    ):
    """
    Filter proteins by their peptide count.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix with a protein identifier column in ``adata.var``.
    min_count : int or None, optional
        Keep peptides whose proteins have at least this many peptides.
    max_count : int or None, optional
        Keep peptides whose proteins have at most this many peptides.
    protein_col : str, optional (default: "protein_id")
        Column in ``adata.var`` containing protein identifiers.
    inplace : bool, optional (default: True)
        If True, modify ``adata`` in place. Otherwise, return a filtered view.

    Returns
    -------
    None or anndata.AnnData
        ``None`` if ``inplace=True``; otherwise the filtered AnnData view.
    """
    check_proteodata(adata)
    if is_proteodata(adata)[1] != "peptide":
        raise ValueError((
            "`AnnData` object must be in ProteoData peptide format."
            ))

    if min_count is None and max_count is None:
        warnings.warn("Pass at least one argument: min_count | max_count")
        adata_copy = None if inplace else adata.copy()
        if adata_copy is not None:
            check_proteodata(adata_copy)
        return adata_copy

    if min_count is not None:
        if min_count < 0:
            raise ValueError("`min_count` must be non-negative.")
    if max_count is not None:
        if max_count < 0:
            raise ValueError("`max_count` must be non-negative.")
    if (min_count is not None and max_count is not None) and (min_count > max_count):
        raise ValueError("`min_count` cannot be greater than `max_count`.")

    if protein_col not in adata.var.columns:
        raise KeyError(f"`protein_col`='{protein_col}' not found in adata.var")

    proteins = adata.var[protein_col]
    counts = proteins.value_counts()

    keep_mask = pd.Series(True, index=counts.index)
    if min_count is not None:
        keep_mask &= counts >= min_count
    if max_count is not None:
        keep_mask &= counts <= max_count
    protein_ids_keep = counts.index[keep_mask]

    var_keep_mask = proteins.isin(protein_ids_keep)

    if inplace:
        adata._inplace_subset_var(var_keep_mask.values)
        check_proteodata(adata)
        n_proteins_removed = len(counts.index) - len(protein_ids_keep)
        n_peptides_removed = int((~var_keep_mask).sum())
        print(
            f"Removed {n_proteins_removed} proteins and "
            f"{n_peptides_removed} peptides."
        )
        return None

    else:
        new_adata = adata[:, var_keep_mask]
        check_proteodata(new_adata)
        n_proteins_removed = len(counts.index) - len(protein_ids_keep)
        n_peptides_removed = int((~var_keep_mask).sum())
        print(
            f"Removed {n_proteins_removed} proteins and "
            f"{n_peptides_removed} peptides."
        )
        return new_adata


def filter_samples_by_category_count(
    adata,
    category_col,
    min_count=None,
    max_count=None,
    inplace=True,
    ):
    """
    Filter observations by the frequency of their category value in a ``.vars``
    metadata column.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix.
    category_col : str
        Column in ``adata.obs`` containing the categories to count.
    min_count : int or None, optional
        Keep categories with at least this many observations.
    max_count : int or None, optional
        Keep categories with at most this many observations.
    inplace : bool, optional (default: True)
        If True, modify ``adata`` in place. Otherwise, return a filtered copy.

    Returns
    -------
    None or anndata.AnnData
        ``None`` if ``inplace=True``; otherwise the filtered AnnData.
    """
    check_proteodata(adata)

    if min_count is None and max_count is None:
        raise ValueError(
            "At least one argument must be passed: min_count | max_count"
        )

    if min_count is not None and min_count < 0:
        raise ValueError("`min_count` must be non-negative.")
    if max_count is not None and max_count < 0:
        raise ValueError("`max_count` must be non-negative.")
    if (
        min_count is not None
        and max_count is not None
        and min_count > max_count
    ):
        raise ValueError("`min_count` cannot be greater than `max_count`.")

    if category_col not in adata.obs.columns:
        raise KeyError(f"`category_col`='{category_col}' not found in adata.obs")

    obs_series = adata.obs[category_col]
    counts = obs_series.value_counts(dropna=False)

    counts_filt = counts
    if min_count is not None:
        counts_filt = counts_filt[counts_filt >= min_count]
    if max_count is not None:
        counts_filt = counts_filt[counts_filt <= max_count]

    obs_keep_mask = obs_series.isin(counts_filt.index)
    removed = int((~obs_keep_mask).sum())
    print(f"Removed {removed} observations.")

    if inplace:
        adata._inplace_subset_obs(obs_keep_mask.values)
        check_proteodata(adata)
        return None

    new_adata = adata[obs_keep_mask, :].copy()
    check_proteodata(new_adata)
    return new_adata


def _validate_remove_zero_variance_vars_input(
    adata,
    group_by,
    atol,
    inplace,
    verbose,
):
    """Validate inputs for ``remove_zero_variance_vars``."""
    if not isinstance(adata, ad.AnnData):
        raise TypeError(
            f"`adata` must be an AnnData object, "
            f"got {type(adata).__name__}."
        )
    if group_by is not None and not isinstance(group_by, str):
        raise TypeError(
            f"`group_by` must be a string or None, "
            f"got {type(group_by).__name__}."
        )
    if not isinstance(atol, (int, float)):
        raise TypeError(
            f"`atol` must be a numeric value, "
            f"got {type(atol).__name__}."
        )
    if atol < 0:
        raise ValueError("`atol` must be non-negative.")
    if not isinstance(inplace, bool):
        raise TypeError(
            f"`inplace` must be a bool, "
            f"got {type(inplace).__name__}."
        )
    if not isinstance(verbose, bool):
        raise TypeError(
            f"`verbose` must be a bool, "
            f"got {type(verbose).__name__}."
        )
    if group_by is not None:
        if group_by not in adata.obs.columns:
            raise KeyError(
                f"`group_by`='{group_by}' not found "
                f"in adata.obs"
            )
        if adata.obs[group_by].isna().any():
            raise ValueError(
                f"`group_by`='{group_by}' column in "
                f"adata.obs contains NaN values."
            )


def remove_zero_variance_vars(
    adata,
    group_by=None,
    atol=1e-8,
    inplace=True,
    verbose=False,
):
    """
    Remove variables with near-zero or zero variance, skipping NaN values.

    Variables whose variance is at or below ``atol`` are removed.
    Variables that are entirely NaN — globally or within any group
    when ``group_by`` is set — are treated as zero variance and also
    removed.

    Parameters
    ----------
    adata : AnnData
        :class:`~anndata.AnnData` annotated data matrix.
    group_by : str | None, optional
        Column in ``adata.obs`` to compute variance per group. When
        set, a variable is removed if its variance is ``<= atol`` or
        all-NaN in *any* group.
    atol : float, optional
        Absolute tolerance threshold. Variables with variance
        ``<= atol`` are considered zero-variance and removed.
    inplace : bool, optional
        Modify ``adata`` in place and return ``None``.
        Otherwise, returns a filtered copy.
    verbose : bool, optional
        Print how many variables were present, removed,
        and remaining.

    Returns
    -------
    None | AnnData
        ``None`` when ``inplace=True``; a new :class:`anndata.AnnData`
        containing only variables with variance ``> atol`` otherwise.

    Raises
    ------
    TypeError
        If any argument has an incorrect type.
    ValueError
        If ``atol`` is negative or the ``group_by`` column
        contains NaN values.
    KeyError
        If ``group_by`` is not a column in ``adata.obs``.

    Warns
    -----
    UserWarning
        Raised when one or more variables are removed because they are
        entirely NaN (globally or within at least one group).

    Examples
    --------
    Build a small protein-level dataset with four variables:
    ``p1`` varies, ``p2`` is constant, ``p3`` is all-NaN, and
    ``p4`` varies.

    >>> import numpy as np
    >>> import pandas as pd
    >>> import anndata as ad
    >>> import proteopy as pr
    >>> X = np.array([
    ...     [1.0, 5.0, np.nan, 7.0],
    ...     [2.0, 5.0, np.nan, 7.0],
    ...     [3.0, 5.0, np.nan, 8.0],
    ... ])
    >>> obs = pd.DataFrame(
    ...     {"sample_id": ["s1", "s2", "s3"]},
    ...     index=["s1", "s2", "s3"],
    ... )
    >>> var = pd.DataFrame(
    ...     {"protein_id": ["p1", "p2", "p3", "p4"]},
    ...     index=["p1", "p2", "p3", "p4"],
    ... )
    >>> adata = ad.AnnData(X=X, obs=obs, var=var)
    >>> pr.pp.remove_zero_variance_vars(adata)
    >>> adata.var_names.tolist()
    ['p1', 'p4']

    With ``group_by``, a variable is removed if it has zero
    variance or is all-NaN in *any* group. Here ``p2`` is
    constant in group A, ``p3`` is all-NaN in group A, and
    ``p4`` is constant in both groups:

    >>> X_grp = np.array([
    ...     [1.0, 5.0, np.nan, 9.0],
    ...     [2.0, 5.0, np.nan, 9.0],
    ...     [3.0, 7.0,  8.0,   9.0],
    ...     [4.0, 8.0,  8.0,   9.0],
    ... ])
    >>> obs_grp = pd.DataFrame(
    ...     {"sample_id": ["s1", "s2", "s3", "s4"],
    ...      "group": ["A", "A", "B", "B"]},
    ...     index=["s1", "s2", "s3", "s4"],
    ... )
    >>> adata = ad.AnnData(X=X_grp, obs=obs_grp, var=var)
    >>> pr.pp.remove_zero_variance_vars(adata, group_by="group")
    >>> adata.var_names.tolist()
    ['p1']
    """
    _validate_remove_zero_variance_vars_input(
        adata, group_by, atol, inplace, verbose,
    )
    check_proteodata(adata)
    X = adata.X
    n_vars = adata.n_vars
    is_sparse = sp.issparse(X)

    keep_mask = np.ones(n_vars, dtype=bool)
    n_allnan = 0

    if group_by is None:
        X_full = X.toarray() if is_sparse else np.asarray(X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            var_all = np.nanvar(X_full, axis=0, ddof=0)
        allnan_mask = np.isnan(var_all)  # np.nanvar(all_nan_vector) == np.nan
        n_allnan = int(allnan_mask.sum())
        keep_mask &= (var_all > atol) & ~allnan_mask
    else:
        groups = adata.obs[group_by].astype("category")
        zero_any = np.zeros(n_vars, dtype=bool)
        allnan_any = np.zeros(n_vars, dtype=bool)

        for g in groups.cat.categories:
            idx = np.where(groups.values == g)[0]
            if idx.size == 0:
                continue
            Xg = X[idx, :]
            Xg_arr = (
                Xg.toarray() if sp.issparse(Xg)
                else np.asarray(Xg)
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                vg = np.nanvar(Xg_arr, axis=0, ddof=0)
            allnan_g = np.isnan(vg)  # np.nanvar(all_nan_vector) == np.nan
            allnan_any |= allnan_g
            zero_any |= (vg <= atol) & ~allnan_g

        n_allnan = int(allnan_any.sum())
        keep_mask &= ~zero_any & ~allnan_any

    if n_allnan > 0:
        warnings.warn(
            f"{n_allnan} variable(s) contained only NaN values"
            f"{' in at least one group' if group_by else ''}"
            " and were treated as zero variance.",
            UserWarning,
            stacklevel=2,
        )

    removed = int((~keep_mask).sum())
    if verbose:
        remaining = int(keep_mask.sum())
        print(
            f"{n_vars} variables present, "
            f"{removed} removed, "
            f"{remaining} remaining."
        )

    if inplace:
        adata._inplace_subset_var(keep_mask)
        check_proteodata(adata)
        return None
    else:
        new_adata = adata[:, keep_mask].copy()
        check_proteodata(new_adata)
        return new_adata


def remove_contaminants(
    adata,
    contaminant_path,
    protein_key="protein_id",
    header_parser: Callable[[str], str] | None = None,
    inplace=True,
    ):
    """
    Remove variables whose protein identifier matches a contaminant FASTA entry.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data.
    contaminant_path : str | Path
        Path to the contaminant list. The file can be in FASTA format, in which
        case the headers are parsed to extract the contaminant ids (see param:
        header_parser); or tabular format TSV/CSV files, in which case the
        first column is extracted as contaminant ids..
    protein_key : str, optional (default: "protein_id")
        Column in ``adata.var`` containing protein identifiers to match.
    header_parser : callable, optional
        Function to extract protein IDs from FASTA headers. Defaults to splitting
        the header on ``"|"`` and returning the second element, falling back to
        the full header if not present.
    inplace : bool, optional (default: False)
        If True, modify ``adata`` in place. Otherwise, return a filtered view.

    Returns
    -------
    None or anndata.AnnData
        ``None`` if ``inplace=True``; otherwise the filtered AnnData view.
    """
    check_proteodata(adata)

    if header_parser is None:
        def header_parser(header: str) -> str:
            parts = header.split("|")
            return parts[1] if len(parts) > 1 else header

    def _load_contaminant_ids_from_fasta(fasta_path: Path) -> set[str]:
        contaminant_ids = set()
        for record in SeqIO.parse(fasta_path, "fasta"):
            parsed = header_parser(record.id)
            if parsed == "":
                warnings.warn(
                    f"Header parser returned empty ID for record '{record.id}'.",
                )
                continue
            contaminant_ids.add(parsed)
        return contaminant_ids

    def _load_contaminant_ids_from_table(table_path: Path, sep: str) -> set[str]:
        series = pd.read_csv(table_path, sep=sep, usecols=[0]).iloc[:, 0]
        series = series.dropna().astype(str)
        return set(series.tolist())

    cont_path = Path(contaminant_path)
    if not cont_path.exists():
        raise FileNotFoundError(f"Contaminant file not found at {cont_path}")

    if protein_key not in adata.var.columns:
        raise KeyError(f"`protein_key`='{protein_key}' not found in adata.var")

    suffix = cont_path.suffix.lower()
    match suffix:
        case ".fasta" | ".fa" | ".faa":
            contaminant_ids = _load_contaminant_ids_from_fasta(cont_path)
        case ".csv":
            contaminant_ids = _load_contaminant_ids_from_table(cont_path, ",")
        case ".tsv":
            contaminant_ids = _load_contaminant_ids_from_table(cont_path, "\t")
        case _:
            raise ValueError(
                "Unsupported contaminant file type. Use FASTA (.fasta/.fa/.faa), "
                "CSV (.csv), or TSV (.tsv).",
            )

    proteins = adata.var[protein_key]
    keep_mask = ~proteins.isin(contaminant_ids)

    _, level = is_proteodata(adata)
    if level == "peptide":
        removed_peptides = int((~keep_mask).sum())
        removed_proteins = int(proteins[~keep_mask].nunique())
        print(
            f"Removed {removed_peptides} contaminating peptides and "
            f"{removed_proteins} contaminating proteins.",
        )
    elif level == "protein":
        removed_proteins = int((~keep_mask).sum())
        print(f"Removed {removed_proteins} contaminating proteins.")
    else:
        removed = int((~keep_mask).sum())
        print(f"Removed {removed} contaminating variables.")

    if inplace:
        adata._inplace_subset_var(keep_mask.values)
        check_proteodata(adata)
        return None

    new_adata = adata[:, keep_mask]
    check_proteodata(new_adata)
    return new_adata
