import re
import warnings
import gc
import pandas as pd
import numpy as np
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
from packaging.version import Version

from proteopy.utils.anndata import check_proteodata


# -- Aggregation level dispatch for v1.9.1

_PEPTIDE_AGGR_PATTERNS = {
    r"[Pp]recursor\.[Ii]d": "Precursor.Id",
    r"[Mm]odified\.[Ss]equence": "Modified.Sequence",
    r"[Ss]tripped\.[Ss]equence": "Stripped.Sequence",
}

_PROTEIN_AGGR_PATTERNS = {
    r"[Pp]rotein\.[Ii]ds": "Protein.Ids",
    r"[Pp]rotein\.[Gg]roups": "Protein.Groups",
}

_ALL_AGGR_PATTERNS = {
    **_PEPTIDE_AGGR_PATTERNS,
    **_PROTEIN_AGGR_PATTERNS,
}


def _resolve_aggr_level(aggr_level):
    """Resolve an aggr_level string to its canonical column name.

    Returns the canonical column name and a boolean indicating
    whether the level corresponds to protein-level aggregation.
    """
    for pattern, canonical in _ALL_AGGR_PATTERNS.items():
        if re.fullmatch(pattern, aggr_level):
            return canonical, pattern in _PROTEIN_AGGR_PATTERNS
    valid = ", ".join(
        f"'{p}'" for p in _ALL_AGGR_PATTERNS
    )
    raise ValueError(
        f"Invalid aggr_level '{aggr_level}'. "
        f"Valid regex patterns: {valid}"
    )


def _resolve_version_handler(version, dispatch):
    """Return the handler callable for the given DIA-NN version string.

    Performs a floor-match against sorted dispatch keys, raising
    ``ValueError`` if ``version`` is below the minimum supported key.
    """
    query = Version(version)
    sorted_keys = sorted(dispatch.keys(), key=Version)

    if query < Version(sorted_keys[0]):
        supported = ", ".join(sorted_keys)
        raise ValueError(
            f"DIA-NN version '{version}' is below the "
            f"minimum supported version '{sorted_keys[0]}'.\n"
            f"Supported versions: {supported}"
        )

    matched_key = sorted_keys[0]
    for key in sorted_keys:
        if Version(key) <= query:
            matched_key = key
        else:
            break

    return dispatch[matched_key]


def _read_diann_v1(
    diann_output_path,
    aggr_level,
    precursor_pval_max,
    gene_pval_max,
    global_precursor_pval_max,
    show_input_stats=False,
    run_parser=None,
    fill_na=None,
):
    """Read a DIA-NN v1.x TSV report into an :class:`~anndata.AnnData`.

    Filters to proteotypic, non-multi-mapping precursors, applies
    Q-value thresholds, aggregates ``Precursor.Quantity`` by
    ``aggr_level``, and returns a samples-x-peptides AnnData object.
    """
    # -- Check args
    aggr_level_options = [
        'Stripped.Sequence',
        'Modified.Sequence',
        'Precursor.Id',
    ]

    if aggr_level not in aggr_level_options:
        raise ValueError(
            f'Wrong option passsed to aggr_level argument: '
            f'{aggr_level}.'
        )

    if run_parser is not None and not callable(run_parser):
        raise ValueError(
            'run_parser arg must either be a function or None.'
        )

    base_required_cols = {
        'Run',
        'Proteotypic',
        'Protein.Ids',
        'Precursor.Quantity',
        'Protein.Q.Value',
        'Global.Q.Value',
        'Q.Value',
        'Protein.Group',
        'Genes',
        'Protein.Names',
        'Stripped.Sequence',
    }

    required_cols = set(base_required_cols)
    required_cols.add(aggr_level)

    if aggr_level == 'Precursor.Id':
        required_cols.update(
            {'Modified.Sequence', 'Precursor.Charge'}
        )
    if aggr_level == 'Modified.Sequence':
        required_cols.add('Modified.Sequence')

    header = pd.read_csv(diann_output_path, sep='\t', nrows=0)
    missing_cols = sorted(required_cols - set(header.columns))

    if missing_cols:
        missing_str = ', '.join(missing_cols)
        raise ValueError(
            'Missing required columns in DIA-NN output: '
            f'{missing_str}.'
        )

    data = pd.read_csv(
        diann_output_path,
        sep='\t',
        header=0,
        usecols=sorted(required_cols),
    )

    if run_parser:
        data['Run'] = data['Run'].apply(run_parser)

    if show_input_stats:
        print(
            'Before Q-value and proteotypicity filtering\n'
            '------'
        )
        proteotypic_fraction = (
            (data['Proteotypic'] == 1).sum() / len(data)
        )
        print(
            f'Proteotypic peptide fraction: '
            f'{proteotypic_fraction:.2f}'
        )

        multimapper_fraction = (
            (data['Protein.Ids'].str.split(';').apply(len) == 1)
            .sum() / len(data)
        )
        print(
            f'Multimapper peptide fraction: '
            f'{multimapper_fraction:.2f}'
        )

        # Q value distr. plots
        fig, axes = plt.subplots(
            nrows=1, ncols=3, figsize=(16, 4)
        )
        plt.subplots_adjust(wspace=0.3)

        sns.histplot(data['Q.Value'], bins=100, ax=axes[0])
        axes[0].set_title('Q.Value distr.')

        if precursor_pval_max:
            axes[0].axvline(
                x=precursor_pval_max,
                color='red',
                linestyle='--',
                linewidth=2,
            )

        sns.histplot(
            data['Global.Q.Value'], bins=100, ax=axes[1]
        )
        axes[1].set_title('Gobal.Q.Value distr.')

        if global_precursor_pval_max:
            axes[1].axvline(
                x=global_precursor_pval_max,
                color='red',
                linestyle='--',
                linewidth=2,
            )

        sns.histplot(
            data['Protein.Q.Value'], bins=100, ax=axes[2]
        )
        axes[2].set_title('Protein.Q.Value distr.')

        if gene_pval_max:
            axes[2].axvline(
                x=gene_pval_max,
                color='red',
                linestyle='--',
                linewidth=2,
            )
        plt.show()

        # Q values stats
        q_stats = data[[
            'Q.Value', 'Protein.Q.Value', 'Global.Q.Value'
        ]].describe()
        print(q_stats)

    # -- Filter ds
    data_sub = data[
        (data['Proteotypic'] == 1)
        & (data['Protein.Ids'].str.split(';')
           .apply(len).eq(1))
    ].copy()
    del data
    gc.collect()

    # ToDo: change to < instead of <=
    if precursor_pval_max:
        data_sub = data_sub[
            data_sub['Q.Value'] <= precursor_pval_max
        ]
    if global_precursor_pval_max:
        data_sub = data_sub[
            data_sub['Global.Q.Value']
            <= global_precursor_pval_max
        ]
    if gene_pval_max:
        data_sub = data_sub[
            data_sub['Protein.Q.Value'] <= gene_pval_max
        ]

    if len(data_sub) == 0:
        raise ValueError('Dataframe after filtering empty')

    if show_input_stats:
        # Q values stats
        q_stats = data_sub[[
            'Q.Value', 'Protein.Q.Value', 'Global.Q.Value'
        ]].describe()
        print(
            '\nAfter Q-value and proteotypicity filtering\n'
            '------'
        )
        print(q_stats)

    # -- Check: how peptides map to proteins
    is_pep_multiprots = (
        data_sub.groupby(
            [aggr_level, 'Run'], observed=True
        )['Protein.Ids'].nunique() > 1
    )

    if is_pep_multiprots.any():
        raise ValueError(
            f'Peptides at aggregation level {aggr_level} '
            'map to multiple proteins. '
            'Not implemented yet.'
        )

    # -- Aggregate precursors
    data_cols = [
        'Run',
        aggr_level,
        'Protein.Ids',
        'Precursor.Quantity',
    ]

    precursor_data = data_sub[data_cols].copy()

    precursor_data_summed = (
        precursor_data.groupby(
            [aggr_level, 'Protein.Ids', 'Run'],
            observed=True,
        )['Precursor.Quantity']
        .sum()
        .reset_index()
    )

    # -- Check: proteotypicity
    assert ((
        precursor_data_summed
        .groupby('Stripped.Sequence', observed=True)
        ['Protein.Ids']
        .nunique().le(1).all()
    )), "Error: Some peptides map to multiple proteins!"

    X = pd.pivot(
        precursor_data_summed,
        index='Run',
        columns=aggr_level,
        values='Precursor.Quantity',
    )

    X = X.sort_index(axis=0).sort_index(axis=1)

    if fill_na is not None:
        X.fillna(fill_na, inplace=True)

    X.columns.name = None
    X.index.name = None

    del precursor_data
    gc.collect()

    # -- obs
    obs = pd.DataFrame(
        {'run_id': X.index}, index=X.index
    )
    obs.index.name = None

    meta_cols = [
        aggr_level,
        'Protein.Ids',
        'Protein.Group',
        'Genes',
        'Protein.Names',
    ]

    if aggr_level == 'Modified.Sequence':
        meta_cols.append('Stripped.Sequence')

    if aggr_level == 'Precursor.Id':
        meta_cols.extend([
            'Stripped.Sequence',
            'Modified.Sequence',
            'Precursor.Charge',
        ])

    precursor_meta = data_sub[meta_cols].copy()

    # Groups contain identical rows
    assert (
        precursor_meta.groupby(aggr_level, observed=True)
        .apply(
            lambda x: x.nunique().eq(1).all(),
            include_groups=False,
        )
        .all()
    )

    var = precursor_meta.groupby(
        aggr_level, observed=True
    ).first()
    var = var.loc[X.columns]
    var[aggr_level] = var.index
    var['peptide_id'] = var.index
    var.index.name = None

    del precursor_meta
    del data_sub
    gc.collect()

    adata = ad.AnnData(
        X=X,
        var=var,
        obs=obs,
    )

    adata.strings_to_categoricals()

    if len(adata.obs_names.unique()) < adata.n_obs:
        adata.obs_names_make_unique()
        warnings.warn(
            'Repeated obs names were present in the data. '
            'They were made unique by numbered suffixes.'
        )

    if len(adata.var_names.unique()) < adata.n_vars:
        adata.var_names_make_unique()
        warnings.warn(
            'Repeated var names were present in the data. '
            'They were made unique by numbered suffixes.'
        )

    return adata


def _read_diann_v1_9_1(
    diann_output_path,
    aggr_level,
    max_precursor_q=None,
    max_protein_q=None,
    max_global_precursor_q=None,
    normalized=False,
    run_parser=None,
    fill_na=None,
    zero_to_na=False,
    verbose=False,
):
    """Read a DIA-NN v1.9.1+ parquet report into an :class:`~anndata.AnnData`.

    Filters decoys and multi-mapping precursors, applies Q-value
    thresholds, aggregates intensities by ``aggr_level``, and returns
    a validated samples-x-peptides AnnData object.
    """
    # -- Validate arguments
    if run_parser is not None and not callable(run_parser):
        raise ValueError(
            "run_parser must be a callable or None."
        )

    aggr_col, is_protein = _resolve_aggr_level(aggr_level)

    if is_protein:
        raise NotImplementedError(
            "Protein-level aggregation not yet "
            "implemented for DIA-NN >= 1.9.1."
        )

    if fill_na is not None and zero_to_na:
        raise ValueError(
            "fill_na and zero_to_na are mutually exclusive."
        )

    # -- Determine columns to read
    intensity_col = (
        "Precursor.Normalised"
        if normalized
        else "Precursor.Quantity"
    )

    base_cols = [
        "Run",
        "Decoy",
        aggr_col,
        "Protein.Ids",
        "Protein.Group",
        "Genes",
        "Protein.Names",
        intensity_col,
    ]

    if max_precursor_q is not None:
        base_cols.append("Q.Value")
    if max_protein_q is not None:
        base_cols.append("Protein.Q.Value")
    if max_global_precursor_q is not None:
        base_cols.append("Global.Q.Value")
    if aggr_col == "Precursor.Id":
        base_cols.extend([
            "Modified.Sequence",
            "Stripped.Sequence",
            "Precursor.Charge",
        ])
    elif aggr_col == "Modified.Sequence":
        base_cols.append("Stripped.Sequence")

    usecols = sorted(set(base_cols))

    # -- Read parquet
    data = pd.read_parquet(
        diann_output_path, columns=usecols,
    )

    if verbose:
        print(
            f"Rows before decoy and proteotypicity "
            f"filtering: {len(data):,}"
        )

    # -- Filter decoys
    data = data[data["Decoy"] == 0].copy()
    data.drop(columns=["Decoy"], inplace=True)

    # -- Filter proteotypicity (single protein mapping)
    proteotypic_mask = (
        data["Protein.Ids"].str.split(";").str.len() == 1
    )
    data = data[proteotypic_mask].copy()

    if verbose:
        print(
            f"Rows after decoy and proteotypicity "
            f"filtering: {len(data):,}"
        )

    # -- Apply Q-value filters
    if max_precursor_q is not None:
        data = data[data["Q.Value"] <= max_precursor_q]
    if max_protein_q is not None:
        data = data[
            data["Protein.Q.Value"] <= max_protein_q
        ]
    if max_global_precursor_q is not None:
        data = data[
            data["Global.Q.Value"]
            <= max_global_precursor_q
        ]

    if len(data) == 0:
        raise ValueError(
            "No rows remain after Q-value filtering."
        )

    if verbose:
        print(
            f"Rows after Q-value filtering: {len(data):,}"
        )

    # -- Parse Run column
    if run_parser is not None:
        data["Run"] = data["Run"].apply(run_parser)

    # -- Pivot to wide format
    if aggr_col == "Precursor.Id":
        X = pd.pivot(
            data,
            index="Run",
            columns=aggr_col,
            values=intensity_col,
        )
    else:
        agg_data = (
            data.groupby(
                [aggr_col, "Protein.Ids", "Run"],
                observed=True,
            )[intensity_col]
            .sum()
            .reset_index()
        )
        X = pd.pivot(
            agg_data,
            index="Run",
            columns=aggr_col,
            values=intensity_col,
        )

    X = X.sort_index(axis=0).sort_index(axis=1)
    X.columns.name = None
    X.index.name = None

    # -- Build obs
    obs = pd.DataFrame(index=X.index)
    obs["sample_id"] = obs.index
    obs.index.name = None

    # -- Build var metadata
    meta_cols = [
        aggr_col,
        "Protein.Ids",
        "Protein.Group",
        "Genes",
        "Protein.Names",
    ]

    if aggr_col == "Modified.Sequence":
        meta_cols.append("Stripped.Sequence")
    elif aggr_col == "Precursor.Id":
        meta_cols.extend([
            "Stripped.Sequence",
            "Modified.Sequence",
            "Precursor.Charge",
        ])

    meta = data[meta_cols].drop_duplicates(
        subset=[aggr_col], keep="first",
    )

    var = meta.set_index(aggr_col)
    var = var.loc[X.columns]
    var["peptide_id"] = var.index
    var["protein_id"] = var["Protein.Ids"]
    var.index.name = None

    del data
    gc.collect()

    # -- Build AnnData
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.strings_to_categoricals()

    # -- Handle zeros and NAs in .X
    if zero_to_na:
        mat = adata.X
        mat[mat == 0] = np.nan
        adata.X = mat

    if fill_na is not None:
        mat = adata.X
        mat = np.where(np.isnan(mat), fill_na, mat)
        adata.X = mat

    check_proteodata(adata)
    return adata


_DIANN_VERSION_DISPATCH = {
    "1.0.0": _read_diann_v1,
    "1.9.1": _read_diann_v1_9_1,
}


def diann(
    diann_output_path,
    aggr_level,
    version="1.0.0",
    **kwargs,
):
    """Read a DIA-NN report into an :class:`~anndata.AnnData` object.

    Parameters
    ----------
    diann_output_path : str | Path
        Path to the DIA-NN output file. TSV for version ``"1.0.0"``;
        parquet for version ``"1.9.1"``.
    aggr_level : str
        Peptide aggregation level. Accepted values (case-insensitive
        regex match):

        - ``"Precursor.Id"`` — one row per charge-modified sequence
          pair; no intensity summing across precursors.
        - ``"Modified.Sequence"`` — sum precursor quantities per
          modified peptide sequence.
        - ``"Stripped.Sequence"`` — sum precursor quantities per
          unmodified peptide sequence.
    version : str, optional
        DIA-NN version string used to select the parsing handler.
        Floor-matched against supported versions.
    **kwargs
        Additional keyword arguments forwarded to the version-specific
        handler. Common options:

        *v1.0.0 handler* (``_read_diann_v1``):

        - ``precursor_pval_max`` *(float)* — maximum ``Q.Value``.
        - ``gene_pval_max`` *(float)* — maximum ``Protein.Q.Value``.
        - ``global_precursor_pval_max`` *(float)* — maximum
          ``Global.Q.Value``.
        - ``show_input_stats`` *(bool)* — print Q-value distributions
          and proteotypicity fractions before and after filtering.
        - ``run_parser`` *(callable | None)* — function applied to
          each ``Run`` value to transform sample identifiers.
        - ``fill_na`` *(float | int | None)* — value used to replace
          ``NaN`` entries in the intensity matrix.

        *v1.9.1 handler* (``_read_diann_v1_9_1``):

        - ``max_precursor_q`` *(float | None)* — maximum ``Q.Value``.
        - ``max_protein_q`` *(float | None)* — maximum
          ``Protein.Q.Value``.
        - ``max_global_precursor_q`` *(float | None)* — maximum
          ``Global.Q.Value``.
        - ``normalized`` *(bool)* — use ``Precursor.Normalised``
          instead of ``Precursor.Quantity`` as the intensity column.
        - ``run_parser`` *(callable | None)* — function applied to
          each ``Run`` value to transform sample identifiers.
        - ``fill_na`` *(float | int | None)* — value used to replace
          ``NaN`` entries in the intensity matrix.
        - ``zero_to_na`` *(bool)* — replace zeros with ``np.nan``
          before returning. Mutually exclusive with ``fill_na``.
        - ``verbose`` *(bool)* — print row counts at each filtering
          step.

    Returns
    -------
    ad.AnnData
        AnnData with shape ``(n_samples, n_peptides)``. Observations
        (``.obs``) contain ``sample_id``; variables (``.var``) contain
        ``peptide_id``, ``protein_id``.

    Raises
    ------
    ValueError
        If ``version`` is below the minimum supported version.
    ValueError
        If ``aggr_level`` does not match any recognised pattern.
    ValueError
        If required columns are absent from the input file (v1.0.0).
    ValueError
        If no rows remain after Q-value and proteotypicity filtering.
    NotImplementedError
        If a protein-level ``aggr_level`` is requested for
        DIA-NN >= 1.9.1.

    Examples
    --------
    Read a DIA-NN v1.0.0 TSV report at stripped-sequence level:

    >>> import proteopy as pr
    >>> adata = pr.read.diann(
    ...     "report.tsv",
    ...     aggr_level="Stripped.Sequence",
    ...     version="1.0.0",
    ...     precursor_pval_max=0.01,
    ...     gene_pval_max=0.01,
    ...     global_precursor_pval_max=0.01,
    ... )

    Read a DIA-NN v1.9.1 parquet report at precursor level with a
    custom run-name parser:

    >>> import proteopy as pr
    >>> adata = pr.read.diann(
    ...     "report.parquet",
    ...     aggr_level="Precursor.Id",
    ...     version="1.9.1",
    ...     max_precursor_q=0.01,
    ...     run_parser=lambda s: s.split("/")[-1].split(".")[0],
    ...     verbose=True,
    ... )
    """
    handler = _resolve_version_handler(
        version, _DIANN_VERSION_DISPATCH
    )
    return handler(
        diann_output_path=diann_output_path,
        aggr_level=aggr_level,
        **kwargs,
    )
