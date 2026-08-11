# Changelog

All notable changes to ProteoPy will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **Plotting** (`pr.pl`): `n_var_per_sample()`, and with it
  `n_peptides_per_sample()` and `n_proteins_per_sample()`, now derive
  every axis and group order deterministically instead of following
  the order rows happen to occupy in the AnnData object. Bars, groups
  and blocks follow the category order of the annotation when the
  column is a Categorical, otherwise the lexicographic order of its
  values — so a user fixes the order once, by storing the annotation
  as an ordered Categorical, instead of relying on how the file was
  read. **This changes existing figures**: a plot whose bars followed
  `.obs_names` will now be sorted.
  - x-axis labels come from `adata.obs["sample_id"]` rather than
    `adata.obs_names`. The drawn strings are unchanged —
    `check_proteodata()` enforces that the two are identical — but an
    AnnData axis index cannot carry a category order, so the column is
    the only source that can.
  - `order` now **subsets**, as its documented semantics always
    claimed: values it omits are excluded from the plot and from the
    printed statistics, instead of being appended after the listed
    ones. Its values are validated against `adata.obs["sample_id"]`.
  - `ascending` combined with `order_by` sorts samples within each
    group; it was silently ignored before. It is still ignored, with a
    warning, when `order` or `group_by` is set.
  - samples with a missing `order_by` value are drawn in a trailing
    block labelled `NA` instead of `nan`.
  - new `proteopy/pl/_utils.py` holds `resolve_default_order()`, the
    shared implementation of the rule, for the remaining `pl`
    functions to adopt.

### Added

- **Preprocessing** (`pr.pp`): `summarize_peptides_by_neighbourhood_union()`
  collapses peptides that overlap in the protein sequence, keeping the
  most abundant member of each group. Peptide positions are resolved
  from a FASTA in the same call. A reimplementation of CCprofiler's
  `summarizeAlternativePeptideSequences(topN = 1)`.
  - groups by positional overlap rather than substring containment, so
    it sees peptide pairs that overlap without either containing the
    other, and needs no separate modification-summarisation step
  - selects the most abundant member instead of aggregating; `top_n`
    sums the leading members instead, with `keep_less` controlling
    undersized groups
  - missing values are deprioritised rather than removed: an
    incomplete peptide sorts last and loses to any complete
    competitor, but survives if its group has no complete member
  - equal totals are resolved by `tie_break_key`, so the result does
    not depend on input row order; the default sorts non-letters after
    letters, favouring the unmodified form of an identifier
  - `on_unknown_protein` and `on_unlocated_peptide` decide whether an
    unresolvable position raises, skips the peptide, or leaves the
    position undefined
  - `.var` is reduced to the peptide-level proteodata columns plus the
    function's own output, since the surviving row's annotations
    describe one member rather than the group; `keep_var_cols` carries
    chosen columns through, aggregated across the group

### Fixed

- **Datasets** (`pr.datasets`) and **Download** (`pr.download`):
  `williams_2018()` no longer conflates measured zeros with missing
  values. Three defects, each masked by another:
  - zeros in `.X` were coerced to `np.nan`, discarding 13,547 genuine
    measurements
  - charge-state summation used pandas' default `min_count=0`, so a
    group with no measurements at all summed to `0.0`, inventing 3,324
  - the same summation skipped `NaN` inside a *partially* measured
    group, reporting a partial total as complete for 260 cells
  - **this changes `.X`**, and therefore the output of
    `pr.download.williams_2018()`. Verified against the PRIDE deposit
    used by the original publication: the two are now bit-identical
    with an identical missingness pattern.

### Changed

- **Datasets** (`pr.datasets`) and **Download** (`pr.download`):
  `williams_2018()` gained a `zero_to_na` parameter (default `False`),
  for consistency with sibling functions. It is mutually exclusive
  with `fill_na`. Note it governs zero semantics only and does not
  restore the summation defects above.
- **Preprocessing** (`pr.pp`): `normalize_median()`
  - now defaults to `log_space=True`
  - renamed the `batch_id` parameter to `group_by`
  - renamed the `zeros_to_na` parameter to `zero_to_na`, for
    consistency with sibling functions
  - no longer accepts sparse `.X` input

## [0.1.1] - 2025-03-24

### Added

- **Preprocessing** (`pr.pp`): `summarize_modifications()` for
  modification summarization
- **Analysis** (`pr.tl`): ANOVA support in `differential_abundance()`
- **Visualization** (`pr.pl`): `binary_heatmap()`, `box()`,
  `volcano()`, `peptides_on_sequence()`,
  `peptides_on_prot_sequence()`; `print_stats` parameter across
  multiple plot functions
- **Datasets** (`pr.datasets`): `williams_2018()` and
  `karayel_2020()` download functions
- **Utilities** (`pr.utils`): Public API with `is_proteodata()`,
  `check_proteodata()`, `is_log_transformed()`
- **Documentation**: Sphinx documentation site; proteoform inference
  and protein-level analysis tutorials

### Changed

- **Reader** (`pr.read`): `diann()` now supports version >=1.9.1
  with automatic version dispatch
- **Preprocessing** (`pr.pp`): `impute_downshift()` now supports
  `group_by`; `normalize_median()` gains `method` parameter;
  `remove_contaminants()` defaults to `inplace=True`
- **Validation**: `is_proteodata()` now checks for NaN in ID
  columns, infinite values in `.X`/layers, and obs/var index sync

### Fixed

- `volcano_plot` type incompatibility and label display
- `n_cat1_per_cat2_hist` minimum bin width

## [0.1.0] - 2025-01-29

Initial release of ProteoPy.

### Added

- **Data import** (`pr.read`): Support for DIA-NN and generic
  long-format tables
- **Annotation** (`pr.ann`): Functions to annotate samples (`.obs`) and
  variables (`.var`)
- **Quality control** (`pr.pp`): Completeness filtering, CV calculation,
  contaminant removal
- **Preprocessing** (`pr.pp`): Median normalization, downshift imputation
- **Differential abundance** (`pr.tl`): t-test, Welch's test, ANOVA with
  multiple testing correction
- **Proteoform inference** (`pr.tl`): COPF algorithm reimplementation for
  detecting functional proteoform groups
- **Visualization** (`pr.pl`): Volcano plots, abundance rank plots, intensity
  distributions, CV plots, correlation matrices, hierarchical clustering
  profiles
- **Datasets** (`pr.datasets`): Built-in example datasets (Karayel 2020)
