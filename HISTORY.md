# Changelog

All notable changes to ProteoPy will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Tools** (`pr.tl`): `peptide_proximity()` tests whether the peptides
  of a proteoform cluster sit closer together in the protein sequence
  than a random grouping of the same size. A reimplementation of
  CCprofiler's `evaluateProteoformLocation`, which is the
  characterisation COPF applies to the proteoform groups it detects.

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
