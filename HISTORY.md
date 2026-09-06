# Changelog

All notable changes to ProteoPy will be documented in this file.
The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

**Plotting** (`pr.pl`)

- `peptide_intensities()`, `proteoform_intensities()`: new `facet_by`
  parameter splitting the samples (`.obs`) across a grid of subplots

**Preprocessing** (`pr.pp`)

- `summarize_peptides_by_neighbourhood_union()`: collapses peptides
  that overlap in the protein sequence, keeping the most abundant
  member of each group. Reimplements CCprofiler's
  `summarizeAlternativePeptideSequences(topN = 1)`

### Fixed

**Datasets** (`pr.datasets`) and **Download** (`pr.download`)

- `williams_2018()` no longer conflates measured zeros with missing
  values. Three defects, each masked by another:
  - zeros in `.X` were coerced to `np.nan`, discarding 13,547 genuine
    measurements
  - charge-state summation used pandas' default `min_count=0`, so a
    group with no measurements summed to `0.0`, inventing 3,324
  - the same summation skipped `NaN` inside a *partially* measured
    group, reporting a partial total as complete for 260 cells
- **This changes `.X`**, and therefore the output of
  `pr.download.williams_2018()`. Verified against the PRIDE deposit
  used by the original publication: the two are now bit-identical,
  with an identical missingness pattern
- `williams_2018()`: new `zero_to_na` parameter (default `False`), for
  consistency with sibling functions; mutually exclusive with
  `fill_na`. Governs zero semantics only, and does not restore the
  summation defects above

### Changed

**Preprocessing** (`pr.pp`)

- `normalize_median()`:
  - now defaults to `log_space=True`
  - `batch_id` parameter renamed to `group_by`
  - `zeros_to_na` parameter renamed to `zero_to_na`, for consistency
    with sibling functions
  - no longer accepts sparse `.X` input

## [0.1.1] - 2025-03-24

### Added

**Preprocessing** (`pr.pp`)

- `summarize_modifications()` for modification summarization

**Analysis** (`pr.tl`)

- ANOVA support in `differential_abundance()`

**Plotting** (`pr.pl`)

- `binary_heatmap()`, `box()`, `volcano()`,
  `peptides_on_sequence()`, `peptides_on_prot_sequence()`
- `print_stats` parameter across multiple plot functions

**Datasets** (`pr.datasets`)

- `williams_2018()` and `karayel_2020()` download functions

**Utilities** (`pr.utils`)

- public API: `is_proteodata()`, `check_proteodata()`,
  `is_log_transformed()`

**Documentation** (`docs/`)

- Sphinx documentation site
- proteoform inference and protein-level analysis tutorials

### Changed

**Reader** (`pr.read`)

- `diann()` supports version >=1.9.1, with automatic version dispatch

**Preprocessing** (`pr.pp`)

- `impute_downshift()`: now supports `group_by`
- `normalize_median()`: new `method` parameter
- `remove_contaminants()`: defaults to `inplace=True`

**Validation** (`pr.utils`)

- `is_proteodata()` now checks for NaN in ID columns, infinite values
  in `.X`/layers, and obs/var index sync

### Fixed

**Plotting** (`pr.pl`)

- `volcano_plot()`: type incompatibility and label display
- `n_cat1_per_cat2_hist()`: minimum bin width

## [0.1.0] - 2025-01-29

Initial release of ProteoPy.

### Added

**Reader** (`pr.read`)

- DIA-NN and generic long-format tables

**Annotation** (`pr.ann`)

- annotation of samples (`.obs`) and variables (`.var`)

**Preprocessing** (`pr.pp`)

- quality control: completeness filtering, CV calculation,
  contaminant removal
- median normalization, downshift imputation

**Analysis** (`pr.tl`)

- differential abundance: t-test, Welch's test, ANOVA with multiple
  testing correction
- proteoform inference: COPF algorithm reimplementation for detecting
  functional proteoform groups

**Plotting** (`pr.pl`)

- volcano, abundance rank, intensity distribution, CV, correlation
  matrix and hierarchical clustering profile plots

**Datasets** (`pr.datasets`)

- built-in example datasets (Karayel 2020)
