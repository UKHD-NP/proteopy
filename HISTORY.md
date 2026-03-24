# Changelog

All notable changes to ProteoPy will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
