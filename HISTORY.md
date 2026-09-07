# Changelog

All notable changes to ProteoPy will be documented in this file.
The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
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

**Plotting** (`pr.pl`)

- `peptide_intensities()`, `proteoform_intensities()`: new `facet_by`
  parameter splitting the samples (`.obs`) across a grid of subplots

**Preprocessing** (`pr.pp`)

- `summarize_peptides_by_neighbourhood_union()`: collapses peptides
  that overlap in the protein sequence, keeping the most abundant
  member of each group. Reimplements CCprofiler's
  `summarizeAlternativePeptideSequences(topN = 1)`
- **Tools** (`pr.tl`): `peptide_proximity()` tests whether the peptides
  of a proteoform cluster sit closer together in the protein sequence
  than a random grouping of the same size. A reimplementation of
  CCprofiler's `evaluateProteoformLocation`, which is the
  characterisation COPF applies to the proteoform groups it detects.

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
