# Changelog

All notable changes to ProteoPy will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Tutorials**: Protein-level analysis notebook

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
