**Beginning of AGENTS.md**
# Repository Guidelines
Package name: `ProteoPy`

This document defines the structure, conventions, and development practices for the ProteoPy package. It serves as a reference for developers and LLM-based assistants to maintain consistency and quality across the repository.

---

## Overview
ProteoPy is a Python package for bottom-up mass-spectrometry data analysis built on the AnnData framework. It provides a cohesive collection of tools to read, process, and analyze proteomics data at the precursor-, peptide-, and protein-levels.

It supports data from pipelines such as DIA-NN, MSFragger, and MaxQuant, offering capabilities for:
- Quality control (QC)
- Pre-processing such as data filtering, normalization, and quantification (aggregation)
- Differential abundance analysis
- Proteoform inference via the COPF algorithm (Bludau et al., 2021)

The design philosophy emphasizes clarity, modularity, and ease of use, allowing bioinformaticians to perform analyses quickly and reproducibly.

---

## Project Structure
The main package resides under `proteopy/`. Submodules are organized by functionality:

```
Avoid prolixity:
 - type checking only in function arguments and function output, not when defining variables.

### Function guidelines

Prioritize validating parameters and input at the beginning of the
function when it is convenient and elegant. Group all checks (type
guards, value-range assertions, mutually-exclusive argument checks,
etc.) before the main logic so readers can quickly see the contract
the function enforces.

Use `check_proteodata()` from `proteopy/utils/anndata.py` to validate
that an AnnData object conforms to ProteoPy assumptions. Call it at the
beginning of every public function and again before returning a new or
modified AnnData. Use `is_proteodata()` when you need to detect *whether*
the data conforms and at which level (it returns
`(True, "peptide")`, `(True, "protein")`, or `(False, None)`).

Both functions enforce the following checks:
 - **Structure**: obs/var indices must be unique and synchronised with
   obs_names/var_names; `.X` must be 2-dimensional.
 - **obs requirements**: `.obs` must contain a `sample_id` column.
   Columns `protein_id` / `peptide_id` must *not* appear in `.obs`
   (they belong in `.var`).
 - **var requirements**: `sample_id` must *not* appear in `.var`
   (it belongs in `.obs`).
 - **Infinite values**: `.X` must not contain `np.inf` / `-np.inf`.
   When the `layers` parameter is provided, the specified
   `adata.layers` matrices are checked as well.
 - **ID columns must not contain NaN**: neither `peptide_id` nor
   `protein_id` may have missing values.
 - **Protein-level**: `.var["protein_id"]` must exist and match
   `.var_names` (same values, same order).
 - **Peptide-level**: `.var["peptide_id"]` and `.var["protein_id"]`
   must both exist. `peptide_id` must match `.var_names`; each
   peptide must map to exactly one `protein_id` (no multi-mapping).

Signature reference:
```python
is_proteodata(adata, *, raise_error=False, layers=None)
    -> tuple[bool, str | None]
check_proteodata(adata, *, layers=None)
    -> tuple[bool, str | None]   # raises ValueError on failure
```

If a function uses `AnnData.X` or a selected layer, always detect whether
that matrix is `scipy.sparse`. Sparse input is supported by warning the
user with `warnings.warn(..., UserWarning, stacklevel=2)`, converting the
selected matrix with `.toarray()`, and running the algorithm on a dense
array. Computed or transformed output remains dense; do not reject sparse
input and do not convert the output back to sparse.


General argument guidelines:
 - essential arguments which should be found in all functions unless it does not make sense:
    adata : AnnData
        Input AnnData with .X (obs x vars) and .var annotations.
        Default=None (do not include this line in docstrings)
    zero_to_na : bool
        If True zeros in the AnnData X matrix will be replaced with np.nan prior to function execution.
        Default=None (do not include this line in docstrings)
    fill_na : float | int | None
        If True, NAs in the AnnData X matrix will be replaced with the argument.
        Default=None (do not include this line in docstrings)
 - selectively relevant arguments:
    layer : str | None
        Add this parameter only when the function explicitly needs or supports
        reading from an `adata.layers` matrix. Functions normally operate on
        `adata.X` and should not expose `layer` by default. When present,
        `None` selects `adata.X` and a string selects that layer.
        Default=None (do not include this line in docstrings)
    metadata_key : str
        When the function requires a metadata (.obs or .var) key by definition, this argument supplies the column. For example the function batch_correct would require the argument batch_key found in .obs. Replace metadata in metadata_key with the expected type of metadata.
        Default=Depends on the function and convention (do not include this line in docstrings)
    groupby : str
        Column in AnnData .var or .obs to perform grouping for the function algorithm (e.g. group by sample 'condition' to compute average peptide intensities across observations).
        Default=None (do not include this line in docstrings)
    verbose : bool
        If True, print status messages describing which input data is being used
        (e.g., matrix, layer, metadata columns), where results are stored
        (e.g., `.obs`, `.var`, `.uns` keys, file paths), progress, file saving
        location etc. Present in most functions across `pp`, `tl`, `pl`, `ann`,
        and other modules. Default=False (do not include this line in
        docstrings)

Module-specific function guidelines
 - pp and tl:
    inplace : bool
        If True, modify AnnData object in place; else return a new AnnData.
        Default=True (do not include this line in docstrings)
    key_added : str
        Metadata (.obs/.var) columns key or .uns slots key to save the computed data in.
        Default=Depends on the function (do not include this line in docstrings)
    random_state : int | None
        If the function performs computations with random components, set the seed to
this number. If None, use the internal function defaults.
        Default=None (do not include this line in docstrings)
    skip_na : bool
        In algorithms which can skip or retain NAs, if true, skip them. For example when
        computing the mean, if skip_na is True it would compute the mean only with
        non-NA values but if False, it will return NA if there were NAs present.
        Default depends on function (do not include this line in docstrings)
 - pl:
    show : bool
        If True, call plt.show() at the end.
        Default=True (do not include this line in docstrings)
    save : bool | str | Path | None
        If True, save to a default filename.
        If str/Path, save to that path. If False, do not save.
        Default=None (do not include this line in docstrings)
    ax : matplotlib.axes.Axes | None
        Matplotlib Axes object to plot onto. If provided, draw on that exact
        object without replacing it. If `None`, create a new figure and axes.
        Always return the Axes object used, including when `show=True` or the
        figure is saved.
        Default=None (do not include this line in docstrings)
    print_stats : bool
        If True, print the statistics underlying the plot as a pandas
        DataFrame before the plot is displayed and the axes object is
        returned. Always prints global summary statistics (e.g. mean, std,
        median, min, max). When `group_by` or `order_by` is provided,
        also prints per-group statistics below the global summary.
        Use `df.to_string(index=False, float_format="%.1f")` for
        tabular output and label sections with headers such as
        `"Global:"` and `f"\nPer {group_by}:"`.
        Default=False (do not include this line in docstrings)
    show_zeros : bool
        Don't display zeros if False.
        Default=True (do not include this line in docstrings)
    log_transform : float | None
        Base for log transformation of the data. 1 will be added to each value before transformation.
        Default=None (do not include this line in docstrings)
    z_transform : bool
        Transform values to have 0-mean and 1-variance
        along the peptide axis. Always uses zeros instead of NaNs if present, even if show_zeros=False.
        Default=False (do not include this line in docstrings)
    color : str | list
        Variable(s) to color observations by e.g. metadata columns, etc.
        Default=None (do not include this line in docstrings)
    color_scheme : str | dict | Sequence | Colormap | callable | None
        Defines the color mapping for groups. Can be a named Matplotlib colormap, a single color, a list/tuple of colors, a dict mapping labels to colors, a Colormap object, or a callable that returns colors. If `None`, the default Matplotlib color cycle is used.
        Default=None (do not include this line in docstrings)
    orderby : str | list | None
        Categorical .obs or .var column by which to order, subset, or
        duplicate observations or variables for plotting. When combined
        with `order`, controls which groups appear and in what sequence.
        Default=None (do not include this line in docstrings)
    order : str | list | None
        Controls ordering and subsetting of observations, variables, or
        categories on the plot axis. Values must be a subset of the
        unique values in the `order_by` column (duplicates not allowed).
        - If `order_by` is None and `order` is None: apply the default
          order rule to the axis ID column (`sample_id`, `peptide_id`
          or `protein_id`) — category order if categorical, else
          lexicographic order of the `str`-coerced values. Never the
          raw .var_names / .obs_names order.
        - If `order_by` is None and `order` is not None: `order`
          specifies the exact items to plot and their sequence. Items
          not listed are excluded (subsetting).
        - If `order_by` is not None and `order` is None: use the unique
          values in the `order_by` column. If categorical, use its
          category order; else the lexicographic order of its
          `str`-coerced values.
        - If `order_by` is not None and `order` is not None: `order`
          defines which `order_by` groups to show and in what sequence.
          Groups absent from `order` are excluded (subsetting). Values
          in `order` must be a subset of unique values in `order_by`.
        Default=None (do not include this line in docstrings)
    ascending : bool | None
        If `order` is None, sort the function relevant axis by a function-relevant
        metric. For example, if the plotting function computes the average var across
        obs and plots this in a barplot, sort the obs bars by ascending var average if
        True, if False sort the obs bars by descending var average. If None, no metric
        order is imposed and the default order rule applies (categories, else
        lexicographic).
        Default=None (do not include this line in docstrings)


**Example of preferred multi-line formatting:**

```python
some_function(
    arg1=arg1,
    arg2=arg2,
    arg3=arg3,
)

def my_function(
    arg1,
    arg2,
    arg3,
):
    raise ValueError(
        "Line one\n"
        "Line two\n"
        "Line three"
    )
```

---

## Module Conventions

### Function Design
All preprocessing (`pp`), tool (`tl`), and annotation (`ann`) functions operate in-place on an `AnnData` object by default.

#### Imports
```python
from proteopy.copf import pairwise_peptide_correlations   # public API
from proteopy.utils.anndata import sanitize_obs            # private helper
from tests.utils.helpers import transform_dendogram_r2py # test helper
```

---

### Data Validation & Matrix Sparsity (MANDATORY)

#### 1) Validate proteomics assumptions
Every public function that accepts an `AnnData` must call
`proteopy/utils/anndata.py:check_proteodata()` at the beginning and
again before returning (if a new `AnnData` is returned or the input is
modified in-place). Pass the `layers` parameter when the function
operates on specific `adata.layers` matrices.

Checks enforced by `check_proteodata()` / `is_proteodata()`:
- **Structure**: unique obs/var indices, 2-D `.X`, synchronised axes.
- **obs**: `sample_id` column required; `protein_id`/`peptide_id`
  must not appear in `.obs`.
- **var**: `sample_id` must not appear in `.var`.
- **Infinite values**: `.X` (and any requested layers) must be free
  of `np.inf` / `-np.inf`.
- **No NaN in ID columns**: `peptide_id` and `protein_id` must not
  contain missing values.
- **Protein-level**: `.var['protein_id']` must exist and match
  `.var_names` (same values in the same order).
- **Peptide-level**: `.var['peptide_id']` and `.var['protein_id']`
  must exist. `peptide_id` matches `.var_names`. Each peptide maps
  to exactly one `protein_id` (no multi-mapping).

Use `is_proteodata()` when you need to detect whether the data
conforms and at which level; it returns `(True, "peptide")`,
`(True, "protein")`, or `(False, None)`.

#### 2) Handle sparse matrices consistently
If a function uses `AnnData.X` or a selected layer:
- Detect sparsity via `scipy.sparse.issparse(Xsrc)`.
- If sparse, emit `UserWarning` with `stacklevel=2` explaining that the
  selected matrix is being densified.
- Convert with `Xsrc.toarray()` and perform the algorithm on a dense matrix.
- Keep computed or transformed output dense. Do not reject sparse input and
  do not convert output back to a sparse format.
- Test that sparse input warns and produces the same dense values as the
  equivalent dense input.

Skeleton pattern:
```python
import warnings

import numpy as np
from scipy import sparse

def example_fn(adata, *, inplace=True, **kwargs):
    # Validate upfront
    from proteopy.utils.anndata import check_proteodata
    check_proteodata(adata)

    Xsrc = adata.X
    if sparse.issparse(Xsrc):
        warnings.warn(
            "Sparse input is being densified for computation.",
            UserWarning,
            stacklevel=2,
        )
        X = Xsrc.toarray()
    else:
        X = np.asarray(Xsrc)

    # ... compute, optionally producing X_new ...

    if inplace:
        adata.X = np.asarray(X_new)
        check_proteodata(adata)  # validate before returning
        return None
    else:
        adata_out = adata.copy()
        adata_out.X = np.asarray(X_new)
        check_proteodata(adata_out)  # validate before returning
        return adata_out
```

#### 3) Data type assumptions for AnnData.X
ProteoPy assumes that `AnnData.X` contains only the following data types:
- `np.nan` (missing values)
- `int` (integer numeric values)
- `float` (floating-point numeric values)

**Infinite value validation**:
- `is_proteodata()` and `check_proteodata()` automatically check
  `.X` for `np.inf` and `-np.inf`.
- When the `layers` parameter is provided, the specified
  `adata.layers` matrices are checked as well.
- A `ValueError` is raised if infinite values are detected.
- Since all public functions must call `check_proteodata()` (see
  subsection 1 above), this validation is automatically enforced.
- No additional infinite value checking is needed in individual
  functions.

**Rationale**: Infinite values can cause unexpected behavior in
statistical computations, correlations, and normalization procedures.
Centralized validation in `check_proteodata()` ensures consistent
error handling across all public functions and guides users to clean
their data appropriately.

---


### Common Function Arguments

| Argument | Description |
|-----------|--------------|
| `adata` | AnnData object with `.X`, `.obs`, and `.var` annotations |
| `group_by` | Column in `adata.var` or `adata.obs` used for grouping |
| `zero_to_na` | Convert zeros in `.X` to `np.nan` |
| `fill_na` | Replace missing values in `.X` with a specified constant |
| `verbose` | Print status messages about input data and output destinations (default: `False`) |

Functions operate on `adata.X` by default. Add a `layer` parameter only when
the function explicitly needs or supports choosing an alternate matrix from
`adata.layers`; do not include it as a routine parameter otherwise.

### Additional Argument Conventions

#### Preprocessing and Tool Modules (`pp`, `tl`)
- `inplace`: modify `adata` directly (default: `True`)
- `key_added`: destination key in `.obs`, `.var`, or `.uns`
- `random_state`: random seed for reproducibility
- `skip_na`: whether to ignore missing values in calculations

#### Plotting Modules (`pl`)
To ensure consistent plotting behavior across `pl.*` modules, adhere to the following argument semantics. (Document defaults in code, but do not repeat default lines in docstrings.)

> Apply `show`, `save` and `ax` consistently across `pl.*` functions and reflect behavior in docstrings with concise wording. Apply further other arguments where relevant.

- `show: bool`
  Call plt.show() at the end of the function (default=True).

- `save: str | Path | None`
  Save the figure: str/Path for a specific path, None to skip saving (default=None).

- `ax: matplotlib.axes.Axes | None`
  Matplotlib Axes object to plot onto. If supplied, plot on that exact object;
  do not create, replace, or clear it. If `None`, create a new figure and axes.
  Always return the Axes object used, so `returned_ax is ax` when one was
  supplied. Showing or saving the figure does not change the return value
  (default=None).

Required implementation pattern:
```python
def example_plot(..., ax=None, show=True, save=None):
    if ax is None:
        _, ax = plt.subplots()

    ax.plot(x, y)

    if save is not None:
        ax.figure.savefig(save)
    if show:
        plt.show()
    return ax


fig, supplied_ax = plt.subplots()
returned_ax = example_plot(..., ax=supplied_ax, show=False)
assert returned_ax is supplied_ax
```

- `print_stats: bool`
  If True, print the statistics underlying the plot as a pandas
  DataFrame before the plot is displayed and the axes object is
  returned. Always prints global summary statistics (e.g. mean, std,
  median, min, max). When `group_by` or `order_by` is provided,
  also prints per-group statistics below the global summary
  (default=False).

- `show_zeros: bool`
  Display zeros in the visualization; if False, hide or mask zeros where applicable (default=True).

- `log_transform: float | None`
  Apply a log transform with the given base (add 1 before transform) (default=None).

- `z_transform: bool`
  Standardize values to mean 0 and variance 1 along the function-relevant axis (default=False).

- `color: str | list`
  Variable name(s) to color observations by (e.g., metadata columns).

- `color_scheme: str | dict | Sequence | Colormap | callable | None`
  Mapping for groups to colors. Accepts a named Matplotlib colormap, a single color, a list/tuple, a dict `{label: color}`, a `Colormap` object, or a callable returning colors. If `None`, use the Matplotlib default cycle.

- `order_by: str | list | None`
  Categorical `.obs` or `.var` column(s) by which to order, subset, or duplicate observations/variables for plotting. When combined with `order`, controls which groups appear and in what sequence.

- `order: str | list | None`
  Controls ordering and subsetting of observations, variables, or categories on the plot axis. Values must be a subset of the unique values in the `order_by` column (duplicates not allowed).
  - If `order_by is None` and `order is None`: apply the default order rule to the axis ID column (`.obs["sample_id"]`, `.var["peptide_id"]` / `.var["protein_id"]`) — category order if categorical, otherwise lexicographic order of the `str`-coerced values. Never the raw `.var_names` / `.obs_names` order. See *Deterministic Ordering in `pl`*.
  - If `order_by is None` and `order is not None`: `order` specifies the exact items to plot and their sequence. Items not listed are excluded (subsetting).
  - If `order_by is not None` and `order is None`: order by the unique values in `order_by`. If `order_by` is categorical, use its category order; otherwise use the lexicographic order of its `str`-coerced values.
  - If `order_by is not None` and `order is not None`: `order` defines which `order_by` groups to show and in what sequence. Groups absent from `order` are excluded (subsetting). Values in `order` must be a subset of unique values in `order_by`.

- `ascending: bool | None`
  When `order` is `None`, sort the relevant axis by a function-relevant metric. For example, if a bar plot shows the mean of vars across obs, `ascending=True` sorts bars by ascending mean; `False` by descending; `None` imposes no metric order, so the default order rule applies (see *Deterministic Ordering in `pl`*).


---

### Deterministic Ordering in `pl` (MANDATORY)

Every axis, group sequence, and legend in a `pl.*` function must have an
order the user can predict and control. Three rules produce that; they
apply to every plotting function without exception.

#### 1) Default order: category order, else lexicographic

Whenever a plotting function needs an order the caller has *not*
specified — the sample sequence on the x axis of
`pr.pl.n_peptides_per_sample()` / `pr.pl.n_proteins_per_sample()`, the
group sequence of a boxplot, the entries of a legend — derive it from
the annotation column supplying those labels:

- **Categorical dtype** (`pd.CategoricalDtype`, ordered or unordered):
  use `series.cat.categories`, in that order. Categories with no rows
  present remain valid positions on the axis, so a subset still plots
  against the full series; drop them only where the function documents
  that it drops empty groups.
- **Any other dtype** (object, str, numeric, bool): sort the unique
  values lexicographically on their string representation
  (`sorted(uniques, key=str)` — sort by the string, return the original
  values so they still match the data). The coercion is deliberate: it
  makes the order identical across functions, calls, and dtypes.

`proteopy/pl/_utils.py: resolve_default_order()` is the shared
implementation. Call it rather than re-deriving the rule.

**Never** fall back to the order rows happen to occupy in the object:
`.obs_names` / `.var_names`, or the order in which values first appear
along the axis. That order is an artefact of how the object was built
(`read.diann`'s pivot, a merge, a filter) and the user has no clean way
to change it. The goal is not that lexicographic order is *right* —
`F1, F10, F2` is plainly wrong for a fractionation — but that it is
consistent across every plot and that the user has one documented way
to override it.

**How the user changes it.** By storing the annotation as an ordered
Categorical, which is the single supported mechanism:

```python
adata.obs["fraction"] = pd.Categorical(
    adata.obs["fraction"],
    categories=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    ordered=True,
)
```

Say so in the docstring of every function whose default order comes
from this rule, and do not add per-function ordering parameters that
duplicate it.

#### 2) A parameter that imposes an order always wins

Rule 1 applies only when no argument imposes an order. When one does,
it overrides rule 1 entirely and the category order is not consulted:

- `order` — the given sequence, verbatim (and it subsets).
- `ascending` — sorts the relevant axis by the function's metric.
  `ascending=None` means no metric order was requested, so rule 1
  applies.
- Any function-specific ordering: `sort_by`, a dendrogram leaf order,
  a clustering result, and so on.

Precedence, highest first: `order` → `ascending` / metric sort →
rule 1 (categories, else lexicographic).

#### 3) `group_by` and `order_by` follow the same rule

The sequence of the groups produced by `group_by`, and the sequence
produced by `order_by` when `order` is `None`, are rule 1 applied to
that column: its category order if it is categorical, otherwise the
lexicographic order of its `str`-coerced values. Not the order the
groups appear in along the axis.

#### 4) Testing

A `pl` function whose order is derived rather than given needs two
tests: one asserting the tick-label sequence follows `cat.categories`
for a categorical annotation, and one asserting lexicographic order for
that same annotation stored as plain strings.

---

### Axis Labels Come from ID Columns, Never from the Index (MANDATORY)

Tick labels, group labels, legend entries, and any other text
identifying a sample, protein, or peptide are read from the annotation
columns:

- observations → `adata.obs["sample_id"]`
- variables → `adata.var["peptide_id"]` / `adata.var["protein_id"]`

**Never** from `.obs_names`, `.var_names`, `.obs.index`, or
`.var.index`.

The drawn strings are the same either way — `check_proteodata()`
enforces that `.obs["sample_id"]` is identical to `.obs_names` and that
`.var["peptide_id"]` / `.var["protein_id"]` match `.var_names`, element
for element — so this is not about which labels appear. It is about dtype: AnnData
keeps `.obs_names` / `.var_names` as a plain string index, which cannot
carry a `CategoricalDtype` and therefore cannot carry a user-defined
category order. A function that labels from the index silently discards
the order the user set with `categories=` and falls back to positional
order. Reading the column keeps rule 1 above reachable; reading the
index makes it unreachable.

Corollaries:

- Use the column for selection and grouping too, not only for the drawn
  label, so ordering and labelling cannot disagree.
- When a plot summarises a level other than the object's variables
  (peptides aggregated into proteins), label from `.var["protein_id"]`.
- Functions that still label from `.obs_names` / `.var_names` are
  non-conforming; migrate them when the function is next touched, and
  add the two ordering tests from rule 4 in the same commit.

---

## Documentation
Documentation is built using Sphinx, with the source files located in `docs/`. Keep function docstrings structured and descriptive, following numpydoc conventions.

---

## Development Workflow

### Setup & Installation
```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -e .
pip install -r requirements/requirements_development.txt
```

### Quality Checks
```bash
flake8 .
pylint $(git ls-files "*.py") --disable=all --enable=E,F
pytest -v tests/
```

---

## Testing
- Add tests in `tests/<subpackage>/test_*.py` corresponding to your module.
- Reuse fixtures from `tests/utils/helpers.py`.
- Place datasets in `tests/data/<feature>/`.
- For stochastic operations, fix random seeds.
- Ensure CI passes all checks and test coverage remains high.
- Prioritize simple and readable tests separated into multiple testing functions
  than convoluted and complex tests.
- If numeric arguments are being tested with multiple numeric examples, create a dictionary with the numbers and the expected output and iterate over these for function testing.

---

## Commit and Pull Request Guidelines
- Write concise, capitalized commit subjects (e.g., `Feature: pl.intensity_distribution_per_obs()`).
- Keep changes logically grouped.
- Pull requests should include:
  - Purpose and motivation
  - Key changes
  - New data or notebooks
  - Test/lint outputs
  - Before/after visuals (if applicable)
  - Linked issues or related discussions

---

## Key Repository Files
```
proteopy/
├── ann/          # Annotation tools for the AnnData object
├── datasets/     # Loading and simulating curated datasets
├── get/          # Retrieval helpers for accessing AnnData content
├── pl/           # Plotting and visualization modules
├── pp/           # Preprocessing and quality control functions for proteomics data
├── read/         # Data import utilities for DIA-NN, MaxQuant, etc.
├── tl/           # Analytical tools and algorithms (e.g. COPF)
└── utils/        # Shared helpers and miscellaneous utilities
tests/             # Mirrors proteopy structure for testing
```

Each submodule contains multiple function-specific files (e.g., `normalization.py`, `imputation.py`). Keep files cohesive and small; group related functions meaningfully.

- Tests: located in `tests/`, following the module structure of `proteopy/`.
- Test data: under `tests/data/`.
- Documentation: Sphinx source files in `docs/`.
- Tutorial datasets: under `data/`.

New assets must remain lightweight and include provenance notes.

---

## Coding Standards
- Python version: 3.10–3.11
- Indentation: 4 spaces
- Line length: 72 characters for docstrings and 79 characters for code.
- Naming conventions:
  - Functions and variables → `snake_case`
  - Classes → `CamelCase`
  - Constants → `UPPER_CASE`

- Formatting and linting:
  - Run `flake8` for style compliance (complexity threshold C901 = 20).
    Keep functions below this threshold by extracting input validation
    and major algorithmic steps into helper functions.
  - Run `pylint` (error level only) before committing
  - Use `black` for auto-formatting

### Style Notes
Prefer f-strings for string interpolation. Use type hints in function signatures and docstrings but avoid verbose variable-level type checking. Perform input type checking at the beginning of the function when possible for good readability.

### Commenting Convention
Strike a balance between "the code is the documentation" and readable
navigation. Do not over-comment obvious code.

- **Section headings** — use `# -- <description>` to mark major logical
  blocks within a function. Reserve these for larger chunks of code,
  not individual lines. They act as scannable signposts when reading
  longer functions.
- **Clarification comments** — use plain `#` for brief notes that
  explain *why* non-obvious code exists, not *what* it does.


### Import Alias Convention
In tutorials, docstring examples, and documentation, always import proteopy as `pr`:
```python
import proteopy as pr

# Example usage in docstrings
>>> import proteopy as pr
>>> adata = pr.datasets.example_peptide_data()
>>> pr.tl.hclust_vars(adata, group_by="condition")
>>> pr.pl.hclust_vars_silhouette_scores(adata, k=5)
```

**Example of type hints**
```python
import pandas as pd
some_function(
    df: pd.DataFrame
)
```

**Example of docstrings**
```python
def preprocess_data(
    adata: ad.AnnData,
    min_proteins: int = 200,
    min_samples: int = 3,
    log_transform: bool = True,
    inplace: True,
) -> ad.AnnData:
    """
    Preprocess an AnnData object by filtering and normalizing cells and genes.

    Parameters
    ----------
    adata : AnnData
        :class:`~anndata.AnnData`
    min_proteins : int, optional
        Minimum number of proteins expressed per sample. Samples with fewer proteins
        are filtered out. Defaults to 200.
    min_samples : int, optional
        Minimum number of samples expressing a protein. Proteins detected in fewer
        samples are removed. Defaults to 3.
    inplace : bool, optional
        If False, return a copy of `adata`. Otherwise, modify in place. Defaults to False.

    Returns
    -------
    AnnData
        The filtered and optionally transformed AnnData object.

    Warns
    -----
    UserWarning
        If `min_proteins` removes more than 50 % of samples.

    Examples
    --------
    >>> import proteopy as pr
    >>> adata = pr.datasets.example_peptide_data()
    >>> pr.pp.preprocess_data(adata, min_proteins=100)
    >>> pr.pp.preprocess_data(adata, inplace=False, min_samples=5)
    >>> pr.pp.preprocess_data(adata, verbose=True)
    Removed 1501 samples.
```


**Example of preferred multi-line formatting:**

```python
some_function(
    arg1=arg1,
    arg2=arg2,
    arg3=arg3,
)

def my_function(
    arg1,
    arg2,
    arg3,
):
    raise ValueError(
        "Line one\n"
        "Line two\n"
        "Line three"
    )
```

---

## Module Conventions

### Function Design
All preprocessing (`pp`), tool (`tl`), and annotation (`ann`) functions operate in-place on an `AnnData` object by default.

#### Imports
```python
from proteopy.copf import pairwise_peptide_correlations   # public API
from proteopy.utils.anndata import sanitize_obs            # private helper
from tests.utils.helpers import transform_dendogram_r2py # test helper
```

---

### Data Validation & Matrix Sparsity (MANDATORY)

#### 1) Validate proteomics assumptions
Every public function that accepts an `AnnData` must call
`proteopy/utils/anndata.py:check_proteodata()` at the beginning and
again before returning (if a new `AnnData` is returned or the input is
modified in-place). Pass the `layers` parameter when the function
operates on specific `adata.layers` matrices.

Checks enforced by `check_proteodata()` / `is_proteodata()`:
- **Structure**: unique obs/var indices, 2-D `.X`, synchronised axes.
- **obs**: `sample_id` column required; `protein_id`/`peptide_id`
  must not appear in `.obs`.
- **var**: `sample_id` must not appear in `.var`.
- **Infinite values**: `.X` (and any requested layers) must be free
  of `np.inf` / `-np.inf`.
- **No NaN in ID columns**: `peptide_id` and `protein_id` must not
  contain missing values.
- **Protein-level**: `.var['protein_id']` must exist and match
  `.var_names` (same values in the same order).
- **Peptide-level**: `.var['peptide_id']` and `.var['protein_id']`
  must exist. `peptide_id` matches `.var_names`. Each peptide maps
  to exactly one `protein_id` (no multi-mapping).

Use `is_proteodata()` when you need to detect whether the data
conforms and at which level; it returns `(True, "peptide")`,
`(True, "protein")`, or `(False, None)`.

#### 2) Handle sparse matrices consistently
If a function uses `AnnData.X` or a selected layer:
- Detect sparsity via `scipy.sparse.issparse(Xsrc)`.
- If sparse, emit `UserWarning` with `stacklevel=2` explaining that the
  selected matrix is being densified.
- Convert with `Xsrc.toarray()` and perform the algorithm on a dense matrix.
- Keep computed or transformed output dense. Do not reject sparse input and
  do not convert output back to a sparse format.
- Test that sparse input warns and produces the same dense values as the
  equivalent dense input.

Skeleton pattern:
```python
import warnings

import numpy as np
from scipy import sparse

def example_fn(adata, *, inplace=True, **kwargs):
    # Validate upfront
    from proteopy.utils.anndata import check_proteodata
    check_proteodata(adata)

    Xsrc = adata.X
    if sparse.issparse(Xsrc):
        warnings.warn(
            "Sparse input is being densified for computation.",
            UserWarning,
            stacklevel=2,
        )
        X = Xsrc.toarray()
    else:
        X = np.asarray(Xsrc)

    # ... compute, optionally producing X_new ...

    if inplace:
        adata.X = np.asarray(X_new)
        check_proteodata(adata)  # validate before returning
        return None
    else:
        adata_out = adata.copy()
        adata_out.X = np.asarray(X_new)
        check_proteodata(adata_out)  # validate before returning
        return adata_out
```

---


### Common Function Arguments

| Argument | Description |
|-----------|--------------|
| `adata` | AnnData object with `.X`, `.obs`, and `.var` annotations |
| `group_by` | Column in `adata.var` or `adata.obs` used for grouping |
| `zero_to_na` | Convert zeros in `.X` to `np.nan` |
| `fill_na` | Replace missing values in `.X` with a specified constant |
| `verbose` | Print status messages about input data and output destinations (default: `False`) |

Functions operate on `adata.X` by default. Add a `layer` parameter only when
the function explicitly needs or supports choosing an alternate matrix from
`adata.layers`; do not include it as a routine parameter otherwise.

### Additional Argument Conventions

#### Cross-Module Arguments (`pp`, `tl`, `pl`, `ann`, and others)
- `verbose`: print status messages describing which input data is being used and where results are stored (default: `False`). Include in most functions.

#### Preprocessing and Tool Modules (`pp`, `tl`)
- `inplace`: modify `adata` directly (default: `True`)
- `key_added`: destination key in `.obs`, `.var`, or `.uns`
- `random_state`: random seed for reproducibility
- `skip_na`: whether to ignore missing values in calculations

#### Plotting Modules (`pl`)
To ensure consistent plotting behavior across `pl.*` modules, adhere to the following argument semantics. (Document defaults in code, but do not repeat default lines in docstrings.)

> Apply `show`, `save` and `ax` consistently across `pl.*` functions and reflect behavior in docstrings with concise wording. Apply further other arguments where relevant.

- `show: bool`
  Call plt.show() at the end of the function (default=True).

- `save: str | Path | None`
  Save the figure: str/Path for a specific path, None to skip saving (default=None).

- `ax: matplotlib.axes.Axes | None`
  Matplotlib Axes object to plot onto. If supplied, plot on that exact object;
  do not create, replace, or clear it. If `None`, create a new figure and axes.
  Always return the Axes object used, so `returned_ax is ax` when one was
  supplied. Showing or saving the figure does not change the return value
  (default=None).

- `print_stats: bool`
  If True, print the statistics underlying the plot as a pandas
  DataFrame before the plot is displayed and the axes object is
  returned. Always prints global summary statistics (e.g. mean, std,
  median, min, max). When `group_by` or `order_by` is provided,
  also prints per-group statistics below the global summary
  (default=False).

- `show_zeros: bool`
  Display zeros in the visualization; if False, hide or mask zeros where applicable (default=True).

- `log_transform: float | None`
  Apply a log transform with the given base (add 1 before transform) (default=None).

- `z_transform: bool`
  Standardize values to mean 0 and variance 1 along the function-relevant axis (default=False).

- `color: str | list`
  Variable name(s) to color observations by (e.g., metadata columns).

- `color_scheme: str | dict | Sequence | Colormap | callable | None`
  Mapping for groups to colors. Accepts a named Matplotlib colormap, a single color, a list/tuple, a dict `{label: color}`, a `Colormap` object, or a callable returning colors. If `None`, use the Matplotlib default cycle.

- `order_by: str | list | None`
  Categorical `.obs` or `.var` column(s) by which to order, subset, or duplicate observations/variables for plotting. When combined with `order`, controls which groups appear and in what sequence.

- `order: str | list | None`
  Controls ordering and subsetting of observations, variables, or categories on the plot axis. Values must be a subset of the unique values in the `order_by` column (duplicates not allowed).
  - If `order_by is None` and `order is None`: apply the default order rule to the axis ID column (`.obs["sample_id"]`, `.var["peptide_id"]` / `.var["protein_id"]`) — category order if categorical, otherwise lexicographic order of the `str`-coerced values. Never the raw `.var_names` / `.obs_names` order. See *Deterministic Ordering in `pl`*.
  - If `order_by is None` and `order is not None`: `order` specifies the exact items to plot and their sequence. Items not listed are excluded (subsetting).
  - If `order_by is not None` and `order is None`: order by the unique values in `order_by`. If `order_by` is categorical, use its category order; otherwise use the lexicographic order of its `str`-coerced values.
  - If `order_by is not None` and `order is not None`: `order` defines which `order_by` groups to show and in what sequence. Groups absent from `order` are excluded (subsetting). Values in `order` must be a subset of unique values in `order_by`.

- `ascending: bool | None`
  When `order` is `None`, sort the relevant axis by a function-relevant metric. For example, if a bar plot shows the mean of vars across obs, `ascending=True` sorts bars by ascending mean; `False` by descending; `None` imposes no metric order, so the default order rule applies.

> The two mandatory `pl` sections above — **Deterministic Ordering in
> `pl`** and **Axis Labels Come from ID Columns, Never from the Index**
> — govern `order`, `order_by`, `group_by`, every default axis or group
> sequence, and every drawn label. Read them before writing a `pl`
> function.


---

## Documentation
Documentation is built using Sphinx, with the source files located in `docs/`. Keep function docstrings structured and descriptive, following numpydoc conventions.

---

## Development Workflow

### Setup & Installation
```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -e .
pip install -r requirements/requirements_development.txt
```

### Quality Checks
```bash
flake8 .
pylint $(git ls-files "*.py") --disable=all --enable=E,F --disable=E0401
pytest -v tests/
```

---

## Testing
- Add tests in `tests/<subpackage>/test_*.py` corresponding to your module.
- Reuse fixtures from `tests/utils/helpers.py`.
- Place datasets in `tests/data/<feature>/`.
- For stochastic operations, fix random seeds.
- Ensure CI passes all checks and test coverage remains high.

---

## Commit and Pull Request Guidelines
- Write concise, capitalized commit subjects (e.g., `Feature: pl.intensity_distribution_per_obs()`).
- Keep changes logically grouped.
- Pull requests should include:
  - Purpose and motivation
  - Key changes
  - New data or notebooks
  - Test/lint outputs
  - Before/after visuals (if applicable)
  - Linked issues or related discussions

---

## Changelog

The project maintains a changelog in `HISTORY.md` following the
[Keep a Changelog](https://keepachangelog.com/) format. All notable changes
should be documented here, even before a release is published.

- Add new entries under the `[Unreleased]` section as changes are made
- Use categories: `Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`,
  `Security`
- When releasing a new version, move unreleased changes to a versioned section
  with the release date (e.g., `[0.2.0] - 2025-03-15`)
- Keep entries concise but descriptive, including module paths where relevant
  (e.g., `pr.pp.normalize_median()`)

---

## Key Repository Files
```
proteopy/
├── .github/workflows/format-code_perform-tests_on_push-pr.yaml
├── AGENTS.md                # This file: repository and agent instructions
├── pyproject.toml
├── README.md
└── docs/                    # Sphinx documentation source
```

---

**End of AGENTS.md**
