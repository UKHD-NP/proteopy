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

If the function uses the AnnData.X matrix, always check if it is a
scipy.sparse matrix. As a general practice, the matrix will be made
non-sparse for its algorithm but if possible sparse operations on the
sparse matrix will be used to obtain the same result. If the function
modifies the AnnData.X matrix inplace or returns a transformed AnnData
matrix and the input was a sparse matrix, it is ensured that the output
AnnData.X matrix is also sparse.


General argument guidelines:
 - essential arguments which should be found in all functions unless it does not make sense: 
    adata : AnnData
        Input AnnData with .X (obs x vars) and .var annotations.
        Default=None (do not include this line in docstrings)
    layer : str | None
        Optional key in `adata.layers`; when set, quantification uses that layer
        Default=None (do not include this line in docstrings)
    zero_to_na : bool
        If True zeros in the AnnData X matrix will be replaced with np.nan prior to function execution.
        Default=None (do not include this line in docstrings)
    fill_na : float | int | None
        If True, NAs in the AnnData X matrix will be replaced with the argument.
        Default=None (do not include this line in docstrings)
 - selectively relevant arguments:
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
    ax : bool
        If `True`, returns the underlying Matplotlib Axes object instead of displaying
        the plot. Useful for further customization or integration into larger figures.
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
        Controls ordering, subsetting, and duplication of observations,
        variables, or categories on the plot axis.
        - If `order_by` is None and `order` is None: use existing
          .var_names or .obs_names order.
        - If `order_by` is None and `order` is not None: `order`
          specifies the exact items to plot and their sequence. Items
          not listed are excluded (subsetting). Items listed more than
          once appear multiple times (duplication).
        - If `order_by` is not None and `order` is None: use the unique
          values in the `order_by` column. If categorical, use its
          category order; if str/object, use sorted order.
        - If `order_by` is not None and `order` is not None: `order`
          defines which `order_by` groups to show and in what sequence.
          Groups absent from `order` are excluded (subsetting). Groups
          listed more than once appear multiple times (duplication).
        Default=None (do not include this line in docstrings)
    ascending : bool | None
        If `order` is None, sort the function relevant axis by a function-relevant
        metric. For example, if the plotting function computes the average var across
        obs and plots this in a barplot, sort the obs bars by ascending var average if
        True, if False sort the obs bars by descending var average.
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

#### 2) Handle sparse `.X` consistently
If a function uses `AnnData.X`:
- Detect sparsity via `scipy.sparse.issparse(adata.X)`.
- Prefer sparse-preserving operations when they can yield identical results.
- If an algorithm requires dense operations, temporarily convert to dense, but:
  - If the input was sparse and the function modifies `AnnData.X` in-place or
    returns a transformed `AnnData`, ensure the output remains sparse (same format
    or CSR by default), unless there is a documented reason not to.

Skeleton pattern:
```python
from scipy import sparse

def example_fn(adata, *, inplace=True, **kwargs):
    # Validate upfront
    from proteopy.utils.anndata import check_proteodata
    check_proteodata(adata)

    X = adata.X
    was_sparse = sparse.issparse(X)

    # Use sparse ops when possible; otherwise densify temporarily
    if was_sparse:
        # try sparse-safe path
        pass
    else:
        pass

    # ... compute, optionally producing X_new ...

    if inplace:
        if was_sparse and not sparse.issparse(adata.X):
            adata.X = sparse.csr_matrix(adata.X)
        check_proteodata(adata)  # validate before returning
        return None
    else:
        adata_out = adata.copy()
        # assign X_new to adata_out.X
        if was_sparse and not sparse.issparse(adata_out.X):
            adata_out.X = sparse.csr_matrix(adata_out.X)
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
| `layer` | Optional key in `adata.layers` specifying quantification data |
| `zero_to_na` | Convert zeros in `.X` to `np.nan` |
| `fill_na` | Replace missing values in `.X` with a specified constant |
| `verbose` | Print status messages about input data and output destinations (default: `False`) |

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

- `ax: bool`
  Return the underlying Matplotlib Axes object instead of displaying the plot (default=None).

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
  Controls ordering, subsetting, and duplication of observations, variables, or categories on the plot axis.
  - If `order_by is None` and `order is None`: use existing `.var_names` / `.obs_names` order.
  - If `order_by is None` and `order is not None`: `order` specifies the exact items to plot and their sequence. Items not listed are excluded (subsetting). Items listed more than once appear multiple times (duplication).
  - If `order_by is not None` and `order is None`: order by the unique values in `order_by`. If `order_by` is categorical, use its category order; if object/string, use sorted order.
  - If `order_by is not None` and `order is not None`: `order` defines which `order_by` groups to show and in what sequence. Groups absent from `order` are excluded (subsetting). Groups listed more than once appear multiple times (duplication).

- `ascending: bool | None`
  When `order` is `None`, sort the relevant axis by a function-relevant metric. For example, if a bar plot shows the mean of vars across obs, `ascending=True` sorts bars by ascending mean; `False` by descending; `None` preserves the derived order.


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
  - Run `flake8` for style compliance
  - Run `pylint` (error level only) before committing
  - Use `black` for auto-formatting

### Style Notes
Prefer f-strings for string interpolation. Use type hints in function signatures and docstrings but avoid verbose variable-level type checking. Perform input type checking at the beginning of the function when possible for good readability.

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

#### 2) Handle sparse `.X` consistently
If a function uses `AnnData.X`:
- Detect sparsity via `scipy.sparse.issparse(adata.X)`.
- Prefer sparse-preserving operations when they can yield identical results.
- If an algorithm requires dense operations, temporarily convert to dense, but:
  - If the input was sparse and the function modifies `AnnData.X` in-place or
    returns a transformed `AnnData`, ensure the output remains sparse (same format
    or CSR by default), unless there is a documented reason not to.

Skeleton pattern:
```python
from scipy import sparse

def example_fn(adata, *, inplace=True, **kwargs):
    # Validate upfront
    from proteopy.utils.anndata import check_proteodata
    check_proteodata(adata)

    X = adata.X
    was_sparse = sparse.issparse(X)

    # Use sparse ops when possible; otherwise densify temporarily
    if was_sparse:
        # try sparse-safe path
        pass
    else:
        pass

    # ... compute, optionally producing X_new ...

    if inplace:
        if was_sparse and not sparse.issparse(adata.X):
            adata.X = sparse.csr_matrix(adata.X)
        check_proteodata(adata)  # validate before returning
        return None
    else:
        adata_out = adata.copy()
        # assign X_new to adata_out.X
        if was_sparse and not sparse.issparse(adata_out.X):
            adata_out.X = sparse.csr_matrix(adata_out.X)
        check_proteodata(adata_out)  # validate before returning
        return adata_out
```

---


### Common Function Arguments

| Argument | Description |
|-----------|--------------|
| `adata` | AnnData object with `.X`, `.obs`, and `.var` annotations |
| `group_by` | Column in `adata.var` or `adata.obs` used for grouping |
| `layer` | Optional key in `adata.layers` specifying quantification data |
| `zero_to_na` | Convert zeros in `.X` to `np.nan` |
| `fill_na` | Replace missing values in `.X` with a specified constant |
| `verbose` | Print status messages about input data and output destinations (default: `False`) |

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

- `ax: bool`
  Return the underlying Matplotlib Axes object instead of displaying the plot (default=None).

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
  Controls ordering, subsetting, and duplication of observations, variables, or categories on the plot axis.
  - If `order_by is None` and `order is None`: use existing `.var_names` / `.obs_names` order.
  - If `order_by is None` and `order is not None`: `order` specifies the exact items to plot and their sequence. Items not listed are excluded (subsetting). Items listed more than once appear multiple times (duplication).
  - If `order_by is not None` and `order is None`: order by the unique values in `order_by`. If `order_by` is categorical, use its category order; if object/string, use sorted order.
  - If `order_by is not None` and `order is not None`: `order` defines which `order_by` groups to show and in what sequence. Groups absent from `order` are excluded (subsetting). Groups listed more than once appear multiple times (duplication).

- `ascending: bool | None`
  When `order` is `None`, sort the relevant axis by a function-relevant metric. For example, if a bar plot shows the mean of vars across obs, `ascending=True` sorts bars by ascending mean; `False` by descending; `None` preserves the derived order.


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
