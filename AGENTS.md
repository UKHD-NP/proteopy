# Repository Guidelines
Package name: CoPro: correlation proteomics
Use the following guidelines for your prompt responses

## Project Structure & Module Organization
This is a python package for bottom-up mass-spectrometry data analysis based on the AnnData data structure. The package is a cohesive collection of functions which enable the user to read raw precursor-, peptide- or protein-level data output by pipelines such as diaNN, msfragger or maxQuant and provide QC, analysis and functions such as filtering, normalization, aggregation, differential abundance analysis etc. In addition, it provides proteoform-inference from peptide-level information via the COPF algorithm (Bludau et al. 2021) to enable proteoform-level analysis.

The functions are intuitive and simple to understand which enables any bioinformatitian to perform a fast analysis of their proteomics data using this package.

### Basic structure
The core Python package sits in `copro/` with subpackages for preprocessing (`pp`), analysis tools (`tl`), plotting (`pl`), acquisition helpers (`get`, `read`), annotations (`ann`), shared utilities (`utils`) and for loading provided datasets (`datasets`). Each of these subpackage directories contains python files with functions. These files are named by the type of functions they contain e.g. normalization can contain multiple normalization functions. However, less functions per file are prefered if this helps with categorically consistent file naming.
Tests mirror this layout in `tests/`. If fixtures require loading data, these are found under `tests/data/`.
Input data for tutorials such as curated datasets, reference proteome and other annotations live in `data/`.
Sphinx documentation source directory is found in `docs/`.
Keep new artefacts lightweight and committed with provenance notes.

### Assumptions of data strucutre
CoPro assumes the following to be able to determine if the query AnnData contains proteomics data:
 - if the data is a protein-level proteomics dataset, it must contain the .var column `protein_id` which contains the same values and in the same order as the .var index and .var_names.
 - if the data is a peptide-level proteomics dataset, it must contain the .var columns `peptide_id` and `protein_id`. The `peptide_id` column contains the same values and in the same order as the .var index and .var_names. The `protein_id` are the proteins that the peptides map to. It must be single mapping, so no peptide should map to more than one protein_id.

### Module structure and function locations 

copro/
├── copro
│   ├── ann: directory containing modules which allow annotation of the anndata object
│   │   ├── generic.py
│   │   └── proteins.py
│   ├── datasets: directory containing modules which allow for the loading of specific curated datasets (usually from publications) or to generate simulated datasets.
│   │   ├── <surnameYEAR>.py
│   │   └── simulate_minimal_proteins_dataset_bulk.py
│   ├── get: directory containing modules for the formatted accession of information in the anndata object
│   │   ├── proteoforms.py
│   │   └── hypothesis_testing.py
│   ├── pl: directory containing modules for plotting data stored in the anndata object
│   │   ├── intensities.py
│   │   ├── stats.py
│   │   ├── stats_per_obs.py
│   │   ├── stats_obs_by_category.py
│   │   ├── stats_per_var.py
│   │   ├── stats_var_by_category.py
│   │   ├── imputation.py
│   │   ├── hypothesis_testing.py
│   │   ├── copf.py
│   │   └── peptide_sequence.py
│   ├── pp: directory containing modules for preprocessing steps of proteomics data in the anndata object
│   │   ├── filtering.py
│   │   ├── stats_obs_by_category.py
│   │   ├── normalization.py
│   │   ├── imputation.py
│   │   ├── peptide_grouping.py
│   │   └── quantification.py
│   ├── tl: directory containing modules which are tools for processing the proteomics data in the anndata object
│   │   ├── copf.py
│   │   └── hypothesis_testing.py
│   ├── read: directory containing modules which allow the automated reading and formatting of typical proteomics raw data to an anndata object
│   │   ├── long.py (generic reader from a long tabular format)
│   │   ├── diann.py
│   │   ├── maxquant.py
│   │   └── fragpipe.py
│   └── utils: directory containing miscellaneous modules which aid in proteomics data wrangling, transformation and analysis
│       ├── anndata.py
│       └── parsers.py
└── tests: tests mirroring copro directory, file and function structure

All public API functions are defined in the respective __init__.py files.

## Coding style
Target Python 3.10–3.11 compatibility, use 4-space indentation, and prefer f-strings for formatting. Follow `snake_case` for functions and variables, `CamelCase` for classes, and `UPPER_CASE` for constants. Keep modules cohesive and add docstrings summarizing inputs/outputs. Follow PEP8 guidelines with exception that the maximum line-length should be 88 characters as per BLACK. Autosave formatting with `flake8` and error-level `pylint` must pass before you push.


## Coding conventions
All pre-processing (pp), tool (tl) and annotation (ann) functions are performed inplace by default.

Internal modules are called in the following way:
```python
from copro.copf import pairwise_pepide_correlations  # public API
from copro.utils.anndata import sanitize_obs  # private API
from tests.utils.helpers import transform_dendogram_r2py  # test helper functions
```
Avoid prolixity:
 - type checking only in function arguments and function output, not when defining variables.

### Function guidelines

Use the funcion is_proteodata() found in copro/utils/anndata.py to check weather the supplied AnnData object conforms to the proteomics data assumptions for the copro package. The is_proteodata() is called at the beginning of the function and before returning the new anndata or modifying the supplied AnnData inplace.
As a reminder, the proteomics data assumptions are that:
 - if the data is a protein-level proteomics dataset, it must contain the .var column `protein_id` which contains the same values and in the same order as the .var index and .var_names.
 - if the data is a peptide-level proteomics dataset, it must contain the .var columns `peptide_id` and `protein_id`. The `peptide_id` column contains the same values and in the same order as the .var index and .var_names. The `protein_id` are the proteins that the peptides map to. It must be single mapping, so no peptide should map to more than one protein_id.

If the function uses the AnnData.X matrix, always check if it is a scipy.sparse matrix. As a general practice, the matrix will be made non-sparse for its algorithm but if possible sparse operations on the sparse matrix will be used to obtain the same result. If the function modifies the AnnData.X matrix inplace or returns an transformed AnnData matrix and the input was a sparse matrix, it is ensured, that the output AnnData.X matrix is also sparse.


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
    groups : list | None
        Restrict plot to particular groups.
        Default=None (do not include this line in docstrings)
    orderby : str | list | None
        Categorical .obs or .var column by which to order the observations or variables for plotting.
        Default=None (do not include this line in docstrings)
    order : str | list | None
        The order by which to present the observations, variables, or categories. If `orderby` is None and `order` is None, the existing .var_names or .obs_names order will be used. If `orderby` is None and `order` is not None, `order` contains the order by which observations or variables will be plotted. If `orderby` is not None and `order` is None, the unique values in the `sortby` column will be used to plot the the axis. If `sortby` is categorical, the category order will be used, if it is str or object the order of occurance will be used. If `sortby` is not None and `order` is not None, `order` defines the order by which the unique `sortby` column values are plotted in the relevant axis.
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
from copro.copf import pairwise_peptide_correlations   # public API
from copro.utils.anndata import sanitize_obs            # private helper
from tests.utils.helpers import transform_dendogram_r2py # test helper
```

---

### Data Validation & Matrix Sparsity (MANDATORY)

#### 1) Validate proteomics assumptions
Every public function that accepts an `AnnData` must call
`copro/utils/anndata.py:check_proteodata()` at the beginning and again before returning
(if a new `AnnData` is returned or the input is modified in-place).

Assumptions enforced by `check_proteodata()`:
- Protein-level datasets: `.var['protein_id']` must exist and match `.var_names`
  (same values in the same order).
- Peptide-level datasets: `.var['peptide_id']` and `.var['protein_id']` must exist.
  - `.var['peptide_id']` matches `.var_names` (same values and order).
  - `.var['protein_id']` contains the single-mapped protein for each peptide
    (no peptide maps to multiple proteins).

The helper function is_proteodata can also be useful to detect if the anndata is a
proteodata dataset and at which level as it returns (bool, str) where the string is
either 'peptide' or 'protein'. 

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
    from copro.utils.anndata import check_proteodata
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
CoPro assumes that `AnnData.X` contains only the following data types:
- `np.nan` (missing values)
- `int` (integer numeric values)
- `float` (floating-point numeric values)

**Infinite value validation**:
- The `is_proteodata()` and `check_proteodata()` functions automatically check for `np.inf` and `-np.inf`
- A `ValueError` is raised if infinite values are detected in `AnnData.X`
- Since all public functions must call `check_proteodata()` (see subsection 1 above), this validation is automatically enforced
- No additional infinite value checking is needed in individual functions

**Rationale**: Infinite values can cause unexpected behavior in statistical computations, correlations, and normalization procedures. Centralized validation in `check_proteodata()` ensures consistent error handling across all public functions and guides users to clean their data appropriately.

---


### Common Function Arguments

| Argument | Description |
|-----------|--------------|
| `adata` | AnnData object with `.X`, `.obs`, and `.var` annotations |
| `group_by` | Column in `adata.var` or `adata.obs` used for grouping |
| `layer` | Optional key in `adata.layers` specifying quantification data |
| `zero_to_na` | Convert zeros in `.X` to `np.nan` |
| `fill_na` | Replace missing values in `.X` with a specified constant |

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

- `groups: list | None`
  Restrict the plot to a subset of groups.

- `order_by: str | list | None`
  Categorical `.obs` or `.var` column(s) to order observations/variables for plotting.

- `order: str | list | None`
  Controls the explicit order of observations/variables or categories.
  - If `order_by is None` and `order is None`: use existing `.var_names` / `.obs_names` order.
  - If `order_by is None` and `order is not None`: use `order` as the explicit order.
  - If `order_by is not None` and `order is None`: order by the unique values in `order_by`.
    If `order_by` is categorical, use its category order; if object/string, use first-occurrence order.
  - If `order_by is not None` and `order is not None`: `order` defines the plotting order of the
    unique values from `order_by`.

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
copro/
├── .git
├── .github/
│   └── workflows/
│       └── format-code_perform-tests_on_push-pr.yaml
├── .gitignore
├── AGENTS.md: Current file which includes instructions for the architecture, code format, development and usage of this repository.
├── pyproject.toml
└── README.md

### Documentation using Sphinx


## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate` – create a local virtual environment.
- `python -m pip install -e .` followed by `pip install -r requirements/requirements_development.txt` – install the package plus dev tooling.
- `flake8 .` – run the same lint check as CI (127-char limit, complexity guard).
- `pylint $(git ls-files "*.py") --disable=all --enable=E,F --disable=E0401` – surface import and error-level issues.
- `pytest -v tests/` – execute the full regression suite.


## Testing Guidelines
Add tests alongside new modules under matching `tests/<submodule>/test_*.py` files. Reuse shared fixtures from `tests/utils/helpers.py`, and place any extra data under `tests/data/<feature>/` with clear filenames. Failing tests that rely on stochastic operations should fix random seeds. Run `pytest -v tests/` locally and ensure CI remains green; aim to cover all new branches and numerical edge cases.

## Commit & Pull Request Guidelines
Write concise, capitalized commit subjects that mirror the existing style (e.g., `Added function: pl.intensity_distribution_per_obs()`). Group related changes and avoid mixed concerns. Pull requests should describe motivation, enumerate key changes, and call out new data files or notebooks. Link to relevant issues, paste command outputs for lint/tests, and attach before/after visuals when altering plots or reports.
