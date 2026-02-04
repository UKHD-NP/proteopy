Contributing
============

We welcome contributions to ProteoPy! This guide will help you get started.

Development Setup
-----------------

1. Fork and clone the repository:

   .. code-block:: bash

      git clone https://github.com/YOUR_USERNAME/proteopy.git
      cd proteopy

2. Create a virtual environment and install development dependencies:

   .. code-block:: bash

      python -m venv .venv
      source .venv/bin/activate
      python -m pip install -e .
      pip install -r requirements/requirements_development.txt

3. Run quality checks before submitting:

   .. code-block:: bash

      flake8 .
      pylint $(git ls-files "*.py") --disable=all --enable=E,F --disable=E0401
      pytest -v tests/

Code Style
----------

- **Python version**: 3.10–3.11
- **Indentation**: 4 spaces
- **Line length**: 79 characters (88 for long strings)
- **Naming**: ``snake_case`` for functions/variables, ``CamelCase`` for classes
- **Formatting**: Use ``black`` for auto-formatting, ``flake8`` for linting

Documentation
-------------

- Use NumPy-style docstrings
- Include type hints in function signatures
- Add docstring examples using ``>>> import proteopy as pr``

Testing
-------

- Add tests in ``tests/<subpackage>/test_*.py``
- Reuse fixtures from ``tests/utils/helpers.py``
- Place test datasets in ``tests/data/<feature>/``
- Fix random seeds for stochastic operations

Commit Guidelines
-----------------

- Write concise, capitalized commit subjects
- Use prefixes: ``Feature:``, ``Fix:``, ``Docs:``, ``Refactor:``, ``Test:``
- Example: ``Feature: pl.intensity_distribution_per_obs()``

Pull Requests
-------------

Please include in your PR description:

- Purpose and motivation
- Key changes
- Test/lint outputs
- Before/after visuals (if applicable)
- Related issues
