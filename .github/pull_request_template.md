<!--
How to use this template:
1. Fill in "Summary" and "Related issues".
2. Keep only the sections below that apply to this PR and delete the rest.
   A PR can need more than one (e.g. Feature + Tests).
3. Tick the boxes you have done. Mark the rest N/A or explain why not.
4. HTML comments like this one are hidden in the rendered PR, so you
   can leave them in.

Conventions (function design, validation, plotting rules, style) are
in AGENTS.md. Read it before opening a code PR.
-->

## Summary

<!--
What changed and why, in 1–3 sentences. Say why, not only what.
-->

## Related issues

<!--
Use "Closes #123" / "Fixes #123" to close an issue on merge, or
"Related to #123" if it should stay open. Write "None" if there is none.
-->

## General checklist

<!-- Applies to every PR. -->

- [ ] Branch name and commit subjects follow the repo conventions
- [ ] `HISTORY.md` updated under `[Unreleased]` (or N/A: no user-facing change)
- [ ] CI is green (flake8, pylint, pytest)

---

<!-- ============================================================ -->
<!-- Keep only the sections that apply. Delete the rest.          -->
<!-- ============================================================ -->

## ✨ Feature

<!--
New public functionality, e.g. a new function in pr.pp / pr.tl / pr.pl /
pr.read / pr.datasets, or a new parameter on an existing function.
-->

**What it adds:**

<!-- e.g. `pr.tl.peptide_proximity()`: tests whether ... -->

**Example usage:**

```python
import proteopy as pr

```

<!-- For pl functions, attach a screenshot of the output. -->

- [ ] Follows the AGENTS.md function and module conventions
- [ ] Tests added for the new behavior
- [ ] Numpydoc docstring with an example
- [ ] Added to the Sphinx API docs (`docs/sphinx/source/`)
- [ ] Relevant tutorial notebooks updated, or none affected
- [ ] Algorithm reimplemented from published work (e.g. CCprofiler, COPF): source cited in the docstring and results checked against the reference

## 🐛 Bug fix

**What was wrong:**

<!-- Observed vs. expected behavior. Include a minimal reproducer if possible. -->

**Root cause:**

<!-- Why it happened. -->

**Fix:**

<!-- How this PR resolves it. Note any user-visible change in results. -->

- [ ] Regression test added that fails without the fix
- [ ] Other functions with the same pattern checked
- [ ] Relevant tutorial notebooks re-run if results change

## 🧪 Tests

<!--
Changes to the test suite only (new coverage, refactoring tests,
fixtures, test data). If tests come with a feature or fix, put them in
that section and delete this one.
-->

**What is covered now that wasn't before:**

- [ ] New test data is lightweight, under `tests/data/<feature>/`, and its provenance is noted
- [ ] Stochastic tests use fixed seeds

## 📚 Documentation

<!--
Docstrings, Sphinx pages, README, AGENTS.md, or tutorial notebooks.
-->

**What changed:**

- [ ] Sphinx build runs without new warnings
- [ ] Notebooks re-run top to bottom; outputs are current
- [ ] Examples use `import proteopy as pr`

## ♻️ Refactor / style

<!--
No change in behavior: restructuring, renaming internals, applying
black, bringing legacy code up to the current conventions.
-->

**What was restructured and why:**

- [ ] No change to public API or numerical results
- [ ] Existing tests pass unchanged (or the test changes are explained)

## ⚙️ CI/CD

<!--
GitHub Actions workflows, pre-commit hooks, lint config, packaging and
release automation.
-->

**What changed:**

**How it was verified:**

<!-- e.g. a link to a workflow run, or a test commit showing the failure/pass. -->

- [ ] Workflow ran successfully on this branch
- [ ] Still runs on all matrix OSes and Python versions

## ⚠️ Breaking changes / dependencies

<!--
Add this alongside any other section when the PR:
- removes, renames, or changes the signature or defaults of a public function
- changes numerical results of an existing function
- adds, removes, or changes dependency bounds in pyproject.toml / requirements
-->

**What breaks / what changed:**

**Migration for users:**

```python
# before

# after
```

- [ ] Noted under `Changed` / `Removed` / `Deprecated` in `HISTORY.md`
- [ ] Deprecation warning added where a soft transition is possible
- [ ] Tutorial notebooks updated to the new API
- [ ] New or changed dependencies are justified and still support Python 3.10–3.11

---

## Notes for reviewers

<!--
Optional: where to focus, open questions, follow-up work, anything
unusual. Delete if not needed.
-->
