# `tests/data/mouse_tissue/` — CCprofiler reference intermediates

Ground truth for the COPF tests (`tests/tl/test_copro.py`,
`tests/tl/test_proximity_analysis.py`). Every file here is output of the
*original* R workflow, so the tests compare ProteoPy against the
implementation that produced the published numbers rather than against
ProteoPy's own earlier behaviour.

## Origin

| | |
| --- | --- |
| Dataset | Mouse tissue SEC-SWATH-MS, 32,690 peptides × 40 fractions |
| Publication | Bludau, I. *et al.* Systematic detection of functional proteoform groups from bottom-up proteomic datasets. *Nat Commun* **12**, 3810 (2021). [doi:10.1038/s41467-021-24030-x](https://doi.org/10.1038/s41467-021-24030-x) |
| Raw data | PRIDE `PXD005044` |
| Reference implementation | [`CCprofiler/CCprofiler`](https://github.com/CCprofiler/CCprofiler) at `31a3043`, branch `proteoformLocationMapping` — the ref the paper's analysis script installs. The release tag `v1.0.1-copf` does **not** contain most of the functions the script calls. |
| Analysis script | `ProteoformAnanlysis/MouseTissue/GetMouseTissueProteoforms_paper.R` |
| Licence | Data CC-BY 4.0 (paper and PRIDE deposition); CCprofiler is Apache-2.0 |

After the reference pipeline's filters, 24,534 peptides over 2,885
proteins remain, of which 63 are called significant at
`proteoform_score >= 0.1` and `proteoform_score_pval_adj <= 0.1`.

## Files

| File | Rows | Content |
| --- | --- | --- |
| `traces_pre-processed_rcopf.tsv` | 24,534 | Peptide intensities, wide: fraction columns `1`–`40` plus `id` |
| `traces_pre-processed_trace-annotations_rcopf.tsv` | 24,534 | Peptide annotation incl. `PeptidePositionStart` / `PeptidePositionEnd` |
| `fraction_annotation.tsv` | 40 | `filename` ↔ fraction `id` |
| `traces_correlations_rcopf.tsv` | — | Per-protein peptide correlations, long form |
| `traces_cluster-dendograms_rcopf.json` | — | `hclust` dendrograms, one per protein |
| `traces_annotation_cluster-assignment_rcopf.tsv` | 24,534 | As above, plus `cluster` ∈ {1, 2, 100}; `100` is CCprofiler's outlier marker |
| `trace_annotation_proteoform-scores_rcopf.tsv` | 24,534 | `proteoform_score`, `_z`, `_dz`, `_pval`, `_pval_adj` |
| `traces_proteoform-assignment_rcopf.tsv` | 24,534 | `proteoform_id` per peptide, with peptide positions |
| `traces_proteoform-location_rcopf.tsv` | 5,057 | `genomLocation_pval` and `genomLocation_pval_lim` per proteoform group; 3,444 populated, 1,613 `NA` |
| `uniprot_mouse-copf-proteins_subset.fasta` | 2,885 | Protein sequences, for resolving peptide positions. **A subset** of the UniProt file the analysis script reads — see the note below |

### Notes that matter for using them

- **`trace_annotation_proteoform-scores_rcopf.tsv` is post-coercion.**
  The analysis script maps a missing `proteoform_score` to `0` and a
  missing `proteoform_score_pval_adj` to `1`. The 1,613 rows at exactly
  `(0, 1)` are *not* scores; excluding them is what makes BH's `n`
  1,272 rather than 2,885.
- **`cluster == 100` is not `1e6`.** ProteoPy marks noise `1e6` and
  numbers real clusters from `0`; CCprofiler numbers them from `1` and
  marks noise `100`. The tests translate explicitly rather than relying
  on either convention.
- **`traces_proteoform-location_rcopf.tsv` came out of an `.rds`.** The
  proximity p-values are not in any TSV the reference workflow writes;
  they exist only inside the final `traces` object, and were exported
  from it with `readRDS` (columns copied verbatim, nothing recomputed).
  The 1,613 `NA` rows are proteins the reference did not test.
- **The FASTA is a subset, and a verified one.** It holds the 2,885
  proteins of the reference peptide set, extracted from
  `uniprot-filtered-organism__Mus+musculus+(Mouse)+[10090]_+AND+revie--.fasta`
  (UniProt, reviewed *Mus musculus*, 17,033 records, the file the
  analysis script reads) with headers and line wrapping unchanged and
  first occurrences kept. Subsetting cannot change a result, because a
  peptide is only ever located in its own protein's sequence — and this
  is checked rather than assumed:
  `test_peptide_positions_from_fasta_vs_rcopf` asserts that positions
  derived from this file equal `PeptidePositionStart` for all 24,534
  peptides, the one unlocatable peptide included.

## Regenerating

The three files added for `test_proximity_analysis.py` --
`traces_proteoform-location_rcopf.tsv`,
`traces_proteoform-assignment_rcopf.tsv` and
`uniprot_mouse-copf-proteins_subset.fasta` -- were produced by
`experiments/exp_06_sequence-proximity-analysis/code/06_build_proteopy_test_assets.py`
in the `fichtner-2026_code` analysis repository, which holds the executed
reference run and the extraction scripts. They are committed here so the test
suite is self-contained.

⚠️ **These files are reference output: do not let a formatter touch them.** The
location table encodes an untested proteoform group as a row with two empty
trailing fields (`ACC<TAB>ACC<TAB><TAB>`), so stripping trailing whitespace
silently turns a 4-field row into a 2-field one. `.pre-commit-config.yaml`
therefore excludes `tests/data/` from `trailing-whitespace`,
`end-of-file-fixer` and `mixed-line-ending`, alongside the exclusions `black`
and `pyupgrade` already carried. The checksums below are the guard: verify them
after any operation that rewrites the working tree.

```
ebe60584dc44563d9d5fd52dd7d0b3ea  uniprot_mouse-copf-proteins_subset.fasta
2f48e0e1b1dc3f11a919d51c26e6c337  traces_proteoform-location_rcopf.tsv
365eca4afe85335a1d04e315ecfba3df  traces_proteoform-assignment_rcopf.tsv
```
