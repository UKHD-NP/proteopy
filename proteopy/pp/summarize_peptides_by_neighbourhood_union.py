"""Peptide summarisation by positional neighbourhood union.

Reimplementation of CCprofiler's
``summarizeAlternativePeptideSequences(topN = 1)`` together with the
``proteinQuantification(topN, keep_less)`` selection it delegates to.

Reference source
    https://github.com/CCprofiler/CCprofiler/tree/proteoformLocationMapping

    Branch ``proteoformLocationMapping``, git ref ``31a3043``, files
    ``R/summarizeRedundantPeptides.R`` and ``R/proteinQuantification.R``.
    Note this is a branch, not a release tag: the functions below are
    absent from ``v1.0.1-copf``.

Algorithm
---------
Peptides are handled one protein at a time. Let ``P`` be the peptides
of a protein. Each ``x`` in ``P`` carries a closed, 1-based interval
``[s(x), e(x)]`` locating where its modification-stripped sequence
first occurs in the protein sequence, or ``UNDEF`` when the sequence
cannot be located there at all.

**Membership.** The neighbourhood of ``x`` is every peptide whose start
or end falls inside ``x``'s interval::

    N(x) = { q in P : s(q) in [s(x), e(x)]  or  e(q) in [s(x), e(x)] }

Every comparison involving ``UNDEF`` is false, so an unlocated peptide
belongs to no neighbourhood and its own neighbourhood is empty.

``N`` is deliberately **not symmetric**. If ``q`` lies strictly inside
``x`` then ``q`` is in ``N(x)`` but ``x`` is not in ``N(q)``, because
the test only asks where the *other* peptide's endpoints land.

**Label.** The label of ``x`` is the union of every neighbourhood that
contains it — a one-hop union, *not* a transitive closure::

    L(x) = union of { N(y) : y in P and x in N(y) }

**Group.** Peptides carrying the same label form a group::

    G(x) = { q in P : L(q) = L(x) }

**Selection.** With ``T(x)`` the total intensity of ``x`` summed over
all samples, propagating missing values, the members of a group are
ordered by::

    key(x) = ( T(x) is missing,  -T(x),  tie_break_key(id(x)) )

Complete peptides come first, then descending abundance, then a
deterministic tie-break on the identifier. The leading ``top_n``
members are kept and the rest discarded, so exactly one row survives
per group.

Worked example
--------------
Protein ``ACDEFGHIKLMNPQRSTVWY`` with five peptides: a chain of four in
which only adjacent pairs overlap, plus ``CDE`` lying strictly inside
``ACDEF``::

    ACDEF [1, 5]   EFGHI [4, 8]   HIKLM [7, 11]   LMNPQ [10, 14]
    CDE   [2, 4]

    N(ACDEF) = {ACDEF, CDE, EFGHI}
    N(CDE)   = {CDE, EFGHI}                 <- ACDEF absent: N is asymmetric
    N(EFGHI) = {ACDEF, CDE, EFGHI, HIKLM}
    N(HIKLM) = {EFGHI, HIKLM, LMNPQ}
    N(LMNPQ) = {HIKLM, LMNPQ}

    L(ACDEF) = L(CDE)   = {ACDEF, CDE, EFGHI, HIKLM}
    L(EFGHI) = L(HIKLM) = {ACDEF, CDE, EFGHI, HIKLM, LMNPQ}
    L(LMNPQ)            = {EFGHI, HIKLM, LMNPQ}

    G1 = {ACDEF, CDE}    G2 = {EFGHI, HIKLM}    G3 = {LMNPQ}

Three groups, so three peptides survive. Two properties are visible
here and both are load-bearing:

* ``CDE`` groups with the container it sits inside, even though the
  membership relation between them runs one way only.
* A transitive closure over the same overlaps would merge all five into
  a single group and keep ONE peptide. The chain's two interior members
  are what separate the rules: each sits in three neighbourhoods, so
  each absorbs the whole set, while the two ends do not.

Missing values
--------------
A peptide with any missing sample has ``T(x)`` missing and sorts LAST,
so any complete competitor beats it however small its observed values
are. This is deprioritisation, not removal:

* With a complete member present, that member wins and the incomplete
  one is discarded like any other loser.
* When **every** member is missing, all totals are missing, nothing
  separates them on abundance, and ``tie_break_key`` alone decides. The
  winner passes through with its missing values intact, for a later
  completeness filter to act on.
* At ``top_n = 1`` no arithmetic happens — the surviving row is copied
  verbatim — so a missing value can be neither created nor spread.
* At ``top_n > 1`` the selected rows are summed and missingness
  **propagates**: a sample missing in any contributor is missing in the
  result, and a group whose members are all missing sums to missing.

Relationship to summarize_overlapping_peptides
----------------------------------------------
A different grouping and a different reduction.
:func:`~proteopy.pp.summarize_overlapping_peptides` groups by substring
containment and *aggregates* the members; this function groups by
position in the protein sequence and *selects* among them. Positional
overlap also sees pairs that containment cannot — ``ACDEF`` and
``EFGHI`` above overlap in the protein while neither contains the
other — and it needs no separate modification-summarisation step,
because two peptidoforms of one stripped sequence share an interval and
therefore a group.
"""

import re
from pathlib import Path
from typing import Any
from collections.abc import Callable, Iterable

import anndata as ad
import numpy as np
import pandas as pd

from proteopy.pp.quantification import (
    _aggregate_var_value,
    _rebuild_adata,
)
from proteopy.utils.anndata import check_proteodata, is_proteodata

# IUPAC one-letter codes. Broader than the canonical twenty on purpose:
# UniProt sequences legitimately contain U (selenocysteine) and the
# ambiguity codes B, J, O, X and Z, and rejecting them would turn a
# real selenopeptide into a false error. Every modification notation
# character -- digits, brackets, parentheses, +, -, lowercase markers --
# is still outside this set.
IUPAC_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWYBJOUXZ"

# What CCprofiler hard-codes in getPepStartSite and in the
# PeptidePositionEnd calculation.
CCPROFILER_MOD_REGEX = r"\(UniMod:[0-9]+\)"

# names(fasta) = gsub(".*\\|(.*?)\\|.*", "\\1", names(fasta)) -- pulls the
# accession out of a `sp|ACC|NAME description` header. Replicated rather
# than improved: a header whose description contains a pipe is handled
# the same (mis)way as the reference.
_FASTA_ACCESSION_RE = re.compile(r".*\|(.*?)\|.*")

_UNRESOLVED_MODES = ("raise", "skip", "keep")
_ID_FROM_MODES = ("top_ranked",)

_START_COL = "peptide_start"
_END_COL = "peptide_end"
_N_COL = "n_grouped"


def letters_first_key(peptide_id: str) -> tuple:
    """Ordering key placing letters before any non-letter character.

    Under plain codepoint order ``(`` sorts before ``A`` while ``[``
    sorts after ``Z``, so the two common modification notations would
    break ties in opposite directions. This key makes both land after
    the letters, so an unmodified identifier wins a tie against any
    annotated form of itself.

    Parameters
    ----------
    peptide_id : str
        Peptide identifier.

    Returns
    -------
    tuple
        Per-character sort key.
    """
    return tuple(
        (0, char) if char.isalpha() else (1, char) for char in peptide_id
    )


def _validate_arguments(
    top_n,
    keep_less,
    id_from,
    on_unknown_protein,
    on_unlocated_peptide,
    tie_break_key,
    zero_to_na,
    fill_na,
):
    """Check every argument before any work is done."""
    if not isinstance(top_n, (int, np.integer)) or top_n < 1:
        raise ValueError(f"top_n must be an integer >= 1, got {top_n!r}.")
    if not isinstance(keep_less, bool):
        raise TypeError(
            f"keep_less must be bool, got {type(keep_less).__name__}."
        )
    if id_from not in _ID_FROM_MODES:
        raise ValueError(
            f"id_from must be one of {_ID_FROM_MODES!r}, got "
            f"{id_from!r}. Only 'top_ranked' is implemented; other "
            "naming schemes may be added later."
        )
    for name, value in (
        ("on_unknown_protein", on_unknown_protein),
        ("on_unlocated_peptide", on_unlocated_peptide),
    ):
        if value not in _UNRESOLVED_MODES:
            raise ValueError(
                f"{name} must be one of {_UNRESOLVED_MODES!r}, "
                f"got {value!r}."
            )
    if not callable(tie_break_key):
        raise TypeError("tie_break_key must be callable.")
    if not isinstance(zero_to_na, bool):
        raise TypeError(
            f"zero_to_na must be bool, " f"got {type(zero_to_na).__name__}."
        )
    if fill_na is not None and (
        isinstance(fill_na, bool) or not isinstance(fill_na, (int, float))
    ):
        raise TypeError(
            f"fill_na must be float, int, or None, "
            f"got {type(fill_na).__name__}."
        )
    if zero_to_na and fill_na is not None:
        raise ValueError("`zero_to_na` and `fill_na` are mutually exclusive.")


def _validate_var(
    adata, peptide_col, protein_col, written_cols, keep_var_cols
):
    """Check the required columns exist and the written ones do not."""
    for col in (peptide_col, protein_col):
        if col not in adata.var.columns:
            raise KeyError(f"'{col}' not found in adata.var")

    # Reserved before existent: a reserved name is a contract error
    # whether or not the caller happens to have such a column.
    reserved = [
        c
        for c in (keep_var_cols or ())
        if c in (peptide_col, protein_col, *written_cols)
    ]
    if reserved:
        raise ValueError(
            "keep_var_cols must not name a column this function "
            f"already writes: {', '.join(repr(c) for c in reserved)}."
        )
    for col in keep_var_cols or ():
        if col not in adata.var.columns:
            raise KeyError(f"'{col}' not found in adata.var")

    # Nothing to classify when there are no peptides; `is_proteodata`
    # reports such an object as level-less, which would be a confusing
    # error for what is a legitimate (if degenerate) input.
    _, level = is_proteodata(adata) if adata.n_vars else (True, "peptide")
    if level != "peptide":
        raise ValueError(
            "summarize_peptides_by_neighbourhood_union requires "
            "peptide-level proteodata: .var must hold 'peptide_id' "
            "matching .var_names and 'protein_id', with each peptide "
            f"mapping to exactly one protein. Detected level: {level!r}."
        )

    clashes = [c for c in written_cols if c in adata.var.columns]
    if clashes:
        raise ValueError(
            "adata.var already contains column(s) that this function "
            f"writes: {', '.join(repr(c) for c in clashes)}. Rename or "
            "drop them first, or pass a different `key_added`. This is "
            "refused rather than overwritten so a second call cannot "
            "silently discard an earlier annotation."
        )


def _read_fasta(path: str | Path) -> dict[str, str]:
    """Read a FASTA into ``{accession: sequence}``.

    Header handling replicates CCprofiler's: the accession is extracted
    with ``.*\\|(.*?)\\|.*`` and a header that does not match is used
    verbatim. Duplicate accessions keep the FIRST occurrence, matching
    Biostrings' name-based subsetting.
    """
    sequences: dict[str, str] = {}
    header, chunks = None, []

    def flush():
        if header is not None and header not in sequences:
            sequences[header] = "".join(chunks)

    with open(path) as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                header = _FASTA_ACCESSION_RE.sub(r"\1", line[1:])
                chunks = []
            else:
                chunks.append(line)
    flush()
    return sequences


def _resolve_annotator(annotator) -> dict[str, str]:
    """Accept a FASTA path or an already-parsed sequence mapping."""
    if isinstance(annotator, (str, Path)):
        return _read_fasta(annotator)
    if isinstance(annotator, dict):
        return {str(k): str(v) for k, v in annotator.items()}
    raise TypeError(
        "annotator must be a path to a FASTA file or a "
        "{accession: sequence} mapping, got "
        f"{type(annotator).__name__}."
    )


def _strip_and_validate(peptides, pattern, allowed):
    """Strip annotations and check what remains is amino acids only.

    This is what makes ``mod_regex`` self-checking: the caller declares
    what to disregard and the alphabet verifies the declaration was
    complete, so no notation-specific pattern has to be hard-coded to
    recognise mass shifts, bracket tags or lowercase markers.
    """
    stripped = [pattern.sub("", pid) for pid in peptides]

    offenders = []
    for pid, seq in zip(peptides, stripped):
        residual = sorted({c for c in seq if c not in allowed})
        if residual:
            offenders.append((pid, residual))

    if offenders:
        shown = "\n".join(
            f"  {pid}  ->  residual characters: "
            + ", ".join(repr(c) for c in residual)
            for pid, residual in offenders[:20]
        )
        more = (
            f"\n  ... and {len(offenders) - 20} more"
            if len(offenders) > 20
            else ""
        )
        raise ValueError(
            f"{len(offenders)} peptide identifier(s) still contain "
            "characters that are not amino acids after `mod_regex` "
            f"was applied:\n{shown}{more}\n"
            "Widen `mod_regex` so it matches every annotation to "
            "disregard, or pass a different `alphabet`."
        )
    return stripped


def _locate(
    peptides,
    stripped,
    proteins,
    sequences,
    on_unknown_protein,
    on_unlocated_peptide,
):
    """Resolve 1-based inclusive positions, applying both policies.

    Returns ``(starts, ends, keep)`` where ``keep`` is a boolean mask of
    the peptides that remain in the analysis. Unresolvable positions are
    ``NaN``, which is what makes them compare False against every
    interval and collapse into the empty label.
    """
    n = len(peptides)
    starts = np.full(n, np.nan)
    keep = np.ones(n, dtype=bool)

    # --- proteins absent from the annotator
    missing = sorted({p for p in proteins if p not in sequences})
    if missing:
        if on_unknown_protein == "raise":
            shown = ", ".join(missing[:20])
            more = (
                f" ... and {len(missing) - 20} more"
                if len(missing) > 20
                else ""
            )
            raise ValueError(
                f"{len(missing)} protein(s) in adata.var are absent "
                f"from `annotator`: {shown}{more}. Set "
                "on_unknown_protein='skip' to discard their peptides, "
                "or on_unknown_protein='keep' to give them NaN "
                "positions, which is the CCprofiler behaviour."
            )
        absent = np.array([p in missing for p in proteins], dtype=bool)
        if on_unknown_protein == "skip":
            keep &= ~absent

    # --- locate each (protein, stripped peptide) pair once, as
    #     CCprofiler does via `by = c("id", "protein_id")`
    cache: dict[tuple, float] = {}
    for i in range(n):
        if not keep[i]:
            continue
        key = (proteins[i], stripped[i])
        if key not in cache:
            seq = sequences.get(proteins[i])
            if seq is None:
                cache[key] = np.nan
            else:
                found = seq.find(stripped[i])
                cache[key] = np.nan if found < 0 else float(found + 1)
        starts[i] = cache[key]

    # --- peptides whose protein IS present but whose sequence is not
    present = np.array([p in sequences for p in proteins], dtype=bool)
    unlocated = keep & present & np.isnan(starts)
    if unlocated.any():
        if on_unlocated_peptide == "raise":
            idx = np.flatnonzero(unlocated)
            shown = "\n".join(
                f"  {peptides[i]} (protein {proteins[i]})" for i in idx[:20]
            )
            more = f"\n  ... and {len(idx) - 20} more" if len(idx) > 20 else ""
            raise ValueError(
                f"{int(unlocated.sum())} peptide(s) were not found in "
                f"their protein sequence:\n{shown}{more}\n"
                "Set on_unlocated_peptide='skip' to discard them, or "
                "on_unlocated_peptide='keep' to give them NaN "
                "positions, which is the CCprofiler behaviour."
            )
        if on_unlocated_peptide == "skip":
            keep &= ~unlocated

    lengths = np.array([len(s) for s in stripped], dtype=float)
    ends = starts + lengths - 1.0
    return starts, ends, keep


def _neighbourhood_union_labels(starts, ends, ids):
    """CCprofiler's positional group label, for one protein.

    Replicates, verbatim in behaviour::

        coPeps <- subset(p_ann,
            ((p_ann[[start]] >= pep_ann[[start]]) &
             (p_ann[[start]] <= pep_ann[[end]])) |
            (p_ann[[end]]   >= pep_ann[[start]]) &
            (p_ann[[end]]   <= pep_ann[[end]]))$id
        new_id <- paste0(sort(unique(unlist(
            lapply(pep_seq, function(x) if (pep %in% x) x)))),
            collapse = ";")

    Two properties are load-bearing and must not be "fixed":

    * **The overlap test is asymmetric.** ``coPeps(x)`` asks only
      whether the *other* peptide's start or end falls inside ``x``'s
      interval, so a strictly-enclosed neighbour is visible from one
      side only. (This turns out to be unobservable in the output: when
      ``q`` is strictly inside ``x`` then ``coPeps(q)`` is a subset of
      ``coPeps(x)``, so the missing edge adds nothing to any union.)
    * **The label is a one-hop union, not a transitive closure.** An
      overlap chain ``A-B-C-D`` with only adjacent pairs overlapping
      yields THREE groups -- ``{A}``, ``{B, C}``, ``{D}`` -- because the
      two interior peptides each sit in three neighbourhoods and so take
      the full union. Transitive closure would give one group and select
      one peptide where the reference selects three.

    Peptides whose positions are ``NaN`` compare False against
    everything, so they receive the empty label and collapse into a
    single group per protein. That is the reference's behaviour, and it
    is how a protein absent from the FASTA is stripped to one peptide.
    """
    n = len(ids)
    if n == 0:
        return []

    s = np.asarray(starts, dtype=float)
    e = np.asarray(ends, dtype=float)

    # member[j, k] is True  <=>  ids[k] in coPeps(ids[j]).
    # Comparisons against NaN are False, matching what R's subset does
    # with an NA condition.
    with np.errstate(invalid="ignore"):
        member = ((s[None, :] >= s[:, None]) & (s[None, :] <= e[:, None])) | (
            (e[None, :] >= s[:, None]) & (e[None, :] <= e[:, None])
        )

    # union[i, k] is True <=> some j has both i and k in coPeps(j),
    # which is the boolean matrix product member.T @ member. float32 is
    # exact here: the counts are bounded by n, far below 2**24.
    counts = member.astype(np.float32)
    union = (counts.T @ counts) > 0

    return [
        ";".join(sorted(ids[k] for k in np.flatnonzero(union[i])))
        for i in range(n)
    ]


def _group_keys(starts, ends, ids, proteins):
    """Assign every peptide a ``(protein, label)`` group key.

    Grouping is confined to a protein, exactly as
    ``protein_id := paste0(protein_id, "_", new_id)`` does.
    """
    keys = np.empty(len(ids), dtype=object)
    order = pd.Series(range(len(proteins)))
    for _, idx in order.groupby(proteins, sort=False):
        rows = idx.to_numpy()
        labels = _neighbourhood_union_labels(
            starts[rows], ends[rows], list(ids[rows])
        )
        for row, label in zip(rows, labels):
            keys[row] = (proteins[row], label)
    return keys


def _select(keys, totals, ids, top_n, keep_less, tie_break_key):
    """Rank each group and pick its representative and contributors.

    The rank reproduces ``peptide_intensity := sum(intensity)`` under
    ``na.rm = FALSE`` followed by ``rank(-peptide_intensity)`` under
    ``na.last = TRUE``: a peptide with any missing sample has a NaN
    total and ranks LAST. Equal totals are then resolved by
    ``tie_break_key`` rather than by row order, so the result does not
    depend on how the input table happened to be sorted.
    """
    is_nan = np.isnan(totals)
    tie_keys = [tie_break_key(pid) for pid in ids]

    def rank_key(i):
        if is_nan[i]:
            return (1, 0.0, tie_keys[i])
        return (0, -float(totals[i]), tie_keys[i])

    members: dict[Any, list] = {}
    for i, key in enumerate(keys):
        members.setdefault(key, []).append(i)

    selections = []
    for key, rows in members.items():
        if len(rows) < top_n and not keep_less:
            continue
        ranked = sorted(rows, key=rank_key)
        selections.append(
            {
                "representative": ranked[0],
                "contributors": ranked[:top_n],
                "members": rows,
            }
        )
    return selections


def summarize_peptides_by_neighbourhood_union(
    adata: ad.AnnData,
    annotator: str | Path | dict[str, str],
    *,
    protein_col: str = "protein_id",
    peptide_col: str = "peptide_id",
    top_n: int = 1,
    keep_less: bool = False,
    id_from: str = "top_ranked",
    mod_regex: str = CCPROFILER_MOD_REGEX,
    alphabet: Iterable[str] = IUPAC_AMINO_ACIDS,
    on_unknown_protein: str = "raise",
    on_unlocated_peptide: str = "raise",
    tie_break_key: Callable[[str], Any] = letters_first_key,
    zero_to_na: bool = False,
    fill_na: float | int | None = None,
    sort_descending_id: bool = True,
    key_added: str = "peptide_ids",
    keep_var_cols: list[str] | None = None,
    inplace: bool = True,
    verbose: bool = False,
) -> ad.AnnData | None:
    """
    Collapse peptides that overlap in the protein sequence.

    Reimplements CCprofiler's
    ``summarizeAlternativePeptideSequences(topN = 1)``. Peptide
    positions are resolved from ``annotator``, peptides are grouped by
    the union of their positional neighbourhoods, and the most abundant
    member of each group is selected while the rest are discarded.

    Unlike :func:`~proteopy.pp.summarize_overlapping_peptides`, which
    groups by substring containment and aggregates, this function
    selects. It also needs no separate modification-summarisation step:
    two peptidoforms of one stripped sequence share an interval, so they
    group and one is selected.

    Parameters
    ----------
    adata : AnnData
        Peptide-level data. Only ``.X`` is read and written.
    annotator : str | Path | dict
        Path to a FASTA file, or a pre-parsed
        ``{accession: sequence}`` mapping, supplying the protein
        sequences that peptide positions are resolved against.
    protein_col, peptide_col : str, optional
        Columns in ``.var`` holding the protein and peptide
        identifiers.
    top_n : int, optional
        How many of each group's most abundant members contribute to
        the output value. ``1`` selects a single peptide and copies its
        intensities; above ``1`` the selected members are summed.
    keep_less : bool, optional
        If False, discard groups with fewer than ``top_n`` members. Has
        no effect at ``top_n=1``, since every group has a member.
    id_from : {'top_ranked'}, optional
        How the surviving row is identified. Only ``'top_ranked'`` is
        implemented: the row takes the identifier of the group's most
        abundant member. This deviates from CCprofiler, which renames
        the row to a comma-joined list of the summed identifiers and
        thereby breaks its own annotation join.
    mod_regex : str, optional
        Everything in an identifier that is not protein sequence. The
        pattern must cover *every* annotation present; whatever it fails
        to match is searched for verbatim, and the ``alphabet`` check
        turns an incomplete pattern into an error rather than a silently
        unlocatable peptide.
    alphabet : iterable of str, optional
        Characters permitted in a stripped identifier. Defaults to the
        IUPAC one-letter codes, which include selenocysteine and the
        ambiguity codes.
    on_unknown_protein : {'raise', 'skip', 'keep'}, optional
        What to do with a protein absent from ``annotator``.
        ``'skip'`` discards its peptides; ``'keep'`` gives them NaN
        positions, so they share the empty label, collapse into one
        group, and the single survivor is removed downstream by a
        peptide-count filter. ``'keep'`` is the CCprofiler behaviour.
    on_unlocated_peptide : {'raise', 'skip', 'keep'}, optional
        What to do with a peptide whose sequence does not occur in its
        protein. ``'keep'`` is the CCprofiler behaviour, which is
        silent about this case; ``'raise'`` is the default because that
        silence is the reference's real blind spot.
    tie_break_key : callable, optional
        Key applied to the peptide identifier to resolve equal totals.
        Defaults to :func:`letters_first_key`, which places ``(`` and
        ``[`` after the letters so an unmodified identifier wins.
    zero_to_na : bool, optional
        If True, treat zeros as missing before ranking.
    fill_na : float | int | None, optional
        Replace missing values with this constant before ranking.
        Mutually exclusive with ``zero_to_na``. Note that ``0`` is not
        faithful: it gives an incomplete peptide a real total and can
        win it a ranking it should have lost.
    sort_descending_id : bool, optional
        Order output variables by descending identifier, matching the
        reference's closing ``setorder(traces, -id)``. Row order is
        load-bearing downstream, where average-linkage clustering breaks
        its own ties by row order.
    key_added : str, optional
        ``.var`` column receiving the ``';'``-joined identifiers of all
        group members.
    inplace : bool, optional
        If True, modify ``adata`` in place. Otherwise return a new
        AnnData.
    verbose : bool, optional
        Print a peptide-count summary.

    Returns
    -------
    AnnData or None
        The summarised object when ``inplace=False``, otherwise None.

        ``.var`` is reduced to ``peptide_id``, ``protein_id``,
        ``peptide_start``, ``peptide_end``, ``key_added``,
        ``n_grouped`` and anything named in ``keep_var_cols``. Other
        annotations are dropped: the surviving row's metadata is one
        member's, not the group's, and carrying it would invite it to
        be read as representative. Layers are dropped for the same
        reason. ``.X`` is always dense, including when the input was
        sparse.

    Raises
    ------
    ValueError
        If an argument is invalid; if ``.var`` already holds a column
        this function writes; if a stripped identifier contains
        non-amino-acid characters; or, under the default policies, if a
        protein is absent from ``annotator`` or a peptide is not found
        in its protein sequence.

    Examples
    --------
    The worked example from the module docstring: a chain of four
    overlapping peptides, plus ``CDE`` lying strictly inside ``ACDEF``.
    It yields three groups — ``{ACDEF, CDE}``, ``{EFGHI, HIKLM}`` and
    ``{LMNPQ}`` — so three of the five peptides survive.

    >>> import numpy as np
    >>> import pandas as pd
    >>> from anndata import AnnData
    >>> import proteopy as pr
    >>> pids = ["ACDEF", "CDE", "EFGHI", "HIKLM", "LMNPQ"]
    >>> var = pd.DataFrame(
    ...     {"peptide_id": pids, "protein_id": ["P1"] * 5},
    ...     index=pids,
    ... )
    >>> obs = pd.DataFrame({"sample_id": ["s1"]}, index=["s1"])
    >>> adata = AnnData(
    ...     X=np.array([[30.0, 5.0, 20.0, 99.0, 40.0]]),
    ...     obs=obs,
    ...     var=var,
    ... )
    >>> out = pr.pp.summarize_peptides_by_neighbourhood_union(
    ...     adata, {"P1": "ACDEFGHIKLMNPQRSTVWY"}, inplace=False,
    ... )

    The most abundant member of each group survives, ordered by
    descending identifier:

    >>> out.var_names.tolist()
    ['LMNPQ', 'HIKLM', 'ACDEF']
    >>> out.X
    array([[40., 99., 30.]])

    Every member is recorded, including the ones that lost:

    >>> out.var["peptide_ids"].tolist()
    ['LMNPQ', 'EFGHI;HIKLM', 'ACDEF;CDE']
    >>> out.var["n_grouped"].tolist()
    [1, 2, 2]

    ``ACDEF`` and ``EFGHI`` overlap in the protein while neither
    contains the other, which is what substring containment misses; and
    ``CDE`` is grouped with the container it sits strictly inside.

    >>> out.var[["peptide_start", "peptide_end"]].values
    array([[10., 14.],
           [ 7., 11.],
           [ 1.,  5.]])
    """
    # -- validate everything before doing any work
    _validate_arguments(
        top_n,
        keep_less,
        id_from,
        on_unknown_protein,
        on_unlocated_peptide,
        tie_break_key,
        zero_to_na,
        fill_na,
    )
    written = (_START_COL, _END_COL, key_added, _N_COL)
    _validate_var(adata, peptide_col, protein_col, written, keep_var_cols)
    check_proteodata(adata)

    sequences = _resolve_annotator(annotator)
    pattern = re.compile(mod_regex)
    allowed = frozenset(alphabet)

    peptides = adata.var[peptide_col].astype(str).to_numpy()
    proteins = adata.var[protein_col].astype(str).to_numpy()

    stripped = _strip_and_validate(peptides, pattern, allowed)
    starts, ends, keep = _locate(
        peptides,
        stripped,
        proteins,
        sequences,
        on_unknown_protein,
        on_unlocated_peptide,
    )

    # -- matrix, densified; only .X is read
    X = adata.X
    # Densified unconditionally: the algorithm ranks and reorders whole
    # columns and gains nothing from sparsity, and a matrix of peptide
    # intensities is not meaningfully sparse -- an absent measurement is
    # missing, not zero.
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    X = X.astype(float, copy=True)
    if zero_to_na:
        X[X == 0] = np.nan
    if fill_na is not None:
        X[np.isnan(X)] = fill_na

    # -- restrict to the peptides that survived the two policies
    kept = np.flatnonzero(keep)
    X = X[:, kept]
    peptides, proteins = peptides[kept], proteins[kept]
    starts, ends = starts[kept], ends[kept]
    # Positionally aligned with the arrays above, so group member
    # indices address it directly.
    source_var = adata.var.iloc[kept]

    # A plain sum, so NaN propagates -- an incomplete peptide gets a NaN
    # total and ranks last.
    totals = X.sum(axis=0) if X.size else np.zeros(len(kept))

    keys = _group_keys(starts, ends, peptides, proteins)
    selections = _select(
        keys, totals, peptides, top_n, keep_less, tie_break_key
    )

    # -- assemble the surviving rows
    if sort_descending_id:
        # setorder(traces, -id). data.table sorts character in the C
        # locale and Python compares by codepoint, which agree for
        # ASCII peptide identifiers.
        selections.sort(
            key=lambda s: peptides[s["representative"]], reverse=True
        )
    else:
        selections.sort(key=lambda s: s["representative"])

    columns, records = [], []
    for sel in selections:
        rep = sel["representative"]
        contributors = sel["contributors"]
        if len(contributors) == 1:
            values = X[:, contributors[0]]
        else:
            values = X[:, contributors].sum(axis=1)
        columns.append(values)
        records.append(
            {
                peptide_col: peptides[rep],
                protein_col: proteins[rep],
                _START_COL: starts[rep],
                _END_COL: ends[rep],
                key_added: ";".join(
                    sorted(peptides[i] for i in sel["members"])
                ),
                _N_COL: len(sel["members"]),
            }
        )
        for col in keep_var_cols or ():
            # Aggregated over every member, not taken from the
            # survivor: the point of dropping annotations is that one
            # member's value is not the group's.
            records[-1][col] = _aggregate_var_value(
                source_var[col].iloc[sel["members"]]
            )

    if columns:
        X_new = np.column_stack(columns)
    else:
        X_new = np.empty((adata.n_obs, 0), dtype=float)

    var_new = pd.DataFrame(
        records,
        columns=[
            peptide_col,
            protein_col,
            _START_COL,
            _END_COL,
            key_added,
            _N_COL,
            *(keep_var_cols or ()),
        ],
    )
    var_new[_START_COL] = var_new[_START_COL].astype(float)
    var_new[_END_COL] = var_new[_END_COL].astype(float)
    var_new[_N_COL] = var_new[_N_COL].astype(int)
    # A plain list, not the Series: pd.Index() inherits a Series' name
    # even when name=None is passed, and a named var index turns
    # `.reset_index()` into a differently-named column downstream.
    var_new.index = pd.Index([str(v) for v in var_new[peptide_col]])
    var_new.index.name = None

    if verbose:
        print(
            f"summarize_peptides_by_neighbourhood_union: "
            f"{adata.n_vars} -> {var_new.shape[0]} peptides across "
            f"{len(selections)} neighbourhood group(s)"
        )

    result = _rebuild_adata(adata, X_new, var_new, inplace)
    check_proteodata(adata if inplace else result)
    return result
