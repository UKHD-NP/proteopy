"""Tests for ``pr.pp.summarize_peptides_by_neighbourhood_union``.

The function reimplements CCprofiler's
``summarizeAlternativePeptideSequences(topN = 1)`` together with the
``proteinQuantification(topN, keep_less)`` selection it delegates to.
Reference source: CCprofiler at git ref ``31a3043`` (branch
``proteoformLocationMapping``), files ``R/summarizeRedundantPeptides.R``
and ``R/proteinQuantification.R``.

Every expected value below is computed BY HAND from a small synthetic
protein, never by calling the code under test.

Three terms are used throughout, matching the algorithm's three layers:

**neighbourhood**
    ``coPeps(x)`` -- the peptides ``q`` of the same protein whose start
    OR end falls inside ``x``'s interval ``[x.start, x.end]``.
**label**
    the union of every neighbourhood that contains ``x``, sorted and
    joined with ``';'``.
**group**
    all peptides carrying an identical label. One row survives per
    group.

Several tests exist specifically because the reference algorithm has
properties that a reasonable reimplementation would silently "correct"
into something cleaner and wrong:

* ``test_overlap_chain_yields_three_groups_not_one`` -- an A-B-C-D
  overlap chain yields THREE groups. Transitive closure would give one,
  selecting one peptide where the reference selects three.
* ``test_keep_collapses_several_unlocated_peptides_of_one_protein`` --
  peptides whose position cannot be resolved share the empty label and
  collapse into a single group per protein. This is how the reference
  eliminates proteins that are absent from the FASTA.
* ``test_keep_leaves_a_lone_unlocated_peptide_untouched`` -- the
  collapse only bites at two or more. The reference's own output
  contains exactly one unlocated peptide, ``P35235``'s
  ``VGQALLQGNTER``, which survives because it is alone.
"""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse

from proteopy.pp import (
    summarize_peptides_by_neighbourhood_union as summarize,
)
from proteopy.utils.anndata import check_proteodata


# -- Synthetic reference protein --------------------------------------
#
# Twenty distinct residues, so every substring occurs exactly once and
# positions are unambiguous by inspection:
#
#     A  C  D  E  F  G  H  I  K  L  M  N  P  Q  R  S  T  V  W  Y
#     1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20

_P1 = "ACDEFGHIKLMNPQRSTVWY"

_FASTA = {"P1": _P1, "P2": _P1}

# Selenocysteine is a real UniProt residue -- dataset 01's FASTA carries
# 33 of them, plus X, B and Z -- so the default alphabet must accept it.
_SELENO = {"S1": "ACUDEF"}

# Intervals used repeatedly below (1-based, inclusive):
#     ACDEF   1..5      HIKLM   7..11     RSTVWY  15..20
#     EFGHI   4..8      LMNPQ  10..14


# -- Helpers ----------------------------------------------------------


def make_adata(peptides, proteins, X, var_extra=None):
    """Build a minimal peptide-level AnnData."""
    X = np.asarray(X, dtype=float)
    samples = [f"s{i + 1}" for i in range(X.shape[0])]
    var = pd.DataFrame(
        {"peptide_id": peptides, "protein_id": proteins},
        index=peptides,
    )
    if var_extra is not None:
        for key, values in var_extra.items():
            var[key] = values
    obs = pd.DataFrame({"sample_id": samples}, index=samples)
    return AnnData(X=X, obs=obs, var=var)


def survivors(adata):
    """Output peptide identifiers as a set."""
    return set(adata.var_names.astype(str))


def write_fasta(tmp_path, text, name="dummy.fasta"):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


class TestSummarizePeptidesByNeighbourhoodUnion:
    """CCprofiler-equivalent peptide summarisation.

    One class per function under test. The topic groupings that
    were previously separate classes are retained below as header
    comments, each naming the class it replaced.
    """

    # -- FASTA handling ---------------------------------------------------

    # (was class TestFasta)
    #
    # Header parsing replicates CCprofiler's ``gsub``.

    def test_accession_is_extracted_between_pipes(self, tmp_path):
        path = write_fasta(
            tmp_path,
            ">sp|P1|SRBS2_MOUSE Sorbin and SH3 domain protein\n"
            "ACDEFGHIKL\nMNPQRSTVWY\n",
        )
        adata = make_adata(["ACDEF"], ["P1"], [[10.0]])
        out = summarize(adata, path, inplace=False)
        assert out.var["peptide_start"].tolist() == [1.0]

    def test_header_without_pipes_is_used_verbatim(self, tmp_path):
        """``iRT_protein`` has no pipes; the gsub leaves it unchanged."""
        path = write_fasta(tmp_path, ">iRT_protein\nACDEFGHIKL\n")
        adata = make_adata(["ACDEF"], ["iRT_protein"], [[10.0]])
        out = summarize(adata, path, inplace=False)
        assert out.var["peptide_start"].tolist() == [1.0]

    def test_duplicate_accession_keeps_the_first(self, tmp_path):
        """Matches Biostrings' name-based subsetting, which takes [1]."""
        path = write_fasta(
            tmp_path,
            ">sp|P1|A\nACDEFGHIKL\n>sp|P1|B\nYYYYYYYYYY\n",
        )
        adata = make_adata(["ACDEF"], ["P1"], [[10.0]])
        out = summarize(adata, path, inplace=False)
        assert out.var["peptide_start"].tolist() == [1.0]

    def test_multiline_sequence_is_concatenated(self, tmp_path):
        path = write_fasta(
            tmp_path,
            ">sp|P1|A\nACDEFGHIKL\nMNPQRSTVWY\n",
        )
        adata = make_adata(["LMNPQ"], ["P1"], [[10.0]])
        out = summarize(adata, path, inplace=False)
        assert out.var["peptide_start"].tolist() == [10.0]

    def test_path_and_dict_are_equivalent(self, tmp_path):
        path = write_fasta(tmp_path, f">sp|P1|A\n{_P1}\n")
        peptides, proteins = ["ACDEF", "RSTVWY"], ["P1", "P1"]
        X = [[10.0, 20.0]]
        from_path = summarize(
            make_adata(peptides, proteins, X),
            path,
            inplace=False,
        )
        from_dict = summarize(
            make_adata(peptides, proteins, X),
            _FASTA,
            inplace=False,
        )
        assert survivors(from_path) == survivors(from_dict)
        np.testing.assert_array_equal(from_path.X, from_dict.X)

    # -- Position annotation ----------------------------------------------

    # (was class TestPositions)
    #
    # Positions match ``seqinr::words.pos(...)[1]``: first occurrence,
    # 1-based, inclusive.

    def test_positions_are_first_occurrence_one_based(self):
        adata = make_adata(
            ["ACDEF", "LMNPQ", "RSTVWY"],
            ["P1", "P1", "P1"],
            [[1.0, 1.0, 1.0]],
        )
        out = summarize(
            adata,
            _FASTA,
            inplace=False,
            sort_descending_id=False,
        )
        assert out.var["peptide_start"].tolist() == [1.0, 10.0, 15.0]
        assert out.var["peptide_end"].tolist() == [5.0, 14.0, 20.0]

    def test_repeated_subsequence_takes_the_first_occurrence(self):
        adata = make_adata(["AC"], ["Q1"], [[1.0]])
        out = summarize(adata, {"Q1": "ACDACD"}, inplace=False)
        assert out.var["peptide_start"].tolist() == [1.0]

    def test_unimod_tags_do_not_shift_positions(self):
        """The tag is stripped before matching, and the END position is
        computed from the STRIPPED length."""
        adata = make_adata(
            ["AC(UniMod:4)DEF"],
            ["P1"],
            [[1.0]],
        )
        out = summarize(adata, _FASTA, inplace=False)
        assert out.var["peptide_start"].tolist() == [1.0]
        assert out.var["peptide_end"].tolist() == [5.0]

    def test_custom_mod_regex_is_honoured(self):
        adata = make_adata(["AC[+57]DEF"], ["P1"], [[1.0]])
        out = summarize(
            adata,
            _FASTA,
            mod_regex=r"\[.*?\]",
            inplace=False,
        )
        assert out.var["peptide_start"].tolist() == [1.0]
        assert out.var["peptide_end"].tolist() == [5.0]

    def test_mod_regex_can_cover_several_notations_at_once(self):
        """One identifier carrying both notations locates correctly
        only when the pattern covers both."""
        adata = make_adata(
            ["AC(UniMod:4)D[+16]EF"],
            ["P1"],
            [[1.0]],
        )
        out = summarize(
            adata,
            _FASTA,
            mod_regex=r"\(UniMod:[0-9]+\)|\[.*?\]",
            inplace=False,
        )
        assert out.var["peptide_start"].tolist() == [1.0]
        assert out.var["peptide_end"].tolist() == [5.0]

    def test_stripping_does_not_rewrite_the_identifier(self):
        """The regex governs location only. The surviving peptide keeps
        its annotations, which is why the reference needs no separate
        modification-summarisation step."""
        adata = make_adata(
            ["AC(UniMod:4)DEF"],
            ["P1"],
            [[1.0]],
        )
        out = summarize(adata, _FASTA, inplace=False)
        assert out.var_names.tolist() == ["AC(UniMod:4)DEF"]

    # -- The stripped sequence must be an amino-acid sequence -------------

    # (was class TestStrippedSequenceAlphabet)
    #
    # After ``mod_regex`` is applied, what remains must be a pure
    # amino-acid sequence. Anything else is a malformed identifier.
    #
    # This is what makes ``mod_regex`` self-checking: the caller declares
    # what to disregard, and the alphabet check verifies the declaration
    # was complete. No notation-specific pattern has to be hard-coded to
    # detect mass shifts, bracket tags, lowercase markers or anything
    # else -- none of them are amino-acid letters.
    #
    # It is a hard error under every policy. An identifier carrying
    # residual notation is a configuration mistake, not a property of the
    # data, so neither ``on_unlocated_peptide`` nor ``on_unknown_protein``
    # suppresses it. Letting it through would give the peptide NaN
    # positions and drop it silently into the empty-label group, which is
    # exactly the behaviour this check exists to prevent.
    #

    def test_residual_mass_shift_raises(self):
        adata = make_adata(["AC[+57]DEF"], ["P1"], [[1.0]])
        with pytest.raises(ValueError, match=r"AC\[\+57\]DEF"):
            summarize(adata, _FASTA, inplace=False)

    def test_partial_mod_regex_coverage_raises(self):
        """Covering one of two notations is not enough."""
        adata = make_adata(
            ["AC(UniMod:4)D[+16]EF"],
            ["P1"],
            [[1.0]],
        )
        with pytest.raises(ValueError):
            summarize(
                adata,
                _FASTA,
                mod_regex=r"\(UniMod:[0-9]+\)",
                inplace=False,
            )

    def test_digits_raise(self):
        adata = make_adata(["ACDEF2"], ["P1"], [[1.0]])
        with pytest.raises(ValueError, match="ACDEF2"):
            summarize(adata, _FASTA, inplace=False)

    def test_lowercase_modification_marker_raises(self):
        """Some search engines mark modified residues in lowercase."""
        adata = make_adata(["ACDEFm"], ["P1"], [[1.0]])
        with pytest.raises(ValueError, match="ACDEFm"):
            summarize(adata, _FASTA, inplace=False)

    def test_error_names_the_offending_characters(self):
        adata = make_adata(["AC[+57]DEF"], ["P1"], [[1.0]])
        with pytest.raises(ValueError) as excinfo:
            summarize(adata, _FASTA, inplace=False)
        message = str(excinfo.value)
        assert "[" in message and "+" in message

    def test_all_offending_peptides_are_reported(self):
        adata = make_adata(
            ["ACDEF1", "EFGHI2"],
            ["P1", "P1"],
            [[1.0, 2.0]],
        )
        with pytest.raises(ValueError) as excinfo:
            summarize(adata, _FASTA, inplace=False)
        assert "ACDEF1" in str(excinfo.value)
        assert "EFGHI2" in str(excinfo.value)

    def test_skip_does_not_suppress_the_alphabet_error(self):
        adata = make_adata(["AC[+57]DEF"], ["P1"], [[1.0]])
        with pytest.raises(ValueError, match=r"AC\[\+57\]DEF"):
            summarize(
                adata,
                _FASTA,
                on_unlocated_peptide="skip",
                inplace=False,
            )

    def test_keep_does_not_suppress_the_alphabet_error(self):
        adata = make_adata(["AC[+57]DEF"], ["P1"], [[1.0]])
        with pytest.raises(ValueError, match=r"AC\[\+57\]DEF"):
            summarize(
                adata,
                _FASTA,
                on_unlocated_peptide="keep",
                inplace=False,
            )

    def test_selenocysteine_is_accepted_by_default(self):
        """``U`` is a real UniProt residue, not notation. Rejecting it
        would turn a legitimate selenopeptide into a false error."""
        adata = make_adata(["ACUDEF"], ["S1"], [[1.0]])
        out = summarize(adata, _SELENO, inplace=False)
        assert out.var["peptide_start"].tolist() == [1.0]
        assert out.var["peptide_end"].tolist() == [6.0]

    def test_ambiguity_codes_are_accepted_by_default(self):
        adata = make_adata(["AXBZ"], ["S2"], [[1.0]])
        out = summarize(adata, {"S2": "AXBZ"}, inplace=False)
        assert out.var["peptide_start"].tolist() == [1.0]

    def test_alphabet_can_be_narrowed_to_the_canonical_twenty(self):
        adata = make_adata(["ACUDEF"], ["S1"], [[1.0]])
        with pytest.raises(ValueError, match="ACUDEF"):
            summarize(
                adata,
                _SELENO,
                alphabet="ACDEFGHIKLMNPQRSTVWY",
                inplace=False,
            )

    def test_a_clean_peptide_that_is_simply_absent_still_reports_as_unlocated(  # noqa: E501
        self,
    ):
        """The two error classes stay distinct: ``WWWWW`` is a valid
        amino-acid sequence, so it reaches the location step and is
        reported as unlocated -- which ``'skip'`` and ``'keep'`` can
        act on."""
        adata = make_adata(
            ["ACDEF", "WWWWW"],
            ["P1", "P1"],
            [[10.0, 99.0]],
        )
        out = summarize(
            adata,
            _FASTA,
            on_unlocated_peptide="skip",
            inplace=False,
        )
        assert survivors(out) == {"ACDEF"}

    # -- Proteins missing from the annotator ------------------------------

    # (was class TestUnknownProtein)
    #
    # A protein absent from the annotator, governed by
    # ``on_unknown_protein: {'raise', 'skip', 'keep'}``.
    #
    # ``'raise'`` is the default: a protein missing from the FASTA is
    # usually a mismatched-input error worth stopping for, and the
    # reference is only half-vocal about it (it prints the accession and
    # carries on).
    #
    # ``'keep'`` reproduces the reference exactly -- every peptide of the
    # protein gets a NaN position, so they share the empty label, collapse
    # into one group, and the single survivor is then removed downstream
    # by the >=2-peptide filter.
    #
    # ``'skip'`` discards those peptides upfront. Measured equivalent on
    # the reference dataset (5 proteins, 23 peptides, identical
    # 24,534-peptide result) but NOT equivalent in general, which
    # ``test_keep_and_skip_differ_for_a_one_peptide_protein`` pins.
    #

    def test_absent_protein_raises_by_default(self):
        adata = make_adata(["ACDEF"], ["ABSENT"], [[1.0]])
        with pytest.raises(ValueError, match="ABSENT"):
            summarize(adata, _FASTA, inplace=False)

    def test_all_missing_proteins_are_reported_not_just_the_first(self):
        adata = make_adata(
            ["ACDEF", "EFGHI"],
            ["GHOST_A", "GHOST_B"],
            [[1.0, 2.0]],
        )
        with pytest.raises(ValueError) as excinfo:
            summarize(adata, _FASTA, inplace=False)
        assert "GHOST_A" in str(excinfo.value)
        assert "GHOST_B" in str(excinfo.value)

    def test_unknown_protein_error_names_the_available_modes(self):
        adata = make_adata(["ACDEF"], ["ABSENT"], [[1.0]])
        with pytest.raises(ValueError, match="skip"):
            summarize(adata, _FASTA, inplace=False)
        with pytest.raises(ValueError, match="keep"):
            summarize(adata, _FASTA, inplace=False)

    def test_skip_discards_the_absent_protein_peptides(self):
        adata = make_adata(
            ["ACDEF", "EFGHI"],
            ["P1", "ABSENT"],
            [[10.0, 99.0]],
        )
        out = summarize(
            adata,
            _FASTA,
            on_unknown_protein="skip",
            inplace=False,
        )
        assert survivors(out) == {"ACDEF"}

    def test_keep_collapses_the_absent_protein_to_one_peptide(self):
        """All three peptides get NaN positions, share the empty label,
        and form ONE group. This is the mechanism by which the reference
        strips a FASTA-absent protein down to a single peptide."""
        adata = make_adata(
            ["ACDEF", "EFGHI", "HIKLM"],
            ["ABSENT", "ABSENT", "ABSENT"],
            [[10.0, 99.0, 5.0]],
        )
        out = summarize(
            adata,
            _FASTA,
            on_unknown_protein="keep",
            inplace=False,
        )
        assert survivors(out) == {"EFGHI"}

    def test_keep_gives_absent_protein_peptides_nan_positions(self):
        adata = make_adata(
            ["ACDEF", "EFGHI"],
            ["ABSENT", "ABSENT"],
            [[10.0, 99.0]],
        )
        out = summarize(
            adata,
            _FASTA,
            on_unknown_protein="keep",
            inplace=False,
        )
        assert np.isnan(out.var["peptide_start"]).all()

    def test_keep_and_skip_differ_for_a_one_peptide_protein(self):
        """The two modes coincide on the reference dataset only because
        a >=2-peptide filter runs downstream. With a single peptide
        there is nothing to collapse, so ``'keep'`` lets it through."""
        peptides, proteins = ["ACDEF", "EFGHI"], ["P1", "ABSENT"]
        X = [[10.0, 99.0]]
        kept = summarize(
            make_adata(peptides, proteins, X),
            _FASTA,
            on_unknown_protein="keep",
            inplace=False,
        )
        skipped = summarize(
            make_adata(peptides, proteins, X),
            _FASTA,
            on_unknown_protein="skip",
            inplace=False,
        )
        assert survivors(kept) == {"ACDEF", "EFGHI"}
        assert survivors(skipped) == {"ACDEF"}

    def test_on_unlocated_peptide_skip_does_not_govern_proteins(self):
        adata = make_adata(["ACDEF"], ["ABSENT"], [[1.0]])
        with pytest.raises(ValueError, match="ABSENT"):
            summarize(
                adata,
                _FASTA,
                on_unlocated_peptide="skip",
                inplace=False,
            )

    def test_on_unlocated_peptide_keep_does_not_govern_proteins(self):
        adata = make_adata(["ACDEF"], ["ABSENT"], [[1.0]])
        with pytest.raises(ValueError, match="ABSENT"):
            summarize(
                adata,
                _FASTA,
                on_unlocated_peptide="keep",
                inplace=False,
            )

    def test_the_two_policies_are_independent(self):
        """Allowing absent proteins does not silence an unlocated
        peptide in a protein that IS present."""
        adata = make_adata(
            ["ACDEF", "WWWWW"],
            ["ABSENT", "P1"],
            [[10.0, 99.0]],
        )
        with pytest.raises(ValueError, match="WWWWW"):
            summarize(
                adata,
                _FASTA,
                on_unknown_protein="keep",
                inplace=False,
            )

    def test_both_policies_together_reproduce_the_reference(self):
        """The combination the COPF pipeline passes: no pre-filter, no
        error, and NaN positions for everything unresolvable."""
        adata = make_adata(
            ["ACDEF", "WWWWW", "EFGHI"],
            ["P1", "P1", "ABSENT"],
            [[10.0, 99.0, 5.0]],
        )
        out = summarize(
            adata,
            _FASTA,
            on_unknown_protein="keep",
            on_unlocated_peptide="keep",
            inplace=False,
        )
        assert survivors(out) == {"ACDEF", "WWWWW", "EFGHI"}

    def test_invalid_on_unknown_protein_raises(self):
        adata = make_adata(["ACDEF"], ["P1"], [[1.0]])
        with pytest.raises(ValueError, match="on_unknown_protein"):
            summarize(
                adata,
                _FASTA,
                on_unknown_protein="ignore",
                inplace=False,
            )

    # -- Peptides not found in their protein sequence ---------------------

    # (was class TestUnlocatedPeptides)
    #
    # A peptide whose sequence does not occur in its protein.
    #
    # The reference is silent about this case -- ``words.pos(...)[1]``
    # yields NA and the peptide continues as one that overlaps nothing.
    # ``'raise'`` is the default because that silence is the reference's
    # real blind spot; ``'keep'`` reproduces it exactly and is what the
    # COPF pipeline passes.
    #

    def test_unlocated_peptide_raises_by_default(self):
        adata = make_adata(["WWWWW"], ["P1"], [[1.0]])
        with pytest.raises(ValueError, match="WWWWW"):
            summarize(adata, _FASTA, inplace=False)

    def test_error_names_the_peptide_and_its_protein(self):
        adata = make_adata(["WWWWW"], ["P1"], [[1.0]])
        with pytest.raises(ValueError, match="P1"):
            summarize(adata, _FASTA, inplace=False)

    def test_unlocated_peptide_error_names_the_available_modes(self):
        adata = make_adata(["WWWWW"], ["P1"], [[1.0]])
        with pytest.raises(ValueError, match="skip"):
            summarize(adata, _FASTA, inplace=False)
        with pytest.raises(ValueError, match="keep"):
            summarize(adata, _FASTA, inplace=False)

    def test_skip_removes_the_peptide_entirely(self):
        adata = make_adata(
            ["ACDEF", "WWWWW"],
            ["P1", "P1"],
            [[10.0, 99.0]],
        )
        out = summarize(
            adata,
            _FASTA,
            on_unlocated_peptide="skip",
            inplace=False,
        )
        assert survivors(out) == {"ACDEF"}

    def test_skip_leaves_no_trace_in_the_provenance(self):
        """Skipped means removed before grouping, so it must not appear
        as a group member either."""
        adata = make_adata(
            ["ACDEF", "WWWWW"],
            ["P1", "P1"],
            [[10.0, 99.0]],
        )
        out = summarize(
            adata,
            _FASTA,
            on_unlocated_peptide="skip",
            inplace=False,
        )
        assert out.var["peptide_ids"].tolist() == ["ACDEF"]

    def test_keep_leaves_a_lone_unlocated_peptide_untouched(self):
        """The reference's own 24,534-peptide output contains exactly
        one unlocated peptide -- ``P35235``'s ``VGQALLQGNTER`` -- and it
        survives precisely because no other peptide of that protein
        shares its empty label."""
        adata = make_adata(
            ["ACDEF", "WWWWW"],
            ["P1", "P1"],
            [[10.0, 99.0]],
        )
        out = summarize(
            adata,
            _FASTA,
            on_unlocated_peptide="keep",
            inplace=False,
        )
        assert survivors(out) == {"ACDEF", "WWWWW"}

    def test_keep_gives_an_unlocated_peptide_nan_positions(self):
        adata = make_adata(
            ["ACDEF", "WWWWW"],
            ["P1", "P1"],
            [[10.0, 99.0]],
        )
        out = summarize(
            adata,
            _FASTA,
            on_unlocated_peptide="keep",
            inplace=False,
            sort_descending_id=False,
        )
        positions = dict(
            zip(out.var_names, out.var["peptide_start"]),
        )
        assert positions["ACDEF"] == 1.0
        assert np.isnan(positions["WWWWW"])

    def test_keep_collapses_several_unlocated_peptides_of_one_protein(
        self,
    ):
        """NaN compares False against everything, so these share the
        empty label and form ONE group -- the mechanism that strips a
        FASTA-absent protein down to a single peptide."""
        adata = make_adata(
            ["WWWWW", "YYYYY", "ACDEF"],
            ["P1", "P1", "P1"],
            [[10.0, 99.0, 5.0]],
        )
        out = summarize(
            adata,
            _FASTA,
            on_unlocated_peptide="keep",
            inplace=False,
        )
        assert survivors(out) == {"YYYYY", "ACDEF"}

    def test_keep_does_not_group_unlocated_peptides_across_proteins(
        self,
    ):
        """The empty label is still scoped to a protein."""
        adata = make_adata(
            ["WWWWW", "YYYYY"],
            ["P1", "P2"],
            [[10.0, 99.0]],
        )
        out = summarize(
            adata,
            _FASTA,
            on_unlocated_peptide="keep",
            inplace=False,
        )
        assert survivors(out) == {"WWWWW", "YYYYY"}

    def test_keep_and_skip_disagree_on_the_peptide_set(self):
        """Pins that the modes are not interchangeable. Choosing wrong
        costs exactly one peptide on the reference dataset, which is
        enough to break set-level agreement."""
        peptides, proteins = ["ACDEF", "WWWWW"], ["P1", "P1"]
        X = [[10.0, 99.0]]
        kept = summarize(
            make_adata(peptides, proteins, X),
            _FASTA,
            on_unlocated_peptide="keep",
            inplace=False,
        )
        skipped = summarize(
            make_adata(peptides, proteins, X),
            _FASTA,
            on_unlocated_peptide="skip",
            inplace=False,
        )
        assert survivors(kept) - survivors(skipped) == {"WWWWW"}

    def test_invalid_on_unlocated_peptide_raises(self):
        adata = make_adata(["ACDEF"], ["P1"], [[1.0]])
        with pytest.raises(
            ValueError,
            match="on_unlocated_peptide",
        ):
            summarize(
                adata,
                _FASTA,
                on_unlocated_peptide="ignore",
                inplace=False,
            )

    # -- Grouping semantics -----------------------------------------------

    # (was class TestGrouping)
    #
    # The three fidelity-critical properties of the grouping rule.

    def test_strictly_contained_peptide_groups_with_its_container(
        self,
    ):
        """``CDEFGHIKLMNPQRST`` spans 2..17; ``HIKLM`` spans 7..11,
        strictly inside it.

        Neighbourhoods -- note these are NOT symmetric, because the
        test only asks whether the OTHER peptide's start or end lands
        inside the anchor's interval::

            coPeps(LONG)  = {LONG, SHORT}   7 and 11 are inside 2..17
            coPeps(SHORT) = {SHORT}         2 and 17 are OUTSIDE 7..11

        Labels -- the union of every neighbourhood containing the
        peptide::

            label(LONG)  = coPeps(LONG)                 = {LONG, SHORT}
            label(SHORT) = coPeps(LONG) | coPeps(SHORT) = {LONG, SHORT}

        Equal labels, so ONE group with members {LONG, SHORT}, and the
        more abundant member survives.

        ⚠️ This test does NOT pin the asymmetry, despite it being real
        in the source. When ``q`` is strictly inside ``x``, every
        peptide reaching ``q``'s interval also reaches ``x``'s, so
        ``coPeps(q) ⊆ coPeps(x)`` and the extra symmetric edge adds
        nothing to any union. Brute-forced over 200,000 random interval
        sets (87,061 containing a strict containment): zero label
        differences between the asymmetric and symmetric rules. The
        asymmetry is therefore unobservable through this API, and
        pinning it needs a unit test on the private label helper.
        """
        adata = make_adata(
            ["CDEFGHIKLMNPQRST", "HIKLM"],
            ["P1", "P1"],
            [[10.0, 99.0]],
        )
        out = summarize(adata, _FASTA, inplace=False)
        assert survivors(out) == {"HIKLM"}
        assert out.var["peptide_ids"].tolist() == [
            "CDEFGHIKLMNPQRST;HIKLM",
        ]

    def test_overlap_chain_yields_three_groups_not_one(self):
        """An A-B-C-D chain with only adjacent pairs overlapping.

        Intervals: ACDEF 1..5, EFGHI 4..8, HIKLM 7..11, LMNPQ 10..14.

        ``coPeps`` sets::

            coPeps(ACDEF) = {ACDEF, EFGHI}
            coPeps(EFGHI) = {ACDEF, EFGHI, HIKLM}
            coPeps(HIKLM) = {EFGHI, HIKLM, LMNPQ}
            coPeps(LMNPQ) = {HIKLM, LMNPQ}

        Labels are the union of every set containing the peptide::

            ACDEF -> {ACDEF, EFGHI, HIKLM}
            EFGHI -> {ACDEF, EFGHI, HIKLM, LMNPQ}
            HIKLM -> {ACDEF, EFGHI, HIKLM, LMNPQ}
            LMNPQ -> {EFGHI, HIKLM, LMNPQ}

        THREE distinct labels: {ACDEF}, {EFGHI, HIKLM}, {LMNPQ}. The
        two interior peptides sit in three neighbourhoods each, so both
        take the full union and group together; the two terminal
        peptides are singletons. Transitive closure would give a single
        group and select ONE peptide instead of three.
        """
        adata = make_adata(
            ["ACDEF", "EFGHI", "HIKLM", "LMNPQ"],
            ["P1"] * 4,
            [[10.0, 20.0, 99.0, 40.0]],
        )
        out = summarize(adata, _FASTA, inplace=False)
        assert survivors(out) == {"ACDEF", "HIKLM", "LMNPQ"}

    def test_chain_middle_group_records_both_members(self):
        adata = make_adata(
            ["ACDEF", "EFGHI", "HIKLM", "LMNPQ"],
            ["P1"] * 4,
            [[10.0, 20.0, 99.0, 40.0]],
        )
        out = summarize(
            adata,
            _FASTA,
            inplace=False,
            sort_descending_id=False,
        )
        provenance = dict(
            zip(out.var_names, out.var["peptide_ids"]),
        )
        assert provenance["HIKLM"] == "EFGHI;HIKLM"
        assert provenance["ACDEF"] == "ACDEF"
        assert provenance["LMNPQ"] == "LMNPQ"

    def test_non_overlapping_peptides_stay_separate(self):
        adata = make_adata(
            ["ACDEF", "RSTVWY"],
            ["P1", "P1"],
            [[10.0, 20.0]],
        )
        out = summarize(adata, _FASTA, inplace=False)
        assert survivors(out) == {"ACDEF", "RSTVWY"}

    def test_overlap_catches_what_substring_containment_misses(self):
        """``ACDEF`` and ``EFGHI`` overlap at positions 4-5, yet neither
        is a substring of the other. Substring containment would leave
        them as two peptides; positional overlap collapses them."""
        adata = make_adata(
            ["ACDEF", "EFGHI"],
            ["P1", "P1"],
            [[10.0, 99.0]],
        )
        out = summarize(adata, _FASTA, inplace=False)
        assert survivors(out) == {"EFGHI"}

    def test_grouping_is_confined_within_protein(self):
        """Identical intervals in different proteins never group."""
        adata = make_adata(
            ["ACDEF", "ACDEFG"],
            ["P1", "P2"],
            [[10.0, 99.0]],
        )
        out = summarize(adata, _FASTA, inplace=False)
        assert survivors(out) == {"ACDEF", "ACDEFG"}

    def test_peptidoforms_of_one_sequence_group_together(self):
        """Two peptidoforms of one stripped sequence share an interval,
        so they group and one is selected. This is why the reference
        needs no separate modification-summarisation step."""
        adata = make_adata(
            ["ACDEF", "AC(UniMod:4)DEF"],
            ["P1", "P1"],
            [[10.0, 99.0]],
        )
        out = summarize(adata, _FASTA, inplace=False)
        assert survivors(out) == {"AC(UniMod:4)DEF"}

    # -- Selection --------------------------------------------------------

    # (was class TestSelection)
    #
    # ``topN = 1`` SELECTS the most abundant member; it does not
    # aggregate.

    def test_most_abundant_is_selected_not_summed(self):
        """Totals across samples: ACDEF = 1 + 2 = 3,
        AC(UniMod:4)DEF = 10 + 20 = 30. The winner's own intensities
        are returned -- NOT 11 and 22."""
        adata = make_adata(
            ["ACDEF", "AC(UniMod:4)DEF"],
            ["P1", "P1"],
            [[1.0, 10.0], [2.0, 20.0]],
        )
        out = summarize(adata, _FASTA, inplace=False)
        assert survivors(out) == {"AC(UniMod:4)DEF"}
        np.testing.assert_array_equal(out.X, np.array([[10.0], [20.0]]))

    def test_abundance_is_the_sum_across_all_samples(self):
        """ACDEF wins on total (60) despite losing in sample 1."""
        adata = make_adata(
            ["ACDEF", "AC(UniMod:4)DEF"],
            ["P1", "P1"],
            [[10.0, 50.0], [50.0, 5.0]],
        )
        out = summarize(adata, _FASTA, inplace=False)
        assert survivors(out) == {"ACDEF"}

    # -- Tie breaking -----------------------------------------------------

    # (was class TestTieBreaking)
    #
    # Equal totals are resolved by ``tie_break_key``, never by input
    # order.
    #
    # CCprofiler uses ``ties.method = "first"``, i.e. whichever row
    # happened to come first, which makes the output depend on how the
    # input table was sorted. A deterministic key removes that
    # dependency.
    #
    # Measured on the reference dataset: of 26,473 groups, 4,752 have
    # more than one member and only 7 have a tied winner -- 6 of them
    # ties between two incomplete peptides, 1 a genuine equal-total tie.
    # Switching from row order to the default key changes the pick in 4
    # of the 7, and none of those 4 peptides reaches the reference's
    # 24,534-peptide set, so the change costs nothing here. That margin
    # is dataset-specific; re-measure on new data.
    #

    def test_ties_are_broken_by_the_ordering_key(self):
        """The default key sorts letters before any non-letter, so the
        unmodified form wins a tie."""
        adata = make_adata(
            ["AC(UniMod:4)DEF", "ACDEF"],
            ["P1", "P1"],
            [[10.0, 10.0]],
        )
        out = summarize(adata, _FASTA, inplace=False)
        assert survivors(out) == {"ACDEF"}

    def test_tie_break_is_independent_of_input_order(self):
        """The discriminating test: both input orders give the same
        winner, which row-order tie-breaking cannot do."""
        X = [[10.0, 10.0]]
        one = summarize(
            make_adata(
                ["ACDEF", "AC(UniMod:4)DEF"],
                ["P1", "P1"],
                X,
            ),
            _FASTA,
            inplace=False,
        )
        other = summarize(
            make_adata(
                ["AC(UniMod:4)DEF", "ACDEF"],
                ["P1", "P1"],
                X,
            ),
            _FASTA,
            inplace=False,
        )
        assert survivors(one) == survivors(other) == {"ACDEF"}

    def test_bracket_notation_also_sorts_after_letters(self):
        adata = make_adata(
            ["AC[+16]DEF", "ACDEF"],
            ["P1", "P1"],
            [[10.0, 10.0]],
        )
        out = summarize(
            adata,
            _FASTA,
            mod_regex=r"\[.*?\]",
            inplace=False,
        )
        assert survivors(out) == {"ACDEF"}

    def test_letters_still_compare_alphabetically(self):
        adata = make_adata(
            ["EFGHI", "ACDEF"],
            ["P1", "P1"],
            [[10.0, 10.0]],
        )
        out = summarize(adata, _FASTA, inplace=False)
        assert survivors(out) == {"ACDEF"}

    def test_custom_tie_break_key_is_honoured(self):
        """Longest-identifier-first instead of the default."""
        adata = make_adata(
            ["ACDEF", "AC(UniMod:4)DEF"],
            ["P1", "P1"],
            [[10.0, 10.0]],
        )
        out = summarize(
            adata,
            _FASTA,
            tie_break_key=lambda pid: -len(pid),
            inplace=False,
        )
        assert survivors(out) == {"AC(UniMod:4)DEF"}

    def test_the_key_only_applies_to_equal_totals(self):
        """Abundance decides first; the key never overrides it."""
        adata = make_adata(
            ["ACDEF", "AC(UniMod:4)DEF"],
            ["P1", "P1"],
            [[1.0, 99.0]],
        )
        out = summarize(adata, _FASTA, inplace=False)
        assert survivors(out) == {"AC(UniMod:4)DEF"}

    # -- Output .var columns ----------------------------------------------

    # (was class TestVarColumns)
    #
    # The surviving row's annotations are NOT representative of its
    # group, so they are not carried over.
    #
    # Only two kinds of column survive: those a peptide-level proteodata
    # object requires, and those this function computes. Everything else
    # is dropped, because keeping it would invite the reader to treat one
    # member's metadata as the whole group's.
    #

    def test_var_holds_exactly_the_expected_columns(self):
        adata = make_adata(
            ["ACDEF", "EFGHI"],
            ["P1", "P1"],
            [[10.0, 99.0]],
        )
        out = summarize(adata, _FASTA, inplace=False)
        assert set(out.var.columns) == {
            "peptide_id",
            "protein_id",
            "peptide_start",
            "peptide_end",
            "peptide_ids",
            "n_grouped",
        }

    def test_extraneous_columns_are_dropped(self):
        adata = make_adata(
            ["ACDEF", "AC(UniMod:4)DEF"],
            ["P1", "P1"],
            [[1.0, 10.0]],
            var_extra={"charge": ["2", "3"], "gene_id": ["g1", "g2"]},
        )
        out = summarize(adata, _FASTA, inplace=False)
        assert "charge" not in out.var.columns
        assert "gene_id" not in out.var.columns

    def test_dropping_happens_even_when_the_values_agree(self):
        """``gene_id`` is identical across the group here, so carrying
        it would be harmless -- and it is still dropped. The rule is
        about provenance, not about whether a value happens to be
        unambiguous."""
        adata = make_adata(
            ["ACDEF", "AC(UniMod:4)DEF"],
            ["P1", "P1"],
            [[1.0, 10.0]],
            var_extra={"gene_id": ["same", "same"]},
        )
        out = summarize(adata, _FASTA, inplace=False)
        assert "gene_id" not in out.var.columns

    def test_key_added_replaces_the_provenance_column_name(self):
        adata = make_adata(
            ["ACDEF", "EFGHI"],
            ["P1", "P1"],
            [[10.0, 99.0]],
        )
        out = summarize(
            adata,
            _FASTA,
            key_added="members",
            inplace=False,
        )
        assert set(out.var.columns) == {
            "peptide_id",
            "protein_id",
            "peptide_start",
            "peptide_end",
            "members",
            "n_grouped",
        }

    # -- Identifier naming ------------------------------------------------

    # (was class TestNaming)
    #
    # ``id_from`` selects how the surviving row is identified. Only
    # ``'top_ranked'`` is implemented; the parameter exists so that other
    # schemes can be added without a signature change.

    def test_default_names_the_row_after_the_top_ranked_member(self):
        adata = make_adata(
            ["ACDEF", "EFGHI"],
            ["P1", "P1"],
            [[10.0, 99.0]],
        )
        out = summarize(adata, _FASTA, inplace=False)
        assert out.var_names.tolist() == ["EFGHI"]

    def test_explicit_top_ranked_matches_the_default(self):
        peptides, proteins = ["ACDEF", "EFGHI"], ["P1", "P1"]
        X = [[10.0, 99.0]]
        default = summarize(
            make_adata(peptides, proteins, X),
            _FASTA,
            inplace=False,
        )
        explicit = summarize(
            make_adata(peptides, proteins, X),
            _FASTA,
            id_from="top_ranked",
            inplace=False,
        )
        assert default.var_names.tolist() == explicit.var_names.tolist()

    def test_unknown_id_from_raises(self):
        adata = make_adata(["ACDEF"], ["P1"], [[1.0]])
        with pytest.raises(ValueError, match="id_from"):
            summarize(adata, _FASTA, id_from="joined", inplace=False)

    def test_unknown_id_from_names_the_supported_value(self):
        adata = make_adata(["ACDEF"], ["P1"], [[1.0]])
        with pytest.raises(ValueError, match="top_ranked"):
            summarize(adata, _FASTA, id_from="longest", inplace=False)

    # -- Provenance -------------------------------------------------------

    # (was class TestProvenance)

    def test_peptide_ids_lists_all_group_members_sorted(self):
        adata = make_adata(
            ["EFGHI", "ACDEF"],
            ["P1", "P1"],
            [[99.0, 10.0]],
        )
        out = summarize(adata, _FASTA, inplace=False)
        assert out.var["peptide_ids"].tolist() == ["ACDEF;EFGHI"]

    def test_n_grouped_counts_group_members(self):
        adata = make_adata(
            ["ACDEF", "EFGHI", "RSTVWY"],
            ["P1", "P1", "P1"],
            [[10.0, 99.0, 5.0]],
        )
        out = summarize(
            adata,
            _FASTA,
            inplace=False,
            sort_descending_id=False,
        )
        counts = dict(zip(out.var_names, out.var["n_grouped"]))
        assert counts == {"EFGHI": 2, "RSTVWY": 1}

    def test_singleton_group_records_only_itself(self):
        adata = make_adata(["ACDEF"], ["P1"], [[10.0]])
        out = summarize(adata, _FASTA, inplace=False)
        assert out.var["peptide_ids"].tolist() == ["ACDEF"]
        assert out.var["n_grouped"].tolist() == [1]

    def test_key_added_renames_the_provenance_column(self):
        adata = make_adata(
            ["ACDEF", "EFGHI"],
            ["P1", "P1"],
            [[10.0, 99.0]],
        )
        out = summarize(
            adata,
            _FASTA,
            key_added="members",
            inplace=False,
        )
        assert out.var["members"].tolist() == ["ACDEF;EFGHI"]
        assert "peptide_ids" not in out.var.columns

    def test_provenance_survives_a_selection_it_did_not_win(self):
        """The losing member is named even though its row is gone."""
        adata = make_adata(
            ["ACDEF", "EFGHI"],
            ["P1", "P1"],
            [[10.0, 99.0]],
        )
        out = summarize(adata, _FASTA, inplace=False)
        assert "ACDEF" in out.var["peptide_ids"].iloc[0]
        assert "ACDEF" not in survivors(out)

    # -- Guards against silently overwriting .var -------------------------

    # (was class TestColumnGuards)
    #
    # Every column the function writes must be absent from the input
    # ``.var``, so a second call or a pre-existing annotation can never
    # be overwritten without the user noticing.

    def test_existing_peptide_start_raises(self):
        adata = make_adata(
            ["ACDEF"],
            ["P1"],
            [[1.0]],
            var_extra={"peptide_start": [99.0]},
        )
        with pytest.raises(ValueError, match="peptide_start"):
            summarize(adata, _FASTA, inplace=False)

    def test_existing_peptide_end_raises(self):
        adata = make_adata(
            ["ACDEF"],
            ["P1"],
            [[1.0]],
            var_extra={"peptide_end": [99.0]},
        )
        with pytest.raises(ValueError, match="peptide_end"):
            summarize(adata, _FASTA, inplace=False)

    def test_existing_provenance_column_raises(self):
        adata = make_adata(
            ["ACDEF"],
            ["P1"],
            [[1.0]],
            var_extra={"peptide_ids": ["stale"]},
        )
        with pytest.raises(ValueError, match="peptide_ids"):
            summarize(adata, _FASTA, inplace=False)

    def test_existing_n_grouped_raises(self):
        adata = make_adata(
            ["ACDEF"],
            ["P1"],
            [[1.0]],
            var_extra={"n_grouped": [7]},
        )
        with pytest.raises(ValueError, match="n_grouped"):
            summarize(adata, _FASTA, inplace=False)

    def test_key_added_avoids_a_provenance_collision(self):
        """Renaming the output column is the documented way past a
        clash on ``peptide_ids``."""
        adata = make_adata(
            ["ACDEF", "EFGHI"],
            ["P1", "P1"],
            [[10.0, 99.0]],
            var_extra={"peptide_ids": ["stale", "stale"]},
        )
        out = summarize(
            adata,
            _FASTA,
            key_added="members",
            inplace=False,
        )
        assert out.var["members"].tolist() == ["ACDEF;EFGHI"]
        # The stale column is no longer written to, so it is simply an
        # extraneous annotation and is dropped like any other.
        assert "peptide_ids" not in out.var.columns

    def test_key_added_collision_is_also_guarded(self):
        adata = make_adata(
            ["ACDEF"],
            ["P1"],
            [[1.0]],
            var_extra={"members": ["stale"]},
        )
        with pytest.raises(ValueError, match="members"):
            summarize(
                adata,
                _FASTA,
                key_added="members",
                inplace=False,
            )

    def test_calling_twice_raises_rather_than_overwriting(self):
        adata = make_adata(
            ["ACDEF", "EFGHI"],
            ["P1", "P1"],
            [[10.0, 99.0]],
        )
        summarize(adata, _FASTA)
        with pytest.raises(ValueError, match="peptide_start"):
            summarize(adata, _FASTA)

    def test_guard_message_points_at_the_remedy(self):
        adata = make_adata(
            ["ACDEF"],
            ["P1"],
            [[1.0]],
            var_extra={"peptide_start": [99.0]},
        )
        with pytest.raises(ValueError, match="drop"):
            summarize(adata, _FASTA, inplace=False)

    # -- Missing-value policy ---------------------------------------------

    # (was class TestMissingValues)
    #
    # Missing values are handled exactly as CCprofiler handles them,
    # with no parameter to choose otherwise.
    #
    # ``proteinQuantification`` sums with ``na.rm = FALSE`` and ranks with
    # ``rank()``'s default ``na.last = TRUE``, so a peptide with any
    # missing sample has a ``NaN`` total and ranks LAST.
    #
    # That is deprioritisation, not removal. A NaN-bearing peptide with no
    # complete competitor still wins its group and is passed through
    # untouched. The reference depends on this: its zero-variance step is
    # what finally removes such peptides, via ``var()`` propagating ``NA``.
    #

    def test_nan_input_is_accepted(self):
        adata = make_adata(
            ["ACDEF", "EFGHI"],
            ["P1", "P1"],
            [[np.nan, 99.0]],
        )
        out = summarize(adata, _FASTA, inplace=False)
        assert out.n_vars == 1

    def test_nan_bearing_peptide_ranks_last(self):
        """ACDEF would win on its observed values alone (100 > 10) but
        loses because one of its samples is missing."""
        adata = make_adata(
            ["ACDEF", "AC(UniMod:4)DEF"],
            ["P1", "P1"],
            [[100.0, 5.0], [np.nan, 5.0]],
        )
        out = summarize(adata, _FASTA, inplace=False)
        assert survivors(out) == {"AC(UniMod:4)DEF"}

    def test_even_a_far_weaker_complete_peptide_outranks_a_nan(self):
        adata = make_adata(
            ["ACDEF", "AC(UniMod:4)DEF"],
            ["P1", "P1"],
            [[9999.0, 1.0], [np.nan, 1.0]],
        )
        out = summarize(adata, _FASTA, inplace=False)
        assert survivors(out) == {"AC(UniMod:4)DEF"}

    def test_nan_survives_when_it_has_no_competitor(self):
        """A singleton group has nothing to lose to, so the NaN
        survives and is passed through unchanged for a later
        completeness filter to remove."""
        adata = make_adata(["ACDEF"], ["P1"], [[np.nan], [5.0]])
        out = summarize(adata, _FASTA, inplace=False)
        assert survivors(out) == {"ACDEF"}
        assert np.isnan(out.X[0, 0])

    def test_all_nan_group_falls_back_to_the_ordering_key(self):
        """Every member has a NaN total, so nothing separates them on
        abundance and ``tie_break_key`` decides. This is the common
        case: 6 of the 7 tied groups on the reference dataset are ties
        between two incomplete peptides."""
        adata = make_adata(
            ["AC(UniMod:4)DEF", "ACDEF"],
            ["P1", "P1"],
            [[np.nan, np.nan]],
        )
        out = summarize(adata, _FASTA, inplace=False)
        assert survivors(out) == {"ACDEF"}

    def test_fill_na_zero_corrupts_the_ranking(self):
        """Documents WHY ``fill_na=0`` is not faithful, so nobody
        reintroduces it as a default. ACDEF has a missing sample, so
        the reference ranks it last and keeps the complete competitor.
        Zero-filling gives ACDEF a real total of 150, which beats 100,
        and the wrong peptide survives."""
        peptides, proteins = ["ACDEF", "AC(UniMod:4)DEF"], ["P1", "P1"]
        X = [[150.0, 50.0], [np.nan, 50.0]]
        faithful = summarize(
            make_adata(peptides, proteins, X),
            _FASTA,
            inplace=False,
        )
        filled = summarize(
            make_adata(peptides, proteins, X),
            _FASTA,
            fill_na=0,
            inplace=False,
        )
        assert survivors(faithful) == {"AC(UniMod:4)DEF"}
        assert survivors(filled) == {"ACDEF"}

    def test_zero_to_na_makes_a_zero_bearing_peptide_lose(self):
        """ACDEF wins on raw totals (100 > 55) but loses once its zero
        is declared missing."""
        peptides, proteins = ["ACDEF", "AC(UniMod:4)DEF"], ["P1", "P1"]
        X = [[0.0, 50.0], [100.0, 5.0]]
        raw = summarize(
            make_adata(peptides, proteins, X),
            _FASTA,
            inplace=False,
        )
        masked = summarize(
            make_adata(peptides, proteins, X),
            _FASTA,
            zero_to_na=True,
            inplace=False,
        )
        assert survivors(raw) == {"ACDEF"}
        assert survivors(masked) == {"AC(UniMod:4)DEF"}

    def test_zero_to_na_is_inert_without_zeros(self):
        adata = make_adata(["ACDEF"], ["P1"], [[7.0]])
        out = summarize(
            adata,
            _FASTA,
            zero_to_na=True,
            inplace=False,
        )
        np.testing.assert_array_equal(out.X, np.array([[7.0]]))

    def test_zero_to_na_with_fill_na_raises(self):
        adata = make_adata(["ACDEF"], ["P1"], [[0.0]])
        with pytest.raises(ValueError, match="mutually exclusive"):
            summarize(
                adata,
                _FASTA,
                zero_to_na=True,
                fill_na=0,
                inplace=False,
            )

    def test_nan_is_never_written_into_a_surviving_value(self):
        """Selection copies a whole peptide's row, so at ``top_n=1`` no
        arithmetic happens and NaN can be neither created nor spread.
        It reaches the output only when the peptide that carried it
        won."""
        adata = make_adata(
            ["ACDEF", "AC(UniMod:4)DEF"],
            ["P1", "P1"],
            [[100.0, 5.0], [np.nan, 5.0]],
        )
        out = summarize(adata, _FASTA, inplace=False)
        assert not np.isnan(out.X).any()

    # -- top_n and keep_less ----------------------------------------------

    # (was class TestTopNAndKeepLess)

    def test_top_n_two_sums_the_two_most_abundant(self):
        """Group members total 30 and 10; their sum is 40 per sample."""
        adata = make_adata(
            ["ACDEF", "AC(UniMod:4)DEF"],
            ["P1", "P1"],
            [[10.0, 30.0]],
        )
        out = summarize(
            adata,
            _FASTA,
            top_n=2,
            keep_less=True,
            inplace=False,
        )
        np.testing.assert_array_equal(out.X, np.array([[40.0]]))

    def test_top_n_two_sums_only_the_top_two(self):
        """Three peptidoforms of one interval; the weakest is dropped
        from the sum, not merely from the identifier."""
        adata = make_adata(
            ["ACDEF", "AC(UniMod:4)DEF", "AC(UniMod:35)DEF"],
            ["P1", "P1", "P1"],
            [[1.0, 30.0, 10.0]],
        )
        out = summarize(
            adata,
            _FASTA,
            top_n=2,
            keep_less=True,
            inplace=False,
        )
        np.testing.assert_array_equal(out.X, np.array([[40.0]]))

    def test_top_n_two_row_is_named_by_the_top_ranked_member(self):
        """DELIBERATE DEVIATION from CCprofiler, which renames the row
        to a comma-joined list of the summed identifiers and thereby
        breaks its own annotation join. ``id_from='top_ranked'`` keeps
        ``peptide_id`` a real peptide; the summed members remain
        recoverable from the provenance column."""
        adata = make_adata(
            ["ACDEF", "AC(UniMod:4)DEF"],
            ["P1", "P1"],
            [[10.0, 30.0]],
        )
        out = summarize(
            adata,
            _FASTA,
            top_n=2,
            keep_less=True,
            inplace=False,
        )
        assert survivors(out) == {"AC(UniMod:4)DEF"}
        assert out.var["peptide_ids"].tolist() == [
            "AC(UniMod:4)DEF;ACDEF",
        ]

    def test_keep_less_false_drops_undersized_groups(self):
        """RSTVWY is a singleton, so it cannot supply two peptides."""
        adata = make_adata(
            ["ACDEF", "AC(UniMod:4)DEF", "RSTVWY"],
            ["P1", "P1", "P1"],
            [[10.0, 30.0, 99.0]],
        )
        out = summarize(
            adata,
            _FASTA,
            top_n=2,
            keep_less=False,
            inplace=False,
        )
        assert survivors(out) == {"AC(UniMod:4)DEF"}

    def test_keep_less_true_keeps_undersized_groups(self):
        adata = make_adata(
            ["ACDEF", "AC(UniMod:4)DEF", "RSTVWY"],
            ["P1", "P1", "P1"],
            [[10.0, 30.0, 99.0]],
        )
        out = summarize(
            adata,
            _FASTA,
            top_n=2,
            keep_less=True,
            inplace=False,
        )
        assert survivors(out) == {"AC(UniMod:4)DEF", "RSTVWY"}

    def test_keep_less_true_sums_what_the_group_has(self):
        adata = make_adata(
            ["ACDEF", "AC(UniMod:4)DEF", "RSTVWY"],
            ["P1", "P1", "P1"],
            [[10.0, 30.0, 99.0]],
        )
        out = summarize(
            adata,
            _FASTA,
            top_n=2,
            keep_less=True,
            inplace=False,
            sort_descending_id=False,
        )
        values = dict(zip(out.var_names, out.X[0]))
        assert values == {"AC(UniMod:4)DEF": 40.0, "RSTVWY": 99.0}

    def test_keep_less_is_inert_at_top_n_one(self):
        """Every group has at least one member, so the filter cannot
        remove anything. Pinned so the interaction is documented."""
        adata = make_adata(
            ["ACDEF", "AC(UniMod:4)DEF", "RSTVWY"],
            ["P1", "P1", "P1"],
            [[10.0, 30.0, 99.0]],
        )
        strict = summarize(
            adata,
            _FASTA,
            top_n=1,
            keep_less=False,
            inplace=False,
        )
        lenient = summarize(
            adata,
            _FASTA,
            top_n=1,
            keep_less=True,
            inplace=False,
        )
        assert survivors(strict) == survivors(lenient)
        np.testing.assert_array_equal(strict.X, lenient.X)

    def test_top_n_below_one_raises(self):
        adata = make_adata(["ACDEF"], ["P1"], [[1.0]])
        with pytest.raises(ValueError, match="top_n"):
            summarize(adata, _FASTA, top_n=0, inplace=False)

    # -- Output ordering --------------------------------------------------

    # (was class TestOrdering)
    #
    # Row order is load-bearing: average-linkage clustering downstream
    # breaks ties by row order, so the reference's closing
    # ``setorder(traces, -id)`` must be reproducible.

    def test_output_is_ordered_by_descending_identifier(self):
        adata = make_adata(
            ["ACDEF", "RSTVWY", "LMNPQ"],
            ["P1", "P1", "P1"],
            [[1.0, 2.0, 3.0]],
        )
        out = summarize(adata, _FASTA, inplace=False)
        assert out.var_names.tolist() == ["RSTVWY", "LMNPQ", "ACDEF"]

    def test_descending_order_can_be_disabled(self):
        adata = make_adata(
            ["ACDEF", "RSTVWY", "LMNPQ"],
            ["P1", "P1", "P1"],
            [[1.0, 2.0, 3.0]],
        )
        out = summarize(
            adata,
            _FASTA,
            sort_descending_id=False,
            inplace=False,
        )
        assert out.var_names.tolist() == ["ACDEF", "RSTVWY", "LMNPQ"]

    def test_matrix_columns_follow_the_reordered_var(self):
        adata = make_adata(
            ["ACDEF", "RSTVWY", "LMNPQ"],
            ["P1", "P1", "P1"],
            [[1.0, 2.0, 3.0]],
        )
        out = summarize(adata, _FASTA, inplace=False)
        values = dict(zip(out.var_names, out.X[0]))
        assert values == {"ACDEF": 1.0, "RSTVWY": 2.0, "LMNPQ": 3.0}

    # -- API contract -----------------------------------------------------

    # (was class TestContract)

    def test_inplace_returns_none_and_mutates(self):
        adata = make_adata(
            ["ACDEF", "EFGHI"],
            ["P1", "P1"],
            [[10.0, 99.0]],
        )
        assert summarize(adata, _FASTA) is None
        assert survivors(adata) == {"EFGHI"}

    def test_copy_leaves_the_input_untouched(self):
        adata = make_adata(
            ["ACDEF", "EFGHI"],
            ["P1", "P1"],
            [[10.0, 99.0]],
        )
        summarize(adata, _FASTA, inplace=False)
        assert adata.n_vars == 2
        assert "peptide_start" not in adata.var.columns

    def test_inplace_and_copy_agree(self):
        peptides = ["ACDEF", "EFGHI", "RSTVWY"]
        proteins = ["P1"] * 3
        X = [[10.0, 99.0, 5.0]]
        copied = summarize(
            make_adata(peptides, proteins, X),
            _FASTA,
            inplace=False,
        )
        mutated = make_adata(peptides, proteins, X)
        summarize(mutated, _FASTA)
        assert mutated.var_names.tolist() == copied.var_names.tolist()
        np.testing.assert_array_equal(mutated.X, copied.X)

    def test_missing_peptide_column_raises(self):
        adata = make_adata(["ACDEF"], ["P1"], [[1.0]])
        with pytest.raises(KeyError, match="absent_col"):
            summarize(
                adata,
                _FASTA,
                peptide_col="absent_col",
                inplace=False,
            )

    def test_missing_protein_column_raises(self):
        adata = make_adata(["ACDEF"], ["P1"], [[1.0]])
        with pytest.raises(KeyError, match="absent_col"):
            summarize(
                adata,
                _FASTA,
                protein_col="absent_col",
                inplace=False,
            )

    def test_layers_are_dropped_from_the_output(self):
        """The function reads and writes ``.X`` only. A layer cannot be
        carried through honestly -- at ``top_n > 1`` the values are
        summed, and summing someone else's matrix on their behalf would
        be a guess -- so layers are discarded rather than silently
        desynchronised from ``.X``."""
        adata = make_adata(
            ["ACDEF", "EFGHI"],
            ["P1", "P1"],
            [[10.0, 99.0]],
        )
        adata.layers["raw"] = np.array([[1.0, 2.0]])
        out = summarize(adata, _FASTA, inplace=False)
        assert not out.layers

    def test_sparse_input_yields_sparse_output(self):
        adata = make_adata(
            ["ACDEF", "EFGHI"],
            ["P1", "P1"],
            [[10.0, 99.0]],
        )
        adata.X = sparse.csr_matrix(adata.X)
        out = summarize(adata, _FASTA, inplace=False)
        assert sparse.issparse(out.X)
        np.testing.assert_array_equal(
            out.X.toarray(),
            np.array([[99.0]]),
        )

    def test_output_passes_check_proteodata(self):
        adata = make_adata(
            ["ACDEF", "EFGHI", "RSTVWY"],
            ["P1", "P1", "P2"],
            [[10.0, 99.0, 5.0]],
        )
        out = summarize(adata, _FASTA, inplace=False)
        assert check_proteodata(out)[0]

    def test_peptide_id_column_matches_var_names(self):
        adata = make_adata(
            ["ACDEF", "EFGHI"],
            ["P1", "P1"],
            [[10.0, 99.0]],
        )
        out = summarize(adata, _FASTA, inplace=False)
        assert out.var["peptide_id"].tolist() == out.var_names.tolist()

    def test_var_index_is_unnamed(self):
        """A named var index turns ``.reset_index()`` into a
        differently-named column, which breaks the COPF tools
        downstream. Caught only by running the pipeline, so pinned
        here."""
        adata = make_adata(
            ["ACDEF", "EFGHI"],
            ["P1", "P1"],
            [[10.0, 99.0]],
        )
        out = summarize(adata, _FASTA, inplace=False)
        assert out.var.index.name is None

    def test_obs_is_preserved(self):
        adata = make_adata(
            ["ACDEF", "EFGHI"],
            ["P1", "P1"],
            [[10.0, 99.0], [1.0, 2.0]],
        )
        out = summarize(adata, _FASTA, inplace=False)
        assert out.obs_names.tolist() == ["s1", "s2"]

    def test_empty_input_is_handled(self):
        adata = make_adata([], [], np.empty((1, 0)))
        out = summarize(adata, _FASTA, inplace=False)
        assert out.n_vars == 0

    def test_verbose_reports_the_peptide_reduction(self, capsys):
        adata = make_adata(
            ["ACDEF", "EFGHI"],
            ["P1", "P1"],
            [[10.0, 99.0]],
        )
        summarize(adata, _FASTA, verbose=True, inplace=False)
        assert "2" in capsys.readouterr().out
