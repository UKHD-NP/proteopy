"""Tests for :func:`proteopy.read.long`.

The tests cover the public ``long()`` API end-to-end. Private helpers
in ``proteopy/read/long.py`` are exercised indirectly.
"""

import numpy as np
import pandas as pd
import pytest

from proteopy.read import long


# ------------------------------------------------------------------
# Helper constructors
# ------------------------------------------------------------------

def _make_peptide_intensities(
    sample_ids=None,
    peptide_ids=None,
    protein_ids=None,
    intensities=None,
    include_protein_id=True,
):
    """Build a minimal long-form peptide-level intensities DataFrame.

    Defaults produce 2 samples x 2 peptides (all from one protein),
    with distinct, asymmetric intensity values so that pivot/alignment
    bugs are catchable.

    Parameters
    ----------
    sample_ids, peptide_ids, protein_ids : list[str] | None
        Long-format row values. Lists must be the same length.
    intensities : list[float] | None
        Long-format intensity values, same length as the id lists.
    include_protein_id : bool
        When False, the ``protein_id`` column is omitted from the
        returned DataFrame (used to test var-annotation resolution).
    """
    if sample_ids is None:
        sample_ids = ["s1", "s1", "s2", "s2"]
    if peptide_ids is None:
        peptide_ids = ["PEP1", "PEP2", "PEP1", "PEP2"]
    if protein_ids is None:
        protein_ids = ["PROT1", "PROT1", "PROT1", "PROT1"]
    if intensities is None:
        intensities = [1.0, 2.0, 3.0, 4.0]

    data = {
        "sample_id": sample_ids,
        "peptide_id": peptide_ids,
        "intensity": intensities,
    }
    if include_protein_id:
        data["protein_id"] = protein_ids
    return pd.DataFrame(data)


def _make_protein_intensities(
    sample_ids=None,
    protein_ids=None,
    intensities=None,
):
    """Build a minimal long-form protein-level intensities DataFrame.

    Defaults produce 2 samples x 2 proteins with distinct values.
    """
    if sample_ids is None:
        sample_ids = ["s1", "s1", "s2", "s2"]
    if protein_ids is None:
        protein_ids = ["PROT1", "PROT2", "PROT1", "PROT2"]
    if intensities is None:
        intensities = [1.0, 2.0, 3.0, 4.0]
    return pd.DataFrame({
        "sample_id": sample_ids,
        "protein_id": protein_ids,
        "intensity": intensities,
    })


def _make_sample_annotation(sample_ids, extra=None):
    """Build a sample annotation DataFrame with an optional extra col."""
    data = {"sample_id": list(sample_ids)}
    if extra:
        data.update(extra)
    return pd.DataFrame(data)


def _make_peptide_annotation(peptide_ids, protein_ids=None, extra=None):
    """Build a peptide annotation DataFrame."""
    data = {"peptide_id": list(peptide_ids)}
    if protein_ids is not None:
        data["protein_id"] = list(protein_ids)
    if extra:
        data.update(extra)
    return pd.DataFrame(data)


def _make_protein_annotation(protein_ids, extra=None):
    """Build a protein annotation DataFrame."""
    data = {"protein_id": list(protein_ids)}
    if extra:
        data.update(extra)
    return pd.DataFrame(data)


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

class TestLong:
    """Comprehensive tests for :func:`proteopy.read.long`."""

    # -- Argument validation ------------------------------------------

    def test_level_required(self):
        """A missing ``level`` argument is a hard error."""
        df = _make_peptide_intensities()
        with pytest.raises(ValueError, match="level is required"):
            long(df)

    def test_level_invalid_value(self):
        """``level`` must be one of peptide/protein."""
        df = _make_peptide_intensities()
        with pytest.raises(ValueError, match="level must be one of"):
            long(df, level="gene")

    def test_fill_na_and_zero_to_na_mutually_exclusive(self):
        """``fill_na`` and ``zero_to_na`` cannot both be set."""
        df = _make_peptide_intensities()
        with pytest.raises(ValueError, match="mutually exclusive"):
            long(df, level="peptide", fill_na=0.0, zero_to_na=True)

    def test_column_map_invalid_key_for_protein_level(self):
        """``peptide_id`` is not a valid key at protein level."""
        df = _make_protein_intensities()
        with pytest.raises(
            ValueError,
            match="not supported at protein level",
        ):
            long(
                df,
                level="protein",
                column_map={"peptide_id": "seq"},
            )

    def test_empty_intensities_raises(self):
        """An empty intensities table fails validation."""
        with pytest.raises(
            ValueError,
            match="Intensities DataFrame is empty",
        ):
            long(pd.DataFrame(), level="protein")

    def test_peptide_level_missing_protein_id_everywhere(self):
        """Peptide level without any protein_id source raises."""
        df = _make_peptide_intensities(include_protein_id=False)
        with pytest.raises(ValueError, match="protein_id"):
            long(df, level="peptide")

    def test_peptide_conflicting_protein_mapping(self):
        """Same peptide mapped to multiple proteins raises."""
        df = _make_peptide_intensities(
            sample_ids=["s1", "s2"],
            peptide_ids=["PEP1", "PEP1"],
            protein_ids=["PROT1", "PROT2"],
            intensities=[1.0, 2.0],
        )
        with pytest.raises(ValueError, match="exactly one"):
            long(df, level="peptide")

    def test_duplicate_sample_peptide_rows_raise(self):
        """Duplicate (sample, peptide) rows are not allowed."""
        df = _make_peptide_intensities(
            sample_ids=["s1", "s1", "s2", "s2"],
            peptide_ids=["PEP1", "PEP1", "PEP1", "PEP2"],
            protein_ids=["PROT1"] * 4,
            intensities=[1.0, 2.0, 3.0, 4.0],
        )
        with pytest.raises(ValueError, match="duplicate"):
            long(df, level="peptide")

    # -- Peptide-level happy paths ------------------------------------

    def test_peptide_minimal_with_protein_in_intensities(self):
        """Basic case: intensities carry ``protein_id`` directly."""
        df = _make_peptide_intensities()
        adata = long(df, level="peptide")

        assert adata.shape == (2, 2)
        assert list(adata.obs_names) == ["s1", "s2"]
        assert list(adata.var_names) == ["PEP1", "PEP2"]
        assert list(adata.obs["sample_id"]) == ["s1", "s2"]
        assert list(adata.var["peptide_id"]) == ["PEP1", "PEP2"]
        assert list(adata.var["protein_id"]) == ["PROT1", "PROT1"]
        np.testing.assert_allclose(
            adata.X,
            np.array([[1.0, 2.0], [3.0, 4.0]]),
        )

    def test_peptide_protein_id_resolved_from_var_annotation(self):
        """``protein_id`` is looked up from ``var_annotation``."""
        df = _make_peptide_intensities(include_protein_id=False)
        var_ann = _make_peptide_annotation(
            ["PEP1", "PEP2"],
            protein_ids=["PROT_X", "PROT_Y"],
        )
        adata = long(df, level="peptide", var_annotation=var_ann)

        assert list(adata.var_names) == ["PEP1", "PEP2"]
        assert list(adata.var["protein_id"]) == ["PROT_X", "PROT_Y"]

    def test_peptide_intensities_protein_id_takes_precedence(self):
        """Intensities ``protein_id`` wins over annotation."""
        df = _make_peptide_intensities()  # proteins all "PROT1"
        var_ann = _make_peptide_annotation(
            ["PEP1", "PEP2"],
            protein_ids=["FROM_ANN", "FROM_ANN"],
        )
        adata = long(df, level="peptide", var_annotation=var_ann)

        # The intensities value must win.
        assert list(adata.var["protein_id"]) == ["PROT1", "PROT1"]

    def test_peptide_unresolved_peptide_raises(self):
        """Annotation missing a peptide -> ValueError."""
        df = _make_peptide_intensities(include_protein_id=False)
        var_ann = _make_peptide_annotation(
            ["PEP1"],  # PEP2 absent
            protein_ids=["PROT1"],
        )
        with pytest.raises(ValueError, match="could not be mapped"):
            long(df, level="peptide", var_annotation=var_ann)

    # -- Protein-level happy paths ------------------------------------

    def test_protein_minimal(self):
        """Basic protein-level read produces a valid AnnData."""
        df = _make_protein_intensities()
        adata = long(df, level="protein")

        assert adata.shape == (2, 2)
        assert list(adata.obs_names) == ["s1", "s2"]
        assert list(adata.var_names) == ["PROT1", "PROT2"]
        assert list(adata.obs["sample_id"]) == ["s1", "s2"]
        assert list(adata.var["protein_id"]) == ["PROT1", "PROT2"]
        assert "peptide_id" not in adata.var.columns
        np.testing.assert_allclose(
            adata.X,
            np.array([[1.0, 2.0], [3.0, 4.0]]),
        )

    def test_protein_with_annotation(self):
        """Extra annotation columns flow into ``.var``."""
        df = _make_protein_intensities()
        ann = _make_protein_annotation(
            ["PROT1", "PROT2"],
            extra={"gene_name": ["GENE_A", "GENE_B"]},
        )
        adata = long(df, level="protein", var_annotation=ann)

        assert "gene_name" in adata.var.columns
        assert list(adata.var["gene_name"]) == ["GENE_A", "GENE_B"]

    # -- Annotation merging -------------------------------------------

    def test_sample_annotation_columns_merged(self):
        """Sample annotation columns land in ``.obs`` with order."""
        df = _make_peptide_intensities()
        sample_ann = _make_sample_annotation(
            ["s1", "s2"],
            extra={"group": ["A", "B"]},
        )
        adata = long(
            df, level="peptide", sample_annotation=sample_ann,
        )

        assert "group" in adata.obs.columns
        assert list(adata.obs.loc[["s1", "s2"], "group"]) == ["A", "B"]

    def test_peptide_annotation_columns_merged(self):
        """Peptide annotation columns land in ``.var``."""
        df = _make_peptide_intensities()
        var_ann = _make_peptide_annotation(
            ["PEP1", "PEP2"],
            extra={"charge": [2, 3]},
        )
        adata = long(df, level="peptide", var_annotation=var_ann)

        assert "charge" in adata.var.columns
        assert list(adata.var.loc[["PEP1", "PEP2"], "charge"]) == [2, 3]

    def test_duplicate_sample_annotation_warns_and_dedupes(self):
        """Duplicate annotation rows warn and keep the first."""
        df = _make_peptide_intensities()
        sample_ann = pd.DataFrame({
            "sample_id": ["s1", "s1", "s2"],
            "group": ["FIRST", "SECOND", "OTHER"],
        })
        with pytest.warns(UserWarning, match="Duplicate sample"):
            adata = long(
                df, level="peptide", sample_annotation=sample_ann,
            )

        # First occurrence kept.
        assert adata.obs.loc["s1", "group"] == "FIRST"
        assert adata.obs.loc["s2", "group"] == "OTHER"

    def test_duplicate_peptide_annotation_warns_and_dedupes(self):
        """Duplicate peptide annotation rows warn and keep first."""
        df = _make_peptide_intensities()
        var_ann = pd.DataFrame({
            "peptide_id": ["PEP1", "PEP1", "PEP2"],
            "charge": [2, 3, 4],
        })
        with pytest.warns(UserWarning, match="Duplicate peptide"):
            adata = long(df, level="peptide", var_annotation=var_ann)

        assert adata.var.loc["PEP1", "charge"] == 2
        assert adata.var.loc["PEP2", "charge"] == 4

    # -- column_map remapping -----------------------------------------

    def test_column_map_remaps_peptide_level(self):
        """Non-standard peptide columns are canonicalized."""
        df = pd.DataFrame({
            "run": ["s1", "s1", "s2", "s2"],
            "seq": ["PEP1", "PEP2", "PEP1", "PEP2"],
            "prot": ["PROT1", "PROT1", "PROT1", "PROT1"],
            "quant": [1.0, 2.0, 3.0, 4.0],
        })
        adata = long(
            df,
            level="peptide",
            column_map={
                "sample_id": "run",
                "peptide_id": "seq",
                "protein_id": "prot",
                "intensity": "quant",
            },
        )

        assert adata.shape == (2, 2)
        assert list(adata.obs_names) == ["s1", "s2"]
        assert list(adata.var_names) == ["PEP1", "PEP2"]
        assert list(adata.var["protein_id"]) == ["PROT1", "PROT1"]

    def test_column_map_remaps_protein_level(self):
        """Non-standard protein columns are canonicalized."""
        df = pd.DataFrame({
            "run": ["s1", "s1", "s2", "s2"],
            "prot": ["PROT1", "PROT2", "PROT1", "PROT2"],
            "quant": [1.0, 2.0, 3.0, 4.0],
        })
        adata = long(
            df,
            level="protein",
            column_map={
                "sample_id": "run",
                "protein_id": "prot",
                "intensity": "quant",
            },
        )

        assert adata.shape == (2, 2)
        assert list(adata.obs_names) == ["s1", "s2"]
        assert list(adata.var_names) == ["PROT1", "PROT2"]

    # -- Missing-value handling ---------------------------------------

    def test_fill_na_replaces_missing_intensities(self):
        """Missing (sample, peptide) pairs become ``fill_na``."""
        # Drop the (s2, PEP2) row to create a missing entry.
        df = _make_peptide_intensities(
            sample_ids=["s1", "s1", "s2"],
            peptide_ids=["PEP1", "PEP2", "PEP1"],
            protein_ids=["PROT1", "PROT1", "PROT1"],
            intensities=[1.0, 2.0, 3.0],
        )
        adata = long(df, level="peptide", fill_na=0.0)

        np.testing.assert_allclose(
            adata.X,
            np.array([[1.0, 2.0], [3.0, 0.0]]),
        )
        assert not np.isnan(adata.X).any()

    def test_zero_to_na_converts_zeros(self):
        """Explicit zero intensities are converted to NaN in ``.X``."""
        df = _make_peptide_intensities(
            intensities=[0.0, 2.0, 3.0, 4.0],
        )
        adata = long(df, level="peptide", zero_to_na=True)

        expected = np.array([[np.nan, 2.0], [3.0, 4.0]])
        # Compare NaN-aware.
        np.testing.assert_array_equal(
            np.isnan(adata.X), np.isnan(expected),
        )
        np.testing.assert_allclose(
            adata.X[~np.isnan(adata.X)],
            expected[~np.isnan(expected)],
        )

    # -- Observation ordering -----------------------------------------

    def test_obs_order_default_follows_pivot_sort(self):
        """Without sorting, obs order is the pivot's sort order."""
        # Intentionally present rows in non-alphabetical order.
        df = _make_peptide_intensities(
            sample_ids=["s3", "s3", "s1", "s1", "s2", "s2"],
            peptide_ids=["PEP1", "PEP2"] * 3,
            protein_ids=["PROT1"] * 6,
            intensities=[7.0, 8.0, 1.0, 2.0, 3.0, 4.0],
        )
        adata = long(df, level="peptide")

        # pivot().sort_index() -> alphabetical observation order.
        assert list(adata.obs_names) == ["s1", "s2", "s3"]

    def test_sort_obs_by_annotation_uses_annotation_order(self):
        """Annotation order wins when ``sort_obs_by_annotation``."""
        df = _make_peptide_intensities()
        # Annotation lists s2 before s1.
        sample_ann = _make_sample_annotation(["s2", "s1"])

        adata = long(
            df,
            level="peptide",
            sample_annotation=sample_ann,
            sort_obs_by_annotation=True,
        )
        assert list(adata.obs_names) == ["s2", "s1"]

    def test_sort_obs_annotation_extra_samples_are_ignored(self):
        """Annotation entries not in intensities are dropped."""
        df = _make_peptide_intensities()
        sample_ann = _make_sample_annotation(
            ["s_missing", "s2", "s1"],
        )

        adata = long(
            df,
            level="peptide",
            sample_annotation=sample_ann,
            sort_obs_by_annotation=True,
        )
        # Only the intersection, in annotation order.
        assert list(adata.obs_names) == ["s2", "s1"]

    def test_sort_obs_intensity_samples_missing_from_annotation_come_last(
        self,
    ):
        """Samples absent from annotation appear after annotated."""
        df = _make_peptide_intensities(
            sample_ids=["s1", "s1", "s2", "s2", "s3", "s3"],
            peptide_ids=["PEP1", "PEP2"] * 3,
            protein_ids=["PROT1"] * 6,
            intensities=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
        # Annotation only lists s2 -> s1, s3 are unannotated.
        sample_ann = _make_sample_annotation(["s2"])

        adata = long(
            df,
            level="peptide",
            sample_annotation=sample_ann,
            sort_obs_by_annotation=True,
        )
        # s2 first (from annotation), then remaining from pivot order.
        assert list(adata.obs_names) == ["s2", "s1", "s3"]
