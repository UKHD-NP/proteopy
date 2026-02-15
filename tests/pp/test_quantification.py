import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse

from proteopy.pp.quantification import summarize_modifications
from proteopy.utils.anndata import check_proteodata


# ------------------------------------------------------------------
# Helper constructors
# ------------------------------------------------------------------

def _make_peptide_adata(
    X=None,
    peptide_ids=None,
    protein_ids=None,
    obs_names=None,
    extra_var_cols=None,
):
    """
    Build a minimal valid peptide-level AnnData.

    Defaults produce two observations and four peptides
    belonging to two stripped groups:

        PEPTIDEA          -> stripped "PEPTIDEA"
        PEPTIDEA (Oxidation)  -> stripped "PEPTIDEA"
        PEPTIDEB          -> stripped "PEPTIDEB"
        PEPTIDEB (Phospho)    -> stripped "PEPTIDEB"

    Intensities (2 obs x 4 vars):
        [[1, 2, 3, 4],
         [5, 6, 7, 8]]
    """
    if peptide_ids is None:
        peptide_ids = [
            "PEPTIDEA",
            "PEPTIDEA (Oxidation)",
            "PEPTIDEB",
            "PEPTIDEB (Phospho)",
        ]
    if protein_ids is None:
        protein_ids = ["P1"] * 2 + ["P2"] * 2
    if obs_names is None:
        obs_names = ["s1", "s2"]
    if X is None:
        X = np.array(
            [[1, 2, 3, 4],
             [5, 6, 7, 8]],
            dtype=float,
        )

    var = pd.DataFrame(
        {"peptide_id": peptide_ids, "protein_id": protein_ids},
        index=peptide_ids,
    )
    if extra_var_cols:
        for col, vals in extra_var_cols.items():
            var[col] = vals

    obs = pd.DataFrame(
        {"sample_id": obs_names},
        index=obs_names,
    )
    return AnnData(X=X, obs=obs, var=var)


def _make_single_group_adata():
    """
    Three peptidoforms that all strip to 'SEQA':
        SEQA
        SEQA (Ox)
        SEQA (Ph)
    Intensities: [[10, 20, 30]]
    """
    pids = ["SEQA", "SEQA (Ox)", "SEQA (Ph)"]
    X = np.array([[10.0, 20.0, 30.0]])
    return _make_peptide_adata(
        X=X,
        peptide_ids=pids,
        protein_ids=["P1", "P1", "P1"],
        obs_names=["s1"],
    )


def _make_protein_level_adata():
    """Protein-level AnnData (no peptide_id column)."""
    var_names = ["P1", "P2"]
    var = pd.DataFrame(
        {"protein_id": var_names},
        index=var_names,
    )
    obs = pd.DataFrame(
        {"sample_id": ["s1"]},
        index=["s1"],
    )
    return AnnData(
        X=np.array([[1.0, 2.0]]),
        obs=obs,
        var=var,
    )


class TestSummarizeModifications:
    """Tests for summarize_modifications."""

    # --------------------------------------------------------------
    # Basic grouping
    # --------------------------------------------------------------

    def test_basic_grouping_strips_modifications(self):
        """Peptides with modification annotations are grouped
        by their stripped (bare) sequence."""
        adata = _make_peptide_adata()
        result = summarize_modifications(adata, inplace=False)

        assert list(result.var_names) == [
            "PEPTIDEA", "PEPTIDEB",
        ]
        assert result.shape == (2, 2)

    def test_unmodified_peptide_passes_through(self):
        """A peptide without modifications remains unchanged
        after grouping."""
        pids = ["SOLO"]
        X = np.array([[42.0]])
        adata = _make_peptide_adata(
            X=X,
            peptide_ids=pids,
            protein_ids=["P1"],
            obs_names=["s1"],
        )
        result = summarize_modifications(
            adata, inplace=False,
        )

        assert list(result.var_names) == ["SOLO"]
        np.testing.assert_array_equal(result.X, [[42.0]])

    # --------------------------------------------------------------
    # Aggregation methods
    # --------------------------------------------------------------

    def test_aggregation_methods(self):
        """All four methods produce correct aggregated
        values."""
        adata = _make_peptide_adata()
        expected = {
            "sum": np.array(
                [[3.0, 7.0], [11.0, 15.0]],
            ),
            "mean": np.array(
                [[1.5, 3.5], [5.5, 7.5]],
            ),
            "median": np.array(
                [[1.5, 3.5], [5.5, 7.5]],
            ),
            "max": np.array(
                [[2.0, 4.0], [6.0, 8.0]],
            ),
        }
        for method, exp in expected.items():
            result = summarize_modifications(
                adata, method=method, inplace=False,
            )
            np.testing.assert_allclose(
                result.X, exp,
                err_msg=(
                    f"method='{method}' produced "
                    f"wrong values"
                ),
            )

    def test_aggregation_methods_single_group(self):
        """Methods applied to a single group of three
        peptidoforms."""
        adata = _make_single_group_adata()
        expected = {
            "sum": np.array([[60.0]]),
            "mean": np.array([[20.0]]),
            "median": np.array([[20.0]]),
            "max": np.array([[30.0]]),
        }
        for method, exp in expected.items():
            result = summarize_modifications(
                adata, method=method, inplace=False,
            )
            np.testing.assert_allclose(
                result.X, exp,
                err_msg=(
                    f"method='{method}' on single group"
                ),
            )

    # --------------------------------------------------------------
    # inplace behaviour
    # --------------------------------------------------------------

    def test_inplace_true_modifies_original(self):
        """inplace=True modifies the original AnnData and
        returns None."""
        adata = _make_peptide_adata()
        returned = summarize_modifications(
            adata, inplace=True,
        )

        assert returned is None
        assert adata.shape == (2, 2)
        assert list(adata.var_names) == [
            "PEPTIDEA", "PEPTIDEB",
        ]

    def test_inplace_false_returns_copy(self):
        """inplace=False returns a new AnnData; original is
        unchanged."""
        adata = _make_peptide_adata()
        original_var_names = list(adata.var_names)
        original_shape = adata.shape

        result = summarize_modifications(
            adata, inplace=False,
        )

        assert result is not adata
        assert list(adata.var_names) == original_var_names
        assert adata.shape == original_shape
        assert result.shape == (2, 2)

    # --------------------------------------------------------------
    # skip_na
    # --------------------------------------------------------------

    def test_skip_na_true_ignores_nan(self):
        """skip_na=True aggregates over non-NaN values
        only."""
        n = np.nan
        X = np.array([[1.0, n, 3.0, 4.0]])
        adata = _make_peptide_adata(X=X, obs_names=["s1"])

        result = summarize_modifications(
            adata, method="sum",
            skip_na=True, inplace=False,
        )
        np.testing.assert_allclose(
            result.X, [[1.0, 7.0]],
        )

    def test_skip_na_false_propagates_nan(self):
        """skip_na=False produces NaN when any group member
        is NaN."""
        n = np.nan
        X = np.array([[1.0, n, 3.0, 4.0]])
        adata = _make_peptide_adata(X=X, obs_names=["s1"])

        result = summarize_modifications(
            adata, method="sum",
            skip_na=False, inplace=False,
        )
        assert np.isnan(result.X[0, 0])
        np.testing.assert_allclose(result.X[0, 1], 7.0)

    def test_skip_na_false_with_mean(self):
        """skip_na=False propagates NaN for mean aggregation
        too."""
        n = np.nan
        X = np.array([[10.0, n, 3.0, 4.0]])
        adata = _make_peptide_adata(X=X, obs_names=["s1"])

        result = summarize_modifications(
            adata, method="mean",
            skip_na=False, inplace=False,
        )
        assert np.isnan(result.X[0, 0])
        np.testing.assert_allclose(result.X[0, 1], 3.5)

    # --------------------------------------------------------------
    # sort
    # --------------------------------------------------------------

    def test_sort_true_alphabetical_order(self):
        """sort=True orders output variables
        alphabetically."""
        pids = [
            "ZZZ", "ZZZ (Ox)",
            "AAA", "AAA (Ph)",
        ]
        X = np.array([[1.0, 2.0, 3.0, 4.0]])
        adata = _make_peptide_adata(
            X=X,
            peptide_ids=pids,
            protein_ids=["P1", "P1", "P2", "P2"],
            obs_names=["s1"],
        )
        result = summarize_modifications(
            adata, sort=True, inplace=False,
        )
        assert list(result.var_names) == ["AAA", "ZZZ"]

    def test_sort_false_preserves_first_appearance_order(
        self,
    ):
        """sort=False preserves the order of first
        appearance."""
        pids = [
            "ZZZ", "ZZZ (Ox)",
            "AAA", "AAA (Ph)",
        ]
        X = np.array([[1.0, 2.0, 3.0, 4.0]])
        adata = _make_peptide_adata(
            X=X,
            peptide_ids=pids,
            protein_ids=["P1", "P1", "P2", "P2"],
            obs_names=["s1"],
        )
        result = summarize_modifications(
            adata, sort=False, inplace=False,
        )
        assert list(result.var_names) == ["ZZZ", "AAA"]

    # --------------------------------------------------------------
    # keep_var_cols
    # --------------------------------------------------------------

    def test_keep_var_cols_none_has_default_columns_only(
        self,
    ):
        """With keep_var_cols=None, output .var has only the
        mandatory columns."""
        adata = _make_peptide_adata(
            extra_var_cols={
                "gene": ["G1", "G1", "G2", "G2"],
            },
        )
        result = summarize_modifications(
            adata, keep_var_cols=None, inplace=False,
        )
        expected_cols = {
            "peptide_id", "protein_id",
            "n_peptidoforms", "n_modifications",
        }
        assert set(result.var.columns) == expected_cols

    def test_keep_var_cols_carries_over_extra_columns(self):
        """Specifying keep_var_cols includes those columns in
        the output .var."""
        adata = _make_peptide_adata(
            extra_var_cols={
                "gene": ["G1", "G1", "G2", "G2"],
            },
        )
        result = summarize_modifications(
            adata, keep_var_cols=["gene"], inplace=False,
        )
        assert "gene" in result.var.columns
        assert result.var.loc["PEPTIDEA", "gene"] == "G1"
        assert result.var.loc["PEPTIDEB", "gene"] == "G2"

    def test_keep_var_cols_joins_differing_values(self):
        """When group members have different values for a
        kept column, they are joined with ';'."""
        adata = _make_peptide_adata(
            extra_var_cols={
                "source": ["db1", "db2", "db1", "db1"],
            },
        )
        result = summarize_modifications(
            adata, keep_var_cols=["source"], inplace=False,
        )
        assert (
            result.var.loc["PEPTIDEA", "source"] == "db1;db2"
        )
        assert (
            result.var.loc["PEPTIDEB", "source"] == "db1"
        )

    def test_keep_var_cols_with_nan_values(self):
        """NaN entries in a kept column are dropped before
        aggregation."""
        adata = _make_peptide_adata(
            extra_var_cols={
                "note": ["x", np.nan, "y", np.nan],
            },
        )
        result = summarize_modifications(
            adata, keep_var_cols=["note"], inplace=False,
        )
        assert result.var.loc["PEPTIDEA", "note"] == "x"
        assert result.var.loc["PEPTIDEB", "note"] == "y"

    def test_keep_var_cols_all_nan_produces_nan(self):
        """When all values in a kept column are NaN for a
        group, the result is NaN."""
        adata = _make_peptide_adata(
            extra_var_cols={
                "note": [np.nan, np.nan, "y", np.nan],
            },
        )
        result = summarize_modifications(
            adata, keep_var_cols=["note"], inplace=False,
        )
        assert pd.isna(
            result.var.loc["PEPTIDEA", "note"],
        )
        assert result.var.loc["PEPTIDEB", "note"] == "y"

    # --------------------------------------------------------------
    # zero_to_na
    # --------------------------------------------------------------

    def test_zero_to_na_converts_zeros_before_aggregation(
        self,
    ):
        """Zeros are replaced with NaN before aggregation
        when zero_to_na=True."""
        X = np.array([[0.0, 2.0, 3.0, 4.0]])
        adata = _make_peptide_adata(X=X, obs_names=["s1"])

        result = summarize_modifications(
            adata, method="sum",
            zero_to_na=True, skip_na=True,
            inplace=False,
        )
        np.testing.assert_allclose(
            result.X, [[2.0, 7.0]],
        )

    def test_zero_to_na_with_skip_na_false(self):
        """zero_to_na=True combined with skip_na=False
        propagates NaN from zeros."""
        X = np.array([[0.0, 2.0, 3.0, 4.0]])
        adata = _make_peptide_adata(X=X, obs_names=["s1"])

        result = summarize_modifications(
            adata, method="sum",
            zero_to_na=True, skip_na=False,
            inplace=False,
        )
        assert np.isnan(result.X[0, 0])
        np.testing.assert_allclose(result.X[0, 1], 7.0)

    # --------------------------------------------------------------
    # fill_na
    # --------------------------------------------------------------

    def test_fill_na_replaces_nan_before_aggregation(self):
        """NaN values are filled before aggregation."""
        n = np.nan
        X = np.array([[n, 2.0, 3.0, 4.0]])
        adata = _make_peptide_adata(X=X, obs_names=["s1"])

        result = summarize_modifications(
            adata, method="sum",
            fill_na=0.0, inplace=False,
        )
        np.testing.assert_allclose(
            result.X, [[2.0, 7.0]],
        )

    def test_fill_na_with_nonzero_value(self):
        """fill_na works with an arbitrary constant."""
        n = np.nan
        X = np.array([[n, n, 3.0, 4.0]])
        adata = _make_peptide_adata(X=X, obs_names=["s1"])

        result = summarize_modifications(
            adata, method="sum",
            fill_na=100.0, inplace=False,
        )
        np.testing.assert_allclose(
            result.X[0, 0], 200.0,
        )

    # --------------------------------------------------------------
    # layer
    # --------------------------------------------------------------

    def test_layer_uses_specified_layer(self):
        """When layer is specified, data comes from that
        layer instead of .X."""
        adata = _make_peptide_adata()
        layer_data = np.array(
            [[10.0, 20.0, 30.0, 40.0],
             [50.0, 60.0, 70.0, 80.0]],
        )
        adata.layers["raw"] = layer_data

        result = summarize_modifications(
            adata, layer="raw",
            method="sum", inplace=False,
        )
        np.testing.assert_allclose(
            result.X,
            [[30.0, 70.0], [110.0, 150.0]],
        )

    # --------------------------------------------------------------
    # verbose
    # --------------------------------------------------------------

    def test_verbose_prints_message(self, capsys):
        """verbose=True prints a status message to
        stdout."""
        adata = _make_peptide_adata()
        summarize_modifications(
            adata, verbose=True, inplace=False,
        )
        captured = capsys.readouterr().out
        assert "Stripping modifications" in captured
        assert "4 peptides" in captured
        assert "2 unique stripped sequences" in captured

    def test_verbose_false_prints_nothing(self, capsys):
        """verbose=False produces no stdout output."""
        adata = _make_peptide_adata()
        summarize_modifications(
            adata, verbose=False, inplace=False,
        )
        captured = capsys.readouterr().out
        assert captured == ""

    # --------------------------------------------------------------
    # Sparse matrix preservation
    # --------------------------------------------------------------

    def test_sparse_input_produces_sparse_output(self):
        """A sparse .X matrix stays sparse in the output."""
        adata = _make_peptide_adata()
        adata.X = sparse.csr_matrix(adata.X)
        assert sparse.issparse(adata.X)

        result = summarize_modifications(
            adata, method="sum", inplace=False,
        )
        assert sparse.issparse(result.X)

    def test_sparse_output_has_correct_values(self):
        """Sparse output matches the expected dense
        values."""
        adata = _make_peptide_adata()
        adata.X = sparse.csr_matrix(adata.X)

        result = summarize_modifications(
            adata, method="sum", inplace=False,
        )
        expected = np.array(
            [[3.0, 7.0], [11.0, 15.0]],
        )
        np.testing.assert_allclose(
            result.X.toarray(), expected,
        )

    def test_sparse_inplace_preserves_sparsity(self):
        """inplace=True on sparse input keeps .X sparse."""
        adata = _make_peptide_adata()
        adata.X = sparse.csr_matrix(adata.X)

        summarize_modifications(adata, inplace=True)
        assert sparse.issparse(adata.X)

    def test_dense_input_stays_dense(self):
        """A dense .X matrix remains dense in the output."""
        adata = _make_peptide_adata()
        assert not sparse.issparse(adata.X)

        result = summarize_modifications(
            adata, method="sum", inplace=False,
        )
        assert not sparse.issparse(result.X)

    # --------------------------------------------------------------
    # n_peptidoforms
    # --------------------------------------------------------------

    def test_n_peptidoforms_counts_variants(self):
        """n_peptidoforms reflects total variants per
        stripped sequence group."""
        adata = _make_peptide_adata()
        result = summarize_modifications(
            adata, inplace=False,
        )

        val_a = result.var.loc[
            "PEPTIDEA", "n_peptidoforms"
        ]
        assert val_a == 2
        val_b = result.var.loc[
            "PEPTIDEB", "n_peptidoforms"
        ]
        assert val_b == 2

    def test_n_peptidoforms_single_variant(self):
        """An unmodified peptide alone counts as 1
        peptidoform."""
        pids = ["SOLO"]
        X = np.array([[1.0]])
        adata = _make_peptide_adata(
            X=X,
            peptide_ids=pids,
            protein_ids=["P1"],
            obs_names=["s1"],
        )
        result = summarize_modifications(
            adata, inplace=False,
        )
        assert (
            result.var.loc["SOLO", "n_peptidoforms"] == 1
        )

    def test_n_peptidoforms_three_variants(self):
        """Three peptidoforms sharing the same stripped
        sequence."""
        adata = _make_single_group_adata()
        result = summarize_modifications(
            adata, inplace=False,
        )
        assert (
            result.var.loc["SEQA", "n_peptidoforms"] == 3
        )

    # --------------------------------------------------------------
    # n_modifications (position-aware counting)
    # --------------------------------------------------------------

    def test_n_modifications_basic(self):
        """Each unique (position, text) pair counts as one
        modification."""
        adata = _make_peptide_adata()
        result = summarize_modifications(
            adata, inplace=False,
        )

        val_a = result.var.loc[
            "PEPTIDEA", "n_modifications"
        ]
        assert val_a == 1
        val_b = result.var.loc[
            "PEPTIDEB", "n_modifications"
        ]
        assert val_b == 1

    def test_n_modifications_unmodified_contributes_zero(
        self,
    ):
        """A group with only unmodified peptides has 0
        modifications."""
        pids = ["BARE"]
        X = np.array([[5.0]])
        adata = _make_peptide_adata(
            X=X,
            peptide_ids=pids,
            protein_ids=["P1"],
            obs_names=["s1"],
        )
        result = summarize_modifications(
            adata, inplace=False,
        )
        assert (
            result.var.loc["BARE", "n_modifications"] == 0
        )

    def test_n_modifications_same_mod_same_pos_dedup(self):
        """Same modification at same position across
        peptidoforms counts only once."""
        pids = [
            "ABC (Ox)DEF",
            "ABC (Ox)DEF (Ph)",
        ]
        X = np.array([[1.0, 2.0]])
        adata = _make_peptide_adata(
            X=X,
            peptide_ids=pids,
            protein_ids=["P1", "P1"],
            obs_names=["s1"],
        )
        result = summarize_modifications(
            adata, inplace=False,
        )
        val = result.var.loc[
            "ABCDEF", "n_modifications"
        ]
        assert val == 2

    def test_n_modifications_same_text_different_pos(self):
        """Same modification text at different positions
        counts as distinct modifications."""
        pids = [
            "A (Ox)BCD",
            "ABCD (Ox)",
        ]
        X = np.array([[1.0, 2.0]])
        adata = _make_peptide_adata(
            X=X,
            peptide_ids=pids,
            protein_ids=["P1", "P1"],
            obs_names=["s1"],
        )
        result = summarize_modifications(
            adata, inplace=False,
        )
        assert (
            result.var.loc["ABCD", "n_modifications"] == 2
        )

    def test_n_modifications_multiple_mods_in_one_peptide(
        self,
    ):
        """A single peptide with multiple modifications
        contributes all of them."""
        pids = ["A (Ox)B (Ph)C"]
        X = np.array([[1.0]])
        adata = _make_peptide_adata(
            X=X,
            peptide_ids=pids,
            protein_ids=["P1"],
            obs_names=["s1"],
        )
        result = summarize_modifications(
            adata, inplace=False,
        )
        assert (
            result.var.loc["ABC", "n_modifications"] == 2
        )

    # --------------------------------------------------------------
    # Custom mod_regex
    # --------------------------------------------------------------

    def test_custom_mod_regex(self):
        """A custom regex strips different annotation
        formats."""
        pids = [
            "PEP[Ox]TIDE",
            "PEP[Ph]TIDE",
            "PEPTIDE",
        ]
        X = np.array([[1.0, 2.0, 3.0]])
        adata = _make_peptide_adata(
            X=X,
            peptide_ids=pids,
            protein_ids=["P1", "P1", "P1"],
            obs_names=["s1"],
        )
        result = summarize_modifications(
            adata, mod_regex=r"\[.*?\]",
            method="sum", inplace=False,
        )
        assert list(result.var_names) == ["PEPTIDE"]
        np.testing.assert_allclose(result.X, [[6.0]])
        val_pf = result.var.loc[
            "PEPTIDE", "n_peptidoforms"
        ]
        assert val_pf == 3
        val_nm = result.var.loc[
            "PEPTIDE", "n_modifications"
        ]
        assert val_nm == 2

    def test_custom_mod_regex_no_matches(self):
        """When the regex matches nothing, peptides pass
        through unchanged."""
        adata = _make_peptide_adata()
        result = summarize_modifications(
            adata, mod_regex=r"\[NOMATCH\]",
            method="sum", inplace=False,
        )
        assert result.shape[1] == 4

    # --------------------------------------------------------------
    # Error conditions
    # --------------------------------------------------------------

    def test_error_protein_level_data(self):
        """Protein-level AnnData raises ValueError."""
        adata = _make_protein_level_adata()
        with pytest.raises(
            ValueError, match="peptide-level",
        ):
            summarize_modifications(adata)

    def test_error_invalid_method(self):
        """An unsupported method string raises
        ValueError."""
        adata = _make_peptide_adata()
        with pytest.raises(
            ValueError, match="method must be one of",
        ):
            summarize_modifications(adata, method="min")

    def test_error_both_zero_to_na_and_fill_na(self):
        """Setting both zero_to_na and fill_na raises
        ValueError."""
        adata = _make_peptide_adata()
        with pytest.raises(
            ValueError,
            match="Cannot set both zero_to_na and fill_na",
        ):
            summarize_modifications(
                adata, zero_to_na=True, fill_na=0.0,
            )

    def test_error_conflicting_protein_ids(self):
        """Peptides that strip to the same sequence but map
        to different protein_ids raise ValueError."""
        pids = ["SHARED", "SHARED (Ox)"]
        X = np.array([[1.0, 2.0]])
        adata = _make_peptide_adata(
            X=X,
            peptide_ids=pids,
            protein_ids=["P1", "P2"],
            obs_names=["s1"],
        )
        with pytest.raises(
            ValueError, match="multiple protein_ids",
        ):
            summarize_modifications(
                adata, inplace=False,
            )

    def test_error_keep_var_cols_missing_column(self):
        """keep_var_cols with a column not in adata.var
        raises KeyError."""
        adata = _make_peptide_adata()
        with pytest.raises(
            KeyError, match="not found in adata.var",
        ):
            summarize_modifications(
                adata, keep_var_cols=["nonexistent"],
            )

    def test_error_keep_var_cols_multiple_missing(self):
        """All missing keep_var_cols entries are
        reported."""
        adata = _make_peptide_adata()
        with pytest.raises(
            KeyError, match="nonexistent",
        ):
            summarize_modifications(
                adata,
                keep_var_cols=[
                    "nonexistent", "also_bad",
                ],
            )

    def test_error_keep_var_cols_reserved_peptide_id(self):
        """keep_var_cols containing 'peptide_id' raises
        ValueError."""
        adata = _make_peptide_adata()
        with pytest.raises(
            ValueError, match="reserved columns",
        ):
            summarize_modifications(
                adata, keep_var_cols=["peptide_id"],
            )

    def test_error_keep_var_cols_reserved_protein_id(self):
        """keep_var_cols containing 'protein_id' raises
        ValueError."""
        adata = _make_peptide_adata()
        with pytest.raises(
            ValueError, match="reserved columns",
        ):
            summarize_modifications(
                adata, keep_var_cols=["protein_id"],
            )

    def test_error_keep_var_cols_reserved_n_peptidoforms(
        self,
    ):
        """keep_var_cols containing 'n_peptidoforms' raises
        ValueError."""
        adata = _make_peptide_adata(
            extra_var_cols={
                "n_peptidoforms": [1, 1, 1, 1],
            },
        )
        with pytest.raises(
            ValueError, match="reserved columns",
        ):
            summarize_modifications(
                adata,
                keep_var_cols=["n_peptidoforms"],
            )

    def test_error_keep_var_cols_reserved_n_modifications(
        self,
    ):
        """keep_var_cols containing 'n_modifications' raises
        ValueError."""
        adata = _make_peptide_adata(
            extra_var_cols={
                "n_modifications": [0, 0, 0, 0],
            },
        )
        with pytest.raises(
            ValueError, match="reserved columns",
        ):
            summarize_modifications(
                adata,
                keep_var_cols=["n_modifications"],
            )

    def test_error_invalid_mod_regex(self):
        """A malformed regex raises ValueError with a
        descriptive message."""
        adata = _make_peptide_adata()
        with pytest.raises(
            ValueError, match="Invalid mod_regex",
        ):
            summarize_modifications(
                adata, mod_regex=r"(unclosed",
            )

    def test_error_layer_with_infinite_values(self):
        """A layer containing infinite values is rejected
        upfront."""
        adata = _make_peptide_adata()
        layer_data = np.array(
            [[1.0, np.inf, 3.0, 4.0],
             [5.0, 6.0, 7.0, 8.0]],
        )
        adata.layers["bad"] = layer_data
        with pytest.raises(
            ValueError, match="infinite",
        ):
            summarize_modifications(
                adata, layer="bad",
            )

    def test_var_column_named_stripped_not_clobbered(self):
        """A user .var column named '_stripped' is preserved
        when included via keep_var_cols."""
        adata = _make_peptide_adata(
            extra_var_cols={
                "_stripped": ["a", "b", "c", "d"],
            },
        )
        result = summarize_modifications(
            adata, keep_var_cols=["_stripped"],
            inplace=False,
        )
        assert "_stripped" in result.var.columns
        val_a = result.var.loc["PEPTIDEA", "_stripped"]
        assert val_a == "a;b"
        val_b = result.var.loc["PEPTIDEB", "_stripped"]
        assert val_b == "c;d"

    # --------------------------------------------------------------
    # Output validation (check_proteodata)
    # --------------------------------------------------------------

    def test_output_passes_check_proteodata_inplace(self):
        """After inplace=True, the AnnData still passes
        check_proteodata."""
        adata = _make_peptide_adata()
        summarize_modifications(adata, inplace=True)
        check_proteodata(adata)

    def test_output_passes_check_proteodata_copy(self):
        """The returned AnnData from inplace=False passes
        check_proteodata."""
        adata = _make_peptide_adata()
        result = summarize_modifications(
            adata, inplace=False,
        )
        check_proteodata(result)

    def test_output_peptide_id_matches_var_names(self):
        """Output .var['peptide_id'] matches .var_names
        exactly."""
        adata = _make_peptide_adata()
        result = summarize_modifications(
            adata, inplace=False,
        )

        np.testing.assert_array_equal(
            result.var["peptide_id"].values,
            result.var_names.values,
        )

    def test_output_protein_id_is_single_mapped(self):
        """Each output peptide maps to exactly one
        protein_id (no multi-mapping)."""
        adata = _make_peptide_adata()
        result = summarize_modifications(
            adata, inplace=False,
        )

        for pid in result.var["protein_id"]:
            assert ";" not in str(pid)
            assert "," not in str(pid)

    # --------------------------------------------------------------
    # Edge cases
    # --------------------------------------------------------------

    def test_all_nan_matrix_sum(self):
        """An all-NaN matrix produces all-NaN output with
        sum."""
        n = np.nan
        X = np.array([[n, n, n, n]])
        adata = _make_peptide_adata(
            X=X, obs_names=["s1"],
        )

        result = summarize_modifications(
            adata, method="sum", inplace=False,
        )
        assert np.all(np.isnan(result.X))

    def test_all_nan_matrix_mean(self):
        """An all-NaN matrix produces all-NaN output with
        mean."""
        n = np.nan
        X = np.array([[n, n, n, n]])
        adata = _make_peptide_adata(
            X=X, obs_names=["s1"],
        )

        result = summarize_modifications(
            adata, method="mean", inplace=False,
        )
        assert np.all(np.isnan(result.X))

    def test_all_zero_matrix_with_zero_to_na(self):
        """An all-zero matrix with zero_to_na=True produces
        all-NaN output for sum."""
        X = np.zeros((1, 4))
        adata = _make_peptide_adata(
            X=X, obs_names=["s1"],
        )

        result = summarize_modifications(
            adata, method="sum",
            zero_to_na=True, inplace=False,
        )
        assert np.all(np.isnan(result.X))

    def test_multiple_observations_independent(self):
        """Each observation is aggregated
        independently."""
        X = np.array(
            [[1.0, 2.0, 3.0, 4.0],
             [10.0, 20.0, 30.0, 40.0],
             [100.0, 200.0, 300.0, 400.0]],
        )
        adata = _make_peptide_adata(
            X=X,
            obs_names=["s1", "s2", "s3"],
        )
        result = summarize_modifications(
            adata, method="sum", inplace=False,
        )
        expected = np.array(
            [[3.0, 7.0],
             [30.0, 70.0],
             [300.0, 700.0]],
        )
        np.testing.assert_allclose(result.X, expected)

    def test_mixed_nan_across_observations(self):
        """NaN patterns can differ across
        observations."""
        n = np.nan
        X = np.array(
            [[1.0, n, 3.0, 4.0],
             [n, 6.0, 7.0, 8.0]],
        )
        adata = _make_peptide_adata(X=X)

        result = summarize_modifications(
            adata, method="sum",
            skip_na=True, inplace=False,
        )
        np.testing.assert_allclose(
            result.X,
            [[1.0, 7.0], [6.0, 15.0]],
        )

    def test_mixed_nan_skip_na_false_per_observation(self):
        """skip_na=False propagates NaN per observation
        independently."""
        n = np.nan
        X = np.array(
            [[1.0, n, 3.0, 4.0],
             [5.0, 6.0, 7.0, 8.0]],
        )
        adata = _make_peptide_adata(X=X)

        result = summarize_modifications(
            adata, method="sum",
            skip_na=False, inplace=False,
        )
        assert np.isnan(result.X[0, 0])
        np.testing.assert_allclose(result.X[0, 1], 7.0)
        np.testing.assert_allclose(result.X[1, 0], 11.0)
        np.testing.assert_allclose(result.X[1, 1], 15.0)

    def test_sparse_with_nan_values(self):
        """Sparse input with stored NaN values aggregates
        correctly and the output remains sparse."""
        n = np.nan
        X_dense = np.array([[1.0, n, 3.0, 4.0]])
        adata = _make_peptide_adata(
            X=sparse.csr_matrix(X_dense),
            obs_names=["s1"],
        )

        result = summarize_modifications(
            adata, method="sum",
            skip_na=True, inplace=False,
        )
        assert sparse.issparse(result.X)
        np.testing.assert_allclose(
            result.X.toarray(), [[1.0, 7.0]],
        )
