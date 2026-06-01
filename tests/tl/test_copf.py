"""
Unit tests for :func:`proteopy.tl.copf.pairwise_var_correlations`.

The first test (``test_pairwise_var_correlations_vs_rcopf``) is the
parity test against the rCOPF reference output and is the primary
correctness anchor for this module. Subsequent tests cover the
public contract of ``pairwise_var_correlations`` in isolation.
"""
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from pytest import approx

from proteopy.tl.copf import pairwise_var_correlations
from proteopy.utils.hash import md5_hash_list

TEST_DIR = Path(__file__).parent.parent
DATA_DIR = TEST_DIR / "data"


class TestPairwiseVarCorrelations:
    """Tests for :func:`pairwise_var_correlations`."""

    # -- rCOPF reference fixtures (re-declared locally to avoid
    # cross-module fixture coupling with tests/tl/test_copro.py)

    @pytest.fixture
    def peptide_intensities_wide(self):
        """Mouse-tissue pre-processed peptide intensities (samples x
        peptides) used as the rCOPF reference input."""
        path = DATA_DIR / "mouse_tissue" \
            / "traces_pre-processed_rcopf.tsv"
        df = pd.read_csv(path, sep="\t", header=0)
        df = df.rename(columns={"id": "peptide_id"})
        return df

    @pytest.fixture
    def peptide_protein_annotations(self):
        """Peptide-to-protein mapping for the mouse-tissue dataset."""
        path = (
            DATA_DIR / "mouse_tissue"
            / "traces_pre-processed_trace-annotations_rcopf.tsv"
        )
        annotations = pd.read_csv(path, sep="\t", header=0)
        annotations = annotations.rename(columns={"id": "peptide_id"})
        return annotations

    @pytest.fixture
    def peptide_intensities_long(
        self, peptide_intensities_wide, peptide_protein_annotations,
    ):
        """Long-format peptide intensities annotated with protein_id."""
        annotations = peptide_protein_annotations[
            ["peptide_id", "protein_id"]
        ]
        merged = peptide_intensities_wide.merge(
            annotations, on="peptide_id",
        )
        long = pd.melt(
            merged, id_vars=("protein_id", "peptide_id"),
        )
        long = long.rename(
            columns={"value": "intensity", "variable": "sample"},
        )
        return long

    @pytest.fixture
    def rcopf_correlations(self):
        """rCOPF reference pairwise peptide correlations frame."""
        path = DATA_DIR / "mouse_tissue" \
            / "traces_correlations_rcopf.tsv"
        col_names = ["pepA", "pepB", "PCC", "protein_id"]
        return pd.read_csv(path, sep="\t", names=col_names)

    @pytest.fixture
    def rcopf_correlations_upper(self, rcopf_correlations):
        """rCOPF reference correlations restricted to the upper
        triangle of unique pairs (self-pairs and duplicates removed)."""
        ref = rcopf_correlations.set_index("protein_id")
        ref = ref[ref["PCC"] != 1]

        def _sorted_pair(row):
            return tuple(sorted([row["pepA"], row["pepB"]]))

        ref = ref.copy()
        ref["sorted_pair"] = ref.apply(_sorted_pair, axis=1)
        ref = ref.drop_duplicates(subset=["sorted_pair"])
        ref = ref.drop(columns=["sorted_pair"])
        ref = ref.sort_values(["pepA", "pepB"]).sort_index()
        return ref

    # -- single-use helpers (live inside the class)

    @staticmethod
    def _build_peptide_adata_from_long(peptide_intensities_long):
        """
        Build a peptide-level AnnData from long-format intensities.

        The long-format frame must have columns: ``protein_id``,
        ``peptide_id``, ``sample``, ``intensity``. NaNs are not
        allowed in ``intensity``. Peptides whose protein has fewer
        than 3 peptides are dropped, mirroring the COPF >= 3-vars
        constraint enforced by ``pairwise_var_correlations``.
        """
        if peptide_intensities_long["intensity"].isna().any():
            raise AssertionError(
                "Input long frame contains NaNs; the parity test "
                "requires a NaN-free intensity matrix."
            )
        # Drop proteins with < 2 peptides so the parity dataset
        # passes the COPF >= 2-vars-per-group validation.
        per_protein = (
            peptide_intensities_long[["peptide_id", "protein_id"]]
            .drop_duplicates()
            .groupby("protein_id")
            .size()
        )
        eligible = per_protein[per_protein >= 2].index
        peptide_intensities_long = peptide_intensities_long[
            peptide_intensities_long["protein_id"].isin(eligible)
        ]
        wide = peptide_intensities_long.pivot(
            index="sample", columns="peptide_id", values="intensity",
        )
        peptide_to_protein = (
            peptide_intensities_long[["peptide_id", "protein_id"]]
            .drop_duplicates()
            .set_index("peptide_id")["protein_id"]
        )
        var = pd.DataFrame(
            {
                "peptide_id": wide.columns.to_numpy(),
                "protein_id": peptide_to_protein.reindex(
                    wide.columns,
                ).to_numpy(),
            },
            index=wide.columns,
        )
        obs = pd.DataFrame(
            {"sample_id": wide.index.to_numpy()},
            index=wide.index,
        )
        return ad.AnnData(
            X=wide.to_numpy(dtype=float),
            obs=obs,
            var=var,
        )

    @staticmethod
    def _make_protein_adata(
        intensities,
        *,
        var_names=None,
        obs_names=None,
        extra_obs=None,
    ):
        """Build a protein-level AnnData from a dense intensity matrix.

        ``intensities`` is a 2-D array of shape (samples, proteins).
        """
        intensities = np.asarray(intensities, dtype=float)
        n_obs, n_vars = intensities.shape
        if var_names is None:
            var_names = [f"P{i}" for i in range(n_vars)]
        if obs_names is None:
            obs_names = [f"S{i}" for i in range(n_obs)]
        var = pd.DataFrame(
            {"protein_id": list(var_names)},
            index=list(var_names),
        )
        obs_data = {"sample_id": list(obs_names)}
        if extra_obs is not None:
            for k, v in extra_obs.items():
                obs_data[k] = list(v)
        obs = pd.DataFrame(obs_data, index=list(obs_names))
        return ad.AnnData(X=intensities, obs=obs, var=var)

    @staticmethod
    def _make_peptide_adata(
        intensities,
        *,
        peptide_ids,
        protein_ids,
        obs_names=None,
        extra_obs=None,
    ):
        """Build a peptide-level AnnData from a dense intensity matrix.

        ``intensities`` is a 2-D array of shape (samples, peptides).
        """
        intensities = np.asarray(intensities, dtype=float)
        n_obs, _ = intensities.shape
        if obs_names is None:
            obs_names = [f"S{i}" for i in range(n_obs)]
        var = pd.DataFrame(
            {
                "peptide_id": list(peptide_ids),
                "protein_id": list(protein_ids),
            },
            index=list(peptide_ids),
        )
        obs_data = {"sample_id": list(obs_names)}
        if extra_obs is not None:
            for k, v in extra_obs.items():
                obs_data[k] = list(v)
        obs = pd.DataFrame(obs_data, index=list(obs_names))
        return ad.AnnData(X=intensities, obs=obs, var=var)

    # =================================================================
    # 1) PARITY TEST AGAINST rCOPF -- highest priority
    # =================================================================

    def test_pairwise_var_correlations_vs_rcopf(
        self, peptide_intensities_long, rcopf_correlations_upper,
    ):
        """
        Equality of correlations vs the rCOPF reference output.

        Builds a peptide-level AnnData from the mouse-tissue
        pre-processed peptide intensities and verifies the pairwise
        Pearson correlations match the rCOPF reference within 1e-14.
        """
        adata = self._build_peptide_adata_from_long(
            peptide_intensities_long,
        )

        # Fixture drops proteins with < 2 peptides (COPF constraint
        # enforced by pairwise_var_correlations). Confirm before
        # comparing against the reference.
        peptides_per_protein = adata.var.groupby(
            "protein_id", observed=True,
        ).size()
        assert (peptides_per_protein >= 2).all()

        result = pairwise_var_correlations(
            adata, group_by="protein_id", inplace=False,
        )
        assert result is not None
        corrs = result.uns["pairwise_correlations;protein_id;;;"]

        # Normalise to match reference shape and sort identically
        corrs = corrs.rename(
            columns={
                "group_id": "protein_id",
                "varA": "pepA",
                "varB": "pepB",
                "corr": "PCC",
            },
        )
        corrs = corrs[["protein_id", "pepA", "pepB", "PCC"]] \
            .set_index("protein_id")
        corrs = corrs.sort_values(["pepA", "pepB"]).sort_index()

        # Restrict reference to the same protein set as the filtered
        # adata so the row counts and ordering match.
        eligible_proteins = set(adata.var["protein_id"])
        ref = rcopf_correlations_upper[
            rcopf_correlations_upper.index.isin(eligible_proteins)
        ]
        ref = ref.sort_values(["pepA", "pepB"]).sort_index()

        pep_cols = ["pepA", "pepB"]
        assert corrs[pep_cols].equals(ref[pep_cols])
        assert corrs["PCC"].to_numpy() == approx(
            ref["PCC"].to_numpy(), abs=1e-14,
        )

    # =================================================================
    # 2) Classical unit tests
    # =================================================================

    # -- 2.1 Basic ungrouped protein-level case

    def test_ungrouped_default_key_and_columns(self):
        """No ``group_by``: default key is ``pairwise_correlations;;;;``."""
        rng = np.random.default_rng(42)
        intensities = rng.standard_normal((6, 3))
        adata = self._make_protein_adata(intensities)

        result = pairwise_var_correlations(adata, inplace=False)

        key = "pairwise_correlations;;;;"
        assert key in result.uns
        df = result.uns[key]
        assert list(df.columns) == ["varA", "varB", "corr", "pval"]
        assert len(df) == 3  # 3 choose 2 = 3
        assert "group_id" not in df.columns
        assert ((df["corr"] >= -1) & (df["corr"] <= 1)).all()
        assert isinstance(df.index, pd.RangeIndex)

    # -- 2.2 Grouped by .var column

    def test_group_by_changes_key_and_grouping(self):
        """``group_by`` partitions vars by a .var column."""
        rng = np.random.default_rng(7)
        intensities = rng.standard_normal((5, 4))
        adata = self._make_peptide_adata(
            intensities,
            peptide_ids=["pep1", "pep2", "pep3", "pep4"],
            protein_ids=["prot1", "prot1", "prot2", "prot2"],
        )

        result = pairwise_var_correlations(
            adata, group_by="protein_id", inplace=False,
        )

        key = "pairwise_correlations;protein_id;;;"
        assert key in result.uns
        df = result.uns[key]
        assert list(df.columns) == [
            "group_id", "varA", "varB", "corr", "pval",
        ]
        assert set(df["group_id"]) == {"prot1", "prot2"}
        # Each two-peptide protein produces exactly 1 pair
        assert (
            df.groupby("group_id").size().to_dict()
            == {"prot1": 1, "prot2": 1}
        )

    # -- 2.3 Peptide-level wrapper forwards group_by='protein_id'

    def test_pairwise_peptide_correlations_wrapper(self):
        """Wrapper fixes ``group_by='protein_id'`` for peptide adata."""
        from proteopy.tl.copf import pairwise_peptide_correlations

        rng = np.random.default_rng(0)
        intensities = rng.standard_normal((5, 4))
        adata = self._make_peptide_adata(
            intensities,
            peptide_ids=["p1", "p2", "p3", "p4"],
            protein_ids=["A", "A", "B", "B"],
        )

        result = pairwise_peptide_correlations(adata, inplace=False)

        key = "pairwise_correlations;protein_id;;;"
        assert key in result.uns
        df = result.uns[key]
        assert "group_id" in df.columns
        assert set(df["group_id"]) == {"A", "B"}

    def test_pairwise_peptide_correlations_rejects_protein_level(
        self,
    ):
        """Wrapper raises a clear error on protein-level proteodata."""
        from proteopy.tl.copf import pairwise_peptide_correlations

        rng = np.random.default_rng(0)
        intensities = rng.standard_normal((5, 3))
        adata = self._make_protein_adata(intensities)

        with pytest.raises(
            ValueError,
            match=r"requires peptide-level proteodata",
        ):
            pairwise_peptide_correlations(adata)

    # -- 2.4 inplace contract

    def test_inplace_true_returns_none_and_mutates(self):
        """``inplace=True`` mutates the input and returns None."""
        rng = np.random.default_rng(1)
        intensities = rng.standard_normal((5, 3))
        adata = self._make_protein_adata(intensities)

        ret = pairwise_var_correlations(adata, inplace=True)

        assert ret is None
        assert "pairwise_correlations;;;;" in adata.uns

    def test_inplace_false_returns_copy_and_keeps_input_clean(self):
        """``inplace=False`` returns a new AnnData; input untouched."""
        rng = np.random.default_rng(2)
        intensities = rng.standard_normal((5, 3))
        adata = self._make_protein_adata(intensities)

        result = pairwise_var_correlations(adata, inplace=False)

        assert result is not None
        assert result is not adata
        assert "pairwise_correlations;;;;" in result.uns
        assert "pairwise_correlations;;;;" not in adata.uns

    # -- 2.5 Custom key_added

    def test_custom_key_added_used_verbatim(self):
        """``key_added`` overrides the default storage key."""
        rng = np.random.default_rng(3)
        intensities = rng.standard_normal((5, 3))
        adata = self._make_protein_adata(intensities)

        result = pairwise_var_correlations(
            adata, key_added="my_corrs", inplace=False,
        )

        assert "my_corrs" in result.uns
        assert "pairwise_correlations;;;;" not in result.uns

    # -- 2.6 method='spearman' on monotone non-linear data

    def test_spearman_differs_from_pearson_on_monotone_data(self):
        """Spearman and Pearson disagree on non-linear monotone data."""
        # Strictly monotone non-linear pair: Spearman == 1, Pearson < 1
        x_vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        y_vals = x_vals ** 3  # monotone but non-linear
        z_vals = np.array([5.0, 1.0, 4.0, 2.0, 6.0, 3.0])  # noisy
        intensities = np.column_stack([x_vals, y_vals, z_vals])
        adata = self._make_protein_adata(intensities)

        pear = pairwise_var_correlations(
            adata, method="pearson", inplace=False,
        ).uns["pairwise_correlations;;;;"]
        spear = pairwise_var_correlations(
            adata, method="spearman", inplace=False,
        ).uns["pairwise_correlations;;;;"]

        # Locate the (P0, P1) pair in both frames
        pear_row = pear[
            (pear["varA"] == "P0") & (pear["varB"] == "P1")
        ].iloc[0]
        spear_row = spear[
            (spear["varA"] == "P0") & (spear["varB"] == "P1")
        ].iloc[0]
        assert spear_row["corr"] == approx(1.0, abs=1e-12)
        assert pear_row["corr"] < 1.0
        assert pear_row["corr"] != approx(spear_row["corr"])

    # -- 2.7 Layer support

    def test_layer_argument_uses_layer_matrix(self):
        """When ``layer`` is set, correlations come from that layer."""
        rng = np.random.default_rng(4)
        intensities_main = rng.standard_normal((5, 3))
        intensities_layer = rng.standard_normal((5, 3))
        adata = self._make_protein_adata(intensities_main)
        adata.layers["log"] = intensities_layer

        result = pairwise_var_correlations(
            adata, layer="log", inplace=False,
        )

        key = "pairwise_correlations;;;log;"
        assert key in result.uns
        df_layer = result.uns[key]

        # Reference: build a fresh adata with X = the layer matrix
        ref_adata = self._make_protein_adata(intensities_layer)
        df_ref = pairwise_var_correlations(
            ref_adata, inplace=False,
        ).uns["pairwise_correlations;;;;"]

        np.testing.assert_allclose(
            df_layer["corr"].to_numpy(),
            df_ref["corr"].to_numpy(),
            atol=1e-12,
        )

    # -- 2.8 batch_key Fisher pooling (ungrouped)

    def test_batch_key_pooling_ungrouped(self):
        """Batched output has var_z_between column and NaN pvals."""
        rng = np.random.default_rng(5)
        intensities = rng.standard_normal((8, 3))
        batches = ["b1"] * 4 + ["b2"] * 4
        adata = self._make_protein_adata(
            intensities, extra_obs={"batch": batches},
        )

        result = pairwise_var_correlations(
            adata, batch_key="batch", inplace=False,
        )

        key = "pairwise_correlations;;batch;;"
        assert key in result.uns
        df = result.uns[key]
        assert list(df.columns) == [
            "varA", "varB", "corr", "pval", "var_z_between",
        ]
        assert len(df) == 3  # 3 choose 2 upper triangle
        assert df["pval"].isna().all()
        assert df["var_z_between"].notna().all()

    # -- 2.9 batch_key skip on too-small batch

    def test_batch_key_skips_small_batch(self, capsys):
        """Batches with <4 obs are skipped, verbose prints the skip."""
        rng = np.random.default_rng(6)
        intensities = rng.standard_normal((6, 3))
        # b_small has only 2 obs; b_big has 4 -> b_small is skipped
        batches = ["b_small", "b_small", "b_big", "b_big",
                   "b_big", "b_big"]
        adata = self._make_protein_adata(
            intensities, extra_obs={"batch": batches},
        )

        result = pairwise_var_correlations(
            adata,
            batch_key="batch",
            inplace=False,
            verbose=True,
        )

        captured = capsys.readouterr().out
        assert "Skipping batch=b_small" in captured

        df = result.uns["pairwise_correlations;;batch;;"]
        # b_big has 4 obs -> w = max(4-3, 0) = 1 > 0, all 3 pairs kept
        assert len(df) == 3

    # -- 2.10 min_contrib_batches enforced as global precondition

    def test_min_contrib_batches_too_high_raises(self):
        """Threshold exceeding eligible batch count raises."""
        rng = np.random.default_rng(7)
        intensities = rng.standard_normal((8, 3))
        batches = ["b1"] * 4 + ["b2"] * 4
        adata = self._make_protein_adata(
            intensities, extra_obs={"batch": batches},
        )

        with pytest.raises(
            ValueError, match=r"min_contrib_batches=99",
        ):
            pairwise_var_correlations(
                adata,
                batch_key="batch",
                min_contrib_batches=99,
                inplace=False,
            )

    # -- 2.11 Validation errors (parametrised)

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"method": "kendall"}, r"method must be"),
            ({"inplace": "yes"}, r"inplace must be a bool"),
            ({"group_by": "nonexistent"}, r"group_by"),
            ({"group_by": "protein_id"}, r">= 2 vars per group"),
            ({"layer": "nope"}, r"layer 'nope' not found in \.layers"),
            ({"batch_key": "missing"}, r"batch_key 'missing'"),
            ({"min_contrib_batches": 0}, r"min_contrib_batches"),
            ({"min_contrib_batches": True}, r"min_contrib_batches"),
            ({"min_contrib_batches": False}, r"min_contrib_batches"),
            ({"min_wsum": -1}, r"min_wsum"),
            ({"min_wsum": True}, r"min_wsum"),
            ({"min_wsum": False}, r"min_wsum"),
            ({"key_added": ""}, r"key_added"),
        ],
    )
    def test_validation_errors(self, kwargs, match):
        """Bad scalar parameters raise descriptive ValueErrors."""
        rng = np.random.default_rng(8)
        intensities = rng.standard_normal((5, 3))
        adata = self._make_protein_adata(intensities)
        with pytest.raises(ValueError, match=match):
            pairwise_var_correlations(adata, **kwargs)

    # -- 2.12 n_obs < 3 raises

    def test_n_obs_below_three_raises(self):
        """n_obs<3 raises with a clear message."""
        intensities = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        )
        adata = self._make_protein_adata(intensities)
        with pytest.raises(ValueError, match=r"3 observations"):
            pairwise_var_correlations(adata)

    # -- 2.13 zero-variance column raises

    def test_zero_variance_column_raises(self):
        """Constant protein column triggers zero-variance error."""
        intensities = np.array(
            [
                [1.0, 2.0, 5.0],
                [3.0, 2.0, 6.0],
                [5.0, 2.0, 7.0],
                [7.0, 2.0, 8.0],
            ],
        )
        adata = self._make_protein_adata(intensities)
        with pytest.raises(
            ValueError, match=r"variance below",
        ):
            pairwise_var_correlations(adata)

    # -- 2.14 Ungrouped n_vars < 2 raises

    def test_ungrouped_single_var_raises(self):
        """One var with ``group_by=None`` raises: no pair to form."""
        intensities = np.array([[1.0], [2.0], [3.0], [4.0]])
        adata = self._make_protein_adata(
            intensities, var_names=["only"],
        )
        with pytest.raises(
            ValueError, match=r">= 2 vars",
        ):
            pairwise_var_correlations(adata)

    # -- 2.16 var_subset: name list selects vars

    def test_var_subset_name_list_restricts_pairs(self):
        """``var_subset`` as list of names restricts which vars
        contribute pairs; result matches running on the same subset."""
        rng = np.random.default_rng(11)
        intensities = rng.standard_normal((6, 4))
        adata = self._make_protein_adata(
            intensities, var_names=["a", "b", "c", "d"],
        )

        subset = ["a", "c", "d"]
        result = pairwise_var_correlations(
            adata, var_subset=subset, inplace=False,
        )
        hash7 = md5_hash_list(subset, n_chars=7)
        df = result.uns[f"pairwise_correlations;;;;{hash7}"]

        # Only pairs from the subset appear
        assert set(df["varA"]).union(df["varB"]) == set(subset)
        # 3 choose 2 = 3 pairs
        assert len(df) == 3

        # Equivalence: pre-subsetting the AnnData yields the same
        # correlations
        sub_adata = adata[:, subset].copy()
        ref = pairwise_var_correlations(
            sub_adata, inplace=False,
        ).uns["pairwise_correlations;;;;"]
        np.testing.assert_allclose(
            df["corr"].to_numpy(),
            ref["corr"].to_numpy(),
            atol=1e-14,
        )

    # -- 2.17 var_subset: boolean mask selects vars

    def test_var_subset_boolean_mask_restricts_pairs(self):
        """``var_subset`` as boolean mask of length n_vars works."""
        rng = np.random.default_rng(12)
        intensities = rng.standard_normal((6, 4))
        adata = self._make_protein_adata(
            intensities, var_names=["a", "b", "c", "d"],
        )

        mask = [True, False, True, True]
        result = pairwise_var_correlations(
            adata, var_subset=mask, inplace=False,
        )
        # Mask resolves to ["a", "c", "d"] in var_names order
        hash7 = md5_hash_list(["a", "c", "d"], n_chars=7)
        df = result.uns[f"pairwise_correlations;;;;{hash7}"]

        assert set(df["varA"]).union(df["varB"]) == {"a", "c", "d"}
        assert len(df) == 3

    # -- 2.18 var_subset: input order does not affect output order

    def test_var_subset_input_order_irrelevant_to_pair_order(self):
        """Reordering ``var_subset`` does not change emitted pairs:
        the resolver enforces adata.var_names order so the column
        invariant holds across runs."""
        rng = np.random.default_rng(13)
        intensities = rng.standard_normal((6, 4))
        adata = self._make_protein_adata(
            intensities, var_names=["a", "b", "c", "d"],
        )

        # Both inputs resolve to ["a", "b", "c"] in var_names order,
        # so they share the same hash slot.
        hash7 = md5_hash_list(["a", "b", "c"], n_chars=7)
        key = f"pairwise_correlations;;;;{hash7}"
        df1 = pairwise_var_correlations(
            adata, var_subset=["a", "b", "c"], inplace=False,
        ).uns[key]
        df2 = pairwise_var_correlations(
            adata, var_subset=["c", "a", "b"], inplace=False,
        ).uns[key]

        pd.testing.assert_frame_equal(df1, df2)

    # -- 2.19 var_subset works with group_by and batch_key

    def test_var_subset_with_group_by_and_batch_key(self):
        """``var_subset`` composes with grouping and batched pooling."""
        rng = np.random.default_rng(14)
        intensities = rng.standard_normal((8, 5))
        # 5 peptides across 2 proteins (3 + 2)
        adata = self._make_peptide_adata(
            intensities,
            peptide_ids=["p1", "p2", "p3", "p4", "p5"],
            protein_ids=["A", "A", "A", "B", "B"],
            extra_obs={"batch": ["b1"] * 4 + ["b2"] * 4},
        )

        # Keep p1+p2 from A (2 peptides -> valid) and both from B
        subset = ["p1", "p2", "p4", "p5"]
        result = pairwise_var_correlations(
            adata,
            group_by="protein_id",
            batch_key="batch",
            var_subset=subset,
            inplace=False,
        )

        hash7 = md5_hash_list(subset, n_chars=7)
        key = f"pairwise_correlations;protein_id;batch;;{hash7}"
        df = result.uns[key]
        assert set(df["group_id"]) == {"A", "B"}
        # A: C(2,2)=1 pair; B: C(2,2)=1 pair
        assert (
            df.groupby("group_id").size().to_dict()
            == {"A": 1, "B": 1}
        )

    # -- 2.20 var_subset validation errors

    @pytest.mark.parametrize(
        "subset,match",
        [
            ([], r"empty"),
            (["a", "a"], r"duplicate"),
            (["zzz"], r"not in adata\.var_names"),
            ([True, False], r"length 2 but adata\.n_vars is 3"),
            (["a", True], r"only var names .* or only booleans"),
            ("a", r"sequence of var names or a boolean mask"),
        ],
    )
    def test_var_subset_validation_errors(self, subset, match):
        """Malformed ``var_subset`` inputs raise descriptive errors."""
        rng = np.random.default_rng(15)
        intensities = rng.standard_normal((5, 3))
        adata = self._make_protein_adata(
            intensities, var_names=["a", "b", "c"],
        )
        with pytest.raises(ValueError, match=match):
            pairwise_var_correlations(
                adata, var_subset=subset, inplace=False,
            )

    # -- 2.21 var_subset shrinking a group below 2 raises

    def test_var_subset_shrinks_group_below_two_raises(self):
        """A subset that leaves a group with <2 vars raises."""
        rng = np.random.default_rng(16)
        intensities = rng.standard_normal((5, 4))
        adata = self._make_peptide_adata(
            intensities,
            peptide_ids=["p1", "p2", "p3", "p4"],
            protein_ids=["A", "A", "B", "B"],
        )
        # Drop one peptide from A -> A has only 1 var
        with pytest.raises(
            ValueError, match=r">= 2 vars per group",
        ):
            pairwise_var_correlations(
                adata,
                group_by="protein_id",
                var_subset=["p1", "p3", "p4"],
                inplace=False,
            )

    # -- 2.22 fill_na=None + NaN in working matrix fails fast

    def test_nan_in_x_with_fill_na_none_raises(self):
        """NaN in .X with fill_na=None raises at the public API."""
        rng = np.random.default_rng(20)
        intensities = rng.standard_normal((5, 3))
        intensities[2, 1] = np.nan
        adata = self._make_protein_adata(intensities)

        with pytest.raises(
            ValueError, match=r"adata\.X contains NaN values",
        ):
            pairwise_var_correlations(adata)

    def test_nan_in_x_with_fill_na_set_succeeds(self):
        """fill_na set to a constant replaces NaNs and lets the run pass."""
        rng = np.random.default_rng(21)
        intensities = rng.standard_normal((5, 3))
        intensities[2, 1] = np.nan
        adata = self._make_protein_adata(intensities)

        result = pairwise_var_correlations(
            adata, fill_na=0.0, inplace=False,
        )
        assert result is not None
        df = result.uns["pairwise_correlations;;;;"]
        assert not df.empty

    def test_nan_in_unused_columns_with_var_subset_succeeds(self):
        """NaN in a column excluded by var_subset doesn't trigger the guard."""
        rng = np.random.default_rng(22)
        intensities = rng.standard_normal((5, 3))
        intensities[:, 2] = np.nan  # NaN only in P2
        adata = self._make_protein_adata(intensities)

        result = pairwise_var_correlations(
            adata, var_subset=["P0", "P1"], inplace=False,
        )
        assert result is not None

    def test_nan_in_layer_with_fill_na_none_raises_with_layer_name(
        self,
    ):
        """Error message names the offending layer when one is used."""
        rng = np.random.default_rng(23)
        intensities = rng.standard_normal((5, 3))
        layer_mat = intensities.copy()
        layer_mat[1, 0] = np.nan
        adata = self._make_protein_adata(intensities)
        adata.layers["log"] = layer_mat

        with pytest.raises(
            ValueError,
            match=r"adata\.layers\['log'\] contains NaN values",
        ):
            pairwise_var_correlations(adata, layer="log")
