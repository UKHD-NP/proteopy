import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from proteopy.utils.anndata import is_proteodata, check_proteodata


class TestIsProteodata:
    def test_returns_true_for_valid_peptide_data(self):
        peptides = ["PEP1", "PEP2", "PEP3"]
        proteins = ["PROT_A", "PROT_B", "PROT_C"]
        obs_names = [f"obs{i}" for i in range(3)]
        adata = AnnData(
            np.arange(9).reshape(3, 3),
            obs=pd.DataFrame({"sample_id": obs_names}, index=obs_names),
            var=pd.DataFrame(index=peptides),
        )
        adata.var["peptide_id"] = peptides
        adata.var["protein_id"] = proteins

        assert adata.var["peptide_id"].is_unique
        assert is_proteodata(adata) == (True, "peptide")

    def test_peptide_data_requires_protein_column(self):
        peptides = ["PEP1", "PEP2"]
        obs_names = [f"obs{i}" for i in range(2)]
        adata = AnnData(
            np.arange(4).reshape(2, 2),
            obs=pd.DataFrame({"sample_id": obs_names}, index=obs_names),
            var=pd.DataFrame(index=peptides),
        )
        adata.var["peptide_id"] = peptides

        assert is_proteodata(adata) == (False, None)

        with pytest.raises(ValueError, match="no 'protein_id' column"):
            is_proteodata(adata, raise_error=True)

    def test_peptide_id_must_be_unique(self):
        peptides = ["PEP1", "PEP1"]
        proteins = ["PROT_A", "PROT_B"]
        with pytest.warns(UserWarning, match="Variable names are not unique"):
            adata = AnnData(
                np.arange(4).reshape(2, 2),
                var=pd.DataFrame(index=peptides)
            )
        adata.var["peptide_id"] = peptides
        adata.var["protein_id"] = proteins

        with pytest.raises(ValueError, match="Duplicate names detected"):
            is_proteodata(adata)

    def test_peptide_id_must_match_axis(self):
        peptides = ["PEP1", "PEP2"]
        obs_names = [f"obs{i}" for i in range(2)]
        adata = AnnData(
            np.arange(4).reshape(2, 2),
            obs=pd.DataFrame({"sample_id": obs_names}, index=obs_names),
            var=pd.DataFrame(index=peptides),
        )
        adata.var["peptide_id"] = ["PEP1", "PEP_DIFFERENT"]
        adata.var["protein_id"] = ["PROT1", "PROT2"]

        assert is_proteodata(adata) == (False, None)

        with pytest.raises(ValueError, match="does not match AnnData.var_names"):
            is_proteodata(adata, raise_error=True)

    def test_peptide_multiple_protein_mapping_returns_false(self):
        peptides = ["PEP1", "PEP2"]
        obs_names = [f"obs{i}" for i in range(2)]
        adata = AnnData(
            np.arange(4).reshape(2, 2),
            obs=pd.DataFrame({"sample_id": obs_names}, index=obs_names),
            var=pd.DataFrame(index=peptides),
        )
        adata.var["peptide_id"] = peptides
        adata.var["protein_id"] = ["PROT1;PROT2", "PROT3"]

        assert is_proteodata(adata) == (False, None)
        with pytest.raises(ValueError, match="multiple proteins"):
            is_proteodata(adata, raise_error=True)

    def test_returns_true_for_valid_protein_data(self):
        proteins = ["PROT_A", "PROT_B"]
        obs_names = [f"obs{i}" for i in range(2)]
        adata = AnnData(
            np.arange(4).reshape(2, 2),
            obs=pd.DataFrame({"sample_id": obs_names}, index=obs_names),
            var=pd.DataFrame(index=proteins),
        )
        adata.var["protein_id"] = proteins

        assert adata.var["protein_id"].is_unique
        assert is_proteodata(adata) == (True, "protein")

    def test_protein_id_must_match_axis(self):
        proteins = ["PROT_A", "PROT_B"]
        obs_names = [f"obs{i}" for i in range(2)]
        adata = AnnData(
            np.arange(4).reshape(2, 2),
            obs=pd.DataFrame({"sample_id": obs_names}, index=obs_names),
            var=pd.DataFrame(index=proteins),
        )
        adata.var["protein_id"] = ["PROT_A", "PROT_C"]

        assert is_proteodata(adata) == (False, None)

        with pytest.raises(ValueError, match="does not match AnnData.var_names"):
            is_proteodata(adata, raise_error=True)

    def test_protein_id_must_be_unique(self):
        proteins = ["PROT_A", "PROT_A"]
        with pytest.warns(UserWarning, match="Variable names are not unique"):
            adata = AnnData(
                np.arange(4).reshape(2, 2),
                var=pd.DataFrame(index=proteins)
            )
        adata.var["protein_id"] = proteins

        with pytest.raises(ValueError, match="Duplicate names detected"):
            is_proteodata(adata)

    def test_missing_required_columns_returns_false(self):
        proteins = ["PROT_A", "PROT_B"]
        adata = AnnData(np.arange(4).reshape(2, 2), var=pd.DataFrame(index=proteins))
        adata.var["unrelated"] = ["foo", "bar"]

        assert is_proteodata(adata) == (False, None)

    def test_empty_var_returns_false(self):
        adata = AnnData(np.arange(4).reshape(2, 2))

        assert is_proteodata(adata) == (False, None)

    def test_rejects_non_anndata_input(self):
        with pytest.raises(TypeError, match="expects an AnnData object"):
            is_proteodata(object())

    # -- NaN in ID columns ------------------------------------

    def test_nan_in_peptide_id_returns_false(self):
        peptides = ["PEP1", "PEP2"]
        obs_names = ["obs0", "obs1"]
        adata = AnnData(
            np.arange(4).reshape(2, 2),
            obs=pd.DataFrame(
                {"sample_id": obs_names}, index=obs_names,
                ),
            var=pd.DataFrame(index=peptides),
            )
        adata.var["peptide_id"] = ["PEP1", None]
        adata.var["protein_id"] = ["PROT_A", "PROT_B"]

        assert is_proteodata(adata) == (False, None)

        with pytest.raises(
            ValueError,
            match="'peptide_id'.*missing values",
            ):
            is_proteodata(adata, raise_error=True)

    def test_nan_in_protein_id_peptide_level_returns_false(self):
        peptides = ["PEP1", "PEP2"]
        obs_names = ["obs0", "obs1"]
        adata = AnnData(
            np.arange(4).reshape(2, 2),
            obs=pd.DataFrame(
                {"sample_id": obs_names}, index=obs_names,
                ),
            var=pd.DataFrame(index=peptides),
            )
        adata.var["peptide_id"] = peptides
        adata.var["protein_id"] = ["PROT_A", np.nan]

        assert is_proteodata(adata) == (False, None)

        with pytest.raises(
            ValueError,
            match="'protein_id'.*missing values",
            ):
            is_proteodata(adata, raise_error=True)

    def test_nan_in_protein_id_protein_level_returns_false(self):
        proteins = ["PROT_A", "PROT_B"]
        obs_names = ["obs0", "obs1"]
        adata = AnnData(
            np.arange(4).reshape(2, 2),
            obs=pd.DataFrame(
                {"sample_id": obs_names}, index=obs_names,
                ),
            var=pd.DataFrame(index=proteins),
            )
        adata.var["protein_id"] = ["PROT_A", None]

        assert is_proteodata(adata) == (False, None)

        with pytest.raises(
            ValueError,
            match="'protein_id'.*missing values",
            ):
            is_proteodata(adata, raise_error=True)

    # -- layers parameter -------------------------------------

    def test_layers_missing_key_returns_false(self):
        proteins = ["PROT_A", "PROT_B"]
        obs_names = ["obs0", "obs1"]
        adata = AnnData(
            np.arange(4).reshape(2, 2),
            obs=pd.DataFrame(
                {"sample_id": obs_names}, index=obs_names,
                ),
            var=pd.DataFrame(index=proteins),
            )
        adata.var["protein_id"] = proteins

        result = is_proteodata(adata, layers="nonexistent")
        assert result == (False, None)

        with pytest.raises(
            ValueError,
            match="Layer 'nonexistent' not found",
            ):
            is_proteodata(
                adata,
                raise_error=True,
                layers="nonexistent",
                )

    def test_layers_with_infinite_values_returns_false(self):
        proteins = ["PROT_A", "PROT_B"]
        obs_names = ["obs0", "obs1"]
        X = np.arange(4, dtype=float).reshape(2, 2)
        adata = AnnData(
            X,
            obs=pd.DataFrame(
                {"sample_id": obs_names}, index=obs_names,
                ),
            var=pd.DataFrame(index=proteins),
            )
        adata.var["protein_id"] = proteins
        layer = X.copy()
        layer[0, 0] = np.inf
        adata.layers["raw"] = layer

        result = is_proteodata(adata, layers="raw")
        assert result == (False, None)

        with pytest.raises(
            ValueError,
            match="layers\\['raw'\\].*infinite values",
            ):
            is_proteodata(
                adata,
                raise_error=True,
                layers="raw",
                )

    def test_layers_valid_passes(self):
        proteins = ["PROT_A", "PROT_B"]
        obs_names = ["obs0", "obs1"]
        X = np.arange(4, dtype=float).reshape(2, 2)
        adata = AnnData(
            X,
            obs=pd.DataFrame(
                {"sample_id": obs_names}, index=obs_names,
                ),
            var=pd.DataFrame(index=proteins),
            )
        adata.var["protein_id"] = proteins
        adata.layers["raw"] = X.copy()

        assert is_proteodata(adata, layers="raw") == (
            True, "protein",
            )

    def test_layers_multiple_keys(self):
        proteins = ["PROT_A", "PROT_B"]
        obs_names = ["obs0", "obs1"]
        X = np.arange(4, dtype=float).reshape(2, 2)
        adata = AnnData(
            X,
            obs=pd.DataFrame(
                {"sample_id": obs_names}, index=obs_names,
                ),
            var=pd.DataFrame(index=proteins),
            )
        adata.var["protein_id"] = proteins
        adata.layers["raw"] = X.copy()
        bad_layer = X.copy()
        bad_layer[1, 1] = -np.inf
        adata.layers["bad"] = bad_layer

        # First layer ok, second has inf
        with pytest.raises(
            ValueError,
            match="layers\\['bad'\\].*infinite values",
            ):
            is_proteodata(
                adata,
                raise_error=True,
                layers=["raw", "bad"],
                )

    # -- check_proteodata with layers -------------------------

    def test_check_proteodata_propagates_layers(self):
        proteins = ["PROT_A", "PROT_B"]
        obs_names = ["obs0", "obs1"]
        X = np.arange(4, dtype=float).reshape(2, 2)
        adata = AnnData(
            X,
            obs=pd.DataFrame(
                {"sample_id": obs_names}, index=obs_names,
                ),
            var=pd.DataFrame(index=proteins),
            )
        adata.var["protein_id"] = proteins
        adata.layers["raw"] = X.copy()

        # Valid layer passes
        assert check_proteodata(adata, layers="raw") == (
            True, "protein",
            )

        # Layer with inf raises
        bad_layer = X.copy()
        bad_layer[0, 0] = np.inf
        adata.layers["bad"] = bad_layer

        with pytest.raises(
            ValueError,
            match="layers\\['bad'\\].*infinite values",
            ):
            check_proteodata(adata, layers="bad")
