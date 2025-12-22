import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from proteopy.utils.anndata import is_proteodata


class TestIsProteodata:
    def test_returns_true_for_valid_peptide_data(self):
        peptides = ["PEP1", "PEP2", "PEP3"]
        proteins = ["PROT_A", "PROT_B", "PROT_C"]
        adata = AnnData(np.arange(9).reshape(3, 3), var=pd.DataFrame(index=peptides))
        adata.var["peptide_id"] = peptides
        adata.var["protein_id"] = proteins

        assert adata.var["peptide_id"].is_unique
        assert is_proteodata(adata) == (True, "peptide")

    def test_peptide_data_requires_protein_column(self):
        peptides = ["PEP1", "PEP2"]
        adata = AnnData(np.arange(4).reshape(2, 2), var=pd.DataFrame(index=peptides))
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
        adata = AnnData(np.arange(4).reshape(2, 2), var=pd.DataFrame(index=peptides))
        adata.var["peptide_id"] = ["PEP1", "PEP_DIFFERENT"]
        adata.var["protein_id"] = ["PROT1", "PROT2"]

        assert is_proteodata(adata) == (False, None)

        with pytest.raises(ValueError, match="does not match AnnData.var_names"):
            is_proteodata(adata, raise_error=True)

    def test_peptide_multiple_protein_mapping_returns_false(self):
        peptides = ["PEP1", "PEP2"]
        adata = AnnData(np.arange(4).reshape(2, 2), var=pd.DataFrame(index=peptides))
        adata.var["peptide_id"] = peptides
        adata.var["protein_id"] = ["PROT1;PROT2", "PROT3"]

        assert is_proteodata(adata) == (False, None)
        with pytest.raises(ValueError, match="multiple proteins"):
            is_proteodata(adata, raise_error=True)

    def test_returns_true_for_valid_protein_data(self):
        proteins = ["PROT_A", "PROT_B"]
        adata = AnnData(np.arange(4).reshape(2, 2), var=pd.DataFrame(index=proteins))
        adata.var["protein_id"] = proteins

        assert adata.var["protein_id"].is_unique
        assert is_proteodata(adata) == (True, "protein")

    def test_protein_id_must_match_axis(self):
        proteins = ["PROT_A", "PROT_B"]
        adata = AnnData(np.arange(4).reshape(2, 2), var=pd.DataFrame(index=proteins))
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
