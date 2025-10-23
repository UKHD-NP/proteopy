import numpy as np
import pandas as pd

from copro.read.peptides_generic import peptides_long, peptides_long_from_df


def _build_test_frames():
    intensities_df = pd.DataFrame(
        [
            {"filename": "sample_a", "peptide_id": "pep_1", "protein_id": "prot_1", "intensities": 10.0},
            {"filename": "sample_a", "peptide_id": "pep_2", "protein_id": "prot_2", "intensities": np.nan},
            {"filename": "sample_b", "peptide_id": "pep_1", "protein_id": "prot_1", "intensities": 1e-8},
            {"filename": "sample_b", "peptide_id": "pep_2", "protein_id": "prot_2", "intensities": 5.5},
        ]
    )

    filename_annotation_df = pd.DataFrame(
        [
            {"filename": "sample_a", "condition": "control"},
            {"filename": "sample_b", "condition": "treated"},
        ]
    )

    peptides_annotation_df = pd.DataFrame(
        [
            {"peptide_id": "pep_1", "sequence": "AAAA"},
            {"peptide_id": "pep_2", "sequence": "BBBB"},
        ]
    )

    return intensities_df, filename_annotation_df, peptides_annotation_df


def test_peptides_long_from_df_basic():
    intensities_df, filename_annotation_df, peptides_annotation_df = _build_test_frames()

    adata = peptides_long_from_df(
        intensities_df,
        filename_annotation_df=filename_annotation_df,
        peptides_annotation_df=peptides_annotation_df,
        fill_na=0.0,
    )

    assert adata.X.shape == (2, 2)
    np.testing.assert_allclose(
        adata.X,
        np.array([[10.0, 0.0], [1e-8, 5.5]]),
        atol=1e-12,
    )

    assert adata.obs_names.tolist() == ["sample_a", "sample_b"]
    assert adata.obs["condition"].tolist() == ["control", "treated"]

    assert adata.var_names.tolist() == ["pep_1", "pep_2"]
    assert adata.var.loc["pep_1", "protein_id"] == "prot_1"
    assert adata.var.loc["pep_2", "sequence"] == "BBBB"


def test_peptides_long_basic(tmp_path):
    intensities_df, filename_annotation_df, peptides_annotation_df = _build_test_frames()

    intensities_path = tmp_path / "intensities.tsv"
    filename_annotation_path = tmp_path / "filename_annotation.tsv"
    peptide_annotation_path = tmp_path / "peptide_annotation.tsv"

    intensities_df.to_csv(intensities_path, sep="\t", index=False)
    filename_annotation_df.to_csv(filename_annotation_path, sep="\t", index=False)
    peptides_annotation_df.to_csv(peptide_annotation_path, sep="\t", index=False)

    adata = peptides_long(
        str(intensities_path),
        filename_annotation=str(filename_annotation_path),
        peptides_annotation_path=str(peptide_annotation_path),
        fill_na=0.0,
    )

    assert adata.obs_names.tolist() == ["sample_a", "sample_b"]
    assert adata.obs["condition"].tolist() == ["control", "treated"]

    assert adata.var_names.tolist() == ["pep_1", "pep_2"]
    assert adata.var.loc["pep_1", "protein_id"] == "prot_1"
    assert adata.var.loc["pep_2", "sequence"] == "BBBB"

    np.testing.assert_allclose(
        adata.X,
        np.array([[10.0, 0.0], [1e-8, 5.5]]),
        atol=1e-12,
    )
