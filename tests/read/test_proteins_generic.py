import numpy as np
import pandas as pd

from copro.read.proteins_generic import proteins_long, proteins_long_from_df


def _build_protein_frames():
    intensities_df = pd.DataFrame(
        [
            {"filename": "sample_1", "protein_id": "prot_a", "intensities": 2.0},
            {"filename": "sample_1", "protein_id": "prot_b", "intensities": np.nan},
            {"filename": "sample_2", "protein_id": "prot_a", "intensities": 1e-9},
            {"filename": "sample_2", "protein_id": "prot_b", "intensities": 3.5},
        ]
    )

    filename_annotation_df = pd.DataFrame(
        [
            {"filename": "sample_1", "condition": "baseline"},
            {"filename": "sample_2", "condition": "stimulated"},
        ]
    )

    protein_annotation_df = pd.DataFrame(
        [
            {"protein_id": "prot_a", "description": "alpha"},
            {"protein_id": "prot_b", "description": "beta"},
        ]
    )

    return intensities_df, filename_annotation_df, protein_annotation_df


def test_proteins_long_from_df_basic():
    intensities_df, filename_annotation_df, protein_annotation_df = _build_protein_frames()

    adata = proteins_long_from_df(
        intensities_df,
        filename_annotation_df=filename_annotation_df,
        protein_annotation_df=protein_annotation_df,
        fill_na=0.0,
        )

    assert adata.X.shape == (2, 2)
    np.testing.assert_allclose(
        adata.X,
        np.array([[2.0, 0.0], [1e-9, 3.5]]),
        atol=1e-12,
        )

    assert adata.obs_names.tolist() == ["sample_1", "sample_2"]
    assert adata.obs["condition"].tolist() == ["baseline", "stimulated"]

    assert adata.var_names.tolist() == ["prot_a", "prot_b"]
    assert adata.var.loc["prot_a", "description"] == "alpha"


def test_proteins_long_basic(tmp_path):
    intensities_df, filename_annotation_df, protein_annotation_df = _build_protein_frames()

    intensities_path = tmp_path / "proteins.tsv"
    filename_annotation_path = tmp_path / "filename_annotation.tsv"
    protein_annotation_path = tmp_path / "protein_annotation.tsv"

    intensities_df.to_csv(intensities_path, sep="\t", index=False)
    filename_annotation_df.to_csv(filename_annotation_path, sep="\t", index=False)
    protein_annotation_df.to_csv(protein_annotation_path, sep="\t", index=False)

    adata = proteins_long(
        str(intensities_path),
        filename_annotation_path=str(filename_annotation_path),
        protein_annotation_path=str(protein_annotation_path),
        fill_na=0.0,
        )

    assert adata.obs_names.tolist() == ["sample_1", "sample_2"]
    assert adata.obs["condition"].tolist() == ["baseline", "stimulated"]

    assert adata.var_names.tolist() == ["prot_a", "prot_b"]
    assert adata.var.loc["prot_b", "description"] == "beta"

    np.testing.assert_allclose(
        adata.X,
        np.array([[2.0, 0.0], [1e-9, 3.5]]),
        atol=1e-12,
        )
