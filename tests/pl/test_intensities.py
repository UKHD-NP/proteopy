import anndata as ad
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from proteopy.pl import peptide_intensities, proteoform_intensities

matplotlib.use("Agg")

PEPTIDES = ["PEPA", "PEPB"]
SAMPLES = ["S1", "S2", "S3", "S4"]


@pytest.fixture
def adata():
    var = pd.DataFrame(
        {
            "peptide_id": PEPTIDES,
            "protein_id": ["P1", "P1"],
            "proteoform_id": ["P1_1", "P1_2"],
        },
        index=PEPTIDES,
    )
    obs = pd.DataFrame(
        {
            "sample_id": SAMPLES,
            "condition": ["A", "A", "B", "B"],
            "batch": ["x", "y", "x", "y"],
        },
        index=SAMPLES,
    )
    X = np.array(
        [
            [1.0, 10.0],
            [3.0, 20.0],
            [100.0, 5.0],
            [300.0, 7.0],
        ]
    )
    return ad.AnnData(X=X, obs=obs, var=var)


def _peptide_ydata(ax, n_samples=len(SAMPLES)):
    """Return the y values of each drawn peptide line, in draw order."""
    lines = [
        np.asarray(line.get_ydata(), dtype=float)
        for line in ax.lines
        if len(line.get_ydata()) == n_samples
    ]
    assert len(lines) == len(PEPTIDES)
    return lines


def _zscore(values):
    values = np.asarray(values, dtype=float)
    return (values - values.mean()) / values.std()


def _plot(adata, **kwargs):
    ax = peptide_intensities(
        adata,
        protein_ids="P1",
        show=False,
        **kwargs,
    )
    lines = _peptide_ydata(ax)
    plt.close("all")
    return lines


def test_z_transform_true_standardizes_across_all_samples(adata):
    lines = _plot(adata, z_transform=True)
    expected = [_zscore(adata.X[:, i]) for i in range(adata.n_vars)]
    assert len(lines) == len(expected)
    for got, exp in zip(lines, expected):
        np.testing.assert_allclose(got, exp)


def test_z_transform_obs_key_standardizes_within_groups(adata):
    lines = _plot(adata, z_transform="condition")
    expected = [
        np.concatenate([_zscore(adata.X[:2, i]), _zscore(adata.X[2:, i])])
        for i in range(adata.n_vars)
    ]
    for got, exp in zip(lines, expected):
        np.testing.assert_allclose(got, exp)
    # Each condition is centered on zero
    for got in lines:
        assert got[:2].mean() == pytest.approx(0)
        assert got[2:].mean() == pytest.approx(0)


def test_z_transform_obs_key_list_uses_key_combinations(adata):
    # Every condition x batch combination holds one sample: zero
    # standard deviation, so all values are centered to 0
    lines = _plot(adata, z_transform=["condition", "batch"])
    for got in lines:
        np.testing.assert_allclose(got, np.zeros(len(SAMPLES)))


def test_z_transform_missing_group_label_is_nan(adata):
    adata.obs["condition"] = ["A", "A", "B", np.nan]
    ax = peptide_intensities(
        adata,
        protein_ids="P1",
        z_transform="condition",
        show=False,
    )
    # S4 becomes NaN and is not drawn; S3 is alone in group B
    lines = _peptide_ydata(ax, n_samples=3)
    plt.close("all")
    for got in lines:
        assert got[:2].mean() == pytest.approx(0)
        assert got[2] == pytest.approx(0)


def test_z_transform_matches_facets(adata):
    axes = peptide_intensities(
        adata,
        protein_ids="P1",
        facet_by="condition",
        z_transform="condition",
        show=False,
    )
    plt.close("all")
    assert len(axes) == 2
    for facet_ax in axes:
        for line in facet_ax.lines:
            ydata = np.asarray(line.get_ydata(), dtype=float)
            if len(ydata) == 2:
                assert ydata.mean() == pytest.approx(0)


def test_proteoform_intensities_accepts_obs_key(adata):
    ax = proteoform_intensities(
        adata,
        "P1",
        z_transform="condition",
        show=False,
    )
    lines = _peptide_ydata(ax)
    plt.close("all")
    for got in lines:
        assert got[:2].mean() == pytest.approx(0)


def test_z_transform_sparse_input_matches_dense(adata):
    from scipy import sparse

    dense = _plot(adata, z_transform="condition")
    adata_sparse = adata.copy()
    adata_sparse.X = sparse.csr_matrix(adata_sparse.X)
    sparse_lines = _plot(adata_sparse, z_transform="condition")
    for got, exp in zip(sparse_lines, dense):
        np.testing.assert_allclose(got, exp)


@pytest.mark.parametrize(
    "z_transform, error",
    [
        ("missing", KeyError),
        (["condition", "missing"], KeyError),
        ([], ValueError),
        (["condition", "condition"], ValueError),
        ([1], TypeError),
        (1.5, TypeError),
    ],
)
def test_z_transform_invalid_values_raise(adata, z_transform, error):
    with pytest.raises(error):
        peptide_intensities(
            adata,
            protein_ids="P1",
            z_transform=z_transform,
            show=False,
        )
