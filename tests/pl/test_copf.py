"""Smoke tests for :mod:`proteopy.pl.copf`."""
import matplotlib

matplotlib.use("Agg")

import anndata as ad  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from seaborn.matrix import ClusterGrid  # noqa: E402

import pytest  # noqa: E402

from proteopy.pl.copf import (  # noqa: E402
    pairwise_correlation_heatmap,
    peptide_intensity_heatmap,
)
from proteopy.tl.copf import pairwise_var_correlations  # noqa: E402


def _make_peptide_adata():
    """Build a small peptide-level AnnData with two proteins."""
    rng = np.random.default_rng(0)
    intensities = rng.standard_normal((6, 4))
    peptide_ids = ["p1", "p2", "p3", "p4"]
    protein_ids = ["A", "A", "B", "B"]
    obs_names = [f"S{i}" for i in range(6)]
    var = pd.DataFrame(
        {
            "peptide_id": peptide_ids,
            "protein_id": protein_ids,
            "modification": ["unmod", "mod", "unmod", "mod"],
        },
        index=peptide_ids,
    )
    obs = pd.DataFrame(
        {
            "sample_id": obs_names,
            "condition": ["ctrl", "ctrl", "ctrl", "case", "case", "case"],
        },
        index=obs_names,
    )
    return ad.AnnData(X=intensities, obs=obs, var=var)


def test_pairwise_correlation_heatmap_smoke():
    """Clustermap renders and returns a ClusterGrid."""
    adata = _make_peptide_adata()
    pairwise_var_correlations(adata, group_by="protein_id")
    corrs_key = "pairwise_correlations;protein_id;;;"
    assert corrs_key in adata.uns

    g = pairwise_correlation_heatmap(
        adata,
        corrs_key=corrs_key,
        group_id="A",
        show=False,
    )
    assert isinstance(g, ClusterGrid)


def test_pairwise_correlation_heatmap_no_margin_color_layout():
    """Without margin_color: dendrograms top/left, no legend added."""
    adata = _make_peptide_adata()
    pairwise_var_correlations(adata, group_by="protein_id")
    corrs_key = "pairwise_correlations;protein_id;;;"

    g = pairwise_correlation_heatmap(
        adata,
        corrs_key=corrs_key,
        group_id="A",
        show=False,
    )
    assert isinstance(g, ClusterGrid)
    heatmap_pos = g.ax_heatmap.get_position()
    row_dendro_pos = g.ax_row_dendrogram.get_position()
    col_dendro_pos = g.ax_col_dendrogram.get_position()
    # Row dendrogram is to the left of the heatmap.
    assert row_dendro_pos.x1 <= heatmap_pos.x0 + 1e-6
    # Column dendrogram is above the heatmap.
    assert col_dendro_pos.y0 >= heatmap_pos.y1 - 1e-6
    # No margin-color legend is attached.
    assert len(g.figure.legends) == 0


def test_pairwise_correlation_heatmap_margin_color():
    """Margin color bars render top/left and a legend is added."""
    adata = _make_peptide_adata()
    pairwise_var_correlations(adata, group_by="protein_id")
    corrs_key = "pairwise_correlations;protein_id;;;"

    g = pairwise_correlation_heatmap(
        adata,
        corrs_key=corrs_key,
        group_id="A",
        margin_color="modification",
        show=False,
    )
    assert isinstance(g, ClusterGrid)

    heatmap_pos = g.ax_heatmap.get_position()
    row_colors_pos = g.ax_row_colors.get_position()
    col_colors_pos = g.ax_col_colors.get_position()
    row_dendro_pos = g.ax_row_dendrogram.get_position()
    col_dendro_pos = g.ax_col_dendrogram.get_position()
    # Row colors strip sits between the row dendrogram and the heatmap.
    assert row_colors_pos.x1 <= heatmap_pos.x0 + 1e-6
    assert row_dendro_pos.x1 <= row_colors_pos.x0 + 1e-6
    # Column colors strip sits above the heatmap.
    assert col_colors_pos.y0 >= heatmap_pos.y1 - 1e-6
    # Column dendrogram does not overlap the column colors strip.
    assert col_dendro_pos.y0 >= col_colors_pos.y1 - 1e-6
    # Margin legend is attached to the figure.
    assert len(g.figure.legends) == 1


def test_pairwise_correlation_heatmap_corrs_key_inferred():
    """`corrs_key=None` infers the single matching `adata.uns` slot."""
    adata = _make_peptide_adata()
    pairwise_var_correlations(adata, group_by="protein_id")

    g = pairwise_correlation_heatmap(
        adata,
        group_id="A",
        show=False,
    )
    assert isinstance(g, ClusterGrid)


def test_pairwise_correlation_heatmap_corrs_key_ambiguous():
    """`corrs_key=None` raises when multiple matching slots exist."""
    adata = _make_peptide_adata()
    pairwise_var_correlations(adata, group_by="protein_id")
    pairwise_var_correlations(
        adata,
        group_by="protein_id",
        key_added="pairwise_correlations;protein_id;alt;;",
    )

    with pytest.raises(ValueError, match="Multiple pairwise correlation"):
        pairwise_correlation_heatmap(
            adata,
            group_id="A",
            show=False,
        )


def test_pairwise_correlation_heatmap_var_order():
    """Explicit var_order takes effect when cluster=False."""
    adata = _make_peptide_adata()
    pairwise_var_correlations(adata, group_by="protein_id")
    corrs_key = "pairwise_correlations;protein_id;;;"

    g = pairwise_correlation_heatmap(
        adata,
        corrs_key=corrs_key,
        group_id="A",
        cluster=False,
        var_order=["p2", "p1"],
        show=False,
    )
    assert isinstance(g, ClusterGrid)


def test_pairwise_correlation_heatmap_corrs_key_dataframe():
    """`corrs_key` accepts a DataFrame and bypasses adata.uns lookup."""
    adata = _make_peptide_adata()
    pairwise_var_correlations(adata, group_by="protein_id")
    stored_key = "pairwise_correlations;protein_id;;;"
    df = adata.uns[stored_key]
    # Strip the slot so any accidental lookup would fail loudly.
    del adata.uns[stored_key]

    g = pairwise_correlation_heatmap(
        adata,
        corrs_key=df,
        group_id="A",
        show=False,
    )
    assert isinstance(g, ClusterGrid)


def test_pairwise_correlation_heatmap_corrs_key_dataframe_bad_group_id():
    """DataFrame input still validates `group_id` against the frame."""
    adata = _make_peptide_adata()
    pairwise_var_correlations(adata, group_by="protein_id")
    df = adata.uns["pairwise_correlations;protein_id;;;"]

    with pytest.raises(ValueError, match="not found in corrs_key DataFrame"):
        pairwise_correlation_heatmap(
            adata,
            corrs_key=df,
            group_id="DOES_NOT_EXIST",
            show=False,
        )


# -- peptide_intensity_heatmap


def test_peptide_intensity_heatmap_smoke():
    """Intensity clustermap renders and returns a ClusterGrid."""
    adata = _make_peptide_adata()
    pairwise_var_correlations(adata, group_by="protein_id")
    corrs_key = "pairwise_correlations;protein_id;;;"

    g = peptide_intensity_heatmap(
        adata,
        corrs_key=corrs_key,
        group_id="A",
        show=False,
    )
    assert isinstance(g, ClusterGrid)
    # Rows restricted to the group's peptides; cols are all samples.
    assert g.data2d.shape == (2, 6)
    assert set(g.data2d.index) == {"p1", "p2"}
    assert set(g.data2d.columns) == set(adata.obs_names)


def test_peptide_intensity_heatmap_no_cluster_layout():
    """row/col_cluster=False hides dendrograms but still renders."""
    adata = _make_peptide_adata()
    pairwise_var_correlations(adata, group_by="protein_id")
    corrs_key = "pairwise_correlations;protein_id;;;"

    g = peptide_intensity_heatmap(
        adata,
        corrs_key=corrs_key,
        group_id="A",
        row_cluster=False,
        col_cluster=False,
        show=False,
    )
    assert isinstance(g, ClusterGrid)
    # Natural orders preserved.
    assert list(g.data2d.index) == ["p1", "p2"]
    assert list(g.data2d.columns) == list(adata.obs_names)


def test_peptide_intensity_heatmap_order_by():
    """order_by sorts sample columns by an .obs annotation."""
    adata = _make_peptide_adata()
    pairwise_var_correlations(adata, group_by="protein_id")
    corrs_key = "pairwise_correlations;protein_id;;;"

    g = peptide_intensity_heatmap(
        adata,
        corrs_key=corrs_key,
        group_id="A",
        col_cluster=False,
        order_by="condition",
        show=False,
    )
    # condition = case, case, case, ctrl, ctrl, ctrl (stable-sorted)
    assert list(g.data2d.columns) == ["S3", "S4", "S5", "S0", "S1", "S2"]


def test_peptide_intensity_heatmap_explicit_order():
    """`order` defines the column sequence explicitly."""
    adata = _make_peptide_adata()
    pairwise_var_correlations(adata, group_by="protein_id")
    corrs_key = "pairwise_correlations;protein_id;;;"

    sample_order = ["S5", "S4", "S3", "S2", "S1", "S0"]
    g = peptide_intensity_heatmap(
        adata,
        corrs_key=corrs_key,
        group_id="A",
        col_cluster=False,
        order=sample_order,
        show=False,
    )
    assert list(g.data2d.columns) == sample_order


def test_peptide_intensity_heatmap_order_conflicts():
    """col_cluster excludes order_by/order; order_by excludes order."""
    adata = _make_peptide_adata()
    pairwise_var_correlations(adata, group_by="protein_id")
    corrs_key = "pairwise_correlations;protein_id;;;"

    with pytest.raises(ValueError, match="col_cluster=True"):
        peptide_intensity_heatmap(
            adata,
            corrs_key=corrs_key,
            group_id="A",
            order_by="condition",
            show=False,
        )
    with pytest.raises(ValueError, match="col_cluster=True"):
        peptide_intensity_heatmap(
            adata,
            corrs_key=corrs_key,
            group_id="A",
            order=list(adata.obs_names),
            show=False,
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        peptide_intensity_heatmap(
            adata,
            corrs_key=corrs_key,
            group_id="A",
            col_cluster=False,
            order_by="condition",
            order=list(adata.obs_names),
            show=False,
        )


def test_peptide_intensity_heatmap_dual_margin_colors():
    """row+col margin colors render two stacked legends."""
    adata = _make_peptide_adata()
    pairwise_var_correlations(adata, group_by="protein_id")
    corrs_key = "pairwise_correlations;protein_id;;;"

    g = peptide_intensity_heatmap(
        adata,
        corrs_key=corrs_key,
        group_id="A",
        row_margin_color="modification",
        col_margin_color="condition",
        show=False,
    )
    assert isinstance(g, ClusterGrid)
    heatmap_pos = g.ax_heatmap.get_position()
    row_colors_pos = g.ax_row_colors.get_position()
    col_colors_pos = g.ax_col_colors.get_position()
    # Row color bar sits to the left of the heatmap.
    assert row_colors_pos.x1 <= heatmap_pos.x0 + 1e-6
    # Column color bar sits above the heatmap.
    assert col_colors_pos.y0 >= heatmap_pos.y1 - 1e-6
    # Two stacked legends (one per margin) are attached.
    assert len(g.figure.legends) == 2


def test_peptide_intensity_heatmap_col_cluster_rejects_nan():
    """col_cluster=True with NaN intensities raises ValueError."""
    adata = _make_peptide_adata()
    adata.X[0, 0] = np.nan
    pairwise_var_correlations(adata, group_by="protein_id", fill_na=0.0)
    corrs_key = "pairwise_correlations;protein_id;;;"

    with pytest.raises(ValueError, match="NaN"):
        peptide_intensity_heatmap(
            adata,
            corrs_key=corrs_key,
            group_id="A",
            col_cluster=True,
            show=False,
        )


def test_peptide_intensity_heatmap_zscore():
    """zscore=True row-standardizes each peptide across samples."""
    adata = _make_peptide_adata()
    pairwise_var_correlations(adata, group_by="protein_id")
    corrs_key = "pairwise_correlations;protein_id;;;"

    g = peptide_intensity_heatmap(
        adata,
        corrs_key=corrs_key,
        group_id="A",
        row_cluster=False,
        col_cluster=False,
        zscore=True,
        show=False,
    )
    arr = g.data2d.to_numpy()
    # Each peptide row should have mean ~0 and std ~1.
    np.testing.assert_allclose(arr.mean(axis=1), 0.0, atol=1e-10)
    np.testing.assert_allclose(arr.std(axis=1, ddof=1), 1.0, atol=1e-10)


def test_peptide_intensity_heatmap_zscore_zero_variance_row():
    """A constant peptide yields an all-NaN row after zscore."""
    adata = _make_peptide_adata()
    pairwise_var_correlations(adata, group_by="protein_id")
    corrs_key = "pairwise_correlations;protein_id;;;"
    # Make p1 constant across samples after correlations are computed
    # (pairwise_var_correlations rejects zero-variance columns).
    adata.X[:, 0] = 5.0

    g = peptide_intensity_heatmap(
        adata,
        corrs_key=corrs_key,
        group_id="A",
        row_cluster=False,
        col_cluster=False,
        zscore=True,
        show=False,
    )
    arr = g.data2d.loc["p1"].to_numpy()
    assert np.isnan(arr).all()


def test_peptide_intensity_heatmap_layer():
    """`layer` selects an alternative intensity matrix."""
    adata = _make_peptide_adata()
    adata.layers["scaled"] = adata.X * 10
    pairwise_var_correlations(adata, group_by="protein_id")
    corrs_key = "pairwise_correlations;protein_id;;;"

    g = peptide_intensity_heatmap(
        adata,
        corrs_key=corrs_key,
        group_id="A",
        layer="scaled",
        row_cluster=False,
        col_cluster=False,
        show=False,
    )
    expected = adata.layers["scaled"][:, [0, 1]].T
    np.testing.assert_allclose(g.data2d.to_numpy(), expected)
