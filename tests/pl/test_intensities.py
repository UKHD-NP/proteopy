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


# -- Missing values break the line


@pytest.fixture
def adata_na():
    peptides = ["PEPA", "PEPB", "PEPC"]
    samples = [f"S{i}" for i in range(1, 7)]
    var = pd.DataFrame(
        {
            "peptide_id": peptides,
            "protein_id": ["P1"] * 3,
            "proteoform_id": ["f1", "f2", "f2"],
        },
        index=peptides,
    )
    obs = pd.DataFrame(
        {"sample_id": samples, "condition": list("AAABBB")},
        index=samples,
    )
    X = np.array(
        [
            [10.0, 50.0, 30.0],
            [20.0, np.nan, np.nan],
            [np.nan, 70.0, 35.0],
            [40.0, 80.0, np.nan],
            [50.0, np.nan, np.nan],
            [60.0, 100.0, 45.0],
        ]
    )
    return ad.AnnData(X=X, obs=obs, var=var)


def _segments(axes):
    """Return the sorted (x positions, y values) of each drawn line."""
    axes = axes if isinstance(axes, list) else [axes]
    segments = []
    for ax in axes:
        for line in ax.lines:
            if line.get_linestyle() == "--" or not len(line.get_xdata()):
                continue
            segments.append(
                (
                    tuple(int(x) for x in line.get_xdata()),
                    tuple(float(y) for y in line.get_ydata()),
                )
            )
    plt.close("all")
    return sorted(segments)


PEPTIDE_SEGMENTS = sorted(
    [
        ((0, 1), (10.0, 20.0)),
        ((3, 4, 5), (40.0, 50.0, 60.0)),
        ((0,), (50.0,)),
        ((2, 3), (70.0, 80.0)),
        ((5,), (100.0,)),
        ((0,), (30.0,)),
        ((2,), (35.0,)),
        ((5,), (45.0,)),
    ]
)


def test_missing_values_break_peptide_lines(adata_na):
    ax = peptide_intensities(adata_na, protein_ids="P1", show=False)
    assert _segments(ax) == PEPTIDE_SEGMENTS


def test_missing_values_break_color_lines(adata_na):
    ax = proteoform_intensities(adata_na, "P1", show=False)
    assert _segments(ax) == PEPTIDE_SEGMENTS


def test_missing_values_break_group_by_lines_only_if_all_missing(
    adata_na,
):
    ax = peptide_intensities(
        adata_na,
        protein_ids="P1",
        group_by="proteoform_id",
        show=False,
    )
    # f2 (PEPB, PEPC) is missing entirely only in S2 and S5
    expected = sorted(
        [
            ((0, 1), (10.0, 20.0)),
            ((3, 4, 5), (40.0, 50.0, 60.0)),
            ((0,), (40.0,)),
            ((2, 3), (52.5, 80.0)),
            ((5,), (72.5,)),
        ]
    )
    assert _segments(ax) == expected


def test_missing_values_break_group_by_legend_complete(adata_na):
    ax = peptide_intensities(
        adata_na,
        protein_ids="P1",
        group_by="proteoform_id",
        show=False,
    )
    legend = (ax if isinstance(ax, list) else [ax])[0].get_legend()
    labels = [text.get_text() for text in legend.get_texts()]
    plt.close("all")
    assert labels == ["f1", "f2"]


def test_missing_values_break_after_log_and_z_transform(adata_na):
    ax = peptide_intensities(
        adata_na,
        protein_ids="P1",
        log_transform=2,
        z_transform=True,
        show=False,
    )
    positions = [x for x, _ in _segments(ax)]
    assert positions == [x for x, _ in PEPTIDE_SEGMENTS]


def test_missing_values_break_lines_in_facets(adata_na):
    axes = peptide_intensities(
        adata_na,
        protein_ids="P1",
        facet_by="condition",
        show=False,
    )
    # PEPA: S1-S2 in facet A; S4-S6 in facet B
    segments = _segments(axes)
    assert ((0, 1), (10.0, 20.0)) in segments
    assert ((0, 1, 2), (40.0, 50.0, 60.0)) in segments
    # PEPB in facet B: S4 and S6 separated by missing S5
    assert ((0,), (80.0,)) in segments
    assert ((2,), (100.0,)) in segments


def test_hidden_zeros_break_lines(adata_na):
    adata_na.X = np.nan_to_num(adata_na.X, nan=0.0)
    ax = peptide_intensities(
        adata_na,
        protein_ids="P1",
        show_zeros=False,
        show=False,
    )
    assert _segments(ax) == PEPTIDE_SEGMENTS


# -- Dashed segments bridge missing values

CONNECTOR_GID = "missing_value_connector"


def _connectors(axes):
    """Return the sorted (x positions, y values) of dashed connectors."""
    axes = axes if isinstance(axes, list) else [axes]
    connectors = []
    for ax in axes:
        for line in ax.lines:
            if line.get_gid() != CONNECTOR_GID:
                continue
            assert line.get_linestyle() == "--"
            connectors.append(
                (
                    tuple(int(x) for x in line.get_xdata()),
                    tuple(float(y) for y in line.get_ydata()),
                )
            )
    plt.close("all")
    return sorted(connectors)


PEPTIDE_CONNECTORS = sorted(
    [
        ((1, 3), (20.0, 40.0)),
        ((0, 2), (50.0, 70.0)),
        ((3, 5), (80.0, 100.0)),
        ((0, 2), (30.0, 35.0)),
        ((2, 5), (35.0, 45.0)),
    ]
)


def test_dashed_connectors_bridge_peptide_gaps(adata_na):
    ax = peptide_intensities(adata_na, protein_ids="P1", show=False)
    assert _connectors(ax) == PEPTIDE_CONNECTORS


def test_dashed_connectors_bridge_color_gaps(adata_na):
    ax = proteoform_intensities(adata_na, "P1", show=False)
    assert _connectors(ax) == PEPTIDE_CONNECTORS


def test_dashed_connectors_use_line_color(adata_na):
    ax = proteoform_intensities(adata_na, "P1", show=False)
    ax = (ax if isinstance(ax, list) else [ax])[0]
    legend = ax.get_legend()
    handle_colors = {
        text.get_text(): handle.get_color()
        for text, handle in zip(legend.get_texts(), legend.legend_handles)
    }
    connector_colors = {
        tuple(int(x) for x in line.get_xdata()): line.get_color()
        for line in ax.lines
        if line.get_gid() == CONNECTOR_GID
    }
    plt.close("all")
    # PEPA (f1) skips S3; PEPB (f2) skips S5
    assert connector_colors[(1, 3)] == handle_colors["f1"]
    assert connector_colors[(3, 5)] == handle_colors["f2"]


def test_dashed_connectors_bridge_group_means(adata_na):
    ax = peptide_intensities(
        adata_na,
        protein_ids="P1",
        group_by="proteoform_id",
        show=False,
    )
    expected = sorted(
        [
            ((1, 3), (20.0, 40.0)),
            ((0, 2), (40.0, 52.5)),
            ((3, 5), (80.0, 72.5)),
        ]
    )
    assert _connectors(ax) == expected


def test_dashed_connectors_do_not_cross_order_by_groups(adata_na):
    ax = peptide_intensities(
        adata_na,
        protein_ids="P1",
        order_by="condition",
        show=False,
    )
    # Gaps spanning the A|B boundary (S3|S4) are left open
    expected = sorted(
        [
            ((0, 2), (50.0, 70.0)),
            ((3, 5), (80.0, 100.0)),
            ((0, 2), (30.0, 35.0)),
        ]
    )
    assert _connectors(ax) == expected


def test_dashed_connectors_in_facets(adata_na):
    axes = peptide_intensities(
        adata_na,
        protein_ids="P1",
        facet_by="condition",
        show=False,
    )
    expected = sorted(
        [
            ((0, 2), (50.0, 70.0)),
            ((0, 2), (30.0, 35.0)),
            ((0, 2), (80.0, 100.0)),
        ]
    )
    assert _connectors(axes) == expected


def test_no_dashed_connectors_without_missing_values(adata):
    ax = peptide_intensities(adata, protein_ids="P1", show=False)
    assert _connectors(ax) == []
