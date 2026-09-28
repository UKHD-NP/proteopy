"""Tests for pr.pl.pairwise_peptide_correlations_heatmap.

Covers: signature contract, return-value semantics, symmetric
clustering and linkage pass-through, cluster toggle, annotation
strips, ID-column labelling, argument validation, and lifecycle
(save / non-mutation).
"""

import inspect
import warnings

import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage as scipy_linkage
import pytest

import proteopy as pr

plt.switch_backend("Agg")


@pytest.fixture(autouse=True)
def close_test_figures():
    before = set(plt.get_fignums())
    yield
    for num in set(plt.get_fignums()) - before:
        plt.close(num)


def _function():
    return getattr(pr.pl, "pairwise_peptide_correlations_heatmap")


def _copf_adata(seed=0):
    """Peptide-level AnnData with the COPF chain already run.

    P1: 5 peptides forming two proteoforms; P2: 3 peptides.
    """
    rng = np.random.default_rng(seed)
    prot_map = {f"pep{i}": "P1" for i in range(5)}
    prot_map.update({f"pep{i}": "P2" for i in range(5, 8)})
    peps = list(prot_map)

    n_obs = 12
    pat_a = rng.normal(size=n_obs)
    pat_b = rng.normal(size=n_obs)
    pat_c = rng.normal(size=n_obs)
    patterns = {
        "pep0": pat_a,
        "pep1": pat_a,
        "pep2": pat_a,
        "pep3": pat_b,
        "pep4": pat_b,
        "pep5": pat_c,
        "pep6": pat_c,
        "pep7": pat_c,
    }
    cols = {
        p: np.abs(patterns[p] + rng.normal(scale=0.1, size=n_obs)) + 1.0
        for p in peps
    }
    X = np.vstack([cols[p] for p in peps]).T

    obs = pd.DataFrame(
        {"sample_id": [f"s{i}" for i in range(n_obs)]},
        index=[f"s{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(
        {"peptide_id": peps, "protein_id": [prot_map[p] for p in peps]},
        index=peps,
    )
    adata = ad.AnnData(X=X, obs=obs, var=var)

    pr.tl.pairwise_peptide_correlations(adata)
    pr.tl.peptide_dendograms_by_correlation(adata)
    pr.tl.peptide_clusters_from_dendograms(
        adata, n_clusters=2, min_peptides_per_cluster=2
    )
    return adata


def _protein_peptides(adata, protein):
    var = adata.var
    return sorted(var.index[var["protein_id"] == protein].tolist())


def tick_labels(ax, axis="y"):
    ticks = ax.get_yticklabels() if axis == "y" else ax.get_xticklabels()
    return [t.get_text() for t in ticks]


def _rendered(ax):
    """Tick labels and the matrix exactly as drawn (NaN for masked)."""
    rows, cols = tick_labels(ax), tick_labels(ax, "x")
    values = ax.collections[0].get_array().astype(float)
    return (
        rows,
        cols,
        np.ma.filled(values, np.nan).reshape(len(rows), len(cols)),
    )


def _recorrelate(adata):
    """Recompute .uns correlations after editing adata.X."""
    pr.tl.pairwise_peptide_correlations(adata)
    return adata


def _spy_clustermap(monkeypatch):
    """Capture kwargs passed to sns.clustermap; call through."""
    import proteopy.pl.copf as copf_mod

    captured = {}
    original = copf_mod.sns.clustermap

    def spy(data, **kwargs):
        captured["data"] = data
        captured["kwargs"] = kwargs
        return original(data, **kwargs)

    monkeypatch.setattr(copf_mod.sns, "clustermap", spy)
    return captured


# -- Signature contract --------------------------------------------------


def test_signature_locked():
    sig = inspect.signature(_function())
    params = list(sig.parameters.values())
    names = [p.name for p in params]
    assert names == [
        "adata",
        "protein",
        "corr_key",
        "margin_color",
        "method",
        "cluster",
        "linkage",
        "color_scheme",
        "cmap",
        "xticklabels",
        "yticklabels",
        "figsize",
        "show",
        "ax",
        "print_stats",
        "save",
    ]
    # protein is required (no default)
    assert sig.parameters["protein"].default is inspect.Parameter.empty
    defaults = sig.parameters
    assert defaults["corr_key"].default == "pairwise_peptide_correlations"
    assert defaults["margin_color"].default == "proteoform_id"
    assert defaults["method"].default == "average"
    assert defaults["cluster"].default is True
    assert defaults["linkage"].default is None
    assert defaults["cmap"].default == "coolwarm"
    assert defaults["xticklabels"].default == "auto"
    assert defaults["yticklabels"].default == "auto"
    assert defaults["show"].default is True
    assert defaults["ax"].default is False
    assert defaults["print_stats"].default is False
    assert defaults["save"].default is None


# -- Return-value semantics ----------------------------------------------


def test_returns_axes_when_ax_true():
    adata = _copf_adata()
    out = _function()(adata, protein="P1", show=False, ax=True)
    assert isinstance(out, Axes)


def test_returns_none_when_ax_false():
    adata = _copf_adata()
    out = _function()(adata, protein="P1", show=False, ax=False)
    assert out is None


# -- Clustering: symmetry, linkage pass-through, toggle -------------------


def test_default_clustering_is_symmetric():
    adata = _copf_adata()
    axm = _function()(adata, protein="P1", show=False, ax=True)
    x = [t.get_text() for t in axm.get_xticklabels()]
    y = [t.get_text() for t in axm.get_yticklabels()]
    assert x == y


def test_cluster_false_orders_plain_ids_lexicographically():
    adata = _copf_adata()
    axm = _function()(adata, protein="P1", cluster=False, show=False, ax=True)
    y = [t.get_text() for t in axm.get_yticklabels()]
    assert y == _protein_peptides(adata, "P1")


def test_cluster_false_follows_peptide_id_categories():
    adata = _copf_adata()
    categories = list(reversed(adata.var["peptide_id"].tolist()))
    adata.var["peptide_id"] = pd.Categorical(
        adata.var["peptide_id"], categories=categories, ordered=True
    )
    axm = _function()(adata, protein="P1", cluster=False, show=False, ax=True)
    y = [t.get_text() for t in axm.get_yticklabels()]
    assert y == ["pep4", "pep3", "pep2", "pep1", "pep0"]


def test_supplied_linkage_shadows_both_axes(monkeypatch):
    adata = _copf_adata()
    captured = _spy_clustermap(monkeypatch)

    # Build a linkage over P1's peptides.
    corr = adata.uns["pairwise_peptide_correlations"].loc[["P1"]]
    from proteopy.utils._matrix_wrangling import (
        reconstruct_symmetric_matrix_from_long,
    )

    mat = reconstruct_symmetric_matrix_from_long(
        corr, var_a_col="pepA", var_b_col="pepB", value_col="PCC"
    ).to_numpy(dtype=float)
    dist = 1.0 - mat
    np.fill_diagonal(dist, 0.0)
    Z = scipy_linkage(
        squareform(np.clip(dist, 0, 2), checks=False), method="complete"
    )

    _function()(adata, protein="P1", linkage=Z, show=False, ax=True)
    kwargs = captured["kwargs"]
    assert kwargs["row_linkage"] is Z
    assert kwargs["col_linkage"] is Z


# -- Annotation strips ----------------------------------------------------


def test_margin_color_string_single_strip(monkeypatch):
    adata = _copf_adata()
    captured = _spy_clustermap(monkeypatch)
    _function()(
        adata, protein="P1", margin_color="cluster_id", show=False, ax=False
    )
    row_colors = captured["kwargs"]["row_colors"]
    assert list(row_colors.columns) == ["cluster_id"]


def test_margin_color_list_multiple_strips(monkeypatch):
    adata = _copf_adata()
    captured = _spy_clustermap(monkeypatch)
    _function()(
        adata,
        protein="P1",
        margin_color=["cluster_id", "proteoform_id"],
        show=False,
        ax=False,
    )
    row_colors = captured["kwargs"]["row_colors"]
    assert list(row_colors.columns) == ["cluster_id", "proteoform_id"]
    # Same strips mirrored to columns (symmetric).
    assert captured["kwargs"]["col_colors"] is row_colors


def test_annotation_distinct_colors_match_categories(monkeypatch):
    adata = _copf_adata()
    captured = _spy_clustermap(monkeypatch)
    _function()(
        adata, protein="P1", margin_color="cluster_id", show=False, ax=False
    )
    strip = captured["kwargs"]["row_colors"]["cluster_id"]
    n_colors = len({tuple(np.atleast_1d(c)) for c in strip})
    n_cats = adata.var.loc[
        _protein_peptides(adata, "P1"), "cluster_id"
    ].nunique()
    assert n_colors == n_cats


# -- Labelling from ID columns -------------------------------------------


def test_tick_labels_from_peptide_id():
    adata = _copf_adata()
    axm = _function()(adata, protein="P1", show=False, ax=True)
    y = [t.get_text() for t in axm.get_yticklabels()]
    assert sorted(y) == _protein_peptides(adata, "P1")


# -- Validation -----------------------------------------------------------


def test_unknown_protein_raises():
    adata = _copf_adata()
    with pytest.raises(ValueError, match="not found"):
        _function()(adata, protein="NOPE", show=False)


def test_missing_corr_key_raises():
    adata = _copf_adata()
    del adata.uns["pairwise_peptide_correlations"]
    with pytest.raises(ValueError, match="pairwise_peptide_correlations"):
        _function()(adata, protein="P1", show=False)


def test_missing_margin_color_column_raises():
    adata = _copf_adata()
    with pytest.raises(KeyError, match="not found in adata.var"):
        _function()(
            adata, protein="P1", margin_color="does_not_exist", show=False
        )


def test_bad_margin_color_type_raises():
    adata = _copf_adata()
    with pytest.raises(TypeError, match="margin_color"):
        _function()(adata, protein="P1", margin_color=123, show=False)


# -- Lifecycle ------------------------------------------------------------


def test_save_writes_png(tmp_path):
    adata = _copf_adata()
    out = tmp_path / "heatmap.png"
    _function()(adata, protein="P1", show=False, save=out)
    assert out.exists() and out.stat().st_size > 0


def test_input_not_mutated():
    adata = _copf_adata()
    var_before = adata.var.copy()
    uns_before = adata.uns["pairwise_peptide_correlations"].copy()
    _function()(adata, protein="P1", show=False)
    pd.testing.assert_frame_equal(adata.var, var_before)
    pd.testing.assert_frame_equal(
        adata.uns["pairwise_peptide_correlations"], uns_before
    )


# -- Robustness: NaN correlations, NaN annotations, colour schemes -------


def _with_nan_correlation(adata):
    corrs = adata.uns["pairwise_peptide_correlations"].copy()
    first = np.flatnonzero(corrs.index == "P1")[0]
    corrs.iloc[first, corrs.columns.get_loc("PCC")] = np.nan
    adata.uns["pairwise_peptide_correlations"] = corrs
    return adata


def test_nan_correlations_are_clustered_as_zero_with_warning():
    adata = _with_nan_correlation(_copf_adata())
    with pytest.warns(UserWarning, match="treated as r = 0"):
        axm = _function()(adata, protein="P1", show=False, ax=True)
    assert tick_labels(axm, "x") == tick_labels(axm)


def test_nan_correlations_cluster_false_draws():
    adata = _with_nan_correlation(_copf_adata())
    out = _function()(adata, protein="P1", cluster=False, show=False, ax=True)
    assert isinstance(out, Axes)


def test_margin_color_with_missing_values(monkeypatch):
    adata = _copf_adata()
    adata.var["region"] = pd.Series(
        ["N-term", None, "C-term", None, "N-term", None, None, None],
        index=adata.var_names,
        dtype=object,
    )
    captured = _spy_clustermap(monkeypatch)
    _function()(adata, protein="P1", margin_color="region", show=False)
    strip = captured["kwargs"]["row_colors"]["region"]
    gray = plt.matplotlib.colors.to_rgba("lightgray")
    assert sum(tuple(c) == gray for c in strip) == 2


def test_color_scheme_dict_keyed_by_raw_values(monkeypatch):
    adata = _copf_adata()
    values = adata.var.loc[_protein_peptides(adata, "P1"), "cluster_id"]
    scheme = {v: "black" for v in values.unique()}
    captured = _spy_clustermap(monkeypatch)
    _function()(
        adata,
        protein="P1",
        margin_color="cluster_id",
        color_scheme=scheme,
        show=False,
    )
    strip = captured["kwargs"]["row_colors"]["cluster_id"]
    assert set(strip) == {"black"}


def test_print_stats_per_margin_color(capsys):
    adata = _copf_adata()
    adata.var = adata.var.drop(columns=["cluster_id"])
    _function()(
        adata,
        protein="P1",
        margin_color="proteoform_id",
        print_stats=True,
        show=False,
    )
    out = capsys.readouterr().out
    assert "Peptide correlation summary" in out
    assert "Per proteoform_id" in out


# -- Margin annotations: default, colours, legends ----------------------


def test_default_margin_is_proteoform_id(monkeypatch):
    adata = _copf_adata()
    captured = _spy_clustermap(monkeypatch)
    _function()(adata, protein="P1", show=False)
    assert list(captured["kwargs"]["row_colors"].columns) == ["proteoform_id"]


def test_multiple_margins_use_disjoint_colours(monkeypatch):
    adata = _copf_adata()
    captured = _spy_clustermap(monkeypatch)
    _function()(
        adata,
        protein="P1",
        margin_color=["cluster_id", "proteoform_id", "protein_id"],
        show=False,
    )
    strips = captured["kwargs"]["row_colors"]
    colours = {
        col: {tuple(np.atleast_1d(c)) for c in strips[col]}
        for col in strips.columns
    }
    assert not colours["cluster_id"] & colours["proteoform_id"]
    assert not colours["cluster_id"] & colours["protein_id"]
    assert not colours["proteoform_id"] & colours["protein_id"]


def test_one_legend_per_margin_column():
    adata = _copf_adata()
    margins = ["cluster_id", "proteoform_id"]
    axm = _function()(
        adata, protein="P1", margin_color=margins, show=False, ax=True
    )
    titles = [leg.get_title().get_text() for leg in axm.figure.legends]
    assert titles == margins


def test_legends_clear_of_all_axes_text_and_inside_figure():
    adata = _copf_adata()
    axm = _function()(
        adata,
        protein="P1",
        margin_color=["cluster_id", "proteoform_id", "protein_id"],
        show=False,
        ax=True,
    )
    fig = axm.figure
    renderer = fig.canvas.get_renderer()
    text_right = max(ax.get_tightbbox(renderer).x1 for ax in fig.axes)
    boxes = [leg.get_window_extent(renderer) for leg in fig.legends]
    assert all(box.x0 >= text_right for box in boxes)
    assert all(box.x1 <= fig.bbox.x1 for box in boxes)
    for upper, lower in zip(boxes, boxes[1:]):
        assert lower.y1 <= upper.y0


# -- Edge cases: input tables, arguments, backends, layout -------------


def test_several_correlations_per_pair_raise():
    adata = _copf_adata()
    corrs = adata.uns["pairwise_peptide_correlations"]
    duplicate = corrs.loc[["P1"]].iloc[[0]].assign(PCC=-0.5)
    adata.uns["pairwise_peptide_correlations"] = pd.concat([corrs, duplicate])
    with pytest.raises(ValueError, match="several correlations per"):
        _function()(adata, protein="P1", show=False)


def test_peptides_filtered_after_correlations_warn_and_are_dropped():
    adata = _copf_adata()
    sub = adata[:, adata.var_names != "pep0"].copy()
    with pytest.warns(UserWarning, match="no longer in adata.var"):
        axm = _function()(sub, protein="P1", show=False, ax=True)
    assert "pep0" not in tick_labels(axm)
    assert len(tick_labels(axm)) == 4


def test_fewer_than_two_remaining_peptides_raise():
    adata = _copf_adata()
    dropped = {"pep1", "pep2", "pep3", "pep4"}
    sub = adata[:, [p for p in adata.var_names if p not in dropped]].copy()
    with pytest.warns(UserWarning):
        with pytest.raises(ValueError, match="Fewer than two"):
            _function()(sub, protein="P1", show=False)


def test_empty_margin_list_raises():
    with pytest.raises(ValueError, match="at least one column"):
        _function()(_copf_adata(), protein="P1", margin_color=[], show=False)


def test_duplicate_margin_columns_raise():
    with pytest.raises(ValueError, match="Duplicate columns"):
        _function()(
            _copf_adata(),
            protein="P1",
            margin_color=["cluster_id", "cluster_id"],
            show=False,
        )


def test_bad_save_type_raises_before_drawing():
    adata = _copf_adata()
    before = set(plt.get_fignums())
    with pytest.raises(TypeError, match="save"):
        _function()(adata, protein="P1", save=123, show=False)
    assert set(plt.get_fignums()) == before


def test_linkage_of_wrong_size_raises():
    adata = _copf_adata()
    Z = scipy_linkage(np.random.default_rng(0).random((3, 2)))
    with pytest.raises(ValueError, match=r"shape \(4, 4\)"):
        _function()(adata, protein="P1", linkage=Z, show=False)


def test_print_stats_groups_follow_legend_order(capsys):
    adata = _copf_adata()
    adata.var["grp"] = [2, 10, 2, 10, 2, 2, 10, 2]
    axm = _function()(
        adata,
        protein="P1",
        margin_color="grp",
        print_stats=True,
        show=False,
        ax=True,
    )
    legend = [t.get_text() for t in axm.figure.legends[0].get_texts()]
    table = capsys.readouterr().out.split("Per grp")[1].splitlines()[2:]
    stats = [line.split()[0] for line in table if line.strip()]
    assert legend == ["10", "2"]
    assert stats == legend


def test_all_nan_correlations_draw_without_runtime_warnings():
    adata = _copf_adata()
    corrs = adata.uns["pairwise_peptide_correlations"].copy()
    corrs.loc["P1", "PCC"] = np.nan
    adata.uns["pairwise_peptide_correlations"] = corrs
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _function()(
            adata,
            protein="P1",
            margin_color="protein_id",
            cluster=False,
            print_stats=True,
            show=False,
        )
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]


def test_missing_copf_columns_point_to_clustering_step():
    adata = _copf_adata()
    adata.var = adata.var.drop(columns=["cluster_id", "proteoform_id"])
    with pytest.raises(KeyError, match="peptide_clusters_from_dendograms"):
        _function()(adata, protein="P1", show=False)


@pytest.mark.parametrize("backend", ["pdf", "svg"])
def test_vector_backends(backend, tmp_path):
    adata = _copf_adata()
    plt.switch_backend(backend)
    try:
        _function()(
            adata, protein="P1", show=False, save=tmp_path / f"h.{backend}"
        )
    finally:
        plt.switch_backend("Agg")
    assert (tmp_path / f"h.{backend}").stat().st_size > 0


# The deliberately tiny figure is too small for seaborn's own layout
@pytest.mark.filterwarnings("ignore:Tight layout not applied:UserWarning")
def test_tall_legends_stay_inside_figure():
    adata = _copf_adata()
    adata.var["per_peptide"] = adata.var["peptide_id"].astype(str)
    axm = _function()(
        adata,
        protein="P1",
        margin_color=["per_peptide", "proteoform_id", "protein_id"],
        figsize=(2.5, 1.2),
        show=False,
        ax=True,
    )
    fig = axm.figure
    renderer = fig.canvas.get_renderer()
    for legend in fig.legends:
        box = legend.get_window_extent(renderer)
        assert box.y0 >= 0
        assert box.x1 <= fig.bbox.x1


# -- Checklist scenarios: missing data, degenerate inputs, rendering -----


def test_all_na_peptide_shown_as_empty_row_and_column():
    adata = _copf_adata()
    adata.X[:, 0] = np.nan  # pep0 never measured
    _recorrelate(adata)
    with pytest.warns(UserWarning) as record:
        axm = _function()(adata, protein="P1", show=False, ax=True)
    messages = " | ".join(str(w.message) for w in record)
    assert "no correlations" in messages
    assert "treated as r = 0" in messages
    rows, cols, values = _rendered(axm)
    assert "pep0" in rows
    assert np.isnan(values[rows.index("pep0")]).all()
    assert np.isnan(values[:, cols.index("pep0")]).all()


def test_constant_peptide_draws_with_default_clustering():
    adata = _copf_adata()
    adata.X[:, 1] = 5.0  # zero variance -> NaN PCC
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # scipy ConstantInputWarning
        _recorrelate(adata)
    with pytest.warns(UserWarning, match="NaN"):
        axm = _function()(adata, protein="P1", show=False, ax=True)
    rows, _, values = _rendered(axm)
    assert np.isnan(values[rows.index("pep1")]).sum() == len(rows) - 1


def test_nan_cells_hatched_and_in_legend():
    adata = _with_nan_correlation(_copf_adata())
    with pytest.warns(UserWarning):
        axm = _function()(adata, protein="P1", show=False, ax=True)
    assert axm.patch.get_hatch()
    titles = [leg.get_title().get_text() for leg in axm.figure.legends]
    assert "correlation" in titles


def test_complete_matrix_has_no_nan_legend():
    axm = _function()(_copf_adata(), protein="P1", show=False, ax=True)
    titles = [leg.get_title().get_text() for leg in axm.figure.legends]
    assert "correlation" not in titles
    assert not axm.patch.get_hatch()


def test_fewer_than_three_samples_warn():
    adata = _recorrelate(_copf_adata()[:2].copy())
    with pytest.warns(UserWarning, match="fewer than 3 samples"):
        _function()(adata, protein="P1", show=False)


def test_two_peptide_protein_renders_2x2():
    adata = _copf_adata()
    adata = _recorrelate(adata[:, ["pep0", "pep1", "pep5", "pep6"]].copy())
    axm = _function()(adata, protein="P1", show=False, ax=True)
    rows, cols, values = _rendered(axm)
    assert values.shape == (2, 2)
    assert np.allclose(np.diag(values), 1.0)
    assert np.allclose(values, values.T)


def test_identical_peptides_render_r_of_one():
    adata = _copf_adata()
    adata.X[:, 1] = adata.X[:, 0]
    _recorrelate(adata)
    rows, cols, values = _rendered(
        _function()(adata, protein="P1", show=False, ax=True)
    )
    assert values[rows.index("pep0"), cols.index("pep1")] == pytest.approx(1)


@pytest.mark.parametrize("cluster", [True, False])
def test_rendered_cells_match_correlations(cluster):
    adata = _copf_adata()
    long = adata.uns["pairwise_peptide_correlations"].loc["P1"]
    truth = {}
    for a, b, r in long[["pepA", "pepB", "PCC"]].itertuples(index=False):
        truth[(a, b)] = truth[(b, a)] = r
    axm = _function()(
        adata, protein="P1", cluster=cluster, show=False, ax=True
    )
    rows, cols, values = _rendered(axm)
    assert rows == cols
    assert np.allclose(np.diag(values), 1.0)
    for i, a in enumerate(rows):
        for j, b in enumerate(cols):
            if i != j:
                assert values[i, j] == pytest.approx(truth[(a, b)])


def test_default_labels_do_not_overlap_for_many_peptides():
    rng = np.random.default_rng(3)
    n_obs, n_pep = 12, 45
    peptides = [f"PEPTIDE{i:02d}K" for i in range(n_pep)]
    obs_names = [f"s{i}" for i in range(n_obs)]
    adata = ad.AnnData(
        X=rng.normal(20, 1, (n_obs, n_pep)),
        obs=pd.DataFrame({"sample_id": obs_names}, index=obs_names),
        var=pd.DataFrame(
            {"peptide_id": peptides, "protein_id": "P9"}, index=peptides
        ),
    )
    _recorrelate(adata)
    axm = _function()(
        adata, protein="P9", margin_color="protein_id", show=False, ax=True
    )
    renderer = axm.figure.canvas.get_renderer()
    boxes = sorted(
        (t.get_window_extent(renderer) for t in axm.get_yticklabels()),
        key=lambda box: box.y0,
    )
    assert all(lower.y1 <= upper.y0 for lower, upper in zip(boxes, boxes[1:]))
