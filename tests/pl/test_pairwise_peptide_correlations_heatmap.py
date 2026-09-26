"""Tests for pr.pl.pairwise_peptide_correlations_heatmap.

Covers: signature contract, return-value semantics, symmetric
clustering and linkage pass-through, cluster toggle, annotation
strips, ID-column labelling, argument validation, and lifecycle
(save / non-mutation).
"""

import inspect

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
    assert defaults["margin_color"].default == "cluster_id"
    assert defaults["method"].default == "average"
    assert defaults["cluster"].default is True
    assert defaults["linkage"].default is None
    assert defaults["cmap"].default == "coolwarm"
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


def test_cluster_false_keeps_input_order():
    adata = _copf_adata()
    axm = _function()(adata, protein="P1", cluster=False, show=False, ax=True)
    y = [t.get_text() for t in axm.get_yticklabels()]
    assert y == _protein_peptides(adata, "P1")


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
