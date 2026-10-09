# `n_peptides_per_sample` / `n_proteins_per_sample` are built with
# `partial_with_docsig`. Pylint reads the partial's fixed `level=`
# as if it had consumed the first positional parameter, so every
# call below is flagged whichever way `adata` is passed. The calls
# are correct; the inference is not.
# pylint: disable=too-many-function-args

import matplotlib
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from proteopy.pl.stats import (  # noqa: E402
    n_peptides_per_sample,
    n_proteins_per_sample,
    n_var_per_sample,
)

# Sample names whose lexicographic order ("F1", "F10", "F2")
# differs from the fraction order a user means ("F1", "F2",
# "F10") and from the order the rows sit in the object.
SAMPLES = ["F2", "F10", "F1"]
FRACTION_ORDER = ["F1", "F2", "F10"]
LEXICOGRAPHIC_ORDER = ["F1", "F10", "F2"]

# Detected peptides per sample: F2 -> 3, F10 -> 1, F1 -> 2.
X = np.array(
    [
        [1.0, 1.0, 1.0],
        [1.0, np.nan, np.nan],
        [1.0, 1.0, np.nan],
    ],
)


def _make_adata(sample_categories=None, group=None) -> AnnData:
    """Peptide-level AnnData with three samples of one protein."""
    var_names = [f"pep_{i}" for i in range(3)]
    var = pd.DataFrame(
        {
            "peptide_id": var_names,
            "protein_id": ["prot_0"] * 3,
        },
        index=var_names,
    )

    sample_id = pd.Series(SAMPLES, index=SAMPLES)
    if sample_categories is not None:
        sample_id = pd.Series(
            pd.Categorical(
                SAMPLES,
                categories=sample_categories,
                ordered=True,
            ),
            index=SAMPLES,
        )

    obs = pd.DataFrame({"sample_id": sample_id}, index=SAMPLES)
    if group is not None:
        obs["group"] = group
    return AnnData(X=X.copy(), obs=obs, var=var)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _xticklabels(ax) -> list:
    # Category tick labels are only populated once drawn.
    ax.get_figure().canvas.draw()
    return [t.get_text() for t in ax.get_xticklabels()]


def _bar_heights(ax) -> list:
    return [round(patch.get_height(), 6) for patch in ax.patches]


# -- Rule 1: category order, else lexicographic ---------------


def test_default_order_is_lexicographic_when_not_categorical():
    adata = _make_adata()
    ax = n_peptides_per_sample(adata, show=False)
    assert _xticklabels(ax) == LEXICOGRAPHIC_ORDER
    # Never the order the samples sit in the object.
    assert _xticklabels(ax) != SAMPLES


def test_default_order_follows_sample_id_categories():
    adata = _make_adata(sample_categories=FRACTION_ORDER)
    ax = n_peptides_per_sample(adata, show=False)
    assert _xticklabels(ax) == FRACTION_ORDER


def test_default_order_carries_the_matching_counts():
    adata = _make_adata(sample_categories=FRACTION_ORDER)
    ax = n_peptides_per_sample(adata, show=False)
    # F1 -> 2, F2 -> 3, F10 -> 1
    assert _bar_heights(ax) == [2.0, 3.0, 1.0]


def test_protein_level_default_order_matches_peptide_level():
    adata = _make_adata()
    ax = n_proteins_per_sample(adata, show=False)
    assert _xticklabels(ax) == LEXICOGRAPHIC_ORDER


# -- Rule 3: group_by and order_by follow the same rule -------


def test_group_by_order_is_lexicographic_when_not_categorical():
    adata = _make_adata(group=["beta", "alpha", "beta"])
    ax = n_peptides_per_sample(
        adata,
        group_by="group",
        show=False,
    )
    assert _xticklabels(ax) == ["alpha", "beta"]


def test_group_by_order_follows_categories():
    group = pd.Categorical(
        ["beta", "alpha", "beta"],
        categories=["beta", "alpha"],
        ordered=True,
    )
    adata = _make_adata(group=group)
    ax = n_peptides_per_sample(
        adata,
        group_by="group",
        show=False,
    )
    assert _xticklabels(ax) == ["beta", "alpha"]


def test_order_by_blocks_are_lexicographic_when_not_categorical():
    adata = _make_adata(group=["beta", "alpha", "beta"])
    ax = n_peptides_per_sample(
        adata,
        order_by="group",
        show=False,
    )
    # alpha block (F10) first, then the beta block (F1, F2)
    assert _xticklabels(ax) == ["F10", "F1", "F2"]


def test_order_by_blocks_follow_categories():
    group = pd.Categorical(
        ["beta", "alpha", "beta"],
        categories=["beta", "alpha"],
        ordered=True,
    )
    adata = _make_adata(group=group)
    ax = n_peptides_per_sample(
        adata,
        order_by="group",
        show=False,
    )
    assert _xticklabels(ax) == ["F1", "F2", "F10"]


def test_order_by_samples_within_a_block_follow_categories():
    adata = _make_adata(
        sample_categories=FRACTION_ORDER,
        group=["alpha", "alpha", "alpha"],
    )
    ax = n_peptides_per_sample(
        adata,
        order_by="group",
        show=False,
    )
    assert _xticklabels(ax) == FRACTION_ORDER


def test_order_by_keeps_unlabelled_samples_in_a_trailing_block():
    adata = _make_adata(group=["alpha", None, "alpha"])
    ax = n_peptides_per_sample(
        adata,
        order_by="group",
        show=False,
    )
    assert _xticklabels(ax) == ["F1", "F2", "F10"]


def test_order_excludes_the_unlabelled_block():
    adata = _make_adata(group=["alpha", None, "alpha"])
    ax = n_peptides_per_sample(
        adata,
        order_by="group",
        order=["alpha"],
        show=False,
    )
    assert _xticklabels(ax) == ["F1", "F2"]


# -- Rule 2: an explicit order parameter wins -----------------


def test_ascending_overrides_the_default_order():
    adata = _make_adata(sample_categories=FRACTION_ORDER)
    ax = n_peptides_per_sample(
        adata,
        ascending=True,
        show=False,
    )
    # F10 -> 1, F1 -> 2, F2 -> 3
    assert _xticklabels(ax) == ["F10", "F1", "F2"]
    assert _bar_heights(ax) == [1.0, 2.0, 3.0]


def test_descending_overrides_the_default_order():
    adata = _make_adata(sample_categories=FRACTION_ORDER)
    ax = n_peptides_per_sample(
        adata,
        ascending=False,
        show=False,
    )
    assert _xticklabels(ax) == ["F2", "F1", "F10"]


def test_ascending_sorts_within_order_by_blocks():
    adata = _make_adata(group=["beta", "alpha", "beta"])
    ax = n_peptides_per_sample(
        adata,
        order_by="group",
        ascending=True,
        show=False,
    )
    # alpha block (F10), then beta sorted by count (F1=2, F2=3)
    assert _xticklabels(ax) == ["F10", "F1", "F2"]
    assert _bar_heights(ax) == [1.0, 2.0, 3.0]


def test_order_overrides_ascending_with_a_warning():
    adata = _make_adata()
    with pytest.warns(UserWarning, match="`ascending` is ignored"):
        ax = n_peptides_per_sample(
            adata,
            order=["F2", "F1", "F10"],
            ascending=True,
            show=False,
        )
    assert _xticklabels(ax) == ["F2", "F1", "F10"]


def test_ascending_warns_when_group_by_is_set():
    adata = _make_adata(group=["beta", "alpha", "beta"])
    with pytest.warns(UserWarning, match="`ascending` is ignored"):
        n_peptides_per_sample(
            adata,
            group_by="group",
            ascending=True,
            show=False,
        )


# -- `order` subsets ------------------------------------------


def test_order_subsets_the_samples():
    adata = _make_adata()
    ax = n_peptides_per_sample(
        adata,
        order=["F2", "F1"],
        show=False,
    )
    assert _xticklabels(ax) == ["F2", "F1"]
    assert _bar_heights(ax) == [3.0, 2.0]


def test_order_subsets_order_by_groups():
    adata = _make_adata(group=["beta", "alpha", "beta"])
    ax = n_peptides_per_sample(
        adata,
        order_by="group",
        order=["alpha"],
        show=False,
    )
    assert _xticklabels(ax) == ["F10"]


def test_order_subsets_group_by_groups():
    adata = _make_adata(group=["beta", "alpha", "beta"])
    ax = n_peptides_per_sample(
        adata,
        group_by="group",
        order=["alpha"],
        show=False,
    )
    assert _xticklabels(ax) == ["alpha"]


def test_order_rejects_values_outside_sample_id():
    adata = _make_adata()
    with pytest.raises(
        ValueError,
        match=r"adata.obs\['sample_id'\]",
    ):
        n_peptides_per_sample(
            adata,
            order=["F2", "F42"],
            show=False,
        )


# -- print_stats reflects what is plotted ---------------------


def test_print_stats_per_sample(capsys):
    adata = _make_adata()
    n_peptides_per_sample(adata, print_stats=True, show=False)
    assert "mean_count" in capsys.readouterr().out


def test_print_stats_with_order_by(capsys):
    adata = _make_adata(group=["beta", "alpha", "beta"])
    n_peptides_per_sample(
        adata,
        order_by="group",
        print_stats=True,
        show=False,
    )
    out = capsys.readouterr().out
    assert "Global:" in out
    assert "Per group:" in out


def test_print_stats_excludes_subset_out_groups(capsys):
    adata = _make_adata(group=["beta", "alpha", "beta"])
    n_peptides_per_sample(
        adata,
        group_by="group",
        order=["alpha"],
        print_stats=True,
        show=False,
    )
    out = capsys.readouterr().out
    assert "alpha" in out
    assert "beta" not in out


# -- Labels come from the ID column, not the index ------------


def test_labels_come_from_sample_id_not_obs_names():
    adata = _make_adata(sample_categories=FRACTION_ORDER)
    # An index cannot carry a category order; the column can,
    # and it is the column the plot must follow.
    assert list(adata.obs_names) == SAMPLES
    ax = n_var_per_sample(adata, show=False)
    assert _xticklabels(ax) == FRACTION_ORDER
