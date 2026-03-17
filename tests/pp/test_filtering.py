import pytest
import numpy as np
import pandas as pd
from anndata import AnnData
import scipy.sparse as sp

from proteopy.pp.filtering import (
    filter_axis,
    filter_proteins_by_peptide_count,
    remove_zero_variance_vars,
    remove_contaminants
    )


def _make_adata_filter_obs_base() -> AnnData:
    """Six obs, five vars with increasing missingness; some zeros present."""
    n = np.nan
    X = np.array(
        [
            [1, 1, 2, 2, 3],                 # obs0: complete
            [n, 1, 2, 2, 3],                 # obs1: 4/5 complete
            [n, n, 2, 2, 3],                 # obs2: 3/5 complete
            [n, n, n, 2, 3],                 # obs3: 2/5 complete
            [0, 1, 2, 2, 3],                 # obs4: complete and a zero
            [0, n, 2, 2, 3],                 # obs5: 4/5 complete and a zero
        ],
        dtype=float,
    )
    obs_names = [f"obs{i}" for i in range(6)]
    var_names = [f"protein_{i}" for i in range(5)]
    obs = pd.DataFrame({"sample_id": obs_names}, index=obs_names)
    var = pd.DataFrame({"protein_id": var_names}, index=var_names)
    return AnnData(X=X, obs=obs, var=var)


def _make_adata_filter_obs_groupby_singletons() -> AnnData:
    """Two vars, two groups"""
    n = np.nan
    X = np.array(
        [
            [n, n],                 # obs0
            [1, n],                 # obs1
            [1, 1],                 # obs2
        ],
        dtype=float,
    )
    obs_names = [f"obs{i}" for i in range(3)]
    var_names = [f"protein_{i}" for i in range(2)]
    obs = pd.DataFrame({"sample_id": obs_names},index=obs_names)
    var = pd.DataFrame(
        {
            "protein_id": var_names,
            "group": ["g1", "g2"],
        },
        index=var_names,
    )
    return AnnData(X=X, obs=obs, var=var)


def _make_adata_filter_obs_groupby() -> AnnData:
    """Five vars, two groups"""
    n = np.nan
    X = np.array(
        [
            [1, 1, 2, 2, 3],         # obs0: both groups complete
            [1, n, 2, 2, 3],         # obs1: group 0 -> 1/2 complete
            [1, 1, 2, 2, n],         # obs2: group 1 -> 2/3 incomplete
            [1, n, 2, 2, n],         # obs3: group 0 -> 1/2 complete, group 1 -> 2/3 complete
            [1, n, 2, n, n],         # obs4: group 0 -> 1/2 complete, group 1 -> 1/3 complete
        ],
        dtype=float,
    )
    obs_names = [f"obs{i}" for i in range(5)]
    var_names = [f"protein_{i}" for i in range(5)]
    obs = pd.DataFrame({"sample_id": obs_names}, index=obs_names)
    var = pd.DataFrame(
        {
            "protein_id": var_names,
            "group": ["g1", "g1", "g2", "g2", "g2"],
        },
        index=var_names,
    )
    return AnnData(X=X, obs=obs, var=var)


def _make_adata_filter_obs_groupby_na() -> AnnData:
    """
    Same as `_make_adata_filter_obs_groupby` but with an added NA group of
    four vars
    """
    n = np.nan
    X = np.array(
        [
            [1, 1, 2, 2, 3, 4, 4, 4, 4],  # obs0: all groups complete
            [1, n, 2, 2, 3, 4, 4, 4, 4],  # obs1: group 0 -> 1/2 complete
            [1, 1, 2, 2, n, 4, 4, 4, 4],  # obs2: group 1 -> 2/3 incomplete
            [1, n, 2, 2, n, 4, 4, 4, 4],  # obs3: group 0 -> 1/2 complete, group 1 -> 2/3 complete
            [1, n, 2, n, n, 4, 4, 4, 4],  # obs4: group 0 -> 1/2 complete, group 1 -> 1/3 complete
        ],
        dtype=float,
    )
    obs_names = [f"obs{i}" for i in range(5)]
    var_names = [f"protein_{i}" for i in range(9)]
    obs = pd.DataFrame({"sample_id": obs_names}, index=obs_names)
    var = pd.DataFrame({
            "protein_id": var_names,
            "group": ["g1", "g1", "g2", "g2", "g2", np.nan, np.nan, np.nan, np.nan],
        }, index=var_names,
    )
    return AnnData(X=X, obs=obs, var=var)


def _make_adata_filter_var_base() -> AnnData:
    """Five obs, six vars with increasing missingness across vars; some zeros."""
    n = np.nan
    X = np.array(
        [
            [1, n, n, n, 0, 0],
            [1, 1, n, n, 1, n],
            [2, 2, 2, n, 2, 2],
            [2, 2, 2, 2, 2, 2],
            [3, 3, 3, 3, 3, 3],
        ],
        dtype=float,
    )
    obs_names = [f"obs{i}" for i in range(5)]
    obs = pd.DataFrame({"sample_id": obs_names}, index=obs_names)
    var_names = [f"protein_{i}" for i in range(6)]
    var = pd.DataFrame({"protein_id": var_names}, index=var_names)
    return AnnData(X=X, obs=obs, var=var)


def _make_adata_filter_var_groupby_singletons() -> AnnData:
    """Three vars, two groups"""
    n = np.nan
    X = np.array(
        [
            [n, 1, 1],
            [n, n, 1],
        ],
        dtype=float,
    )
    obs_names = [f"obs{i}" for i in range(2)]
    obs = pd.DataFrame({"sample_id": obs_names, "group": ["g1", "g2"]}, index=obs_names)
    var_names = [f"protein_{i}" for i in range(3)]
    var = pd.DataFrame({
            "protein_id": var_names,
            "group": ["g1", "g2", "g2"],
        }, index=var_names,
    )
    return AnnData(X=X, obs=obs, var=var)


def _make_adata_filter_var_groupby() -> AnnData:
    """Five obs with obs groupings; vars differ in completeness per group."""
    n = np.nan
    X = np.array(
        [
            [1, 1, 1, 1, 1],
            [1, n, 1, n, n],
            [2, 2, 2, 2, 2],
            [2, 2, 2, 2, n],
            [3, 3, n, n, n],
        ],
        dtype=float,
    )
    obs_names = [f"obs{i}" for i in range(5)]
    obs = pd.DataFrame({
        "sample_id": obs_names,
        "group": ["g1", "g1", "g2", "g2", "g2"],
        }, index=obs_names,
    )
    var_names = [f"protein_{i}" for i in range(5)]
    var = pd.DataFrame({"protein_id": var_names}, index=var_names)
    return AnnData(X=X, obs=obs, var=var)


def _make_adata_filter_var_groupby_na() -> AnnData:
    """
    Same as `_make_adata_filter_var_groupby` but with an added NA group of
    four obs
    """
    n = np.nan
    X = np.array(
        [
            [1, 1, 1, 1, 1],
            [1, n, 1, n, n],
            [2, 2, 2, 2, 2],
            [2, 2, 2, 2, n],
            [3, 3, n, n, n],
            [4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4],
        ],
        dtype=float,
    )
    obs_names = [f"obs{i}" for i in range(9)]
    obs = pd.DataFrame(
        {
            "sample_id": obs_names,
            "group": ["g1", "g1", "g2", "g2", "g2", np.nan, np.nan, np.nan, np.nan],
        },
        index=obs_names,
    )
    var_names = [f"protein_{i}" for i in range(5)]
    var = pd.DataFrame({
            "protein_id": var_names,
            "group": ["g1", "g1", "g2", "g2", np.nan],
        }, index=var_names,
    )
    return AnnData(X=X, obs=obs, var=var)

# ── helpers: remove_zero_variance_vars ──────────────────────────────


def _make_adata_rzv_base() -> AnnData:
    """6 obs × 5 vars.  p0-p2 vary, p3 constant, p4 near-constant (<1e-8).

    Expected kept (atol=1e-8): [p0, p1, p2].
    """
    n = np.nan
    X = np.array(
        [
            [1, 1, 2, 2, 3],
            [n, 1, 2, 2, 3],
            [n, n, 4, 2, 3],
            [n, n, n, 2, 3.00001],
            [0, 2, 3, 2, 3],
            [0, n, 2, 2, 3],
        ],
        dtype=float,
    )
    obs_names = [f"s{i}" for i in range(6)]
    var_names = ["p0", "p1", "p2", "p3", "p4"]
    obs = pd.DataFrame({"sample_id": obs_names}, index=obs_names)
    var = pd.DataFrame({"protein_id": var_names}, index=var_names)
    return AnnData(X=X, obs=obs, var=var)


def _make_adata_rzv_one_allnan_col() -> AnnData:
    """6 obs × 5 vars.  p2 is entirely NaN; p3-p4 near-constant.

    Expected kept (atol=1e-8): [p0, p1].
    Warning: 1 all-NaN variable.
    """
    n = np.nan
    X = np.array(
        [
            [1, 1, n, 2, 3],
            [n, 1, n, 2, 3],
            [n, n, n, 2, 3],
            [n, n, n, 2, 3.00001],
            [0, 2, n, 2, 3],
            [0, n, n, 2, 3],
        ],
        dtype=float,
    )
    obs_names = [f"s{i}" for i in range(6)]
    var_names = ["p0", "p1", "p2", "p3", "p4"]
    obs = pd.DataFrame({"sample_id": obs_names}, index=obs_names)
    var = pd.DataFrame({"protein_id": var_names}, index=var_names)
    return AnnData(X=X, obs=obs, var=var)


def _make_adata_rzv_all_constant() -> AnnData:
    """6 obs × 5 vars.  Every column is constant or near-constant.

    Expected kept (atol=1e-8): [] (empty).
    """
    n = np.nan
    X = np.array(
        [
            [0, 1, 2, 2, 3],
            [n, 1, 2, 2, 3],
            [n, n, 2, 2, 3],
            [n, n, n, 2, 3.00001],
            [0, 1, 2, 2, 3],
            [0, n, 2, 2, 3],
        ],
        dtype=float,
    )
    obs_names = [f"s{i}" for i in range(6)]
    var_names = ["p0", "p1", "p2", "p3", "p4"]
    obs = pd.DataFrame({"sample_id": obs_names}, index=obs_names)
    var = pd.DataFrame({"protein_id": var_names}, index=var_names)
    return AnnData(X=X, obs=obs, var=var)


def _make_adata_rzv_all_vary() -> AnnData:
    """3 obs × 3 vars.  All columns have clear variance.

    Expected kept (atol=1e-8): [p0, p1, p2] (nothing removed).
    """
    X = np.array(
        [[1.0, 10.0, 100.0],
         [2.0, 20.0, 200.0],
         [3.0, 30.0, 300.0]],
    )
    obs_names = ["s0", "s1", "s2"]
    var_names = ["p0", "p1", "p2"]
    obs = pd.DataFrame({"sample_id": obs_names}, index=obs_names)
    var = pd.DataFrame({"protein_id": var_names}, index=var_names)
    return AnnData(X=X, obs=obs, var=var)


def _make_adata_rzv_single_var_varies() -> AnnData:
    """3 obs × 1 var.  The single variable has variance.

    Expected kept: [p0].
    """
    X = np.array([[1.0], [2.0], [3.0]])
    obs_names = ["s0", "s1", "s2"]
    var_names = ["p0"]
    obs = pd.DataFrame({"sample_id": obs_names}, index=obs_names)
    var = pd.DataFrame({"protein_id": var_names}, index=var_names)
    return AnnData(X=X, obs=obs, var=var)


def _make_adata_rzv_single_var_constant() -> AnnData:
    """3 obs × 1 var.  The single variable is constant.

    Expected kept: [] (empty).
    """
    X = np.array([[5.0], [5.0], [5.0]])
    obs_names = ["s0", "s1", "s2"]
    var_names = ["p0"]
    obs = pd.DataFrame({"sample_id": obs_names}, index=obs_names)
    var = pd.DataFrame({"protein_id": var_names}, index=var_names)
    return AnnData(X=X, obs=obs, var=var)


def _make_adata_rzv_multiple_allnan() -> AnnData:
    """3 obs × 4 vars.  p1, p2, p3 are all-NaN; p0 varies.

    Expected kept: [p0].  Warning: 3 all-NaN variables.
    """
    n = np.nan
    X = np.array(
        [
            [1.0, n, n, n],
            [2.0, n, n, n],
            [3.0, n, n, n],
        ],
    )
    obs_names = ["s0", "s1", "s2"]
    var_names = ["p0", "p1", "p2", "p3"]
    obs = pd.DataFrame({"sample_id": obs_names}, index=obs_names)
    var = pd.DataFrame({"protein_id": var_names}, index=var_names)
    return AnnData(X=X, obs=obs, var=var)


def _make_adata_rzv_atol_boundary() -> AnnData:
    """2 obs × 3 vars.  var(p0)=1.0, var(p1)=1.0, var(p2)=6.25.

    With atol=1.0: p0 and p1 removed (<=), p2 kept.
    """
    X = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 2.0, 5.0],
        ],
    )
    obs_names = ["s0", "s1"]
    var_names = ["p0", "p1", "p2"]
    obs = pd.DataFrame({"sample_id": obs_names}, index=obs_names)
    var = pd.DataFrame({"protein_id": var_names}, index=var_names)
    return AnnData(X=X, obs=obs, var=var)


def _make_adata_rzv_atol_zero() -> AnnData:
    """3 obs × 3 vars.  p0 constant (var=0), p1 tiny variance (~1e-16),
    p2 clear variance.

    With atol=0.0: p0 removed (var==0), p1 kept (var>0), p2 kept.
    """
    X = np.array(
        [
            [5.0, 1.0, 1.0],
            [5.0, 1.0 + 1e-8, 2.0],
            [5.0, 1.0, 3.0],
        ],
    )
    obs_names = ["s0", "s1", "s2"]
    var_names = ["p0", "p1", "p2"]
    obs = pd.DataFrame({"sample_id": obs_names}, index=obs_names)
    var = pd.DataFrame({"protein_id": var_names}, index=var_names)
    return AnnData(X=X, obs=obs, var=var)


def _make_adata_rzv_groupby() -> AnnData:
    """5 obs × 5 vars, 2 groups (g1: 2 obs, g2: 3 obs).

    p0: varies in both groups → kept
    p1: varies in g1, constant in g2 → removed
    p2: constant in both → removed
    p3: constant in g1, varies in g2 → removed (zero in g1)
    p4: constant in g1, varies in g2 → removed (zero in g1)

    Expected kept: [p0].
    """
    n = np.nan
    X = np.array(
        [
            [1, 1, 1, 1, 1],
            [2, 3, 1, n, 1],
            [2, 2, 2, 2, 2],
            [2, 2, 2, 2, n],
            [3, n, n, 5, 3],
        ],
        dtype=float,
    )
    obs_names = [f"s{i}" for i in range(5)]
    obs = pd.DataFrame(
        {"sample_id": obs_names,
         "group": ["g1", "g1", "g2", "g2", "g2"]},
        index=obs_names,
    )
    var_names = [f"p{i}" for i in range(5)]
    var = pd.DataFrame({"protein_id": var_names}, index=var_names)
    return AnnData(X=X, obs=obs, var=var)


def _make_adata_rzv_groupby_singletons() -> AnnData:
    """5 obs × 3 vars, each obs is its own group (5 singleton groups).

    Variance of a singleton is always 0 → all vars removed.
    Expected kept: [].
    """
    n = np.nan
    X = np.array(
        [
            [n, 1, 1],
            [n, n, 1],
            [n, 2, 3],
            [n, 3, 1],
            [n, n, 2],
        ],
        dtype=float,
    )
    obs_names = [f"s{i}" for i in range(5)]
    obs = pd.DataFrame(
        {"sample_id": obs_names,
         "group": ["g1", "g2", "g3", "g4", "g5"]},
        index=obs_names,
    )
    var_names = ["p0", "p1", "p2"]
    var = pd.DataFrame({"protein_id": var_names}, index=var_names)
    return AnnData(X=X, obs=obs, var=var)


def _make_adata_rzv_groupby_allnan_one_group() -> AnnData:
    """4 obs × 2 vars, 2 groups (A: 2 obs, B: 2 obs).

    p0: all-NaN in group A, varies in B → removed (all-NaN in any group)
    p1: varies in both groups → kept

    Expected kept: [p1].  Warning: 1 all-NaN in at least one group.
    """
    n = np.nan
    X = np.array(
        [
            [n, 1.0],
            [n, 2.0],
            [5.0, 3.0],
            [6.0, 4.0],
        ],
    )
    obs_names = ["s0", "s1", "s2", "s3"]
    obs = pd.DataFrame(
        {"sample_id": obs_names,
         "group": ["A", "A", "B", "B"]},
        index=obs_names,
    )
    var_names = ["p0", "p1"]
    var = pd.DataFrame({"protein_id": var_names}, index=var_names)
    return AnnData(X=X, obs=obs, var=var)


def _make_adata_rzv_groupby_single_group() -> AnnData:
    """4 obs × 3 vars, all obs in one group ("A").

    p0: varies → kept;  p1: constant → removed;  p2: varies → kept.
    Equivalent to non-grouped result.
    """
    X = np.array(
        [
            [1.0, 5.0, 10.0],
            [2.0, 5.0, 20.0],
            [3.0, 5.0, 30.0],
            [4.0, 5.0, 40.0],
        ],
    )
    obs_names = ["s0", "s1", "s2", "s3"]
    obs = pd.DataFrame(
        {"sample_id": obs_names,
         "group": ["A", "A", "A", "A"]},
        index=obs_names,
    )
    var_names = ["p0", "p1", "p2"]
    var = pd.DataFrame({"protein_id": var_names}, index=var_names)
    return AnnData(X=X, obs=obs, var=var)


def _make_adata_rzv_peptide_level() -> AnnData:
    """4 obs × 4 vars, peptide-level data (has peptide_id + protein_id).

    pep0: varies → kept;  pep1: constant → removed;
    pep2: varies → kept;  pep3: all-NaN → removed.
    Expected kept: [pep0, pep2].
    """
    n = np.nan
    X = np.array(
        [
            [1.0, 5.0, 10.0, n],
            [2.0, 5.0, 20.0, n],
            [3.0, 5.0, 30.0, n],
            [4.0, 5.0, 40.0, n],
        ],
    )
    obs_names = ["s0", "s1", "s2", "s3"]
    var_names = ["pep0", "pep1", "pep2", "pep3"]
    obs = pd.DataFrame({"sample_id": obs_names}, index=obs_names)
    var = pd.DataFrame(
        {
            "peptide_id": var_names,
            "protein_id": ["prot_A", "prot_A", "prot_B", "prot_B"],
        },
        index=var_names,
    )
    return AnnData(X=X, obs=obs, var=var)


def _make_adata_rzv_float32() -> AnnData:
    """3 obs × 3 vars, float32 dtype.  p1 constant.

    Expected kept: [p0, p2].
    """
    X = np.array(
        [[1.0, 5.0, 10.0],
         [2.0, 5.0, 20.0],
         [3.0, 5.0, 30.0]],
        dtype=np.float32,
    )
    obs_names = ["s0", "s1", "s2"]
    var_names = ["p0", "p1", "p2"]
    obs = pd.DataFrame({"sample_id": obs_names}, index=obs_names)
    var = pd.DataFrame({"protein_id": var_names}, index=var_names)
    return AnnData(X=X, obs=obs, var=var)

def _make_adata_remove_contaminants_base() -> AnnData:
    """5 obs, 5 vars"""
    X = np.array(
        [
            [1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2],
            [3, 3, 3, 3, 3],
            [4, 4, 4, 4, 4],
            [5, 5, 5, 5, 5],
        ],
        dtype=float,
    )
    obs_names = [f"obs{i}" for i in range(5)]
    obs = pd.DataFrame({"sample_id": obs_names}, index=obs_names)
    var_names = [f"protein_{i}" for i in range(5)]
    var = pd.DataFrame({
        "protein_id": var_names
    }, index=var_names)
    return AnnData(X=X, obs=obs, var=var)


def _make_adata_remove_contaminants_peptide_level() -> AnnData:
    """4 obs × 5 vars, peptide-level data for contaminant filtering tests."""
    X = np.array(
        [
            [10, 20, 30, 40, 50],
            [11, 21, 31, 41, 51],
            [12, 22, 32, 42, 52],
            [13, 23, 33, 43, 53],
        ],
        dtype=float,
    )
    obs_names = [f"obs{i}" for i in range(4)]
    var_names = [f"pep{i}" for i in range(5)]
    obs = pd.DataFrame({"sample_id": obs_names}, index=obs_names)
    var = pd.DataFrame(
        {
            "peptide_id": var_names,
            "protein_id": [
                "protein_0",
                "protein_1",
                "protein_1",
                "protein_2",
                "protein_3",
            ],
        },
        index=var_names,
    )
    return AnnData(X=X, obs=obs, var=var)

def test_filter_axis_obs_min_fraction():
    adata = _make_adata_filter_obs_base()

    cases = {
        0.8: ["obs0", "obs1", "obs4", "obs5"],
        1.0: ["obs0", "obs4"],
        0.0: list(adata.obs_names),
    }
    for min_fraction, expected in cases.items():
        filtered = filter_axis(
            adata,
            axis=0,
            min_fraction=min_fraction,
            inplace=False,
        )
        assert list(filtered.obs_names) == expected

        adata_inplace = adata.copy()
        returned = filter_axis(
            adata_inplace,
            axis=0,
            min_fraction=min_fraction,
            inplace=True,
        )
        assert returned is None
        assert list(adata_inplace.obs_names) == expected


def test_filter_axis_obs_min_count():
    adata = _make_adata_filter_obs_base()

    cases = {
        4: ["obs0", "obs1", "obs4", "obs5"],
        5: ["obs0", "obs4"],
        0: list(adata.obs_names),
    }
    for min_count, expected in cases.items():
        filtered = filter_axis(
            adata,
            axis=0,
            min_count=min_count,
            inplace=False,
        )
        assert list(filtered.obs_names) == expected

        adata_inplace = adata.copy()
        returned = filter_axis(
            adata_inplace,
            axis=0,
            min_count=min_count,
            inplace=True,
        )
        assert returned is None
        assert list(adata_inplace.obs_names) == expected


def test_filter_axis_obs_min_fraction_and_min_count():
    adata = _make_adata_filter_obs_base()

    cases = {
        (0.4, 3): ["obs0", "obs1", "obs2", "obs4", "obs5"],
        (1.0, 5): ["obs0", "obs4"],
        (0.0, 0): list(adata.obs_names),
    }
    for (min_fraction, min_count), expected in cases.items():
        filtered = filter_axis(
            adata,
            axis=0,
            min_fraction=min_fraction,
            min_count=min_count,
            inplace=False,
        )
        assert list(filtered.obs_names) == expected

        adata_inplace = adata.copy()
        returned = filter_axis(
            adata_inplace,
            axis=0,
            min_fraction=min_fraction,
            min_count=min_count,
            inplace=True,
        )
        assert returned is None
        assert list(adata_inplace.obs_names) == expected


def test_filter_axis_obs_zero_to_na():
    adata = _make_adata_filter_obs_base()

    filtered = filter_axis(
        adata,
        axis=0,
        min_count=4,
        zero_to_na=True,
        inplace=False,
    )
    # zeros become missing → only obs0 stays fully observed
    assert list(filtered.obs_names) == ["obs0", "obs1", "obs4"]

    adata_inplace = adata.copy()
    returned = filter_axis(
        adata_inplace,
        axis=0,
        min_count=4,
        zero_to_na=True,
        inplace=True,
    )
    assert returned is None
    assert list(adata_inplace.obs_names) == ["obs0", "obs1", "obs4"]


def test_filter_axis_obs_groupby_singletons():
    adata = _make_adata_filter_obs_groupby_singletons()

    fraction_cases = {
        0.8: ["obs1", "obs2"],
        1.0: ["obs1", "obs2"],
    }

    for min_fraction, expected in fraction_cases.items():
        filtered = filter_axis(
            adata,
            axis=0,
            min_fraction=min_fraction,
            group_by="group",
            inplace=False,
        )
        assert list(filtered.obs_names) == expected

        adata_inplace = adata.copy()
        returned = filter_axis(
            adata_inplace,
            axis=0,
            min_fraction=min_fraction,
            group_by="group",
            inplace=True,
        )
        assert returned is None
        assert list(adata_inplace.obs_names) == expected

    count_cases = {
        1: ["obs1", "obs2"],
    }
    for min_count, expected in count_cases.items():
        filtered = filter_axis(
            adata,
            axis=0,
            min_count=min_count,
            group_by="group",
            inplace=False,
        )
        assert list(filtered.obs_names) == expected

        adata_inplace = adata.copy()
        returned = filter_axis(
            adata_inplace,
            axis=0,
            min_count=min_count,
            group_by="group",
            inplace=True,
        )
        assert returned is None
        assert list(adata_inplace.obs_names) == expected

def test_filter_axis_obs_groupby_multiple():
    adata = _make_adata_filter_obs_groupby()

    fraction_cases = {
        2 / 3: ["obs0", "obs1", "obs2", "obs3"],
        0.8: ["obs0", "obs1", "obs2"],
    }
    for min_fraction, expected in fraction_cases.items():
        filtered = filter_axis(
            adata,
            axis=0,
            min_fraction=min_fraction,
            group_by="group",
            inplace=False,
        )
        assert list(filtered.obs_names) == expected

        adata_inplace = adata.copy()
        returned = filter_axis(
            adata_inplace,
            axis=0,
            min_fraction=min_fraction,
            group_by="group",
            inplace=True,
        )
        assert returned is None
        assert list(adata_inplace.obs_names) == expected

    count_cases = {
        2: ["obs0", "obs1", "obs2", "obs3"],
        3: ["obs0", "obs1"],
    }
    for min_count, expected in count_cases.items():
        filtered = filter_axis(
            adata,
            axis=0,
            min_count=min_count,
            group_by="group",
            inplace=False,
        )
        assert list(filtered.obs_names) == expected

        adata_inplace = adata.copy()
        returned = filter_axis(
            adata_inplace,
            axis=0,
            min_count=min_count,
            group_by="group",
            inplace=True,
        )
        assert returned is None
        assert list(adata_inplace.obs_names) == expected


def test_filter_axis_obs_groupby_with_nan_group():
    adata = _make_adata_filter_obs_groupby_na()

    fraction_cases = {
        2 / 3: ["obs0", "obs1", "obs2", "obs3"],
        0.8: ["obs0", "obs1", "obs2"],
    }
    for min_fraction, expected in fraction_cases.items():
        filtered = filter_axis(
            adata,
            axis=0,
            min_fraction=min_fraction,
            group_by="group",
            inplace=False,
        )
        assert list(filtered.obs_names) == expected

        adata_inplace = adata.copy()
        returned = filter_axis(
            adata_inplace,
            axis=0,
            min_fraction=min_fraction,
            group_by="group",
            inplace=True,
        )
        assert returned is None
        assert list(adata_inplace.obs_names) == expected

    count_cases = {
        2: ["obs0", "obs1", "obs2", "obs3"],
        3: ["obs0", "obs1"],
    }
    for min_count, expected in count_cases.items():
        filtered = filter_axis(
            adata,
            axis=0,
            min_count=min_count,
            group_by="group",
            inplace=False,
        )
        assert list(filtered.obs_names) == expected

        adata_inplace = adata.copy()
        returned = filter_axis(
            adata_inplace,
            axis=0,
            min_count=min_count,
            group_by="group",
            inplace=True,
        )
        assert returned is None
        assert list(adata_inplace.obs_names) == expected

def test_filter_axis_var_min_fraction():
    adata = _make_adata_filter_var_base()

    cases = {
        0.8: ["protein_0", "protein_1", "protein_4", "protein_5"],
        1.0: ["protein_0", "protein_4"],
        0.0: list(adata.var_names),
    }
    for min_fraction, expected in cases.items():
        filtered = filter_axis(
            adata,
            axis=1,
            min_fraction=min_fraction,
            inplace=False,
        )
        assert list(filtered.var_names) == expected

        adata_inplace = adata.copy()
        returned = filter_axis(
            adata_inplace,
            axis=1,
            min_fraction=min_fraction,
            inplace=True,
        )
        assert returned is None
        assert list(adata_inplace.var_names) == expected


def test_filter_axis_var_min_count():
    adata = _make_adata_filter_var_base()

    cases = {
        4: ["protein_0", "protein_1", "protein_4", "protein_5"],
        5: ["protein_0", "protein_4"],
        0.0: list(adata.var_names),
    }
    for min_count, expected in cases.items():
        filtered = filter_axis(
            adata,
            axis=1,
            min_count=min_count,
            inplace=False,
        )
        assert list(filtered.var_names) == expected

        adata_inplace = adata.copy()
        returned = filter_axis(
            adata_inplace,
            axis=1,
            min_count=min_count,
            inplace=True,
        )
        assert returned is None
        assert list(adata_inplace.var_names) == expected


def test_filter_axis_var_min_fraction_and_min_count():
    adata = _make_adata_filter_var_base()

    cases = {
        (0.4, 3): ["protein_0", "protein_1", "protein_2", "protein_4", "protein_5"],
        (1.0, 5): ["protein_0", "protein_4"],
        (0.0, 0): list(adata.var_names),
    }
    for (min_fraction, min_count), expected in cases.items():
        filtered = filter_axis(
            adata,
            axis=1,
            min_fraction=min_fraction,
            min_count=min_count,
            inplace=False,
        )
        assert list(filtered.var_names) == expected

        adata_inplace = adata.copy()
        returned = filter_axis(
            adata_inplace,
            axis=1,
            min_fraction=min_fraction,
            min_count=min_count,
            inplace=True,
        )
        assert returned is None
        assert list(adata_inplace.var_names) == expected


def test_filter_axis_var_zero_to_na():
    adata = _make_adata_filter_var_base()

    filtered = filter_axis(
        adata,
        axis=1,
        min_count=4,
        zero_to_na=True,
        inplace=False,
    )
    assert list(filtered.var_names) == ["protein_0", "protein_1", "protein_4"]

    adata_inplace = adata.copy()
    returned = filter_axis(
        adata_inplace,
        axis=1,
        min_count=4,
        zero_to_na=True,
        inplace=True,
    )
    assert returned is None
    assert list(adata_inplace.var_names) == ["protein_0", "protein_1", "protein_4"]


def test_filter_axis_var_groupby_singletons():
    adata = _make_adata_filter_var_groupby_singletons()

    fraction_cases = {
        0.8: ["protein_1", "protein_2"],
        1.0: ["protein_1", "protein_2"],
    }
    for min_fraction, expected in fraction_cases.items():
        filtered = filter_axis(
            adata,
            axis=1,
            min_fraction=min_fraction,
            group_by="group",
            inplace=False,
        )
        assert list(filtered.var_names) == expected

        adata_inplace = adata.copy()
        returned = filter_axis(
            adata_inplace,
            axis=1,
            min_fraction=min_fraction,
            group_by="group",
            inplace=True,
        )
        assert returned is None
        assert list(adata_inplace.var_names) == expected

    count_cases = {
        1: ["protein_1", "protein_2"],
    }
    for min_count, expected in count_cases.items():
        filtered = filter_axis(
            adata,
            axis=1,
            min_count=min_count,
            group_by="group",
            inplace=False,
        )
        assert list(filtered.var_names) == expected

        adata_inplace = adata.copy()
        returned = filter_axis(
            adata_inplace,
            axis=1,
            min_count=min_count,
            group_by="group",
            inplace=True,
        )
        assert returned is None
        assert list(adata_inplace.var_names) == expected


def test_filter_axis_var_groupby():
    adata = _make_adata_filter_var_groupby()

    fraction_cases = {
        2 / 3: ["protein_0", "protein_1", "protein_2", "protein_3"],
        0.8: ["protein_0", "protein_1", "protein_2"],
    }
    for min_fraction, expected in fraction_cases.items():
        filtered = filter_axis(
            adata,
            axis=1,
            min_fraction=min_fraction,
            group_by="group",
            inplace=False,
        )
        assert list(filtered.var_names) == expected

        adata_inplace = adata.copy()
        returned = filter_axis(
            adata_inplace,
            axis=1,
            min_fraction=min_fraction,
            group_by="group",
            inplace=True,
        )
        assert returned is None
        assert list(adata_inplace.var_names) == expected

    count_cases = {
        2: ["protein_0", "protein_1", "protein_2", "protein_3"],
        3: ["protein_0", "protein_1"],
    }
    for min_count, expected in count_cases.items():
        filtered = filter_axis(
            adata,
            axis=1,
            min_count=min_count,
            group_by="group",
            inplace=False,
        )
        assert list(filtered.var_names) == expected

        adata_inplace = adata.copy()
        returned = filter_axis(
            adata_inplace,
            axis=1,
            min_count=min_count,
            group_by="group",
            inplace=True,
        )
        assert returned is None
        assert list(adata_inplace.var_names) == expected


def test_filter_axis_var_groupby_with_nan_group():
    adata = _make_adata_filter_var_groupby_na()

    fraction_cases = {
        2 / 3: ["protein_0", "protein_1", "protein_2", "protein_3"],
        0.8: ["protein_0", "protein_1", "protein_2"],
    }
    for min_fraction, expected in fraction_cases.items():
        filtered = filter_axis(
            adata,
            axis=1,
            min_fraction=min_fraction,
            group_by="group",
            inplace=False,
        )
        assert list(filtered.var_names) == expected

        adata_inplace = adata.copy()
        returned = filter_axis(
            adata_inplace,
            axis=1,
            min_fraction=min_fraction,
            group_by="group",
            inplace=True,
        )
        assert returned is None
        assert list(adata_inplace.var_names) == expected

    count_cases = {
        2: ["protein_0", "protein_1", "protein_2", "protein_3"],
        3: ["protein_0", "protein_1"],
    }
    for min_count, expected in count_cases.items():
        filtered = filter_axis(
            adata,
            axis=1,
            min_count=min_count,
            group_by="group",
            inplace=False,
        )
        assert list(filtered.var_names) == expected

        adata_inplace = adata.copy()
        returned = filter_axis(
            adata_inplace,
            axis=1,
            min_count=min_count,
            group_by="group",
            inplace=True,
        )
        assert returned is None
        assert list(adata_inplace.var_names) == expected


def _make_peptide_adata() -> AnnData:
    X = np.zeros((3, 6))
    var_names = [f"pep{i}" for i in range(6)]
    var = pd.DataFrame(
        {
            "peptide_id": var_names,
            "protein_id": ["P1", "P2", "P2", "P3", "P3", "P3"],
        },
        index=var_names,
    )
    obs_names = [f"obs{i}" for i in range(3)]
    obs = pd.DataFrame({"sample_id": obs_names}, index=obs_names)
    return AnnData(X=X, obs=obs, var=var)


def test_filter_proteins_by_peptide_count_min():
    io = {
        0: ["pep0", "pep1", "pep2", "pep3", "pep4", "pep5"],
        2: ["pep1", "pep2", "pep3", "pep4", "pep5"],
        4: [],
    }

    for min_count, expected in io.items():
        adata = _make_peptide_adata()
        filtered = filter_proteins_by_peptide_count(
            adata,
            min_count=min_count,
            inplace=False,
        )

        assert list(filtered.var_names) == expected
        assert list(adata.var_names) == [f"pep{i}" for i in range(6)]

        adata_inplace = _make_peptide_adata()
        returned = filter_proteins_by_peptide_count(
            adata_inplace,
            min_count=min_count,
            inplace=True,
        )

        assert returned is None
        assert list(adata_inplace.var_names) == expected


def test_filter_proteins_by_peptide_count_max():
    io = {
        0: [],
        1: ["pep0"],
        2: ["pep0", "pep1", "pep2"],
    }

    for max_count, expected in io.items():
        adata = _make_peptide_adata()
        filtered = filter_proteins_by_peptide_count(
            adata,
            max_count=max_count,
            inplace=False,
        )

        assert list(filtered.var_names) == expected
        assert list(adata.var_names) == [f"pep{i}" for i in range(6)]

        adata_inplace = _make_peptide_adata()
        returned = filter_proteins_by_peptide_count(
            adata_inplace,
            max_count=max_count,
            inplace=True,
        )

        assert returned is None
        assert list(adata_inplace.var_names) == expected


def test_filter_proteins_by_peptide_count_min_and_max():
    io = {
        (2,2): ["pep1", "pep2"],
        (2,3): ["pep1", "pep2", "pep3", "pep4", "pep5"],
    }

    for (min_count,max_count), expected in io.items():
        adata = _make_peptide_adata()
        filtered = filter_proteins_by_peptide_count(
            adata,
            min_count=min_count,
            max_count=max_count,
            inplace=False,
        )

        assert list(filtered.var_names) == expected
        assert list(adata.var_names) == [f"pep{i}" for i in range(6)]

        adata_inplace = _make_peptide_adata()
        returned = filter_proteins_by_peptide_count(
            adata_inplace,
            min_count=min_count,
            max_count=max_count,
            inplace=True,
        )

        assert returned is None
        assert list(adata_inplace.var_names) == expected


def test_filter_proteins_by_peptide_count_min_gt_max_raises():
    adata = _make_peptide_adata()

    with pytest.raises(ValueError):
        filter_proteins_by_peptide_count(
            adata,
            min_count=3,
            max_count=2,
            inplace=False,
        )


def test_filter_proteins_by_peptide_count_requires_peptide_level():
    X = np.zeros((2, 2))
    var_names = ["prot1", "prot2"]
    var = pd.DataFrame({"protein_id": var_names}, index=var_names)
    obs_names = ["obs0", "obs1"]
    obs = pd.DataFrame({"sample_id": obs_names}, index=obs_names)
    adata = AnnData(X=X, obs=obs, var=var)

    with pytest.raises(ValueError):
        filter_proteins_by_peptide_count(adata, min_count=1)

class TestRemoveZeroVarianceVars:
    """Tests for ``remove_zero_variance_vars``."""

    # ── A. Basic functionality (non-grouped) ────────────────────────

    @pytest.mark.parametrize("inplace", [True, False])
    def test_basic_removal(self, inplace):
        adata = _make_adata_rzv_base()
        original = adata.copy()
        result = remove_zero_variance_vars(adata, inplace=inplace)
        target = adata if inplace else result
        if inplace:
            assert result is None
        else:
            assert list(original.var_names) == ["p0", "p1", "p2", "p3", "p4"]
        assert list(target.var_names) == ["p0", "p1", "p2"]
        assert target.n_obs == original.n_obs

    @pytest.mark.parametrize("inplace", [True, False])
    def test_all_nan_column_removed_with_warning(self, inplace):
        adata = _make_adata_rzv_one_allnan_col()
        with pytest.warns(
            UserWarning,
            match=r"1 variable\(s\) contained only NaN values",
        ):
            result = remove_zero_variance_vars(adata, inplace=inplace)
        target = adata if inplace else result
        assert list(target.var_names) == ["p0", "p1"]

    @pytest.mark.parametrize("inplace", [True, False])
    def test_all_vars_zero_variance_returns_empty(self, inplace):
        adata = _make_adata_rzv_all_constant()
        result = remove_zero_variance_vars(adata, inplace=inplace)
        target = adata if inplace else result
        assert list(target.var_names) == []
        assert target.n_obs == 6

    @pytest.mark.parametrize("inplace", [True, False])
    def test_no_vars_removed_when_all_vary(self, inplace):
        adata = _make_adata_rzv_all_vary()
        result = remove_zero_variance_vars(adata, inplace=inplace)
        target = adata if inplace else result
        assert list(target.var_names) == ["p0", "p1", "p2"]

    @pytest.mark.parametrize("inplace", [True, False])
    def test_single_variable_kept(self, inplace):
        adata = _make_adata_rzv_single_var_varies()
        result = remove_zero_variance_vars(adata, inplace=inplace)
        target = adata if inplace else result
        assert list(target.var_names) == ["p0"]

    @pytest.mark.parametrize("inplace", [True, False])
    def test_single_variable_removed(self, inplace):
        adata = _make_adata_rzv_single_var_constant()
        result = remove_zero_variance_vars(adata, inplace=inplace)
        target = adata if inplace else result
        assert list(target.var_names) == []

    def test_multiple_all_nan_columns_warning_count(self):
        adata = _make_adata_rzv_multiple_allnan()
        with pytest.warns(
            UserWarning,
            match=r"3 variable\(s\) contained only NaN values",
        ):
            filtered = remove_zero_variance_vars(adata, inplace=False)
        assert list(filtered.var_names) == ["p0"]

    # ── B. atol boundary behavior ───────────────────────────────────

    def test_atol_boundary_equal_variance_removed(self):
        adata = _make_adata_rzv_atol_boundary()
        filtered = remove_zero_variance_vars(
            adata, atol=1.0, inplace=False,
        )
        assert list(filtered.var_names) == ["p2"]

    def test_atol_zero_keeps_tiny_variance(self):
        adata = _make_adata_rzv_atol_zero()
        filtered = remove_zero_variance_vars(
            adata, atol=0.0, inplace=False,
        )
        assert list(filtered.var_names) == ["p1", "p2"]

    def test_large_atol_removes_everything(self):
        adata = _make_adata_rzv_all_vary()
        filtered = remove_zero_variance_vars(
            adata, atol=1e10, inplace=False,
        )
        assert list(filtered.var_names) == []

    def test_negative_atol_raises(self):
        adata = _make_adata_rzv_base()
        with pytest.raises(
            ValueError, match=r"`atol` must be non-negative.",
        ):
            remove_zero_variance_vars(adata, atol=-2)

    # ── C. Grouped path (group_by) ─────────────────────────────────

    @pytest.mark.parametrize("inplace", [True, False])
    def test_groupby_removes_zero_in_any_group(self, inplace):
        adata = _make_adata_rzv_groupby()
        result = remove_zero_variance_vars(
            adata, group_by="group", inplace=inplace,
        )
        target = adata if inplace else result
        assert list(target.var_names) == ["p0"]

    @pytest.mark.parametrize("inplace", [True, False])
    def test_groupby_singleton_groups_removes_all(self, inplace):
        adata = _make_adata_rzv_groupby_singletons()
        with pytest.warns(
            UserWarning,
            match=r"at least one group",
        ):
            result = remove_zero_variance_vars(
                adata, group_by="group", inplace=inplace,
            )
        target = adata if inplace else result
        assert list(target.var_names) == []

    def test_groupby_all_nan_in_one_group_warns(self):
        adata = _make_adata_rzv_groupby_allnan_one_group()
        with pytest.warns(
            UserWarning, match=r"at least one group",
        ):
            filtered = remove_zero_variance_vars(
                adata, group_by="group", inplace=False,
            )
        assert list(filtered.var_names) == ["p1"]

    def test_groupby_single_group_matches_global(self):
        adata = _make_adata_rzv_groupby_single_group()
        filtered_grouped = remove_zero_variance_vars(
            adata, group_by="group", inplace=False,
        )
        adata2 = _make_adata_rzv_groupby_single_group()
        filtered_global = remove_zero_variance_vars(
            adata2, group_by=None, inplace=False,
        )
        assert (
            list(filtered_grouped.var_names)
            == list(filtered_global.var_names)
            == ["p0", "p2"]
        )

    def test_groupby_categorical_column(self):
        adata = _make_adata_rzv_groupby()
        adata.obs["group"] = pd.Categorical(adata.obs["group"])
        filtered = remove_zero_variance_vars(
            adata, group_by="group", inplace=False,
        )
        assert list(filtered.var_names) == ["p0"]

    def test_groupby_nan_in_column_raises(self):
        adata = _make_adata_rzv_base()
        adata.obs["group"] = ["a", "b", np.nan, "a", "b", "a"]
        with pytest.raises(
            ValueError,
            match=r"`group_by`='group' column in adata.obs contains NaN",
        ):
            remove_zero_variance_vars(adata, group_by="group")

    def test_groupby_missing_column_raises(self):
        adata = _make_adata_rzv_base()
        with pytest.raises(
            KeyError,
            match=r"`group_by`='missing' not found in adata.obs",
        ):
            remove_zero_variance_vars(adata, group_by="missing")

    # ── D. Type validation (parametrized) ───────────────────────────

    @pytest.mark.parametrize("bad_adata", ["not-anndata", 42, None])
    def test_invalid_adata_type(self, bad_adata):
        with pytest.raises(
            TypeError, match=r"`adata` must be an AnnData object",
        ):
            remove_zero_variance_vars(adata=bad_adata)

    @pytest.mark.parametrize("bad_group_by", [123, True, [1]])
    def test_invalid_group_by_type(self, bad_group_by):
        adata = _make_adata_rzv_base()
        with pytest.raises(
            TypeError, match=r"`group_by` must be a string or None",
        ):
            remove_zero_variance_vars(adata, group_by=bad_group_by)

    @pytest.mark.parametrize("bad_atol", ["tiny", None])
    def test_invalid_atol_type(self, bad_atol):
        adata = _make_adata_rzv_base()
        with pytest.raises(
            TypeError, match=r"`atol` must be a numeric value",
        ):
            remove_zero_variance_vars(adata, atol=bad_atol)

    @pytest.mark.parametrize("bad_inplace", ["True", 1])
    def test_invalid_inplace_type(self, bad_inplace):
        adata = _make_adata_rzv_base()
        with pytest.raises(
            TypeError, match=r"`inplace` must be a bool",
        ):
            remove_zero_variance_vars(adata, inplace=bad_inplace)

    @pytest.mark.parametrize("bad_verbose", [1, "yes"])
    def test_invalid_verbose_type(self, bad_verbose):
        adata = _make_adata_rzv_base()
        with pytest.raises(
            TypeError, match=r"`verbose` must be a bool",
        ):
            remove_zero_variance_vars(adata, verbose=bad_verbose)

    # ── E. Verbose output ───────────────────────────────────────────

    def test_verbose_reports_correct_counts(self, capsys):
        adata = _make_adata_rzv_base()
        remove_zero_variance_vars(
            adata, inplace=True, verbose=True,
        )
        captured = capsys.readouterr()
        assert "5 variables present" in captured.out
        assert "2 removed" in captured.out
        assert "3 remaining" in captured.out

    def test_verbose_false_prints_nothing(self, capsys):
        adata = _make_adata_rzv_base()
        remove_zero_variance_vars(
            adata, inplace=True, verbose=False,
        )
        captured = capsys.readouterr()
        assert captured.out == ""

    # ── F. Property / invariant tests ───────────────────────────────

    def test_inplace_false_preserves_original(self):
        adata = _make_adata_rzv_base()
        original_var_names = list(adata.var_names)
        original_X = adata.X.copy()
        _ = remove_zero_variance_vars(adata, inplace=False)
        assert list(adata.var_names) == original_var_names
        np.testing.assert_array_equal(adata.X, original_X)

    def test_obs_and_var_metadata_preserved(self):
        adata = _make_adata_rzv_base()
        adata.var["extra_col"] = ["a", "b", "c", "d", "e"]
        filtered = remove_zero_variance_vars(adata, inplace=False)
        assert list(filtered.obs.columns) == list(adata.obs.columns)
        assert list(filtered.obs.index) == list(adata.obs.index)
        assert "extra_col" in filtered.var.columns
        assert list(filtered.var["extra_col"]) == ["a", "b", "c"]

    def test_idempotency(self):
        adata = _make_adata_rzv_base()
        first = remove_zero_variance_vars(adata, inplace=False)
        second = remove_zero_variance_vars(first, inplace=False)
        assert list(first.var_names) == list(second.var_names)
        np.testing.assert_array_equal(
            np.asarray(first.X), np.asarray(second.X),
        )

    def test_kept_var_values_unchanged(self):
        adata = _make_adata_rzv_base()
        original_X = adata.X.copy()
        filtered = remove_zero_variance_vars(adata, inplace=False)
        kept_idx = [
            list(adata.var_names).index(v)
            for v in filtered.var_names
        ]
        np.testing.assert_array_equal(
            np.asarray(filtered.X),
            original_X[:, kept_idx],
        )

    # ── G. Data type variants ──────────────────────────────────────

    def test_float32_matrix(self):
        adata = _make_adata_rzv_float32()
        filtered = remove_zero_variance_vars(adata, inplace=False)
        assert list(filtered.var_names) == ["p0", "p2"]

    # ── H. Peptide-level data ──────────────────────────────────────

    @pytest.mark.parametrize("inplace", [True, False])
    def test_peptide_level_data_basic(self, inplace):
        adata = _make_adata_rzv_peptide_level()
        with pytest.warns(UserWarning, match=r"1 variable\(s\)"):
            result = remove_zero_variance_vars(
                adata, inplace=inplace,
            )
        target = adata if inplace else result
        assert list(target.var_names) == ["pep0", "pep2"]
        assert "peptide_id" in target.var.columns
        assert "protein_id" in target.var.columns

    def test_peptide_level_data_with_groupby(self):
        adata = _make_adata_rzv_peptide_level()
        adata.obs["group"] = ["A", "A", "B", "B"]
        with pytest.warns(UserWarning, match=r"at least one group"):
            filtered = remove_zero_variance_vars(
                adata, group_by="group", inplace=False,
            )
        assert "peptide_id" in filtered.var.columns
        assert "protein_id" in filtered.var.columns
class TestRemoveContaminants:
    @pytest.fixture
    def fasta(self, tmp_path):
        fasta_content = (
        ">sp|protein_1\n"
        "AAAA\n"
        ">sp|protein_2\n"
        "CCCC\n"
        )
        fasta_path = tmp_path / "test.fasta"
        fasta_path.write_text(fasta_content)
        return fasta_path

    @pytest.fixture
    def csv_file(self, tmp_path):
        csv_path = tmp_path / "contaminants.csv"
        csv_path.write_text(
            "contaminant,source\n"
            "protein_2,db\n"
            "protein_4,db\n",
        )
        return csv_path

    @pytest.fixture
    def tsv_file(self, tmp_path):
        tsv_path = tmp_path / "contaminants.tsv"
        tsv_path.write_text(
            "contaminant\tcomment\n"
            "protein_0\ta\n"
            "protein_3\tb\n",
        )
        return tsv_path

    # ── A. Basic functionality ────────────────────────────────────

    @pytest.mark.parametrize("inplace", [True, False])
    def test_fasta_filters_expected_proteins(self, fasta, inplace):
        adata = _make_adata_remove_contaminants_base()
        original_var_names = list(adata.var_names)

        result = remove_contaminants(
            adata,
            contaminant_path=fasta,
            inplace=inplace,
        )

        target = adata if inplace else result
        assert list(target.var_names) == [
            "protein_0", "protein_3", "protein_4",
        ]
        assert target.n_obs == 5

        if inplace:
            assert result is None
        else:
            assert list(adata.var_names) == original_var_names

    def test_no_matching_contaminants_keeps_all_variables(self, tmp_path):
        fasta_path = tmp_path / "none_match.fasta"
        fasta_path.write_text(
            ">sp|not_present_a\nAAAA\n"
            ">sp|not_present_b\nCCCC\n",
        )

        adata = _make_adata_remove_contaminants_base()
        filtered = remove_contaminants(
            adata,
            contaminant_path=fasta_path,
            inplace=False,
        )
        assert list(filtered.var_names) == list(adata.var_names)

    # ── B. Input format and parser behavior ──────────────────────

    def test_csv_filters_using_first_column(self, csv_file):
        adata = _make_adata_remove_contaminants_base()
        filtered = remove_contaminants(
            adata,
            contaminant_path=csv_file,
            inplace=False,
        )
        assert list(filtered.var_names) == ["protein_0", "protein_1", "protein_3"]

    def test_tsv_filters_using_first_column(self, tsv_file):
        adata = _make_adata_remove_contaminants_base()
        filtered = remove_contaminants(
            adata,
            contaminant_path=tsv_file,
            inplace=False,
        )
        assert list(filtered.var_names) == ["protein_1", "protein_2", "protein_4"]

    def test_custom_protein_key_column(self, fasta):
        adata = _make_adata_remove_contaminants_base()
        adata.var["uniprot_id"] = [
            "u0", "protein_1", "protein_2", "u3", "u4",
        ]

        filtered = remove_contaminants(
            adata,
            contaminant_path=fasta,
            protein_key="uniprot_id",
            inplace=False,
        )
        assert list(filtered.var_names) == ["protein_0", "protein_3", "protein_4"]

    def test_custom_header_parser_is_used(self, tmp_path):
        fasta_path = tmp_path / "custom_header.fasta"
        fasta_path.write_text(
            ">contam__protein_0\nAAAA\n"
            ">contam__protein_4\nCCCC\n",
        )

        adata = _make_adata_remove_contaminants_base()
        filtered = remove_contaminants(
            adata,
            contaminant_path=fasta_path,
            header_parser=lambda h: h.split("__")[1],
            inplace=False,
        )
        assert list(filtered.var_names) == ["protein_1", "protein_2", "protein_3"]

    def test_header_parser_empty_id_warns_and_skips(self, tmp_path):
        fasta_path = tmp_path / "empty_id.fasta"
        fasta_path.write_text(
            ">sp|protein_1\nAAAA\n"
            ">sp|protein_2\nCCCC\n",
        )

        adata = _make_adata_remove_contaminants_base()
        with pytest.warns(
            UserWarning,
            match=r"Header parser returned empty ID",
        ):
            filtered = remove_contaminants(
                adata,
                contaminant_path=fasta_path,
                header_parser=lambda _: "",
                inplace=False,
            )
        assert list(filtered.var_names) == list(adata.var_names)

    # ── C. Output messaging by proteodata level ──────────────────

    def test_prints_protein_level_summary(self, fasta, capsys):
        adata = _make_adata_remove_contaminants_base()
        remove_contaminants(
            adata,
            contaminant_path=fasta,
            inplace=False,
        )
        out = capsys.readouterr().out
        assert "Removed 2 contaminating proteins." in out

    def test_prints_peptide_level_summary(self, tmp_path, capsys):
        fasta_path = tmp_path / "peptide_contam.fasta"
        fasta_path.write_text(">sp|protein_1\nAAAA\n")

        adata = _make_adata_remove_contaminants_peptide_level()
        filtered = remove_contaminants(
            adata,
            contaminant_path=fasta_path,
            inplace=False,
        )
        out = capsys.readouterr().out

        assert "Removed 2 contaminating peptides" in out
        assert "1 contaminating proteins" in out
        assert list(filtered.var_names) == ["pep0", "pep3", "pep4"]
        assert "peptide_id" in filtered.var.columns
        assert "protein_id" in filtered.var.columns

    # ── D. Error handling ─────────────────────────────────────────

    def test_missing_contaminant_file_raises(self, tmp_path):
        adata = _make_adata_remove_contaminants_base()
        missing_path = tmp_path / "does_not_exist.fasta"

        with pytest.raises(FileNotFoundError, match=r"Contaminant file not found"):
            remove_contaminants(
                adata,
                contaminant_path=missing_path,
                inplace=False,
            )

    def test_missing_protein_key_raises(self, fasta):
        adata = _make_adata_remove_contaminants_base()

        with pytest.raises(KeyError, match=r"`protein_key`='missing_key'"):
            remove_contaminants(
                adata,
                contaminant_path=fasta,
                protein_key="missing_key",
                inplace=False,
            )

    def test_unsupported_file_type_raises(self, tmp_path):
        path = tmp_path / "contaminants.txt"
        path.write_text("protein_1\n")
        adata = _make_adata_remove_contaminants_base()

        with pytest.raises(ValueError, match=r"Unsupported contaminant file type"):
            remove_contaminants(
                adata,
                contaminant_path=path,
                inplace=False,
            )

    def test_invalid_adata_fails_proteodata_validation(self, fasta):
        adata = _make_adata_remove_contaminants_base()
        adata.obs = pd.DataFrame(index=adata.obs_names)

        with pytest.raises(ValueError, match=r"sample_id"):
            remove_contaminants(
                adata,
                contaminant_path=fasta,
                inplace=False,
            )

    # ── E. Invariants and data preservation ──────────────────────

    def test_inplace_false_does_not_modify_original(self, fasta):
        adata = _make_adata_remove_contaminants_base()
        original_var_names = list(adata.var_names)

        filtered = remove_contaminants(
            adata,
            contaminant_path=fasta,
            inplace=False,
        )

        assert list(adata.var_names) == original_var_names
        assert list(filtered.var_names) != original_var_names

    def test_obs_and_remaining_var_metadata_preserved(self, fasta):
        adata = _make_adata_remove_contaminants_base()
        adata.obs["batch"] = ["A", "A", "B", "B", "C"]
        adata.var["annotation"] = ["x", "y", "z", "w", "v"]

        filtered = remove_contaminants(
            adata,
            contaminant_path=fasta,
            inplace=False,
        )

        assert list(filtered.obs.columns) == list(adata.obs.columns)
        assert list(filtered.obs_names) == list(adata.obs_names)
        assert "annotation" in filtered.var.columns
        assert list(filtered.var["annotation"]) == ["x", "w", "v"]

    def test_sparse_input_remains_sparse_after_filtering(self, fasta):
        adata = _make_adata_remove_contaminants_base()
        adata.X = sp.csr_matrix(adata.X)

        filtered = remove_contaminants(
            adata,
            contaminant_path=fasta,
            inplace=False,
        )

        assert sp.issparse(filtered.X)