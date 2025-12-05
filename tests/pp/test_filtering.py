import numpy as np
import pandas as pd
from anndata import AnnData

from copro.pp.filtering import filter_axis


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
    obs = pd.DataFrame(index=obs_names)
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
    obs = pd.DataFrame(index=obs_names)
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
    obs = pd.DataFrame(index=obs_names)
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
    obs = pd.DataFrame(index=obs_names)
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
    obs = pd.DataFrame(index=[f"obs{i}" for i in range(5)])
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
    obs = pd.DataFrame({"group": ["g1", "g2"]}, index=[f"obs{i}" for i in range(2)])
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
    obs = pd.DataFrame({
        "group": ["g1", "g1", "g2", "g2", "g2"]
        }, index=[f"obs{i}" for i in range(5)],
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
    obs = pd.DataFrame(
        {"group": ["g1", "g1", "g2", "g2", "g2", np.nan, np.nan, np.nan, np.nan]},
        index=[f"obs{i}" for i in range(9)],
    )
    var_names = [f"protein_{i}" for i in range(5)]
    var = pd.DataFrame({
            "protein_id": var_names,
            "group": ["g1", "g1", "g2", "g2", np.nan],
        }, index=var_names,
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
