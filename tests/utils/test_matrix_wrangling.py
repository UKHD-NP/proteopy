import numpy as np
import pandas as pd
import pytest

from proteopy.utils._matrix_wrangling import (
    reconstruct_symmetric_matrix_from_long,
)


def test_basic_reconstruction():
    # labels a-c
    #
    # [[ x, 0.1, 0.5],        [[ 1  , 0.1, 0.5],
    #  [ x, 1  , x  ],   ==>   [ 0.1, 1  , 0.4],
    #  [ x  0.4, 1  ]]         [ 0.5, 0.4, 1  ]]
    df = pd.DataFrame(
        {
            "colA": ["a", "a", "b", "c", "c"],
            "colB": ["b", "c", "b", "b", "c"],
            "value": [0.1, 0.5, 1, 0.4, 1],
        }
    )
    expected = pd.DataFrame(
        {
            "a": [1.0, 0.1, 0.5],
            "b": [0.1, 1.0, 0.4],
            "c": [0.5, 0.4, 1.0],
        },
        index=["a", "b", "c"],
    )

    result = reconstruct_symmetric_matrix_from_long(df, "colA", "colB", 2)
    assert np.isclose(result, expected, atol=1e-4).all().all()


def test_result_is_symmetric_with_sorted_labels():
    df = pd.DataFrame(
        {
            "a": ["c", "c", "a"],
            "b": ["a", "b", "b"],
            "v": [0.5, 0.4, 0.1],
        }
    )
    result = reconstruct_symmetric_matrix_from_long(df, "a", "b", "v")

    assert list(result.index) == ["a", "b", "c"]
    assert list(result.columns) == ["a", "b", "c"]
    assert np.allclose(result.values, result.values.T)


def test_lower_triangle_mirrored_from_upper():
    # Only upper-triangle pairs supplied.
    df = pd.DataFrame(
        {
            "a": ["a", "a", "b"],
            "b": ["b", "c", "c"],
            "v": [0.2, 0.3, 0.4],
        }
    )
    result = reconstruct_symmetric_matrix_from_long(df, "a", "b", "v")

    assert result.loc["b", "a"] == 0.2
    assert result.loc["c", "a"] == 0.3
    assert result.loc["c", "b"] == 0.4


def test_diagonal_value():
    df = pd.DataFrame({"a": ["x"], "b": ["y"], "v": [0.7]})

    default = reconstruct_symmetric_matrix_from_long(df, "a", "b", "v")
    assert list(np.diag(default.values)) == [1.0, 1.0]

    custom = reconstruct_symmetric_matrix_from_long(
        df, "a", "b", "v", diagonal=0.0
    )
    assert list(np.diag(custom.values)) == [0.0, 0.0]


def test_column_selectors_int_vs_str_equivalent():
    df = pd.DataFrame(
        {
            "a": ["a", "a", "b"],
            "b": ["b", "c", "c"],
            "v": [0.2, 0.3, 0.4],
        }
    )
    by_name = reconstruct_symmetric_matrix_from_long(df, "a", "b", "v")
    by_index = reconstruct_symmetric_matrix_from_long(df, 0, 1, 2)

    assert np.allclose(by_name.values, by_index.values)
    assert list(by_name.index) == list(by_index.index)


def test_generic_labels_and_values():
    # Non-"peptide" labels and a non-correlation value column.
    df = pd.DataFrame(
        {
            "src": ["CityA", "CityA", "CityB"],
            "dst": ["CityB", "CityC", "CityC"],
            "distance": [10.0, 25.0, 7.0],
        }
    )
    result = reconstruct_symmetric_matrix_from_long(
        df, "src", "dst", "distance", diagonal=0.0
    )
    assert result.loc["CityA", "CityB"] == 10.0
    assert result.loc["CityC", "CityA"] == 25.0
    assert result.loc["CityA", "CityA"] == 0.0


def test_single_pair_minimal_input():
    df = pd.DataFrame({"a": ["p"], "b": ["q"], "v": [0.9]})
    result = reconstruct_symmetric_matrix_from_long(df, "a", "b", "v")

    assert result.shape == (2, 2)
    assert result.loc["p", "q"] == 0.9
    assert result.loc["q", "p"] == 0.9


def test_missing_pair_raises():
    # Labels a, b, c present but the (b, c) pair has no value anywhere.
    df = pd.DataFrame(
        {
            "a": ["a", "a"],
            "b": ["b", "c"],
            "v": [0.1, 0.2],
        }
    )
    with pytest.raises(ValueError, match="No value found"):
        reconstruct_symmetric_matrix_from_long(df, "a", "b", "v")


def test_conflicting_values_raise():
    # (a, b) supplied as 0.1 and (b, a) supplied as 0.9.
    df = pd.DataFrame(
        {
            "a": ["a", "b"],
            "b": ["b", "a"],
            "v": [0.1, 0.9],
        }
    )
    with pytest.raises(ValueError, match="Conflicting values"):
        reconstruct_symmetric_matrix_from_long(df, "a", "b", "v")
