"""Tests for shared fractional-threshold resolvers."""

import numpy as np
import pytest

from proteopy.pp._parameter_resolvers import _resolve_min_fraction


@pytest.mark.parametrize(
    "min_fraction, expected",
    {0.19: 0, 0.20: 1, 0.21: 1}.items(),
)
def test_min_fraction_floors_at_integer_boundary(min_fraction, expected):
    assert _resolve_min_fraction(min_fraction, 5) == expected


def test_min_fraction_handles_endpoints_and_different_group_sizes():
    cases = {
        (0.0, 5): 0,
        (1.0, 5): 5,
        (0.5, 0): 0,
        (0.5, 3): 1,
        (0.6, 5): 3,
        (0.56, 25): 14,
    }
    for (min_fraction, n_items), expected in cases.items():
        assert _resolve_min_fraction(min_fraction, n_items) == expected


@pytest.mark.parametrize("value", [True, "0.5", None, np.float16(0.5)])
def test_min_fraction_rejects_invalid_types(value):
    with pytest.raises(TypeError, match="min_fraction"):
        _resolve_min_fraction(value, 5)


@pytest.mark.parametrize("value", [-0.1, 1.1, np.nan, np.inf, -np.inf])
def test_min_fraction_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="min_fraction"):
        _resolve_min_fraction(value, 5)


@pytest.mark.parametrize("value", [True, 5.0, "5"])
def test_min_fraction_rejects_invalid_item_count_types(value):
    with pytest.raises(TypeError, match="n_items"):
        _resolve_min_fraction(0.5, value)


def test_min_fraction_rejects_negative_item_counts():
    with pytest.raises(ValueError, match="n_items"):
        _resolve_min_fraction(0.5, -1)
