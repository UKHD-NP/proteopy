"""Tests for :mod:`proteopy.utils.slot_parsers`."""
import anndata as ad
import numpy as np
import pandas as pd
import pytest

from proteopy.tl.copf import pairwise_var_correlations
from proteopy.utils.slot_parsers import (
    find_pairwise_var_correlations_keys,
    parse_pairwise_var_correlations_result,
    resolve_pairwise_var_correlations_key,
)


def _make_peptide_adata():
    """Build a small peptide-level AnnData with two proteins."""
    rng = np.random.default_rng(0)
    intensities = rng.standard_normal((6, 4))
    peptide_ids = ["p1", "p2", "p3", "p4"]
    protein_ids = ["A", "A", "B", "B"]
    obs_names = [f"S{i}" for i in range(6)]
    var = pd.DataFrame(
        {"peptide_id": peptide_ids, "protein_id": protein_ids},
        index=peptide_ids,
    )
    obs = pd.DataFrame({"sample_id": obs_names}, index=obs_names)
    return ad.AnnData(X=intensities, obs=obs, var=var)


def test_find_returns_empty_when_no_slot_present():
    adata = _make_peptide_adata()
    assert find_pairwise_var_correlations_keys(adata) == []


def test_find_returns_single_key_after_one_run():
    adata = _make_peptide_adata()
    pairwise_var_correlations(adata, group_by="protein_id")
    keys = find_pairwise_var_correlations_keys(adata)
    assert len(keys) == 1
    assert keys[0].startswith("pairwise_correlations;")


def test_find_returns_multiple_keys_for_multiple_runs():
    adata = _make_peptide_adata()
    pairwise_var_correlations(adata, group_by="protein_id")
    # Second run with a custom key produces a distinct slot that
    # still matches the pairwise_var_correlations key format.
    pairwise_var_correlations(
        adata,
        group_by="protein_id",
        key_added="pairwise_correlations;protein_id;alt;;",
    )
    keys = find_pairwise_var_correlations_keys(adata)
    assert len(keys) == 2


def test_find_ignores_unrelated_keys():
    adata = _make_peptide_adata()
    adata.uns["some_other_key"] = pd.DataFrame({"a": [1]})
    adata.uns["pairwise_correlations;but;wrong;value"] = 42
    assert find_pairwise_var_correlations_keys(adata) == []


def test_resolve_returns_single_match_when_corrs_key_is_none():
    adata = _make_peptide_adata()
    pairwise_var_correlations(adata, group_by="protein_id")
    resolved = resolve_pairwise_var_correlations_key(adata, None)
    assert resolved in adata.uns


def test_resolve_passes_explicit_key_through():
    adata = _make_peptide_adata()
    pairwise_var_correlations(adata, group_by="protein_id")
    explicit = "some;arbitrary;value"
    assert resolve_pairwise_var_correlations_key(adata, explicit) == explicit


def test_resolve_raises_when_no_slot_present():
    adata = _make_peptide_adata()
    with pytest.raises(ValueError, match="No pairwise correlation slot"):
        resolve_pairwise_var_correlations_key(adata, None)


def test_resolve_raises_when_multiple_slots_present():
    adata = _make_peptide_adata()
    pairwise_var_correlations(adata, group_by="protein_id")
    # Second run with a custom key produces a distinct slot that
    # still matches the pairwise_var_correlations key format.
    pairwise_var_correlations(
        adata,
        group_by="protein_id",
        key_added="pairwise_correlations;protein_id;alt;;",
    )
    with pytest.raises(
        ValueError,
        match="Multiple pairwise correlation slots",
    ):
        resolve_pairwise_var_correlations_key(adata, None)


def test_parser_roundtrip_grouped_frame():
    adata = _make_peptide_adata()
    pairwise_var_correlations(adata, group_by="protein_id")
    key = find_pairwise_var_correlations_keys(adata)[0]
    _, sym = parse_pairwise_var_correlations_result(
        adata, corrs_key=key, group_id="A"
    )
    assert sym.shape == (2, 2)
    assert list(sym.index) == ["p1", "p2"]


def test_parser_raises_when_group_id_missing_for_grouped_frame():
    adata = _make_peptide_adata()
    pairwise_var_correlations(adata, group_by="protein_id")
    key = find_pairwise_var_correlations_keys(adata)[0]
    with pytest.raises(ValueError, match="grouped; provide a group_id"):
        parse_pairwise_var_correlations_result(adata, corrs_key=key)
