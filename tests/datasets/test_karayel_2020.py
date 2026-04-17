"""Tests for proteopy.datasets.karayel_2020."""
import hashlib

import anndata as ad
import numpy as np
import pytest

from proteopy.datasets import karayel_2020


# -- Expected values -------------------------------------------------

_EXPECTED_SHAPE = (20, 7758)

_EXPECTED_X_HASH = (
    "eb0692166e44df0d32558495a5bcd44e"
    "2bbcb2c8a46be2b8ba468f7a552f3c0d"
)
_EXPECTED_OBS_NAMES_HASH = (
    "fef7fd91a6e93d20b719f61c63098865"
    "bbd3f886dabfa46786cde09e520c0abe"
)
_EXPECTED_VAR_NAMES_HASH = (
    "17d3bd09174bad3544738f30ed2867c4"
    "bd431feccdf36515e5fae415110fc456"
)

_EXPECTED_OBS_COLUMNS = ["sample_id", "cell_type", "replicate"]
_EXPECTED_VAR_COLUMNS = ["protein_id", "gene_id"]
_EXPECTED_CELL_TYPES = [
    "LBaso", "Ortho", "Poly", "ProE&EBaso", "Progenitor",
]
_EXPECTED_REPLICATES = ["rep1", "rep2", "rep3", "rep4"]


# -- Fixtures --------------------------------------------------------

@pytest.fixture(scope="module")
def adata():
    """Load karayel_2020 dataset once for all tests."""
    return karayel_2020()


# -- Helpers ---------------------------------------------------------

def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# -- Content tests ---------------------------------------------------

class TestKarayel2020:
    """Verify structure and content of the karayel_2020 dataset."""

    def test_returns_anndata(self, adata):
        assert isinstance(adata, ad.AnnData)

    def test_shape(self, adata):
        assert adata.shape == _EXPECTED_SHAPE

    def test_obs_columns(self, adata):
        assert adata.obs.columns.tolist() == _EXPECTED_OBS_COLUMNS

    def test_var_columns(self, adata):
        assert adata.var.columns.tolist() == _EXPECTED_VAR_COLUMNS

    def test_cell_types(self, adata):
        assert (
            sorted(adata.obs["cell_type"].unique())
            == _EXPECTED_CELL_TYPES
        )

    def test_replicates(self, adata):
        assert (
            sorted(adata.obs["replicate"].unique())
            == _EXPECTED_REPLICATES
        )

    def test_four_replicates_per_cell_type(self, adata):
        counts = adata.obs.groupby("cell_type").size()
        assert (counts == 4).all()

    def test_obs_names_match_sample_id(self, adata):
        assert (
            list(adata.obs_names)
            == list(adata.obs["sample_id"])
        )

    def test_var_names_match_protein_id(self, adata):
        assert (
            list(adata.var_names)
            == list(adata.var["protein_id"])
        )

    def test_x_dtype(self, adata):
        assert adata.X.dtype == np.float64

    def test_x_contains_nan(self, adata):
        assert np.isnan(adata.X).any()

    def test_x_hash(self, adata):
        h = _sha256(
            np.nan_to_num(adata.X, nan=0.0).tobytes(),
        )
        assert h == _EXPECTED_X_HASH

    def test_obs_names_hash(self, adata):
        h = _sha256(
            ",".join(adata.obs_names).encode(),
        )
        assert h == _EXPECTED_OBS_NAMES_HASH

    def test_var_names_hash(self, adata):
        h = _sha256(
            ",".join(adata.var_names).encode(),
        )
        assert h == _EXPECTED_VAR_NAMES_HASH


# -- fill_na tests ---------------------------------------------------

class TestKarayel2020FillNa:
    """Verify fill_na parameter behaviour."""

    def test_fill_na_zero_removes_nan(self):
        result = karayel_2020(fill_na=0)
        assert not np.isnan(result.X).any()

    def test_fill_na_bool_raises(self):
        with pytest.raises(TypeError, match="fill_na must be"):
            karayel_2020(fill_na=True)

    def test_fill_na_string_raises(self):
        with pytest.raises(TypeError, match="fill_na must be"):
            karayel_2020(fill_na="0")
