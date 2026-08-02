"""Tests for proteopy.datasets.williams_2018."""

import hashlib

import anndata as ad
import numpy as np
import pytest

from proteopy.datasets import williams_2018


# -- Expected values -------------------------------------------------

_EXPECTED_SHAPE = (40, 32690)

_EXPECTED_X_HASH = (
    "2f319804269569ce370f2f3477e3d957" "9118e389a07e27d06dfc2a7a798dfbc3"
)

# Cell-state census of .X. These are pinned individually as well as via
# the hash, because the hash says only THAT the matrix changed, while
# these say WHAT it should contain -- and the distinction between a
# zero and a missing value is the thing most easily broken here.
_EXPECTED_N_NAN = 3584
_EXPECTED_N_ZERO = 13547
_EXPECTED_N_POSITIVE = 1290469
_EXPECTED_OBS_NAMES_HASH = (
    "4a510a6124dd8b917c42f4270353aee2" "0a11fd97d0bbd38200319af5f6b602ee"
)
_EXPECTED_VAR_NAMES_HASH = (
    "35bac1a175466852feb110553409be8c" "f56c6564aaa73a75e4dc910b1cbb2d0e"
)

_EXPECTED_OBS_COLUMNS = ["tissue", "mouse_id", "sample_id"]
_EXPECTED_VAR_COLUMNS = ["protein_id", "gene_id", "peptide_id"]
_EXPECTED_TISSUES = ["BAT", "Brain", "Heart", "Liver", "Quad"]
_EXPECTED_MOUSE_IDS = [
    "101",
    "45",
    "66",
    "68",
    "73",
    "80",
    "C57",
    "DBA",
]


# -- Fixtures --------------------------------------------------------


@pytest.fixture(scope="module")
def adata():
    """Load williams_2018 dataset once for all tests."""
    return williams_2018()


# -- Helpers ---------------------------------------------------------


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _encode_index(index) -> bytes:
    """Encode a pandas Index for hashing.

    Joins the entries with ``","`` and UTF-8 encodes the result so
    that the value is a deterministic byte string that can be fed
    into ``hashlib.sha256``.
    """
    return ",".join(index).encode()


# -- Content tests ---------------------------------------------------


class TestWilliams2018:
    """Verify structure and content of the williams_2018 dataset."""

    def test_returns_anndata(self, adata):
        assert isinstance(adata, ad.AnnData)

    def test_shape(self, adata):
        assert adata.shape == _EXPECTED_SHAPE

    def test_obs_columns(self, adata):
        assert adata.obs.columns.tolist() == _EXPECTED_OBS_COLUMNS

    def test_var_columns(self, adata):
        assert adata.var.columns.tolist() == _EXPECTED_VAR_COLUMNS

    def test_tissues(self, adata):
        assert sorted(adata.obs["tissue"].unique()) == _EXPECTED_TISSUES

    def test_mouse_ids(self, adata):
        assert sorted(adata.obs["mouse_id"].unique()) == _EXPECTED_MOUSE_IDS

    def test_eight_mice_per_tissue(self, adata):
        counts = adata.obs.groupby("tissue").size()
        assert (counts == 8).all()

    def test_obs_names_match_sample_id(self, adata):
        assert list(adata.obs_names) == list(adata.obs["sample_id"])

    def test_var_names_match_peptide_id(self, adata):
        assert list(adata.var_names) == list(adata.var["peptide_id"])

    def test_x_dtype(self, adata):
        assert adata.X.dtype == np.float64

    def test_x_contains_nan(self, adata):
        assert np.isnan(adata.X).any()

    def test_x_hash(self, adata):
        h = _sha256(adata.X.tobytes())
        assert h == _EXPECTED_X_HASH

    def test_obs_names_hash(self, adata):
        h = _sha256(_encode_index(adata.obs_names))
        assert h == _EXPECTED_OBS_NAMES_HASH

    def test_var_names_hash(self, adata):
        h = _sha256(_encode_index(adata.var_names))
        assert h == _EXPECTED_VAR_NAMES_HASH

    def test_fill_na_zero_removes_nan(self):
        result = williams_2018(fill_na=0)
        assert not np.isnan(result.X).any()

    def test_fill_na_string_raises(self):
        with pytest.raises(TypeError, match="fill_na must be"):
            williams_2018(fill_na="0")

    def test_zeros_are_preserved_not_coerced_to_nan(self, adata):
        """A zero is a measurement and must survive as 0.0."""
        assert (adata.X == 0).sum() == _EXPECTED_N_ZERO

    def test_nan_count(self, adata):
        assert np.isnan(adata.X).sum() == _EXPECTED_N_NAN

    def test_positive_count(self, adata):
        assert (adata.X > 0).sum() == _EXPECTED_N_POSITIVE

    def test_cell_states_are_exhaustive(self, adata):
        """No negative intensities, and the three states tile .X."""
        assert (adata.X < 0).sum() == 0
        assert (
            _EXPECTED_N_POSITIVE + _EXPECTED_N_ZERO + _EXPECTED_N_NAN
            == adata.X.size
        )

    def test_zero_to_na_converts_zeros(self):
        """Opt-in flag for callers who want zeros treated as missing."""
        result = williams_2018(zero_to_na=True)
        assert (result.X == 0).sum() == 0
        assert np.isnan(result.X).sum() == _EXPECTED_N_NAN + _EXPECTED_N_ZERO
        assert (result.X > 0).sum() == _EXPECTED_N_POSITIVE

    def test_zero_to_na_with_fill_na_raises(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            williams_2018(zero_to_na=True, fill_na=0)

    def test_zero_to_na_non_bool_raises(self):
        with pytest.raises(TypeError, match="zero_to_na must be bool"):
            williams_2018(zero_to_na="yes")
