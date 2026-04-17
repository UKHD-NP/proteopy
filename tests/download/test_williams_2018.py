"""Tests for proteopy.download.williams_2018."""
import hashlib

import pandas as pd
import pytest

from proteopy.download import williams_2018


# -- Expected values -------------------------------------------------

_EXPECTED_INTENSITIES_HASH = (
    "021410ece8505f9ef1181a4f1bbb5cde"
    "c884011eba53a77e72cc6d6f51f1a531"
)
_EXPECTED_VAR_HASH = (
    "827b32fd2962cd18a7a990d56eab0e64"
    "daa2a244b6226fe2d242106f185b2161"
)
_EXPECTED_SAMPLE_HASH = (
    "8cca98fa3a38df78b78912f3ef7daed5"
    "7f82902485d61d90db5a823c1ed4f031"
)

_EXPECTED_INTENSITIES_COLUMNS = [
    "sample_id", "peptide_id", "intensity",
]
_EXPECTED_VAR_COLUMNS = [
    "peptide_id", "protein_id", "gene_id",
]
_EXPECTED_SAMPLE_COLUMNS = [
    "sample_id", "tissue", "mouse_id",
]
_EXPECTED_TISSUES = [
    "BAT", "Brain", "Heart", "Liver", "Quad",
]


# -- Helpers ---------------------------------------------------------

def _make_paths(tmp_path, ext=".tsv"):
    return (
        tmp_path / f"intensities{ext}",
        tmp_path / f"var_annotation{ext}",
        tmp_path / f"sample_annotation{ext}",
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# -- Content tests ---------------------------------------------------

class TestWilliams2018Download:
    """Verify downloaded file content and structure."""

    @pytest.fixture(scope="class")
    def paths(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("williams_dl")
        p = _make_paths(tmp)
        williams_2018(*p)
        return p

    def test_files_created(self, paths):
        for p in paths:
            assert p.exists()

    def test_intensities_columns(self, paths):
        df = pd.read_csv(paths[0], sep="\t", nrows=0)
        assert (
            df.columns.tolist()
            == _EXPECTED_INTENSITIES_COLUMNS
        )

    def test_var_annotation_columns(self, paths):
        df = pd.read_csv(paths[1], sep="\t", nrows=0)
        assert df.columns.tolist() == _EXPECTED_VAR_COLUMNS

    def test_sample_annotation_columns(self, paths):
        df = pd.read_csv(paths[2], sep="\t", nrows=0)
        assert (
            df.columns.tolist()
            == _EXPECTED_SAMPLE_COLUMNS
        )

    def test_intensities_hash(self, paths):
        assert (
            _sha256(paths[0].read_bytes())
            == _EXPECTED_INTENSITIES_HASH
        )

    def test_var_annotation_hash(self, paths):
        assert (
            _sha256(paths[1].read_bytes())
            == _EXPECTED_VAR_HASH
        )

    def test_sample_annotation_hash(self, paths):
        assert (
            _sha256(paths[2].read_bytes())
            == _EXPECTED_SAMPLE_HASH
        )

    def test_sample_count(self, paths):
        df = pd.read_csv(paths[2], sep="\t")
        assert len(df) == 40

    def test_tissues_in_file(self, paths):
        df = pd.read_csv(paths[2], sep="\t")
        assert (
            sorted(df["tissue"].unique())
            == _EXPECTED_TISSUES
        )


# -- Separator tests -------------------------------------------------

class TestWilliams2018DownloadSeparator:
    """Verify separator auto-detection from file extension."""

    def test_csv_extension_uses_comma(self, tmp_path):
        p = _make_paths(tmp_path, ext=".csv")
        williams_2018(*p)
        df = pd.read_csv(p[0], sep=",", nrows=0)
        assert (
            df.columns.tolist()
            == _EXPECTED_INTENSITIES_COLUMNS
        )

    def test_tsv_extension_uses_tab(self, tmp_path):
        p = _make_paths(tmp_path, ext=".tsv")
        williams_2018(*p)
        df = pd.read_csv(p[0], sep="\t", nrows=0)
        assert (
            df.columns.tolist()
            == _EXPECTED_INTENSITIES_COLUMNS
        )


# -- Error tests -----------------------------------------------------

class TestWilliams2018DownloadErrors:
    """Verify input validation and error handling."""

    def test_file_exists_error(self, tmp_path):
        p = _make_paths(tmp_path)
        williams_2018(*p)
        with pytest.raises(FileExistsError):
            williams_2018(*p)

    def test_force_overwrites(self, tmp_path):
        p = _make_paths(tmp_path)
        williams_2018(*p)
        williams_2018(*p, force=True)
        for path in p:
            assert path.exists()

    def test_overlapping_paths_raises(self, tmp_path):
        same = tmp_path / "same.tsv"
        with pytest.raises(ValueError, match="same path"):
            williams_2018(same, same, tmp_path / "other.tsv")

    def test_invalid_path_type_raises(self, tmp_path):
        with pytest.raises(
            TypeError, match="must be str or Path",
        ):
            williams_2018(
                123,
                tmp_path / "v.tsv",
                tmp_path / "s.tsv",
            )

    def test_invalid_sep_type_raises(self, tmp_path):
        p = _make_paths(tmp_path)
        with pytest.raises(
            TypeError, match="sep must be str or None",
        ):
            williams_2018(*p, sep=123)

    def test_fill_na_bool_raises(self, tmp_path):
        p = _make_paths(tmp_path)
        with pytest.raises(
            TypeError, match="fill_na must be",
        ):
            williams_2018(*p, fill_na=True)

    def test_force_non_bool_raises(self, tmp_path):
        p = _make_paths(tmp_path)
        with pytest.raises(
            TypeError, match="force must be bool",
        ):
            williams_2018(*p, force=1)
